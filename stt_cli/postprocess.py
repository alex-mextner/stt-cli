"""postprocess — let a language model fix what the speech model got wrong, and summarize it.

TWO PASSES, ONE PRINCIPLE: NEVER LOSE THE ORIGINAL
    Correction rewrites segment text. That is a genuinely risky operation — a language model
    asked to "fix" a transcript will happily smooth away a real word it did not expect. So
    whenever a segment is rewritten, the speech model's original reading is pushed into the
    segment's variants first. The raw transcript is therefore always recoverable from the
    archived JSON, and ``--variant raw`` can print it back verbatim.

WHY THE MODEL IS SHOWN CONFIDENCE AND ALTERNATIVES
    A correction pass over bare text is guesswork: the model cannot tell which words were
    heard clearly and which were a coin flip, so it edits uniformly and its mistakes land
    anywhere. Given per-segment confidence and the alternative decodings, it can do the thing
    that is actually useful — leave the confident parts alone and choose between real
    candidates where the speech model was unsure. That is why enabling correction implicitly
    enables variant gathering (see :func:`stt_cli.variants.plan_from_settings`).

SUMMARIZATION IS MAP-REDUCE, BECAUSE MEETINGS ARE LONG
    An hour of talk does not fit in one comfortable prompt, and stuffing it in produces a
    summary that remembers the beginning and the end. Long transcripts are summarized in
    windows and the partial summaries merged, so the middle survives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import llm
from .jsonio import JsonDict, as_dicts, as_list, as_opt_float, as_str
from .models import Segment, Summary, SummarySection, Transcript, Variant
from .timestamps import clock

# One correction window. Small enough that the model keeps every segment in view and does
# not start paraphrasing, large enough that it has the surrounding context to fix a word.
WINDOW_SEGMENTS = 40
WINDOW_CHARS = 6000
# Above this, summarization goes map-reduce instead of one call.
SUMMARY_DIRECT_CHARS = 24_000
SUMMARY_WINDOW_CHARS = 16_000


@dataclass(slots=True)
class FixReport:
    changed: int = 0
    windows: int = 0
    failures: int = 0
    tool: str = ""

    def summary(self) -> str:
        if not self.windows:
            return "correction pass did not run"
        detail = f"{self.changed} segment(s) rewritten by {self.tool} over {self.windows} window(s)"
        return detail + (f"; {self.failures} window(s) failed" if self.failures else "")


# ── correction ────────────────────────────────────────────────────────────────
_FIX_INSTRUCTIONS = """\
You are correcting an automatic speech-to-text transcript. Work ONLY from the evidence given.

For each segment you receive:
  "i"          its index — you must return the same indices, unchanged
  "text"       what the speech model decoded
  "confidence" 0..1, how sure the speech model was of that text
  "alts"       other readings the speech model produced for the same audio, if any

Rules:
1. Keep the original language. Never translate.
2. Leave high-confidence text alone. Fix obvious mis-hearings, wrong word boundaries,
   punctuation and casing. If an "alts" entry is clearly the correct reading, use it.
3. Do NOT merge, split, reorder or delete segments. Do not add commentary.
4. Do NOT invent content. If a segment is unintelligible, return it unchanged.
5. Where you are genuinely unsure between readings, put your best in "text" and the other
   plausible readings in "alts", and lower "confidence" to reflect your uncertainty.

Reply with ONE JSON object and nothing else:
{"segments":[{"i":0,"text":"...","confidence":0.0,"alts":["..."]}]}
"""


async def correct(
    transcript: Transcript, *, tool_name: str, language: str | None = None
) -> FixReport:
    """Run the LLM correction pass over the whole transcript, window by window."""
    tool = llm.resolve(tool_name)
    report = FixReport(tool=tool.name)
    for window in _windows(transcript.segments):
        report.windows += 1
        payload = await llm.ask_json(tool, _fix_prompt(window, language))
        if payload is None:
            report.failures += 1
            continue
        report.changed += _apply_fixes(window, payload, transcript.engine.label())
    return report


def _windows(segments: list[Segment]) -> list[list[Segment]]:
    """Split the transcript into prompt-sized windows of consecutive segments."""
    windows: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        if current and (len(current) >= WINDOW_SEGMENTS or size + len(segment.text) > WINDOW_CHARS):
            windows.append(current)
            current, size = [], 0
        current.append(segment)
        size += len(segment.text)
    if current:
        windows.append(current)
    return windows


def _fix_prompt(window: list[Segment], language: str | None) -> str:
    items = [
        {
            "i": index,
            "text": segment.text,
            "confidence": round(segment.confidence, 3) if segment.confidence is not None else None,
            "alts": [v.text for v in segment.variants],
        }
        for index, segment in enumerate(window)
    ]
    hint = f"\nThe language of this recording is: {language}.\n" if language else ""
    return (
        _FIX_INSTRUCTIONS
        + hint
        + "\nSegments:\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
        + "\n"
    )


def _apply_fixes(window: list[Segment], payload: JsonDict, asr_label: str) -> int:
    """Write the model's corrections back, keeping the speech model's reading as a variant."""
    changed = 0
    for item in as_dicts(payload.get("segments")):
        index = item.get("i")
        if not isinstance(index, int) or not 0 <= index < len(window):
            continue
        segment = window[index]
        new_text = as_str(item.get("text")).strip()
        if new_text and new_text != segment.text:
            segment.variants.insert(
                0,
                Variant(
                    text=segment.text,
                    source=f"asr:{asr_label}",
                    kind="primary",
                    confidence=segment.confidence,
                ),
            )
            segment.text = new_text
            segment.flag("corrected")
            changed += 1
        _absorb_alts(segment, item.get("alts"))
        _absorb_confidence(segment, item.get("confidence"))
    return changed


def _absorb_alts(segment: Segment, alts: object) -> None:
    known = {segment.text.strip().lower(), *(v.text.strip().lower() for v in segment.variants)}
    for alt in as_list(alts):
        text = as_str(alt).strip()
        if text and text.lower() not in known:
            segment.variants.append(Variant(text=text, source="llm", kind="llm"))
            known.add(text.lower())


def _absorb_confidence(segment: Segment, value: object) -> None:
    """Take the model's own certainty as a floor-lowering signal, never as a promotion.

    A language model claiming high confidence about audio it never heard is worthless; a
    language model saying "I am unsure here" is real information. So the number can only
    move the segment's confidence down.
    """
    if not isinstance(value, (int, float)):
        return
    stated = max(0.0, min(1.0, float(value)))
    if segment.confidence is None or stated < segment.confidence:
        segment.confidence = stated
        if stated < 0.55:
            segment.flag("low-confidence")


# ── summarization ─────────────────────────────────────────────────────────────
_SUMMARY_INSTRUCTIONS = """\
You are summarizing a transcript of a real conversation. Be concrete and specific: name the
things that were actually discussed, not the fact that things were discussed.

Rules:
1. Write in the language of the transcript.
2. Every bullet must carry actual content. "They talked about the architecture" is useless;
   "Agreed to move frontend state into the agent process so the preview survives reloads" is
   what a reader needs.
3. Do not invent anything. If something was left unresolved, put it under "questions".
4. "start" on a section is the timestamp in SECONDS where that topic begins, taken from the
   [h:mm:ss] markers in the transcript. Omit it if you are unsure.

Reply with ONE JSON object and nothing else:
{"headline":"one sentence naming what this conversation was",
 "sections":[{"title":"topic","bullets":["..."],"start":0}],
 "decisions":["what was decided"],
 "actions":["who does what next"],
 "questions":["what was left open"]}
"""


async def summarize(transcript: Transcript, *, tool_name: str) -> Summary | None:
    """Produce a structured summary, in one call or by map-reduce for long recordings."""
    tool = llm.resolve(tool_name)
    body = _transcript_text(transcript.segments)
    if len(body) <= SUMMARY_DIRECT_CHARS:
        payload = await llm.ask_json(tool, _SUMMARY_INSTRUCTIONS + "\nTranscript:\n" + body)
        return _to_summary(payload, tool.name) if payload else None
    return await _map_reduce(tool, transcript)


async def _map_reduce(tool: llm.Tool, transcript: Transcript) -> Summary | None:
    """Summarize each part of a long recording, then merge the parts into one summary."""
    parts: list[Summary] = []
    for chunk in _char_windows(transcript.segments, SUMMARY_WINDOW_CHARS):
        payload = await llm.ask_json(tool, _SUMMARY_INSTRUCTIONS + "\nTranscript:\n" + chunk)
        if payload:
            parts.append(_to_summary(payload, tool.name))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    merged = json.dumps([p.to_dict() for p in parts], ensure_ascii=False, indent=1)
    prompt = (
        _SUMMARY_INSTRUCTIONS
        + "\nThese are partial summaries of consecutive parts of ONE conversation, in order."
        + " Merge them into a single coherent summary: fold duplicate topics together, keep"
        + " every distinct decision and action, and preserve the earliest 'start' for each"
        + " merged topic.\n\nPartial summaries:\n"
        + merged
        + "\n"
    )
    final = await llm.ask_json(tool, prompt)
    return _to_summary(final, tool.name) if final else _concat(parts, tool.name)


def _concat(parts: list[Summary], source: str) -> Summary:
    """Fallback merge when the reduce call fails: keep everything rather than nothing."""
    return Summary(
        headline=parts[0].headline,
        sections=[section for part in parts for section in part.sections],
        decisions=list(dict.fromkeys(d for part in parts for d in part.decisions)),
        actions=list(dict.fromkeys(a for part in parts for a in part.actions)),
        questions=list(dict.fromkeys(q for part in parts for q in part.questions)),
        source=f"{source} (partial summaries concatenated; merge call failed)",
    )


def _line(segment: Segment) -> str:
    """One transcript line as the language model sees it: time, optional speaker, text."""
    who = f"{segment.speaker}: " if segment.speaker else ""
    return f"[{clock(segment.start)}] {who}{segment.text}"


def _transcript_text(segments: list[Segment]) -> str:
    return "\n".join(_line(s) for s in segments if s.text.strip())


def _char_windows(segments: list[Segment], limit: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for segment in segments:
        if not segment.text.strip():
            continue
        line = _line(segment)
        if current and size + len(line) > limit:
            windows.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        windows.append("\n".join(current))
    return windows


def _to_summary(payload: JsonDict | None, source: str) -> Summary:
    payload = payload or {}
    sections = [
        SummarySection(
            title=as_str(item.get("title")).strip(),
            bullets=[b for b in (as_str(x).strip() for x in as_list(item.get("bullets"))) if b],
            start=as_opt_float(item.get("start")),
        )
        for item in as_dicts(payload.get("sections"))
    ]
    return Summary(
        headline=as_str(payload.get("headline")).strip(),
        sections=[s for s in sections if s.title or s.bullets],
        decisions=_strings(payload.get("decisions")),
        actions=_strings(payload.get("actions")),
        questions=_strings(payload.get("questions")),
        source=source,
    )


def _strings(value: object) -> list[str]:
    return [text for text in (as_str(item).strip() for item in as_list(value)) if text]
