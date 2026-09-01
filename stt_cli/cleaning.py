"""cleaning — remove what the model invented, and keep a record of what was removed.

This is the second line of defence. The first is :mod:`stt_cli.vad`, which never shows the
model the silence it would hallucinate over; whatever still gets through lands here.

THREE DISTINCT FAILURES, THREE DIFFERENT FIXES
    *Filler phrases* — subtitle credits and channel plugs the model reaches for when the
    audio says nothing. Matched against :mod:`stt_cli.phrases`, with the ambiguous ones
    requiring corroborating evidence (low confidence, or overlapping a silent stretch).

    *Repetition loops* — the decoder falls into a cycle and emits the same phrase until the
    chunk ends. Detected structurally (a repeated n-gram), not by a word list, so it works
    in any language and on phrases nobody has ever seen before.

    *Cross-segment echo* — the same short line repeated across many consecutive segments,
    which is the same failure one level up.

NOTHING IS DELETED SILENTLY
    Every removal is a flag on the segment plus a line in the report. With ``--no-clean``
    the flags are still computed and nothing is dropped, so you can always see exactly what
    the filter would have taken and judge it for yourself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Segment, SpeechSpan
from .phrases import compile_all, load_user_patterns

# A phrase repeated this many times in a row is a decoder loop, not emphasis. Three is
# deliberately forgiving: people do say "no, no, no".
DEFAULT_MAX_REPEATS = 3
# Only short lines can be "echoed" across segments; a repeated long sentence is far more
# likely to be a genuine re-statement (or a chorus) than a decoder fault.
ECHO_MAX_CHARS = 60
ECHO_MIN_RUN = 3


@dataclass(slots=True)
class CleanReport:
    """What the filter did, in enough detail to argue with."""

    dropped: list[tuple[float, str, str]] = field(default_factory=list)  # (start, reason, text)
    collapsed: list[tuple[float, str]] = field(default_factory=list)  # (start, text)
    flagged_low: int = 0

    def summary(self) -> str:
        if not self.dropped and not self.collapsed:
            return "nothing removed"
        parts = []
        if self.dropped:
            parts.append(f"{len(self.dropped)} segment(s) dropped")
        if self.collapsed:
            parts.append(f"{len(self.collapsed)} loop(s) collapsed")
        return ", ".join(parts)

    def detail(self, limit: int = 20) -> list[str]:
        lines = [f"  dropped [{t:8.2f}] ({why}) {text[:70]!r}" for t, why, text in self.dropped]
        lines += [f"  collapsed [{t:8.2f}] {text[:70]!r}" for t, text in self.collapsed]
        return lines[:limit]


def clean(
    segments: list[Segment],
    *,
    speech_spans: list[SpeechSpan],
    home: Path,
    apply: bool = True,
    strict: bool = False,
    max_repeats: int = DEFAULT_MAX_REPEATS,
    confidence_floor: float = 0.55,
) -> tuple[list[Segment], CleanReport]:
    """Flag (and optionally remove) invented text. Returns the kept segments and a report."""
    always, contextual = compile_all(load_user_patterns(home))
    report = CleanReport()

    for segment in segments:
        segment.text = collapse_loops(segment.text, max_repeats, segment, report)
        _mark(segment, speech_spans, always, contextual, strict, confidence_floor, report)

    _mark_echoes(segments, report)
    if not apply:
        return segments, report

    kept: list[Segment] = []
    for segment in segments:
        reason = _drop_reason(segment)
        if reason:
            report.dropped.append((segment.start, reason, segment.text))
        else:
            kept.append(segment)
    report.flagged_low = sum(1 for s in kept if "low-confidence" in s.flags)
    return kept, report


def _drop_reason(segment: Segment) -> str | None:
    for flag in ("hallucination", "empty", "loop-echo"):
        if flag in segment.flags:
            return flag
    return None


def _mark(
    segment: Segment,
    spans: list[SpeechSpan],
    always: list[re.Pattern[str]],
    contextual: list[re.Pattern[str]],
    strict: bool,
    floor: float,
    report: CleanReport,
) -> None:
    """Attach every flag this segment has earned. Never deletes; the caller decides that."""
    text = segment.text.strip()
    if not text:
        segment.flag("empty")
        return
    if segment.confidence is not None and segment.confidence < floor:
        segment.flag("low-confidence")
    if _silence_overlap(segment, spans) > 0.6:
        segment.flag("silence")
    if any(p.search(text) for p in always):
        segment.flag("hallucination")
        return
    if any(p.search(text) for p in contextual) and _contextually_suspect(segment, strict):
        segment.flag("hallucination")


def _contextually_suspect(segment: Segment, strict: bool) -> bool:
    """Is there corroborating evidence that this ordinary-looking phrase was invented?

    In ``strict`` mode the phrase list alone is enough. Otherwise the segment must also look
    wrong: the decoder was unsure of it, or it sits over audio the detector called silence.
    """
    if strict:
        return True
    return "low-confidence" in segment.flags or "silence" in segment.flags


def _silence_overlap(segment: Segment, spans: list[SpeechSpan]) -> float:
    """Fraction of this segment that lies OUTSIDE every detected speech span."""
    if not spans or segment.duration <= 0:
        return 0.0
    covered = 0.0
    for span in spans:
        overlap = min(segment.end, span.end) - max(segment.start, span.start)
        if overlap > 0:
            covered += overlap
    return max(0.0, 1.0 - covered / segment.duration)


# One or more words, repeated back to back. Written over word sequences rather than
# characters so it catches "да да да" and "и так далее и так далее и так далее" alike,
# in any language, without a dictionary.
def collapse_loops(text: str, max_repeats: int, segment: Segment, report: CleanReport) -> str:
    """Squash a phrase repeated more than ``max_repeats`` times back down to one copy."""
    words = text.split()
    if len(words) < 2 * max_repeats:
        return text
    for size in range(1, min(8, len(words) // max_repeats) + 1):
        collapsed = _collapse_ngram(words, size, max_repeats)
        if collapsed is not None:
            segment.flag("loop")
            report.collapsed.append((segment.start, text))
            return " ".join(collapsed)
    return text


def _collapse_ngram(words: list[str], size: int, max_repeats: int) -> list[str] | None:
    """Rewrite runs of an identical ``size``-word group; ``None`` when nothing repeated."""
    out: list[str] = []
    index = 0
    changed = False
    while index < len(words):
        group = words[index : index + size]
        if len(group) < size:
            out.extend(words[index:])
            break
        run = 1
        while words[index + run * size : index + (run + 1) * size] == group:
            run += 1
        if run > max_repeats:
            out.extend(group)
            changed = True
        else:
            out.extend(words[index : index + run * size])
        index += run * size
    return out if changed else None


def _mark_echoes(segments: list[Segment], report: CleanReport) -> None:
    """Flag runs of consecutive segments carrying the same short line — a loop one level up."""
    run_start = 0
    for index in range(1, len(segments) + 1):
        same = index < len(segments) and _norm(segments[index].text) == _norm(
            segments[run_start].text
        )
        if same:
            continue
        run = segments[run_start:index]
        if len(run) >= ECHO_MIN_RUN and _norm(run[0].text) and len(run[0].text) <= ECHO_MAX_CHARS:
            for segment in run[1:]:
                segment.flag("loop-echo")
            report.collapsed.append((run[0].start, f"{run[0].text} (x{len(run)})"))
        run_start = index


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.strip().lower()).strip()
