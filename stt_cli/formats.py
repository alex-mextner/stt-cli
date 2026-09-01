"""formats — render one transcript into whatever the reader actually needs.

Every renderer reads the same :class:`~stt_cli.models.Transcript`, so a run is transcribed
once and can be re-rendered from the archive forever: change your mind about timestamps, ask
for subtitles a month later, get both without touching the GPU again.

WHAT THE OPTIONS MEAN
    Variants and confidence are always *computed* when they were asked for (and always when
    the LLM correction pass runs, because that pass needs them). Whether a reader sees them
    is a separate decision, made here. Showing a variant without its confidence would be
    worse than useless — a reader cannot weigh an alternative they have no measure of — so
    turning variants on turns confidence on with it.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace

from ._errors import unknown_item
from .config import DEFAULT_ALL, FORMATS
from .models import Segment, Summary, Transcript
from .timestamps import Stamper, srt_time, vtt_time


@dataclass(slots=True)
class RenderOptions:
    show_variants: bool = False
    show_flags: bool = False
    # Which reading to print when the LLM correction pass rewrote a segment. The speech
    # model's original is never discarded — it is kept as the segment's ``primary`` variant —
    # so "actually, give me the raw transcript" costs a re-render, not a re-run.
    text_variant: str = "fixed"  # fixed | raw | both
    # Speaker labels and summaries persist on an archived transcript, so a run that did not
    # ask for them can still find them there. Showing them anyway would mean `stt rec.m4a
    # -f srt` silently emitting "S1:" prefixes because someone once passed --diarize.
    show_speakers: bool = True
    show_summary: bool = True

    @property
    def show_confidence(self) -> bool:
        # Variants are only interpretable next to the numbers that rank them.
        return self.show_variants


def expand(names: Iterable[str]) -> list[str]:
    """Resolve the ``--format`` list, expanding ``all`` and rejecting unknown names."""
    out: list[str] = []
    for name in names:
        if name == "all":
            out.extend(DEFAULT_ALL)
        elif name in FORMATS:
            out.append(name)
        else:
            raise unknown_item("format", name, ["all", *FORMATS])
    return list(dict.fromkeys(out))


def extension(fmt: str) -> str:
    return {"speakers": "speakers.txt", "summary": "summary.md", "md": "md"}.get(fmt, fmt)


def render(fmt: str, transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    renderers = {
        "txt": render_txt,
        "md": render_md,
        "json": render_json,
        "srt": render_srt,
        "vtt": render_vtt,
        "csv": render_csv,
        "tsv": render_tsv,
        "speakers": render_speakers,
        "summary": render_summary,
    }
    renderer = renderers.get(fmt)
    if renderer is None:
        raise unknown_item("format", fmt, sorted(renderers))
    return renderer(transcript, stamper, options)


# ── plain text ────────────────────────────────────────────────────────────────
def render_txt(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        lines.append(_line(segment, stamper, options))
        lines.extend(_variant_lines(segment, options))
    return "\n".join(lines) + "\n"


def _line(segment: Segment, stamper: Stamper, options: RenderOptions) -> str:
    parts: list[str] = []
    if stamper.enabled:
        parts.append(f"[{stamper.at(segment.start)}]")
    if segment.speaker and options.show_speakers:
        parts.append(f"{segment.speaker}:")
    if options.show_confidence and segment.confidence is not None:
        parts.append(f"({segment.confidence:.2f})")
    if options.show_flags and segment.flags:
        parts.append(f"<{','.join(segment.flags)}>")
    parts.append(chosen_text(segment, options))
    return " ".join(parts)


def chosen_text(segment: Segment, options: RenderOptions) -> str:
    """The reading to print, honouring ``--text raw|fixed|both``."""
    original = _original(segment)
    if options.text_variant == "raw" and original is not None:
        return original
    if options.text_variant == "both" and original is not None and original != segment.text:
        return f"{segment.text}   |raw| {original}"
    return segment.text


def _original(segment: Segment) -> str | None:
    """The speech model's own reading, kept as a variant when the LLM pass rewrote the text."""
    return next((v.text for v in segment.variants if v.kind == "primary"), None)


def _variant_lines(segment: Segment, options: RenderOptions) -> list[str]:
    if not options.show_variants or not segment.variants:
        return []
    lines = []
    for variant in segment.variants:
        score = f"{variant.confidence:.2f}" if variant.confidence is not None else "--"
        lines.append(f"    ├ alt ({score}, {variant.source}): {variant.text}")
    return lines


# ── markdown ──────────────────────────────────────────────────────────────────
def render_md(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    from pathlib import Path

    media = transcript.media
    head = [
        f"# {Path(media.path).stem}",
        "",
        f"- **Source**: `{media.path}`",
        f"- **Duration**: {media.duration / 60:.1f} min",
        f"- **Engine**: {transcript.engine.label()} (language: {transcript.language or 'auto'})",
        f"- **Voice activity**: {transcript.engine.vad}",
    ]
    if media.recorded_at:
        head.append(
            f"- **Recorded**: {media.recorded_at:%Y-%m-%d %H:%M:%S} ({media.recorded_at_source})"
        )
    if stamper.describe_base():
        head.append(f"- **Timestamps**: {stamper.describe_base()}")
    for warning in transcript.warnings:
        head.append(f"- **Note**: {warning}")
    body = [""]
    if transcript.summary and options.show_summary:
        body += [render_summary(transcript, stamper, options).rstrip(), ""]
    body += ["## Transcript", ""]
    for segment in transcript.segments:
        body.append(_line(segment, stamper, options))
        body.extend(_variant_lines(segment, options))
        body.append("")
    return "\n".join([*head, *body]).rstrip() + "\n"


# ── machine-readable ──────────────────────────────────────────────────────────
def render_json(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n"


def render_srt(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        text = _cue_text(segment, options)
        blocks.append(f"{index}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{text}\n")
    return "\n".join(blocks)


def render_vtt(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    blocks = ["WEBVTT", ""]
    for segment in transcript.segments:
        speaker = segment.speaker if options.show_speakers else None
        body = chosen_text(segment, options)
        text = f"<v {speaker}>{body}" if speaker else body
        blocks.append(f"{vtt_time(segment.start)} --> {vtt_time(segment.end)}\n{text}\n")
    return "\n".join(blocks)


def _cue_text(segment: Segment, options: RenderOptions) -> str:
    """One subtitle cue: the chosen reading, prefixed with the speaker when asked for."""
    body = chosen_text(segment, options)
    if segment.speaker and options.show_speakers:
        return f"{segment.speaker}: {body}"
    return body


def render_csv(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    return _delimited(transcript, stamper, options, delimiter=",")


def render_tsv(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    return _delimited(transcript, stamper, options, delimiter="\t")


def _delimited(
    transcript: Transcript, stamper: Stamper, options: RenderOptions, *, delimiter: str
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(["start", "end", "at", "speaker", "confidence", "flags", "text"])
    for segment in transcript.segments:
        writer.writerow(
            [
                f"{segment.start:.3f}",
                f"{segment.end:.3f}",
                stamper.at(segment.start),
                segment.speaker or "" if options.show_speakers else "",
                "" if segment.confidence is None else f"{segment.confidence:.4f}",
                "|".join(segment.flags),
                chosen_text(segment, options),
            ]
        )
    return buffer.getvalue()


# ── dialogue ──────────────────────────────────────────────────────────────────
def render_speakers(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    """One block per speaker turn — consecutive segments from one voice joined into a line.

    Without diarization every segment is the same unknown speaker, which would collapse the
    whole recording into a single wall of text; that case falls back to plain text so the
    format is never actively worse than what it replaces.
    """
    if not any(s.speaker for s in transcript.segments):
        return render_txt(transcript, stamper, options)
    # Asking for this format IS asking to see the speakers, whatever the run's flags said.
    options = replace(options, show_speakers=True)
    lines: list[str] = []
    for speaker, start, text in _turns(transcript.segments, options):
        prefix = f"[{stamper.at(start)}] " if stamper.enabled else ""
        lines.append(f"{prefix}{speaker}: {text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _turns(segments: list[Segment], options: RenderOptions) -> list[tuple[str, float, str]]:
    """Join consecutive segments from one voice into a single spoken turn."""
    turns: list[tuple[str, float, str]] = []
    for segment in segments:
        speaker = segment.speaker or "SPEAKER"
        body = chosen_text(segment, options).strip()
        if turns and turns[-1][0] == speaker:
            head, start, text = turns[-1]
            turns[-1] = (head, start, f"{text} {body}".strip())
        else:
            turns.append((speaker, segment.start, body))
    return turns


# ── summary ───────────────────────────────────────────────────────────────────
def render_summary(transcript: Transcript, stamper: Stamper, options: RenderOptions) -> str:
    summary = transcript.summary
    if summary is None:
        return "_No summary was produced for this run (re-run with --summary)._\n"
    return "\n".join(_summary_lines(summary, stamper)).rstrip() + "\n"


def _summary_lines(summary: Summary, stamper: Stamper) -> list[str]:
    lines = ["## Summary", ""]
    if summary.headline:
        lines += [summary.headline, ""]
    for section in summary.sections:
        at = (
            f" — {stamper.at(section.start)}"
            if section.start is not None and stamper.enabled
            else ""
        )
        lines.append(f"### {section.title}{at}")
        lines += [f"- {bullet}" for bullet in section.bullets]
        lines.append("")
    for title, items in (
        ("Decisions", summary.decisions),
        ("Action items", summary.actions),
        ("Open questions", summary.questions),
    ):
        if items:
            lines += [f"### {title}", *[f"- {item}" for item in items], ""]
    return lines
