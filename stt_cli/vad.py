"""vad — find the speech and refuse to hand the model anything else.

WHY THIS IS THE CENTRAL DEFENCE
    Whisper-family models are trained to always produce text. Give one thirty seconds of
    room tone and it will not stay quiet — it emits whatever filler dominated its training
    data over silence: the subtitle credit of whoever captioned a lot of YouTube, a
    "продолжение следует", a "Thanks for watching!", or the same phrase forty times in a
    row. Filtering that out afterwards is guesswork. Never generating it is not: if the
    silence is never fed to the model, the model cannot hallucinate over it.

    So voice-activity detection is not an optimization here, it is the fix. Everything in
    ``cleaning`` is the second line of defence for what still slips through.

TWO DETECTORS, ONE SHAPE
    ``silero`` is the neural detector shipped with whisper.cpp — accurate, and it costs a
    fraction of a transcription pass. ``ffmpeg`` is an energy threshold via the
    ``silencedetect`` filter: less precise, but it needs nothing beyond ffmpeg, so a
    machine with no whisper.cpp build still gets the protection. ``auto`` prefers silero
    and quietly falls back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import proc
from ._errors import UsageError
from .models import SpeechSpan

# A span longer than this is split, because a single enormous decode loses timestamp
# accuracy and cannot be parallelized. Ten minutes keeps plenty of conversational context.
MAX_SPAN_SECONDS = 600.0
# Speech separated by less than this is one span: chopping at every breath costs the model
# the context it needs and multiplies process launches for no gain. Callers override it
# from ``min_silence_ms`` so both detectors end up with identical grouping behaviour.
DEFAULT_MERGE_GAP = 0.8


@dataclass(slots=True)
class VadResult:
    spans: list[SpeechSpan]
    method: str
    total_duration: float

    @property
    def speech_duration(self) -> float:
        return sum(s.duration for s in self.spans)

    @property
    def coverage(self) -> float:
        return self.speech_duration / self.total_duration if self.total_duration > 0 else 1.0

    def describe(self) -> str:
        return (
            f"{self.method}: {len(self.spans)} speech span(s), "
            f"{self.speech_duration / 60:.1f} of {self.total_duration / 60:.1f} min "
            f"({self.coverage * 100:.0f}% speech)"
        )


async def detect(
    wav: Path,
    duration: float,
    *,
    mode: str,
    threshold: float = 0.5,
    min_silence_ms: int = 400,
    speech_pad_ms: int = 200,
    min_speech_ms: int = 250,
    silero_binary: Path | None = None,
    silero_model: Path | None = None,
) -> VadResult:
    """Return the speech spans of ``wav`` using the requested (or best available) detector."""
    if mode not in {"auto", "silero", "ffmpeg", "none"}:
        raise UsageError(
            what=f"unknown VAD mode: {mode!r}",
            why="--vad accepts auto, silero, ffmpeg or none",
            how="use --vad auto unless you have a reason not to",
        )
    if mode == "none":
        return VadResult([SpeechSpan(0.0, duration)], "none (whole file)", duration)

    have_silero = bool(silero_binary and silero_model)
    if mode == "silero" and not have_silero:
        raise UsageError(
            what="the silero detector is not available",
            why=(
                "it needs a whisper.cpp build (whisper-vad-speech-segments) and a Silero ggml model"
            ),
            how="run `stt setup` to install whisper.cpp, or use --vad ffmpeg",
        )

    if have_silero and mode in {"auto", "silero"}:
        assert silero_binary is not None and silero_model is not None
        raw = await _silero_spans(
            wav, silero_binary, silero_model, threshold, speech_pad_ms, min_speech_ms
        )
        method = "silero"
    else:
        raw = await _ffmpeg_spans(wav, duration, min_silence_ms / 1000.0, speech_pad_ms / 1000.0)
        method = "ffmpeg silencedetect"

    spans = normalize(raw, duration, min_speech_ms / 1000.0, min_silence_ms / 1000.0)
    if not spans:
        # A detector that finds nothing is far more likely to be misconfigured than to be
        # looking at a genuinely silent file, and returning zero spans would silently
        # produce an empty transcript. Fall back to the whole file and say so.
        return VadResult(
            [SpeechSpan(0.0, duration)], f"{method} (found nothing, using whole file)", duration
        )
    return VadResult(spans, method, duration)


def normalize(
    spans: list[SpeechSpan],
    duration: float,
    min_speech: float,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[SpeechSpan]:
    """Clamp, sort, merge near-touching spans, drop scraps, and split over-long ones.

    Grouping happens here rather than in the detector on purpose: it is the one place both
    detectors pass through, so ``--vad-min-silence`` means the same thing whichever one ran
    (and it routes around a whisper.cpp bug — see :func:`_silero_spans`).
    """
    clamped = [
        SpeechSpan(max(0.0, s.start), min(duration, s.end))
        for s in sorted(spans, key=lambda s: s.start)
        if s.end > s.start
    ]
    merged: list[SpeechSpan] = []
    for span in clamped:
        if merged and span.start - merged[-1].end <= merge_gap:
            merged[-1] = SpeechSpan(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    kept = [s for s in merged if s.duration >= min_speech]
    return [piece for span in kept for piece in _split_long(span)]


def _split_long(span: SpeechSpan) -> list[SpeechSpan]:
    if span.duration <= MAX_SPAN_SECONDS:
        return [span]
    pieces: list[SpeechSpan] = []
    cursor = span.start
    while cursor < span.end:
        end = min(span.end, cursor + MAX_SPAN_SECONDS)
        pieces.append(SpeechSpan(cursor, end))
        cursor = end
    return pieces


_SILERO_LINE = re.compile(r"start\s*=\s*([\d.]+),\s*end\s*=\s*([\d.]+)")


async def _silero_spans(
    wav: Path,
    binary: Path,
    model: Path,
    threshold: float,
    speech_pad_ms: int,
    min_speech_ms: int,
) -> list[SpeechSpan]:
    """Run whisper.cpp's Silero detector; its times are centiseconds, ours are seconds.

    Two upstream quirks are worked around here rather than reported as our failures.
    First, ``-vspd`` appears in the binary's own help but is missing from its argument
    parser, so passing it makes the tool print usage and exit 0 — a silent empty result.
    Second, ``-vsd`` is bound to the minimum *speech* duration (the min-silence branch in
    the parser assigns the same field), so minimum silence cannot be set through this
    binary at all; :func:`normalize` applies it afterwards instead.
    """
    # Deliberately CPU-only: the Silero graph aborts inside ggml's Metal backend
    # ("pre-allocated tensor in a buffer that cannot run the operation"), and the model is
    # small enough that a CPU pass over an hour of audio costs a couple of seconds anyway.
    argv = [
        str(binary), "-f", str(wav), "-vm", str(model),
        "-vt", f"{threshold:.2f}",
        "-vsd", str(min_speech_ms),
        "-vp", str(speech_pad_ms),
        "-np",
    ]  # fmt: skip
    result = await proc.run(argv, check=True, timeout=60 * 60)
    return [
        SpeechSpan(float(a) / 100.0, float(b) / 100.0)
        for a, b in _SILERO_LINE.findall(result.stdout + result.stderr)
    ]


_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")
_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")
_MAX_VOLUME = re.compile(r"max_volume:\s*(-?[\d.]+) dB")


async def _ffmpeg_spans(
    wav: Path, duration: float, min_silence: float, pad: float
) -> list[SpeechSpan]:
    """Invert ffmpeg's silence intervals into speech spans, with an adaptive noise floor."""
    noise_db = await _noise_floor(wav)
    ffmpeg = proc.require("ffmpeg", install_hint=media_hint())
    argv = [
        ffmpeg, "-nostdin", "-hide_banner", "-i", str(wav),
        "-af", f"silencedetect=noise={noise_db:.0f}dB:d={min_silence:.2f}",
        "-f", "null", "-",
    ]  # fmt: skip
    result = await proc.run(argv, timeout=60 * 60)
    log = result.stderr
    starts = [float(x) for x in _SILENCE_START.findall(log)]
    ends = [float(x) for x in _SILENCE_END.findall(log)]
    return _invert(starts, ends, duration, pad)


def _invert(
    starts: list[float], ends: list[float], duration: float, pad: float
) -> list[SpeechSpan]:
    """Turn (silence_start, silence_end) pairs into the speech between them."""
    spans: list[SpeechSpan] = []
    cursor = 0.0
    for i, silence_start in enumerate(starts):
        if silence_start > cursor:
            spans.append(SpeechSpan(max(0.0, cursor - pad), silence_start + pad))
        cursor = ends[i] if i < len(ends) else duration
    if cursor < duration:
        spans.append(SpeechSpan(max(0.0, cursor - pad), duration))
    return spans


async def _noise_floor(wav: Path) -> float:
    """Pick a silence threshold from the recording's own levels, not a fixed guess.

    A quiet phone memo and a loud podcast have wildly different floors; a hardcoded -30 dB
    treats half of one as silence and none of the other. Measuring first costs one cheap
    pass and makes the same defaults work on both.
    """
    ffmpeg = proc.require("ffmpeg", install_hint=media_hint())
    result = await proc.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i",
            str(wav),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout=60 * 60,
    )
    mean = _first_float(_MEAN_VOLUME, result.stderr)
    peak = _first_float(_MAX_VOLUME, result.stderr)
    if mean is None or peak is None:
        return -32.0
    # Sit below the average speech level but well above the true noise floor, and never
    # stray outside the range where silencedetect behaves sensibly.
    candidate = min(mean - 8.0, peak - 25.0)
    return max(-55.0, min(-22.0, candidate))


def _first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def media_hint() -> str:
    from .media import FFMPEG_HINT

    return FFMPEG_HINT
