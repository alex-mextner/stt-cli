"""backends.base — the contract every speech engine satisfies.

An engine's whole job is: given a 16 kHz mono WAV and a decoding temperature, return timed
segments with per-segment confidence. Everything else the tool does — voice-activity
gating, hallucination scrubbing, variants, diarization, LLM correction, rendering — sits
outside and works identically no matter which engine produced the text. That is what makes
adding an engine cheap, and what makes a genuinely better non-Whisper model a drop-in
rather than a rewrite.

CONFIDENCE IS PART OF THE CONTRACT, NOT AN EXTRA
    Confidence is what decides which segments get a second decoding, what the LLM pass is
    told to look at, and what a reader sees marked as shaky. An engine that cannot report
    it must say so (``None``) rather than invent a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..models import Segment


@dataclass(slots=True, frozen=True)
class Availability:
    """Whether an engine can run right now, and what to do when it cannot."""

    ok: bool
    detail: str
    how_to_install: str = ""

    def mark(self) -> str:
        return "available" if self.ok else "unavailable"


@dataclass(slots=True, frozen=True)
class VadProvider:
    """A neural voice-activity detector an engine happens to ship.

    Some engines bring their own detector; others do not, and fall back to the energy-based
    ffmpeg one. Making that a declared capability rather than something the pipeline sniffs
    for with ``hasattr`` means a new engine either provides a detector or does not, and
    everything else keeps working either way.
    """

    binary: Path
    model: Path


@dataclass(slots=True, frozen=True)
class DecodeRequest:
    """One decoding pass over one chunk of audio."""

    wav: Path
    model: str
    language: str | None = None
    temperature: float = 0.0
    offset: float = 0.0  # seconds to add to every timestamp (the span's position in the file)
    threads: int = 0
    word_timestamps: bool = False
    initial_prompt: str | None = None


class Backend(Protocol):
    """What the pipeline needs from a speech engine."""

    name: str

    def availability(self) -> Availability: ...

    def vad_provider(self) -> VadProvider | None:
        """The neural voice-activity detector this engine ships, if it ships one."""

    async def ensure_model(self, model: str) -> None:
        """Fetch the model if it is missing, after checking there is room for it."""

    async def decode(self, request: DecodeRequest) -> list[Segment]:
        """Transcribe one WAV chunk into timed, confidence-bearing segments."""


def confidence_from_logprob(avg_logprob: float | None) -> float | None:
    """Turn an average token log-probability into a 0..1 confidence.

    ``exp(avg_logprob)`` is the geometric mean of the per-token probabilities, which is the
    honest reading of "how sure was the decoder about this text on average". It is only a
    comparable number *within* one model, which is why the thresholds live in settings and
    variants are ranked, never mixed across engines as if the scales matched.
    """
    if avg_logprob is None:
        return None
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def confidence_from_probs(probs: list[float]) -> float | None:
    """Geometric mean of per-token probabilities, for engines that report them directly."""
    usable = [p for p in probs if p > 0.0]
    if not usable:
        return None
    return max(0.0, min(1.0, math.exp(sum(math.log(p) for p in usable) / len(usable))))
