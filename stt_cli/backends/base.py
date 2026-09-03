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
    # Pin the initial prompt to EVERY decode window rather than letting it scroll out of
    # the context after the first one. A glossary is only useful if the model still has it
    # ten minutes in.
    carry_prompt: bool = False
    # Tokens of the decoder's own previous output to carry into the next window. 0 decodes
    # every window independently, which is the only way to stop a repetition loop from
    # feeding itself; see CONTEXT_TOKENS.
    max_context: int = 0


# Named context budgets, in tokens of carried-back output. Whisper clamps the value at
# n_text_ctx/2 = 224, so `full` is the model's real maximum rather than a number we chose.
# `short` exists because the trade-off is not binary: a few dozen tokens carry the casing
# and the proper nouns across a window boundary while giving a loop far less to build on.
CONTEXT_TOKENS: dict[str, int] = {"off": 0, "short": 64, "full": 224}


def context_tokens(name: str) -> int:
    """Resolve a context mode to a token budget, refusing a name that does not exist."""
    from .._errors import unknown_item

    if name not in CONTEXT_TOKENS:
        raise unknown_item("context mode", name, sorted(CONTEXT_TOKENS), plural="context modes")
    return CONTEXT_TOKENS[name]


# One decode request is built per chunk, so a bare print repeats the same line once for
# every chunk of a long recording. Kept here rather than once per engine: both backends had
# their own copy of this set and this function, which is two places for one behaviour to
# drift in — and an engine added later would have written a third.
_warned: set[str] = set()


def warn_once(message: str) -> None:
    """One line per distinct complaint, not one per decoded chunk."""
    import sys

    if message in _warned:
        return
    _warned.add(message)
    print(f"stt: {message}", file=sys.stderr)


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

    async def can_pin_prompt(self) -> bool:
        """Can this engine hold the glossary in EVERY decode window, not just the first?

        A declared capability rather than something the pipeline assumes, because the answer
        depends on the installed binary and it changes the words: whisper drops an initial
        prompt after the first window unless it is re-prepended, so an engine that cannot do
        that decodes as if there were no dictionary at all. The pipeline asks before it
        computes the cache key — a run whose glossary silently did nothing must not be
        stored under an identity that claims it worked, or upgrading the engine would serve
        the un-biased transcript back forever.
        """

    def honours_context_budget(self) -> bool:
        """Can this engine carry a MEASURED amount of its own previous output?

        whisper.cpp takes a token count. mlx-whisper takes a boolean, so `--context short`
        and `--context full` decode identically there — and a run stored under the key for
        `short` would then hold a full-context transcript. The pipeline normalizes the mode
        instead, so the key says what the engine actually did.

        No default: a body written here is never inherited, because backends satisfy this
        protocol structurally rather than by subclassing. A "default" would be a guarantee
        the protocol cannot keep — an engine that omitted the method would raise
        AttributeError on the cache-miss path, which is the path quick tests never take.
        """

    def pinning_the_prompt_costs_context(self) -> bool:
        """Does pinning the glossary force a context budget this engine would not otherwise use?

        Yes on whisper.cpp: with ``-mc 0`` the initial prompt is ignored entirely, so
        carrying a glossary means raising the budget. No on mlx, where re-prepending the
        prompt is independent of carrying previous output. It decides whether the
        zero-context side of ``--context-compare`` can keep the glossary: on an engine
        where pinning is free, dropping it hands the comparison a reading the primary pass
        would never have produced, and then offers that to the LLM as evidence.
        """


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
