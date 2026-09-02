"""diarize — work out who was speaking, and only pay for it when asked.

WHY THIS IS OPT-IN AND INSTALLED SEPARATELY
    Speaker diarization means pyannote.audio, which means PyTorch: roughly two and a half
    gigabytes of wheels, plus a Hugging Face token because the pretrained pipeline is gated.
    Making every user of a speech-to-text tool download that in order to transcribe a voice
    memo would be indefensible. So nothing here is a dependency of the package: ``--diarize``
    checks whether it is present, offers to install it, and says exactly what it will cost
    before doing so.

HOW SPEAKERS ARE ATTACHED
    Diarization and transcription are independent passes over the same audio, producing two
    unrelated sets of intervals. They are joined by overlap: each transcript segment takes
    the speaker whose turns cover the most of it. That is deliberately simple — it is robust
    to the two passes disagreeing about exact boundaries, which they always do, and a segment
    that genuinely spans a handover gets attributed to whoever held most of it rather than
    being split on a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import proc
from ._errors import MissingDependencyError, PermissionDeniedError
from .models import Segment

# The pretrained pipeline. Gated on Hugging Face: the user must accept its terms once and
# supply a token, which is why the error path below is as detailed as it is.
PIPELINE = "pyannote/speaker-diarization-3.1"
INSTALL_SIZE_GIB = 2.5

INSTALL_HINT = (
    "run `stt diarize install` (downloads pyannote.audio + torch, about 2.5 GB), then "
    "`stt login diarization` to get a Hugging Face token and accept the model terms"
)


@dataclass(slots=True, frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def is_installed() -> bool:
    """Is pyannote importable? ``find_spec`` raises when the PARENT package is absent, which
    is the normal case here, so the exception is the answer rather than an error."""
    import importlib.util

    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def install_command() -> list[str]:
    """The command `stt diarize install` runs — surfaced so the user can see it first."""
    uv = proc.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", _python(), "pyannote.audio", "torch"]
    return [_python(), "-m", "pip", "install", "pyannote.audio", "torch"]


def _python() -> str:
    import sys

    return sys.executable


def require_token() -> str:
    """The Hugging Face token the gated pipeline needs, with a real way out if it is missing.

    The lookup lives in :mod:`stt_cli.hf` so that this and ``stt login`` can never disagree
    about which token is in force.
    """
    from . import hf

    token = hf.read_token()
    if token:
        return token
    raise PermissionDeniedError(
        what="speaker diarization needs a Hugging Face token",
        why=f"{PIPELINE} is a gated model and no token is stored or exported",
        how="run `stt login diarization` — it opens the pages and stores the token for you",
    )


async def diarize(wav: Path, *, speakers: int | None = None) -> list[Turn]:
    """Run the diarization pipeline over the normalized audio and return speaker turns."""
    if not is_installed():
        raise MissingDependencyError(
            what="speaker diarization is not installed",
            why="pyannote.audio is not importable",
            how=INSTALL_HINT,
        )
    token = require_token()
    import asyncio

    return await asyncio.to_thread(_run_blocking, wav, speakers, token)


def _run_blocking(wav: Path, speakers: int | None, token: str) -> list[Turn]:
    """The pyannote call itself. Synchronous and slow, so it runs off the event loop."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PIPELINE, use_auth_token=token)
    _to_metal(pipeline)
    kwargs = {"num_speakers": speakers} if speakers else {}
    annotation = pipeline(str(wav), **kwargs)
    return [
        Turn(start=float(segment.start), end=float(segment.end), speaker=str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def _to_metal(pipeline: object) -> None:
    """Move the pipeline onto Apple's GPU when the runtime supports it; CPU otherwise."""
    try:
        import torch

        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))  # type: ignore[attr-defined]
    except Exception:  # any failure here is a performance issue, never a correctness one
        pass


def attach(segments: list[Segment], turns: list[Turn]) -> int:
    """Label each segment with the speaker who covered most of it. Returns the count labelled."""
    if not turns:
        return 0
    labelled = 0
    names = _friendly_names(turns)
    for segment in segments:
        best, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
        if best is not None:
            segment.speaker = names[best]
            labelled += 1
    return labelled


def _friendly_names(turns: list[Turn]) -> dict[str, str]:
    """Rename pyannote's ``SPEAKER_00`` labels to ``S1``, ``S2`` … in order of first speech."""
    order: list[str] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in order:
            order.append(turn.speaker)
    return {label: f"S{index}" for index, label in enumerate(order, start=1)}
