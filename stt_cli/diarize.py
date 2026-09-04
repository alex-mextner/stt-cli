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

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from . import proc
from ._errors import (
    EngineError,
    MissingDependencyError,
    PermissionDeniedError,
    ProcessTimeout,
)
from .models import Segment


class Route(NamedTuple):
    """How the worker gets run: which way, and the argv that does it.

    A bare tuple would have callers reading `route[1]`, which says nothing about what the
    second element is — the same shapelessness `_offline` was rewritten to stop relying on.
    """

    name: str
    argv: list[str]


# The pretrained pipeline. Gated on Hugging Face: the user must accept its terms once and
# supply a token, which is why the error path below is as detailed as it is.
PIPELINE = "pyannote/speaker-diarization-3.1"
INSTALL_SIZE_GIB = 2.5
# How long diarization may take, per second of audio. Diarization runs faster than real time
# on this hardware, so twenty times is not a budget anybody meets by working slowly — it is
# the point past which something has gone wrong. A FIXED hour was the first attempt and was
# wrong in both directions: too long to notice a wedge on a two-minute memo, and too short
# for a three-hour recording, which used to finish because the old in-process call had no
# limit at all. Cutting one of those off after an hour would have been a regression.
TIMEOUT_PER_SECOND = 20.0
# ...and a floor, because the first run also downloads the model.
MINIMUM_TIMEOUT = 1800.0
# Importing torch off a cold disk is not fast, and a probe that gives up too early reports
# "not ready" on a machine that can diarize perfectly well. This is the budget for a STATUS
# question only — `stt doctor`, `stt diarize status` — where the cost of waiting is one
# slow line. The work path asks the same question with its own, much larger budget, because
# there a false negative refuses to diarize a machine that could have.
PROBE_TIMEOUT = 300.0

NO_ROUTE = (
    "pyannote.audio is not importable here and uv is not installed to supply it — "
    "install uv (`brew install uv`), which is how every other optional heavy piece of stt "
    "is supplied without touching a Python you did not create"
)

INSTALL_HINT = (
    "run `stt diarize install` (downloads pyannote.audio + torch, about 2.5 GB), then "
    "`stt login diarization` to get a Hugging Face token and accept the model terms"
)


@dataclass(slots=True, frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


# What `uv run --with` is asked to supply. Two packages, because pyannote does not pull a
# torch build that works on this hardware by itself.
WHEELS = ("pyannote.audio", "torch")


def runner(*, force_uv: bool = False) -> Route | None:
    """The interpreter that will run the worker: ours if it already has pyannote, else uv's.

    NOTHING IS INSTALLED ANYWHERE. This used to `pip install` into `sys.executable`, which on
    a Homebrew Python cannot work at all: PEP 668 marks that interpreter externally managed
    and refuses, and the ways past the refusal (`--break-system-packages`, `--user`) exist to
    be discouraged, because they put two and a half gigabytes of torch into an interpreter
    Homebrew owns and upgrades. `uv run --with` builds an environment for the wheels and
    caches it, so the cost is a one-time resolve rather than a permanent dependency for every
    user — which is exactly what `backends/mlx.py` already does for mlx-whisper.

    Returns the route's NAME alongside its argv — "direct" or "uv". Callers that must treat
    the two differently (`_offline`) then ask which route this is, rather than inspecting the
    argv and guessing; `backends/mlx.py` settled on the same shape for the same reason.

    None means neither route exists: no pyannote here, and no uv to supply it.
    """
    if not force_uv:
        try:
            if importlib.util.find_spec("pyannote.audio") is not None:
                return Route("direct", [sys.executable])
        except (ImportError, ModuleNotFoundError, ValueError):
            pass  # the parent package is absent, which is the ordinary case
    uv = proc.which("uv")
    if uv is None:
        return None
    supply: list[str] = []
    for wheel in WHEELS:
        supply += ["--with", wheel]
    # `--no-project` because otherwise uv looks for a pyproject.toml in the CURRENT
    # DIRECTORY and synchronises whatever it finds: run `stt something.wav --diarize` from
    # inside an unrelated Python project and uv creates that project's `.venv` and then fails
    # resolving its dependencies, which have nothing to do with diarization and everything to
    # do with where the user happened to be standing. stt is not a member of anybody's
    # workspace; it wants an environment holding two wheels and nothing else.
    return Route("uv", [uv, "run", "--quiet", "--no-project", *supply, "python"])


def install_command() -> list[str] | None:
    """What `stt diarize install` runs: a warm-up, not an install.

    It resolves and caches the environment the worker will use, so the download happens once,
    visibly, when the user asked for it — rather than in the middle of their first
    transcription. There is nothing to uninstall afterwards but a uv cache entry.
    """
    # The uv route, and ONLY the uv route. `ready()` has already said no by the time this is
    # called, so a direct interpreter that still HAS pyannote is one whose pyannote is broken
    # — a Python upgrade leaving an incompatible torch wheel behind is the ordinary way.
    # Falling back to it would re-run the very import that just failed, after promising two
    # and a half gigabytes, demanding the disk space for them and asking the user to confirm.
    # Without uv there is no route, and saying so is the only answer that helps.
    route = runner(force_uv=True)
    if route is None:
        return None
    return [*route.argv, "-c", "import pyannote.audio, torch"]


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


async def ready(*, timeout: float = PROBE_TIMEOUT) -> bool:
    """Can this machine diarize right now, WITHOUT fetching anything to find out?

    Two things this must not do, both of which it did in a first version.

    It must not ask `importlib.find_spec` in our own interpreter: the wheels normally live in
    a uv environment we are not running in, so that answers about the wrong Python and
    reports "not installed" on a machine perfectly able to diarize.

    And it must not let uv BUILD the environment while checking whether it exists. `uv run
    --with` resolves and downloads when the cache is cold, so asking "is diarization ready?"
    would have downloaded two and a half gigabytes — from `stt diarize status`, and from
    `stt doctor`, which people run precisely when they do not want surprises. `--offline`
    makes the question a question: a cached environment answers, a cold one fails, and
    failing is the honest "no".

    Never raises. This is a status, and a status that throws is not one.

    The default *timeout* suits a status line. The work path passes its own, because there a
    probe that gives up early does not print a wrong word — it refuses to diarize.
    """
    route = runner()
    if route is None:
        return False
    try:
        answered = await _ask_the_worker([*_offline(route), _worker(), "--probe"], timeout=timeout)
    except Exception:
        # Everything, not just `SttError`. The docstring above says this never raises, and
        # only the diagnosed errors were being caught — a `TimeoutError` from the probe, or
        # anything else `proc.run` lets through, would have come out of `stt doctor` as the
        # traceback that command exists to replace.
        return False
    return bool(answered.get("ready"))


def _offline(route: Route) -> list[str]:
    """The same runner, forbidden to reach the network. A no-op for a direct interpreter.

    Keyed on the route's name rather than on the shape of its argv. The argv test this
    replaces ("is the second word `run`?") was a private agreement with one exact spelling of
    the uv command: reorder it, or put an environment variable in front, and the guard would
    have quietly stopped adding `--offline` — and `stt doctor` would download two and a half
    gigabytes again, which is the one thing this whole path exists to prevent.
    """
    if route.name != "uv":
        return route.argv
    # Inserted after the subcommand rather than over it. Writing `run` back in at index 1
    # would have been a second, quieter assumption about the exact argv this builds — the
    # kind the route name was introduced to retire.
    argv = route.argv
    return [*argv[:2], "--offline", *argv[2:]]


async def diarize(wav: Path, *, speakers: int | None = None) -> list[Turn]:
    """Run the diarization pipeline over the normalized audio and return speaker turns."""
    budget = _long_enough_for(wav)
    route = runner()
    if route is None:
        raise MissingDependencyError(
            what="speaker diarization is not available",
            why=NO_ROUTE,
            how=INSTALL_HINT,
        )
    # Asked before any work, and this is not belt-and-braces. The runner is an ONLINE
    # `uv run --with`, so reaching it with a cold cache resolves and downloads two and a half
    # gigabytes — inside somebody's transcription, with both streams captured, so not even a
    # progress bar reaches them. The download belongs to `stt diarize install`, where it was
    # asked for and can be watched.
    #
    # With THIS command's budget, not the status line's. The probe imports torch, which off a
    # cold disk is slow enough to outrun a short limit — and here giving up early would refuse
    # a machine that was about to succeed, rather than print one pessimistic line.
    if not await ready(timeout=budget):
        raise MissingDependencyError(
            what="speaker diarization is not ready",
            why="the wheels are not in place, and fetching them is not this command's job",
            how=INSTALL_HINT,
        )
    token = require_token()
    said = await _ask_the_worker(
        [
            *route.argv, _worker(),
            "--audio", str(wav),
            "--pipeline", PIPELINE,
            *(["--speakers", str(speakers)] if speakers else []),
        ],
        timeout=budget,
        # In the environment, never in argv. A command line is readable by every user on the
        # machine, and diarizing an hour of audio keeps the process alive for long enough to
        # be caught by anybody running `ps`. This is the variable huggingface_hub reads
        # anyway, so the worker does not have to be told twice.
        secret={"HUGGING_FACE_HUB_TOKEN": token},
    )  # fmt: skip
    if "turns" not in said and "error" not in said:
        # Without this, an unrecognised object read as zero turns and the run finished with a
        # cheerful "found 0 speaker(s)" — the user asked for --diarize, waited for it, and got
        # a transcript with nobody in it and nothing saying why. `_the_object_among` takes the
        # LAST object on stdout, so anything torch prints on its way out could be read as the
        # answer; an answer has to look like one.
        raise EngineError(
            what="speaker diarization returned neither speakers nor a reason",
            why=f"the worker answered with an object that is not a result: {said!r:.200}",
            how="run `stt diarize status` to see whether the environment is intact",
        )
    if "error" in said:
        raise EngineError(
            what="speaker diarization failed",
            why=str(said["error"]),
            how="check the Hugging Face token with `stt diarize status`, then try again",
        )
    return [
        Turn(start=float(t["start"]), end=float(t["end"]), speaker=str(t["speaker"]))
        for t in said.get("turns", [])
    ]


def _long_enough_for(wav: Path) -> float:
    """A limit proportional to the audio, so it fits both a memo and an afternoon."""
    try:
        # The normalized wav this is always handed: 16 kHz, mono, 16-bit. Its length in
        # seconds is its size in bytes over thirty-two thousand, near enough for a timeout.
        seconds = wav.stat().st_size / 32_000
    except OSError:
        seconds = 0.0
    return max(MINIMUM_TIMEOUT, seconds * TIMEOUT_PER_SECOND)


def _worker() -> str:
    """The worker script's path, resolved without importing it — it must not be imported."""
    return str(Path(__file__).with_name("diarize_worker.py"))


async def _ask_the_worker(
    argv: list[str], *, timeout: float, secret: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run the worker and read its one JSON object, or say why that did not happen."""
    try:
        result = await proc.run(argv, timeout=timeout, env=secret)
    except ProcessTimeout as ran_out:
        # `proc.run` catches the raw `TimeoutError` itself and reports it in the vocabulary of
        # a decode: "uv timed out", try a smaller model, split the input. Every word of that
        # is wrong here — uv finished within the first second, the worker is what is slow,
        # there is no smaller diarization model, and no flag raises this limit. So `proc.run`
        # marks its timeouts with a type, and this is the caller that wanted to know.
        raise EngineError(
            what="speaker diarization did not finish",
            why=f"the worker was still running after {timeout / 60:.0f} minutes",
            how="try again, or pass --speakers to give the model less to search for",
        ) from ran_out
    if result.code != 0 and not result.stdout.strip():
        raise EngineError(
            what="speaker diarization could not start",
            why=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no output",
            how=INSTALL_HINT,
        )
    answered = _the_object_among(result.stdout)
    if answered is None:
        raise EngineError(
            what="speaker diarization answered with something that is not a result",
            why=f"no JSON object on stdout; it said {result.stdout.strip()[-200:]!r}",
            how="run `stt diarize status` to see whether the environment is intact",
        )
    return answered


def _the_object_among(stdout: str) -> dict[str, Any] | None:
    """The worker's one JSON object, wherever in the output it ended up.

    Not "the whole stream", which assumes nothing else prints: `uv run --quiet` and a
    download progress bar aimed at stderr are what keep that true, and both are other
    people's programs. Not "the last line" either, which was the first correction and only
    moved the problem — anything torch or pyannote flushes on the way OUT lands after our
    object, and an hour of finished work would be thrown away over a shutdown warning.
    Searching backwards for the first line that parses costs nothing and survives both.
    """
    for line in reversed(stdout.strip().splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None


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
