"""capture — the microphone, as a stream of PCM, through the ffmpeg that is already required.

WHY FFMPEG AND NOT A SOUND LIBRARY
    Python has no way to open a microphone in its standard library, and every package that
    fixes that is a compiled extension. `stt` already requires ffmpeg for everything else it
    does with audio, and ffmpeg's `avfoundation` input opens the same devices Core Audio
    offers, resampled to exactly the 16 kHz mono the models want, in one process that is
    already on the machine. Nothing new to install, and the format conversion is free.

WHAT MACOS STILL DEMANDS
    Microphone access, granted to the application running `stt` — the terminal, not `stt`.
    Unlike Accessibility there is no way to interrogate it, so a refusal shows up as ffmpeg
    failing at once with a message about the input device, which :func:`_explain` turns into
    the sentence that says which checkbox to tick.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .. import proc
from .._errors import EngineError, UsageError
from ..media import FFMPEG_HINT
from .gate import RATE

# How much audio to hand over at a time. Two frames' worth of latency added to the pipeline
# is inaudible, and reading in pieces this size keeps the event loop from waking sixty times
# a second for nothing.
BLOCK_MS = 100
BLOCK_BYTES = RATE * BLOCK_MS // 1000 * 2

_DEVICE_LINE = re.compile(r"^\[AVFoundation.*?\]\s*\[(\d+)\]\s*(.+?)\s*$")
_MICROPHONE_HINT = (
    "grant Microphone to the application running stt in System Settings > Privacy & "
    "Security > Microphone, then start a new terminal"
)


@dataclass(frozen=True)
class Device:
    """One input the system is willing to record from."""

    index: int
    name: str


async def devices() -> list[Device]:
    """Every audio input ffmpeg can see. Listing them is not itself a recording."""
    ffmpeg = proc.require("ffmpeg", install_hint=FFMPEG_HINT)
    # ffmpeg reports the list on stderr and then exits non-zero, by design: there is no
    # input to open. So the exit code says nothing here and is deliberately not checked.
    result = await proc.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ],
        timeout=20.0,
    )
    return _audio_inputs(result.stderr)


def _audio_inputs(report: str) -> list[Device]:
    """The audio half of ffmpeg's device listing, which prints cameras first."""
    found: list[Device] = []
    listening = False
    for line in report.splitlines():
        if "audio devices" in line:
            listening = True
            continue
        if "video devices" in line:
            listening = False
            continue
        match = _DEVICE_LINE.match(line) if listening else None
        if match:
            found.append(Device(index=int(match.group(1)), name=match.group(2)))
    return found


async def resolve(wanted: str | None) -> Device:
    """Turn what the user asked for — nothing, a number, or part of a name — into a device.

    Asking for nothing gets the first one ffmpeg lists, which is not necessarily the input
    macOS itself would pick: avfoundation orders devices its own way and has no notion of a
    system default. On a laptop with a headset plugged in, that difference is the difference
    between the built-in microphone and the one next to the mouth, so `stt mic --list-devices`
    and `--device` are how it is settled rather than guessed.
    """
    available = await devices()
    if not available:
        raise EngineError(
            what="no audio input device was found",
            why="ffmpeg's avfoundation input listed no microphones",
            how=_MICROPHONE_HINT,
        )
    if wanted is None:
        return available[0]
    # `isdecimal`, not `isdigit`: they differ on exactly the characters that then crash.
    # `"²".isdigit()` is True and `int("²")` raises `ValueError`, which is not an `SttError`,
    # so `stt mic --device ²` answered a carefully written usage error with a traceback.
    # Every input that worked before still works — Arabic-Indic digits included, which
    # `isdecimal` accepts and `int` reads.
    if wanted.isdecimal():
        for device in available:
            if device.index == int(wanted):
                return device
        # And no falling through to the name search. It did, and a number that matched no
        # index went on to be matched as a SUBSTRING of the names: `--device 5` on a machine
        # whose second microphone is called "EchoWhiskey536" opened that one, having been
        # asked for a device numbered 5 that does not exist. A number the user typed as an
        # index is wrong as an index or it is nothing.
        raise UsageError(
            what=f"no microphone has index {wanted}",
            why="a number is read as a device index, never as part of a device name",
            how="pick one of: " + ", ".join(f"{d.index} ({d.name})" for d in available),
        )
    matched = [d for d in available if wanted.casefold() in d.name.casefold()]
    if len(matched) == 1:
        return matched[0]
    raise UsageError(
        what=f"no single microphone matches {wanted!r}",
        why="it matched " + (f"{len(matched)} devices" if matched else "nothing"),
        how="pick one of: " + ", ".join(f"{d.index} ({d.name})" for d in available),
    )


@asynccontextmanager
async def listening(device: Device) -> AsyncIterator[AsyncIterator[bytes]]:
    """Open the microphone and yield its audio, closing ffmpeg however the block ends."""
    ffmpeg = proc.require("ffmpeg", install_hint=FFMPEG_HINT)
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "avfoundation", "-i", f":{device.index}",
        "-ac", "1", "-ar", str(RATE),
        # Without this ffmpeg fills a buffer before writing anything down the pipe, and the
        # first words of every session arrive a second late for no reason but bookkeeping.
        "-flush_packets", "1",
        "-f", "s16le", "-",
    ]  # fmt: skip
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # ffmpeg's own complaints are read continuously rather than at the end. `-loglevel error`
    # keeps it quiet most of the time, and "most of the time" is not a guarantee: a pipe
    # nobody reads fills up and blocks the writer, so a device that started complaining
    # repeatedly would stall the microphone itself, silently, in the middle of a session.
    complaints: deque[str] = deque(maxlen=10)
    draining = asyncio.create_task(proc.drain_stderr(process, complaints))
    try:
        yield _blocks(process, complaints)
    finally:
        draining.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await draining
        await _close(process)


async def _blocks(
    process: asyncio.subprocess.Process, complaints: deque[str]
) -> AsyncIterator[bytes]:
    """Audio in usable pieces, until the microphone stops producing it.

    ffmpeg hands over one hardware buffer at a time, which is about 340 samples and does not
    divide into anything — so this gathers them until there is a block's worth rather than
    waking the loop fifty times a second. What it must never do is DISCARD a short read: an
    earlier version dropped anything under one frame, which quietly threw away most of the
    audio and made a live microphone look like a stalled one.
    """
    assert process.stdout is not None
    gathered = bytearray()
    while True:
        piece = await process.stdout.read(BLOCK_BYTES)
        if not piece:
            if gathered:
                yield bytes(gathered)
            await _explain(process, complaints)
            return
        gathered += piece
        if len(gathered) >= BLOCK_BYTES:
            yield bytes(gathered)
            gathered = bytearray()


async def _explain(process: asyncio.subprocess.Process, complaints: deque[str]) -> None:
    """ffmpeg stopped. Say why, in the terms of the thing the user has to go and fix.

    The exit status is consulted when ffmpeg had nothing to say, which is the case that used
    to pass for success. A device unplugged mid-sentence, or an ffmpeg killed by something
    else on the machine, closes the pipe without writing a diagnostic; the audio simply ended,
    the session treated that as the microphone being finished, and `stt mic` printed whatever
    it had and exited zero. Silence is the one thing a dictation tool must never report as a
    normal ending, because it is indistinguishable from the user not having spoken.
    """
    # One turn of the loop, so the drain task can pick up whatever ffmpeg said on its way out.
    await asyncio.sleep(0)
    if complaints:
        raise EngineError(what="the microphone stopped", why=complaints[-1], how=_MICROPHONE_HINT)
    status = await _exit_status(process)
    if status:
        raise EngineError(
            what="the microphone stopped",
            why=f"ffmpeg exited with status {status} without saying why",
            how=_MICROPHONE_HINT,
        )


async def _exit_status(process: asyncio.subprocess.Process) -> int | None:
    """What ffmpeg exited with, waiting briefly for a status that is on its way.

    Its stdout is already closed by the time this is asked, so the status is normally there
    for the taking; the wait covers the moment between the pipe closing and the child being
    reaped. Bounded, because a hung child must not become a hung session — an unknown status
    is reported as no status, and the caller treats that as an ordinary end rather than
    inventing a failure it cannot name.
    """
    with contextlib.suppress(TimeoutError):
        return await asyncio.wait_for(process.wait(), timeout=2.0)
    return process.returncode


async def _close(process: asyncio.subprocess.Process) -> None:
    """Stop recording. A microphone left open is a recording light left on."""
    await proc.end(process, grace=3.0)
