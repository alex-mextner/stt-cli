"""status — what live dictation uses instead of a menu bar icon.

WHY THERE IS NO ICON
    The obvious design is Apple's: an icon in the menu bar and a notification when dictation
    starts. Both want an application. A menu bar item needs an `NSApplication` event loop
    running inside the process, and a notification that says "stt" rather than "Script
    Editor" needs the whole thing packaged as an app bundle with its own identifier. That is
    a real amount of machinery to tell somebody something they asked for two seconds ago.

    So the state is reported in the two places that cost nothing and are already there. A
    single line in the terminal `stt mic` was started from, rewritten in place, which shows
    what is being heard and whether it has settled. And a short system sound when the
    microphone opens and when it closes, for when that terminal is behind another window —
    which is most of the time, because the whole point is to be typing somewhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import shutil
import subprocess
import sys
import unicodedata
import wave
from array import array
from pathlib import Path

# The cue is generated rather than borrowed. macOS ships fourteen sounds and none of them
# says "the microphone is live": `Tink` is a thin tick that reads as an error, `Pop` and
# `Glass` run past a second and a half, and none has an opposite — the open and close sounds
# were two unrelated noises the user had to learn separately. Two tones do the whole job: the
# same pair rising means on and falling means off, which needs no learning at all, and 180 ms
# of it is over before the first word. Written by hand with `wave` and `math`, so this costs
# no dependency and no shipped asset.
# Both cues start on the same note and part company from there: up means the microphone is
# open, down means it is closed. Siblings rather than two noises to learn separately.
_OPEN_TONES = (262.0, 349.0)
_CLOSE_TONES = (262.0, 208.0)
# Bumped whenever the sound changes, so an old cached file is replaced rather than kept.
_CHIME_VERSION = 4
_TONE_MS = 26
# A plucked string, not a beep. The first version held each tone at full volume and faded it
# linearly at both ends, which is a tone generator's envelope and sounds like one. MEASURED
# against a dictation cue that people like (Wispr Flow's): 180 ms long, a peak at fifteen
# percent of full scale, energy around 450 Hz, and a level that falls to two percent within
# seventy milliseconds. All three of this module's first numbers were wrong in the same
# direction — an octave too high, four times too loud, and with no decay at all. It is now
# quieter, lower and shorter than the reference as well: a longer attack takes the edge off
# the onset, which is what makes a tone read as bright however low its pitch is.
#
# The length to match is the AUDIBLE one, not the file's. Wispr's start is a 180 ms file
# holding 47 ms of sound and 130 ms of tail; comparing file lengths made this module's cue
# look the same length as the reference while being more than twice as long to the ear —
# which is what a listener noticed before the measurement did.
_ATTACK_MS = 5
_DECAY_MS = 11
# The last few milliseconds ramp to nothing. An exponential decay is still at a tenth of its
# volume when a fixed-length tone ends, and a step from a tenth to zero is heard as a click —
# the one artefact this envelope exists to avoid, reintroduced at the other end of the note.
_RELEASE_MS = 5
# Under ten percent of full scale. A cue is heard beside whatever else is playing, not over
# it — and this one is heard while somebody is about to speak, so it has to get out of the
# way faster than a notification would.
_AMPLITUDE = 3_000
_CHIME_RATE = 44_100

_LISTENING = "\N{BLACK CIRCLE} listening"
_SETTLED = "\N{CHECK MARK} "
_DRAFT = "\N{HORIZONTAL ELLIPSIS} "


class Line:
    """One line of terminal, rewritten in place. Silent when it is not a terminal."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._width = 0

    def show(self, text: str) -> None:
        if not self.enabled:
            return
        room = max(20, shutil.get_terminal_size((80, 24)).columns - 1)
        painted = _within(text, room)
        width = _columns(painted)
        sys.stderr.write("\r" + painted + " " * max(0, self._width - width))
        sys.stderr.flush()
        self._width = width

    def clear(self) -> None:
        if self.enabled and self._width:
            sys.stderr.write("\r" + " " * self._width + "\r")
            sys.stderr.flush()
            self._width = 0


def _columns(text: str) -> int:
    """How many terminal cells `text` occupies, which is not how many characters it has.

    Everywhere else in `live/` a length is counted in the unit that matters — UTF-16 code
    units for the Quartz call, grapheme clusters for the backspaces — and the status line was
    the one place that assumed a character is a column. It is not for the languages this
    feature exists to serve: a line of dictated Japanese is twice as wide as its length, so it
    wrapped instead of fitting, and the erase that follows it was short by the difference and
    left the tail of the previous line on screen.
    """
    return sum(2 if unicodedata.east_asian_width(mark) in "WF" else 1 for mark in text)


def _within(text: str, room: int) -> str:
    """`text`, trimmed to fit `room` columns, with an ellipsis when anything was dropped."""
    if _columns(text) <= room:
        return text
    kept: list[str] = []
    used = 0
    for mark in text:
        cost = _columns(mark)
        if used + cost > room - 1:
            break
        kept.append(mark)
        used += cost
    return "".join(kept) + "\N{HORIZONTAL ELLIPSIS}"


def note(message: str, *, quiet: bool = False) -> None:
    """One line to stderr, prefixed. What a session says that is not the status line itself.

    It lives here rather than in `commands/mic.py` because `live/` was importing it from
    there while `commands/mic.py` imports `live/` — a dependency pointing both ways, held
    apart only by deferring both imports into function bodies. The library must not reach
    up into the command that drives it.

    `quiet` is not decoration: `-q` says "no status line, sounds or notifications", and this
    was the one channel that ignored it, so somebody who asked for a clean terminal still got
    "loading large-v3-turbo and base" and "press Escape twice to finish" — exactly the lines
    the flag exists to suppress.
    """
    if quiet:
        return
    print(f"stt: {message}", file=sys.stderr)


def render(*, listening: bool, text: str, settled: bool) -> str:
    """The status line: whether the microphone is open, and the sentence in progress."""
    head = _LISTENING if listening else "\N{WHITE CIRCLE} paused"
    if not text:
        return head
    return f"{head}  {_SETTLED if settled else _DRAFT}{text}"


async def opened(device: str, *, quiet: bool = False) -> None:
    # Together, not one after the other: they are two ways of saying the same thing at the
    # same moment, and doing them in turn doubles the wait for no benefit.
    await asyncio.gather(
        _play(_chime(_OPEN_TONES, "open"), quiet=quiet),
        _banner(f"listening on {device}", quiet=quiet),
    )


async def closed(*, said: str = "", quiet: bool = False) -> None:
    words = len(said.split())
    await asyncio.gather(
        _play(_chime(_CLOSE_TONES, "close"), quiet=quiet),
        _banner(f"stopped — {words} word(s) dictated" if words else "stopped", quiet=quiet),
    )


async def _banner(message: str, *, quiet: bool) -> None:
    """A notification, through the one tool that can show one without an application.

    Fire and forget, and silent about its own failure: the user may have notifications off
    for whatever `osascript` is attributed to, and that is their setting, not an error in the
    middle of dictation. `_quoted` is not decoration — the message reaches AppleScript as
    source code, and a stray quote in a device name would otherwise be a syntax error at best.
    """
    if quiet:
        return
    script = f"display notification {_quoted(message)} with title {_quoted('stt')}"
    try:
        process = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
    except OSError:
        return
    await _briefly(process)


def _quoted(text: str) -> str:
    """One AppleScript string literal, with nothing in it that can end the literal early."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + "".join(ch for ch in escaped if ch >= " ") + '"'


def _chime(tones: tuple[float, float], name: str) -> Path | None:
    """The two-tone cue as a file, written once and kept.

    Returns None if it cannot be written — a sound is a convenience, and a dictation session
    that refuses to start because a cache directory is read-only would be a poor trade.
    """
    from .. import config

    where = config.app_home() / "sounds"
    file = where / f"{name}-v{_CHIME_VERSION}.wav"
    if file.is_file():
        return file
    try:
        where.mkdir(parents=True, exist_ok=True)
        with wave.open(str(file), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(_CHIME_RATE)
            handle.writeframes(_two_tones(tones))
    except OSError:
        return None
    return file


def _two_tones(tones: tuple[float, float]) -> bytes:
    """One tone into the next, each faded in and out so neither end clicks."""
    samples = array("h")
    for pitch in tones:
        count = _CHIME_RATE * _TONE_MS // 1000
        for at in range(count):
            value = math.sin(2 * math.pi * pitch * at / _CHIME_RATE)
            samples.append(int(_AMPLITUDE * value * _envelope(at, count)))
    return samples.tobytes()


def _envelope(at: int, count: int) -> float:
    """Struck, then decaying — the shape of something plucked rather than switched on.

    The attack is a few milliseconds because an instant start is heard as a click, and the
    decay is exponential because that is what a physical object does and what the ear reads
    as a note rather than a signal.
    """
    attack = max(1, _CHIME_RATE * _ATTACK_MS // 1000)
    rising = at / attack if at < attack else 1.0
    falling = math.exp(-max(0, at - attack) / (_CHIME_RATE * _DECAY_MS / 1000))
    release = max(1, _CHIME_RATE * _RELEASE_MS // 1000)
    left = count - at
    return min(rising, 1.0) * falling * min(1.0, left / release)


async def _play(sound: Path | None, *, quiet: bool) -> None:
    """Fire and forget. A missing `afplay` or a missing sound file is not worth a word."""
    if quiet or sound is None or not sound.is_file():
        return
    try:
        process = await asyncio.create_subprocess_exec(
            "afplay", str(sound),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
    except OSError:
        return
    await _briefly(process)


async def _briefly(process: asyncio.subprocess.Process) -> None:
    """Wait for a two-hundred-millisecond noise, and not a moment longer than that.

    Long enough for a two-hundred-millisecond noise, which measured at 1.4 seconds because
    `afplay` spends most of that starting up, and short enough that one which wedges is not
    dictation's problem. Whoever calls this must not be in front of anything the user is
    waiting on — see `_through_the_microphone`, where it runs beside the session rather than
    before it. The timeout is not an error, and a process that outlives it is killed rather
    than left to reap itself whenever it feels like it.
    """
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
