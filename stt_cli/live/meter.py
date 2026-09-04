"""meter — what the microphone is actually hearing, in numbers the user can act on.

WHY THIS EXISTS
    `stt mic` can fail in a way that produces nothing at all: no text, no error, no exit
    code. The microphone is open, audio is flowing, and every frame of it is below the
    threshold the detector needs — so nothing is ever sent to a model and nothing is ever
    typed. From the outside that is indistinguishable from a broken installation, from the
    wrong device being open, from a permission never granted, and from the user not having
    spoken. It cost most of a session to tell those apart once, with four throwaway scripts,
    to arrive at one number: the loudest thing heard was 454 and the detector wanted 1500.

    That number is what this module exists to print. Nobody should have to write a program
    to find out whether their microphone is being heard.

WHAT IT DELIBERATELY DOES NOT DO
    Type anything, anywhere. It runs the real capture and the real gate — the same code a
    session runs, so a verdict here means something about a session — and stops short of the
    keyboard entirely.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import capture
from .capture import BLOCK_MS
from .gate import THRESHOLD_MINIMUM, Gate, loudness

# Long enough to say a sentence into, short enough that nobody walks away from it.
LISTEN_SECONDS = 6.0
# A level this far above the room, without ever clearing the threshold, is somebody talking
# too quietly (or too far away) rather than a silent microphone. The two need different
# advice, so they are told apart rather than both called "nothing heard".
AUDIBLE_RATIO = 2.0
# How long something has to stay above the room before it is a voice rather than an impulse.
# A keystroke is tens of milliseconds; the shortest word anybody dictates is longer than this.
SPEECH_MS = 300
# How far the level has to swing across the window before it is a voice rather than a room.
# Measured here: a room moves about two-fold between its quiet and loud tenths, a voice
# seven-fold. Three sits in the gap and does not depend on where the room's level happens
# to be, which is what both of the absolute tests this replaced could not manage.
SPEECH_VARIATION = 3.0


@dataclass
class Reading:
    """What one spell of listening found."""

    device: capture.Device
    levels: list[float] = field(default_factory=list)
    utterances: int = 0
    threshold: float = THRESHOLD_MINIMUM
    # The detector's own estimate of the room, taken from the gate at the end of the window.
    floor: float = 0.0

    @property
    def peak(self) -> float:
        return max(self.levels, default=0.0)

    @property
    def room(self) -> float:
        """The detector's own estimate of the room, which is what its threshold is built on.

        Reported rather than recomputed, so the number the check prints and the number the
        session decides by cannot disagree.
        """
        return self.floor

    @property
    def variation(self) -> float:
        """How much the level moved across the window: the loud tenth over the quiet tenth.

        This, and not any single level, is what tells a voice from a room. Two attempts at
        it by level both failed, in opposite directions and each on the other's case. The
        median said "nothing was said" to somebody who spoke through the whole window,
        because with no pause in it the median IS the voice. The noise floor said "a voice,
        too quiet to count" to an empty room, because the floor tracks the quietest moment
        and ordinary background sits well above its own quietest moment.

        Speech is not defined by its level. It is syllables and the gaps between them, so it
        SWINGS — measured here, a voice moves seven-fold across a window while a steady room
        moves about two-fold. That difference holds wherever the room happens to sit, which
        is what neither absolute test could manage.
        """
        if len(self.levels) < 10:
            return 1.0
        ranked = sorted(self.levels)
        quiet = max(ranked[len(ranked) // 10], 1.0)
        loud = ranked[len(ranked) * 9 // 10]
        return loud / quiet

    @property
    def silent(self) -> bool:
        """Not "quiet" — actually nothing, which is what a muted device delivers."""
        return self.peak == 0.0

    def sustained_ms(self, above: float) -> int:
        """The longest unbroken stretch louder than `above`, in milliseconds.

        This is the difference between a voice and a keyboard, and without it the meter
        cannot tell them apart: typing peaks in the same place a quiet voice does. Measured
        here, a room sat at 150, typing reached 603 in single blocks, and a sentence would
        hold its level across many of them in a row. Peak alone reported the typing as
        "something was said", which is the wrong answer given confidently.
        """
        longest = run = 0
        for level in self.levels:
            run = run + 1 if level > above else 0
            longest = max(longest, run)
        return longest * BLOCK_MS


async def listen(
    device: capture.Device,
    *,
    gate: Gate | None = None,
    seconds: float = LISTEN_SECONDS,
    on_sample: Callable[[float, float], None] | None = None,
) -> Reading:
    """Run the real microphone through the real gate for a while, and report what happened.

    `gate` must be the SAME gate a session would use, threshold and all. A check that
    measures against the built-in default while dictation measures against a configured one
    is worse than no check: somebody who raised `mic_threshold` to stop stray noises being
    typed would be told "heard you" by the very command they ran to find out why `stt mic`
    was typing nothing.

    `on_sample(level, threshold)` is called for every block so a caller can draw a meter;
    it is the only thing here that knows a terminal exists.
    """
    gate = Gate() if gate is None else gate
    reading = Reading(device=device)
    started = time.monotonic()
    async with capture.listening(device) as stream:
        async for block in stream:
            heard = gate.feed(block)
            level = loudness(block[: len(block) // 2 * 2])
            threshold = gate.threshold
            reading.levels.append(level)
            reading.threshold = threshold
            if heard.utterance:
                reading.utterances += 1
            reading.floor = gate.floor
            if on_sample is not None:
                on_sample(level, threshold)
            if time.monotonic() - started > seconds:
                break
    # The gate is holding whatever was still being said when the clock ran out, and a session
    # would finish it with exactly this call. Without it, somebody who spoke steadily through
    # the whole six seconds never closed an utterance, and the check told them they were too
    # quiet — the one answer guaranteed to send them looking in the wrong place.
    if gate.close() is not None:
        reading.utterances += 1
    return reading


def verdict(reading: Reading) -> tuple[bool, str]:
    """Whether dictation would work, and the sentence that says why it would not.

    Four different failures used to look identical from outside, so they are four different
    sentences here, each naming the thing to go and change.
    """
    if reading.silent:
        return False, (
            f"{reading.device.name} delivered pure silence — macOS is feeding stt an empty "
            "stream, which is what a denied Microphone permission looks like"
        )
    if reading.utterances:
        return True, (
            f"heard you: {reading.utterances} utterance(s), loudest {reading.peak:.0f} "
            f"against a threshold of {reading.threshold:.0f}"
        )
    swing = reading.variation
    held = reading.sustained_ms(reading.peak / AUDIBLE_RATIO)
    if swing >= SPEECH_VARIATION and held >= SPEECH_MS:
        return False, (
            f"a voice, and too quiet to count: it reached {reading.peak:.0f} against a "
            f"threshold of {reading.threshold:.0f} — move closer to the microphone, or raise "
            "the input volume in System Settings > Sound > Input. If it is already loud "
            f"there, `stt config set mic_threshold {max(250, int(reading.peak * 0.6))}` lowers "
            "the bar to suit this microphone"
        )
    if swing >= SPEECH_VARIATION:
        return False, (
            f"short noises only — the level moved, but never for longer than {held} ms, "
            f"which is a keyboard or a chair rather than speech (peak {reading.peak:.0f}, "
            f"room {reading.room:.0f})"
        )
    return False, (
        f"nothing but the room: the level barely moved (peak {reading.peak:.0f}, room "
        f"{reading.room:.0f}) — either nothing was said, or the microphone that is open "
        f"({reading.device.name}) is not the one being spoken into; `stt mic --list-devices` "
        "shows the others"
    )


def bar(level: float, threshold: float, *, width: int = 32) -> str:
    """A meter scaled so the threshold sits at a fixed place on it.

    Absolute levels mean nothing to anybody. Where the bar is relative to the mark is the
    whole message: past the mark is heard, short of it is not.
    """
    mark = width // 2
    filled = min(width, int(level / max(threshold, 1.0) * mark))
    painted = ["-"] * width
    for position in range(filled):
        painted[position] = "#"
    if filled <= mark:
        painted[mark] = "|"
    return "".join(painted)
