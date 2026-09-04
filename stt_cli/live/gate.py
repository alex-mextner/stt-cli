"""gate — decide, while the microphone is open, where one utterance ends and the next begins.

WHY LIVE DICTATION NEEDS THIS EVEN MORE THAN A FILE DOES
    `stt_cli.vad` makes the same decision for a recording, and makes it better, because it
    can look at the whole file and afford a neural detector. Neither is true here: the audio
    arrives in twenty-millisecond pieces and a decision has to be made about each one before
    the next arrives. So this is an energy threshold, deliberately: it costs a few thousand
    integer multiplies a second and needs nothing installed.

    The reason it matters is the one `vad` was written for. A Whisper model handed silence
    does not stay quiet, it invents — and in live dictation that invention would be TYPED,
    into whatever window has focus, while the user is not speaking. The gate is what stops
    the microphone's idle hiss from becoming "Продолжение следует..." in somebody's chat.

THE NOISE FLOOR IS LEARNED, NOT CONFIGURED
    A fixed threshold is wrong in both directions: it hears a fan as speech in a quiet room
    and misses a whisper in a loud one. So the floor tracks the quietest recent audio —
    falling to it at once, rising away from it slowly — and speech is what stands far enough
    above it. That way a room's own level is subtracted rather than argued about.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

RATE = 16_000
SAMPLE_BYTES = 2
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_BYTES

# How far above the learned noise floor a frame has to be to count as speech. Chosen against
# a MacBook's own microphone in a room with a fan: three is too eager, ten misses a quiet
# voice at arm's length.
LOUDNESS_RATIO = 6.0
# Nothing below this is speech whatever the floor says. MEASURED on this machine's own
# microphone rather than guessed, because the first guess (60) was wrong in the direction
# that matters: eight seconds of an ordinary quiet room came out at a median RMS of 16 and a
# ninetieth percentile of 30, but with transients — a keystroke, a chair — reaching 1014,
# while actual speech at arm's length sat at a median of 4550. Sixty let every transient
# through. This is an order of magnitude above the room and an order of magnitude below a
# voice, which is the widest gap available.
THRESHOLD_MINIMUM = 250.0
# How low the LEARNED floor may go, which is a different question and used to share the
# constant above. Sharing it was a bug with a plain symptom: the floor could never fall below
# 250, the threshold is the floor times six, so the smallest threshold the detector could
# ever ask for was 1500 — six times the 250 its own comment calls the minimum. On a machine
# whose room sits at an RMS of 150 (measured here; the numbers above were taken in a much
# quieter one) that put the bar at ten times the room, and a voice that did not reach 1500
# was heard, discarded, and never explained to anybody. This clamp only has to keep the floor
# off exact zero — a muted device delivers frames of zero, and zero times six is zero, which
# would switch the adaptive half of the detector off for the rest of the session. The
# absolute bar is `THRESHOLD_MINIMUM`, applied where the comparison happens.
FLOOR_MINIMUM = 20.0
# The floor rises this much per frame while the room is louder than it — about four percent
# a second. Only ever between utterances: see `_one_frame` for what happened when it was
# allowed to climb during one.
FLOOR_RISE = 1.0008

# Speech has to hold for this long before the gate believes it, which is what keeps a door
# closing or a keyboard clack from opening an utterance.
ONSET_MS = 80
# ...and an utterance has to contain at least this much loud audio in total before it counts
# as one. The onset alone is not enough, and finding that out cost a real failure: a single
# noise in a quiet room opened the gate for eighty milliseconds, and the utterance that came
# out of it — a click, a pre-roll and a breath of silence — was handed to `large-v3-turbo`,
# which answered "Thank you." and had it TYPED into the window that had focus. Nobody had
# said anything. A third of a second is longer than any click and shorter than any word.
MIN_SPEECH_MS = 300
# ...and silence for this long before the gate calls the utterance finished. A pause for
# breath is shorter than this; a pause for thought is longer, and a thought is a sentence.
HANGOVER_MS = 700
# Audio kept from before the onset, because the detector needs a few frames to be sure and
# the first consonant is in them. Without it every utterance starts mid-word.
PREROLL_MS = 300
# An utterance longer than this is closed anyway. Somebody who talks without pausing should
# still see text; and the accurate pass has to be given a bounded amount of work.
MAX_UTTERANCE_MS = 25_000


@dataclass
class Heard:
    """What the gate concluded from the audio it was just fed."""

    speaking: bool
    # The finished utterance, set exactly once, on the block that ended it.
    utterance: bytes | None = None
    # True when this block opened an utterance, so the caller can show that it is listening.
    started: bool = False
    # True when an utterance was opened and then thrown away for holding no real speech. The
    # caller has to know, because it may already have typed a guess at it.
    dropped: bool = False


@dataclass
class Gate:
    """The state machine. Feed it PCM; it tells you when an utterance has finished."""

    floor: float = FLOOR_MINIMUM
    # The bar a frame must clear however quiet the room is. Configurable because level alone
    # cannot separate a quiet voice from a loud room: measured here, a keystroke in a silent
    # room reached 1014 while a voice a metre from the laptop reached 812, so any single
    # number is a choice between missing that voice and hearing that keystroke. The default
    # favours hearing the voice — a session that types nothing and explains nothing is the
    # worse failure — and `mic_threshold` in the config raises it for a room where the other
    # mistake hurts more. `stt mic --check` measures both and says which way to move it.
    minimum: float = THRESHOLD_MINIMUM
    _speaking: bool = False
    _partial: bytearray = field(default_factory=bytearray)
    _preroll: bytearray = field(default_factory=bytearray)
    _spare: bytearray = field(default_factory=bytearray)
    _loud_ms: int = 0
    _quiet_ms: int = 0
    _spoken_ms: int = 0

    @property
    def threshold(self) -> float:
        """How loud a frame must be, right now, to count as speech.

        A property rather than an expression written wherever it is needed. It was written
        twice — here and in the meter that exists to REPORT it — and the two promptly
        diverged: the meter kept quoting the built-in minimum after the gate learned to take
        a configured one, so `stt mic --check` would tell somebody their voice cleared a bar
        it had not cleared. A diagnostic that computes the number itself is a diagnostic that
        can disagree with the thing it diagnoses.
        """
        return max(self.floor * LOUDNESS_RATIO, self.minimum)

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def confirmed(self) -> bool:
        """Has enough loud audio arrived for this to be speech rather than a noise?

        The gate opens on eighty milliseconds because it cannot tell a syllable from a click
        any sooner, and only decides at the end. This is the same question asked in the
        middle, for the one caller that needs an answer before then.
        """
        return self._speaking and self._spoken_ms >= MIN_SPEECH_MS

    def pending(self) -> bytes:
        """The utterance so far, for a draft pass over speech that has not finished."""
        return bytes(self._partial)

    def feed(self, pcm: bytes) -> Heard:
        """Take a block of 16 kHz mono signed 16-bit audio and report what changed."""
        self._spare += pcm
        # Not `speaking=self._speaking`: that captured the state BEFORE the frames were fed
        # and was overwritten unconditionally below, which read as though the per-frame
        # handlers might see the old value. They cannot.
        result = Heard(speaking=False)
        usable = len(self._spare) - len(self._spare) % FRAME_BYTES
        for start in range(0, usable, FRAME_BYTES):
            self._one_frame(bytes(self._spare[start : start + FRAME_BYTES]), result)
        del self._spare[:usable]
        result.speaking = self._speaking
        return result

    def close(self) -> bytes | None:
        """Whatever is still being spoken when the microphone is switched off."""
        if not self._speaking or not self._partial:
            return None
        return self._finish()

    def _one_frame(self, frame: bytes, result: Heard) -> None:
        level = loudness(frame)
        # The floor is a model of the ROOM, and while somebody is talking you are not
        # observing the room. Learning from speech frames too meant the floor climbed four
        # percent a second through a long sentence until the sentence no longer cleared its
        # own threshold: measured, a steady quiet voice was cut off after thirteen seconds
        # and every following second raised the floor further, so the rest of what they said
        # was never sent to a model at all. The comment above claimed this could not happen
        # because the climb was slow. Slow is not the same as bounded.
        if not self._speaking:
            self._learn(level)
        if level > self.threshold:
            self._louder(frame, result)
        else:
            self._quieter(frame, result)

    def _learn(self, level: float) -> None:
        """Drop to a quieter room at once; climb away from it only grudgingly.

        Never below `FLOOR_MINIMUM`, which exists only to keep the floor off exact zero: a
        muted device delivers frames of zero, and a floor of zero multiplied by anything is
        still zero, so a single silent frame used to switch the adaptive half of the detector
        off for the rest of the session. It is deliberately NOT the bar a frame has to clear
        — that is `THRESHOLD_MINIMUM`, and the two sharing one constant is what put the
        smallest possible threshold at six times its documented value.
        """
        settled = min(level, self.floor) if level < self.floor else self.floor * FLOOR_RISE
        self.floor = max(FLOOR_MINIMUM, settled)

    def _louder(self, frame: bytes, result: Heard) -> None:
        self._quiet_ms = 0
        if self._speaking:
            self._partial += frame
            self._spoken_ms += FRAME_MS
            if len(self._partial) >= _bytes_for(MAX_UTTERANCE_MS):
                self._close_it(result)
            return
        self._loud_ms += FRAME_MS
        self._remember(frame)
        if self._loud_ms >= ONSET_MS:
            self._speaking = True
            self._spoken_ms = self._loud_ms
            self._partial = bytearray(self._preroll)
            self._preroll.clear()
            result.started = True

    def _quieter(self, frame: bytes, result: Heard) -> None:
        self._loud_ms = 0
        if not self._speaking:
            self._remember(frame)
            return
        # The trailing silence is kept: a model given a word that stops at its last sample
        # tends to guess at what came next, and a breath of quiet after it does not.
        self._partial += frame
        self._quiet_ms += FRAME_MS
        if self._quiet_ms >= HANGOVER_MS:
            self._close_it(result)

    def _close_it(self, result: Heard) -> None:
        """End the utterance, and say whether it turned out to be one.

        Both answers matter to the caller. A finished utterance goes to the accurate model;
        a discarded one — a click, a chair, eighty milliseconds of nothing — has to be
        reported too, because by then the fast model may already have heard a word in it and
        typed that word into somebody's window. Silence about it left the word there.
        """
        spoken = self._finish()
        if spoken is None:
            result.dropped = True
        else:
            result.utterance = spoken

    def _remember(self, frame: bytes) -> None:
        self._preroll += frame
        spare = len(self._preroll) - _bytes_for(PREROLL_MS)
        if spare > 0:
            del self._preroll[:spare]

    def _finish(self) -> bytes | None:
        """Close the utterance, and throw it away if it was never really one.

        Returning nothing is the whole point. An utterance made of a click, its pre-roll and
        the silence after it contains no speech, and a Whisper model handed audio with no
        speech in it does not stay quiet — it invents, and here the invention is typed.
        """
        spoken = bytes(self._partial) if self._spoken_ms >= MIN_SPEECH_MS else None
        self._partial = bytearray()
        self._speaking = False
        self._quiet_ms = 0
        self._loud_ms = 0
        self._spoken_ms = 0
        return spoken


def loudness(frame: bytes) -> float:
    """Root mean square of one frame, which is all the detector needs to know about it."""
    samples = array("h")
    samples.frombytes(frame)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _bytes_for(milliseconds: int) -> int:
    return RATE * milliseconds // 1000 * SAMPLE_BYTES
