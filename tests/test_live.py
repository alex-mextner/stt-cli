"""Live dictation: the gate, the in-place rewrite, and the order the two passes write in.

Nothing here touches a microphone, a model or a keyboard. The parts that do are three thin
modules over `ctypes` and `ffmpeg` (`quartz`, `tap`, `capture`); everything that decides
WHAT is typed and WHEN is here, driven by a fake keyboard and two fake models whose timing
the test controls — which is the only way to reproduce the failure these tests exist for,
where an answer about one sentence arrives after an answer about the next one.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from stt_cli.live.server import WhisperServer

from stt_cli.live import capture as capture_mod
from stt_cli.live import session as live
from stt_cli.live import typist as typist_mod
from stt_cli.live.gate import HANGOVER_MS, RATE, Gate
from stt_cli.live.typist import Typist, plan


def tone(ms: int, amplitude: int) -> bytes:
    """A block of audio: loud enough to be speech, or quiet enough to be a room."""
    count = RATE * ms // 1000
    return struct.pack(
        f"<{count}h",
        *[int(amplitude * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(count)],
    )


SPEECH = tone(1200, 6000)
SILENCE = tone(HANGOVER_MS + 400, 20)


async def _ready(value):
    """An already-finished awaitable, for patching over a coroutine that does real work."""
    return value


def a_server(
    process: object = None,
    *,
    port: int = 65000,
    link: object = None,
    draining: bool = False,
) -> WhisperServer:
    """A `WhisperServer` without a launched process behind it, through its REAL constructor.

    It used to be built with `__new__` and a hand-written list of six private fields, copied
    into seven tests and already inconsistent between them — some set the locks, some did
    not, so what a test exercised depended on which copy it started from, and the next field
    added to `__init__` would have been silently unset in every one. The constructor is
    callable here because it no longer creates its drain task; `start` does. `draining` opts
    into that task for the tests that stop a server and need something to cancel.
    """
    from stt_cli.live.server import WhisperServer

    server = WhisperServer(process, port)  # type: ignore[arg-type]
    server._link = link  # type: ignore[assignment]
    if draining:
        server._begin_draining()
    return server


class FakeKeyboard:
    """Records what would have been typed, and keeps the resulting line."""

    def __init__(self) -> None:
        self.line = ""
        self.events: list[str] = []

    def type_text(self, text: str) -> None:
        self.events.append(f"+{text}")
        self.line += text

    def press_backspace(self, times: int) -> None:
        self.events.append(f"-{times}")
        self.line = self.line[: len(self.line) - times] if times else self.line


class FakeModel:
    """A transcriber that answers with whatever the test queued, when the test allows."""

    def __init__(self, *answers: str, hold: asyncio.Event | None = None) -> None:
        self.answers = list(answers)
        self.hold = hold
        self.asked = 0

    async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
        del pcm, prompt
        self.asked += 1
        if self.hold is not None:
            await self.hold.wait()
        return self.answers.pop(0) if self.answers else ""


async def feed(*blocks: bytes):
    for block in blocks:
        yield block
        await asyncio.sleep(0)


# ── the gate ──────────────────────────────────────────────────────────────────


def test_a_quiet_room_is_never_handed_to_the_model() -> None:
    """The whole reason the gate exists. A Whisper model given silence does not stay quiet,
    it invents — and here the invention would be TYPED into somebody's window."""
    gate = Gate()
    heard = gate.feed(tone(5000, 15))
    assert not heard.speaking
    assert heard.utterance is None
    assert gate.close() is None


def test_speech_becomes_one_utterance_with_the_first_syllable_in_it() -> None:
    """Detection needs a few frames to be sure, and the first consonant is in them — so the
    utterance has to start before the moment the detector made up its mind."""
    gate = Gate()
    assert not gate.feed(tone(400, 15)).speaking
    started = gate.feed(SPEECH)
    assert started.started and started.speaking
    finished = gate.feed(SILENCE)
    assert finished.utterance is not None
    spoken = len(finished.utterance) / (RATE * 2)
    assert spoken > 1.2, "the pre-roll and the trailing silence are both kept"
    assert not gate.speaking


def test_a_microphone_switched_off_mid_sentence_keeps_the_sentence() -> None:
    gate = Gate()
    gate.feed(SPEECH)
    assert gate.close() is not None


def test_a_quiet_voice_in_a_quiet_room_is_still_heard() -> None:
    """The bug this pins made `stt mic` produce nothing at all, and say nothing about it.

    The detector's floor was clamped at 250 and the threshold is the floor times six, so the
    smallest bar it could ever set was 1500 — six times the 250 its own comment calls the
    minimum. A voice quieter than that was heard, measured, discarded, and never mentioned.
    An amplitude of 800 is a real voice on a laptop microphone a metre away; it has to get
    through a silent room.
    """
    gate = Gate()
    gate.feed(tone(400, 20))  # the room, so the floor settles where the room is
    started = gate.feed(tone(1200, 800))
    assert started.speaking, "a quiet voice is a voice"
    assert gate.feed(SILENCE).utterance is not None


def test_the_bar_a_frame_has_to_clear_never_starts_above_its_documented_minimum() -> None:
    """The two constants used to be one, and one of them was multiplied by six. Pinning the
    arithmetic is what stops them being merged again by somebody tidying up."""
    from stt_cli.live.gate import LOUDNESS_RATIO, THRESHOLD_MINIMUM

    gate = Gate()
    assert max(gate.floor * LOUDNESS_RATIO, THRESHOLD_MINIMUM) == THRESHOLD_MINIMUM


def test_a_room_full_of_typing_still_does_not_open_the_gate() -> None:
    """The other side of lowering the bar, and the reason it was raised in the first place:
    a keystroke reached 603 in this room, and single loud blocks must not become an
    utterance. Duration is what separates them, not level."""
    gate = Gate()
    gate.feed(tone(400, 20))
    for _ in range(6):
        gate.feed(tone(30, 900))  # a click: loud, and over almost at once
        gate.feed(tone(200, 25))
    assert gate.close() is None, "clicks in a quiet room are not a sentence"


def test_a_room_that_needs_a_higher_bar_can_be_given_one() -> None:
    """Level alone cannot separate a quiet voice from a loud room, so the number has to be
    somebody's choice rather than a constant. The default favours hearing the voice; a room
    where stray noises get typed instead needs the other trade, without editing the code."""
    from stt_cli import config
    from stt_cli.live.dictation import gate_for

    assert gate_for(config.Settings()).minimum == Gate().minimum, "nobody chose: the default"
    assert gate_for(config.Settings(mic_threshold=1500)).minimum == 1500

    picky = Gate(minimum=1500)
    picky.feed(tone(400, 20))
    assert not picky.feed(tone(1200, 800)).speaking, "800 is below a bar set at 1500"

    # And zero keeps meaning "nobody chose", not "let everything through" — a gate with a
    # threshold of zero opens on the room and hands it to the model continuously.
    assert gate_for(config.Settings(mic_threshold=0.0)).minimum > 0


async def test_a_noise_the_microphone_ends_on_does_not_stay_typed() -> None:
    """The flagship failure, through a door nobody had tried: the stream ending mid-noise.

    A click opens the gate, the fast model reads a word into it and types that word, and then
    the microphone is unplugged before the pause that would have thrown the noise away. The
    gate discards it — under three hundred milliseconds of speech — so no accurate pass ever
    runs, and without this the invented word stays in the window for good.
    """
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=FakeModel("померещилось"), settled=FakeModel("")
    )

    async def a_click_and_then_the_cable_goes() -> None:
        yield tone(400, 20)  # the room, so the floor settles
        yield tone(100, 6000)  # a click: loud, far too short to be a word
        yield tone(150, 20)  # quiet enough to schedule a draft of it
        # and the microphone ends here, before the gate's hangover has expired

    await dictating.run(a_click_and_then_the_cable_goes())
    await asyncio.sleep(0.05)

    assert keys.line == "", "whatever was guessed at the click was taken back"


async def test_the_check_measures_against_the_threshold_the_session_would_use(monkeypatch) -> None:
    """A diagnostic that answers a different question than the tool it diagnoses is worse
    than none: somebody who raised the threshold to stop stray noises being typed would be
    told "heard you" by the very command they ran to find out why nothing was typed.

    Driven through `meter.listen` with a raised gate, because the first version of this test
    handed a `Reading` straight to `verdict` and so never touched the line that was wrong.
    """
    from stt_cli.live import meter

    @contextlib.asynccontextmanager
    async def a_voice_that_clears_the_default_bar_and_not_this_one(_device):
        async def blocks():
            for _ in range(4):
                yield tone(100, 20)  # the room
            for word in range(12):  # a voice: well over 250, nowhere near 1500
                yield tone(100, 20 if word % 4 == 3 else 900)  # words, with gaps between
            yield tone(HANGOVER_MS + 400, 20)

        yield blocks()

    monkeypatch.setattr(
        meter.capture, "listening", a_voice_that_clears_the_default_bar_and_not_this_one
    )
    device = capture_mod.Device(0, "Built-in")

    raised = await meter.listen(device, gate=Gate(minimum=1500))
    assert raised.threshold == 1500, "the bar reported is the bar the session would apply"
    assert raised.utterances == 0
    assert "1500" in meter.verdict(raised)[1], "and the advice quotes it, not the default"

    default = await meter.listen(device, gate=Gate())
    assert default.utterances == 1, "the same voice is heard when nobody raised the bar"


async def test_without_lsof_the_microphone_refuses_rather_than_trusting_the_port(
    monkeypatch,
) -> None:
    """The check that stops microphone audio going to the wrong process must not be skippable.

    The port is chosen by binding a socket and closing it, so something else can take it in
    the gap and be sent every recorded sentence. `lsof` is what proves the listener is the
    process stt started — and skipping the check when `lsof` is missing meant the one defence
    could be removed by arranging for it not to be found. A security check with a fallback is
    not a security check.
    """
    from stt_cli import proc
    from stt_cli._errors import MissingDependencyError

    class Alive:
        returncode = None
        pid = 4242

        def terminate(self) -> None: ...
        def kill(self) -> None: ...

        async def wait(self) -> int:
            return 0

    server = a_server(Alive(), draining=True)  # `stop()` cancels the drain on its way out
    monkeypatch.setattr(proc, "which", lambda name: None)

    with pytest.raises(MissingDependencyError) as raised:
        await server._confirm_it_is_ours()
    assert "lsof" in raised.value.why
    assert "microphone audio" in raised.value.why, "it says what the check is protecting"


def test_a_threshold_that_switches_itself_off_is_refused() -> None:
    """`inf` and `nan` are numbers to Python and to nobody else: every real level compares
    false against them, so either one leaves the microphone open, the detector never opening,
    nothing typed and nothing said about it. The same hole was closed once in
    `--idle-minutes`; it is closed here for every numeric setting at once, because the next
    threshold somebody adds would have had it too."""
    from stt_cli import config
    from stt_cli._errors import UsageError
    from stt_cli.live.dictation import gate_for

    for refused in ("inf", "-inf", "nan", "NaN"):
        with pytest.raises(UsageError):
            config.coerce("mic_threshold", refused)
    assert config.coerce("mic_threshold", "1500") == 1500.0
    assert config.coerce("mic_threshold", "0") == 0.0

    # Both doors have to give the same answer. The typed value and the hand-edited one used
    # to disagree — one refused a value the other loaded and honoured — so the file path is
    # asked here too.
    assert not config._a_real_number(float("inf"))
    assert not config._a_real_number(float("nan"))
    assert config._a_real_number(1500.0)

    # A negative threshold is not a quieter one: it is below every level a microphone can
    # produce, so it would open the gate on the room continuously. Refused by meaning it as
    # "nobody chose", which is what any value not above zero means.
    assert gate_for(config.Settings(mic_threshold=-5.0)).minimum == Gate().minimum


async def test_a_device_argument_that_looks_like_a_number_but_is_not_one(monkeypatch) -> None:
    """`str.isdigit` and `int` disagree about exactly one class of character, and the gap is
    a traceback: `"²".isdigit()` is True, `int("²")` raises `ValueError`, and a ValueError is
    not an `SttError` — so a carefully written usage error was answered with a stack trace."""
    from stt_cli._errors import UsageError
    from stt_cli.live import capture

    async def listed():
        return [capture.Device(0, "Built-in"), capture.Device(1, "Headset")]

    monkeypatch.setattr(capture, "devices", listed)
    with pytest.raises(UsageError):
        await capture.resolve("\N{SUPERSCRIPT TWO}")
    # And the digits that always worked still do, including the non-Latin ones `int` reads.
    assert (await capture.resolve("1")).index == 1
    assert (await capture.resolve("\N{ARABIC-INDIC DIGIT ONE}")).index == 1


def test_the_status_line_is_measured_in_columns_not_characters() -> None:
    """Every other length in `live/` is counted in the unit that matters — UTF-16 units for
    the Quartz call, grapheme clusters for the backspaces. The status line assumed a
    character is a column, which is false for exactly the languages dictation exists for: a
    line of Japanese is twice as wide as it is long, so it wrapped instead of fitting and the
    erase after it was short by the difference."""
    from stt_cli.live import status

    assert status._columns("abc") == 3
    assert status._columns("日本語") == 6, "each of those takes two cells"
    assert status._columns("привет") == 6, "Cyrillic is one cell per letter"

    trimmed = status._within("日本語日本語", 6)
    assert status._columns(trimmed) <= 6, "it fits the room it was given"
    assert trimmed.endswith("\N{HORIZONTAL ELLIPSIS}"), "and says that something was dropped"


def test_a_voice_is_told_from_a_room_by_how_much_the_level_moves() -> None:
    """The advice this check gives depends on telling a room from a voice, and two attempts
    to do it by LEVEL both failed, each on the other's case.

    The median said "nothing was said" to somebody who spoke through the whole window — the
    instruction is "say something for six seconds", so doing exactly that leaves the median
    sitting on the voice. The noise floor then said "a voice, and too quiet to count" to an
    empty room, because the floor tracks the quietest moment of the window and ordinary
    background sits well above its own quietest moment. Speech is syllables and the gaps
    between them, so what separates it from a room is that it SWINGS."""
    from stt_cli.live import meter

    # A voice filling the whole window: syllables and the gaps between them, so the level
    # swings even though it never stops. Both earlier attempts got this case wrong — the
    # median because with no pause the median IS the voice, and the noise floor because it
    # then judged an empty room to be a quiet voice.
    talking = meter.Reading(device=capture_mod.Device(0, "Built-in"), threshold=1500.0, floor=90.0)
    # Words of half a second with gaps between them, which is what speech looks like as a
    # level: long runs above, short drops between. Not one loud block in three — that is a
    # rattle, and the check is right to call it one.
    talking.levels = [120.0 if at % 8 < 2 else 900.0 for at in range(60)]

    assert talking.variation > 3.0, "a voice swings"
    working, said = meter.verdict(talking)
    assert not working
    assert "too quiet to count" in said, "the branch that names the real problem"
    assert "not the one being spoken into" not in said, "and not the one that misdirects"

    # And an ordinary room, where nothing was said: it must not be reported as a quiet voice.
    quiet = meter.Reading(device=capture_mod.Device(0, "Built-in"), threshold=250.0, floor=21.0)
    quiet.levels = [40.0 + (at % 5) * 10 for at in range(60)]

    assert quiet.variation < 3.0, "a room does not"
    silent_working, silent_said = meter.verdict(quiet)
    assert not silent_working
    assert "nothing but the room" in silent_said


def test_quiet_means_quiet() -> None:
    """`-q` says "no status line, sounds or notifications". `note` ignored it, so a session
    started for a script still announced which models it was loading and told the user to
    press Escape twice — exactly the lines the flag exists to suppress."""
    import contextlib as ctx
    import io

    from stt_cli.live import status

    said = io.StringIO()
    with ctx.redirect_stderr(said):
        status.note("loading two large models", quiet=True)
    assert said.getvalue() == ""

    heard = io.StringIO()
    with ctx.redirect_stderr(heard):
        status.note("loading two large models")
    assert "loading two large models" in heard.getvalue(), "and it still speaks when it may"


async def test_both_models_load_at_once_and_neither_is_left_behind(monkeypatch) -> None:
    """The wait before dictation will listen is the whole of what somebody experiences, and
    it used to be the SUM of two loads for no reason but the order of a loop.

    The second half is why this is not a one-line `gather`: if one model fails while the
    other is still loading, the one that succeeded is a process holding a gigabyte open with
    nobody left to use it.
    """
    from stt_cli.live import dictation

    running: list[str] = []
    stopped: list[str] = []

    class Loaded:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            stopped.append(self.name)

    async def load(_binary, path, _language, _threads):
        running.append(path.name)
        await asyncio.sleep(0)  # both are in flight across this point
        if path.name == "broken":
            raise RuntimeError("this model will not load")
        return Loaded(path.name)

    monkeypatch.setattr(dictation, "_load", load)
    monkeypatch.setattr(dictation, "WhisperServer", Loaded)

    both = await dictation._load_together(Path("bin"), [Path("big"), Path("small")], "ru", 0)
    assert [server.name for server in both] == ["big", "small"], "in the order asked for"
    assert running == ["big", "small"], "and started before either had finished"

    running.clear()
    with pytest.raises(RuntimeError):
        await dictation._load_together(Path("bin"), [Path("big"), Path("broken")], "ru", 0)
    assert stopped == ["big"], "the one that did start is not left holding memory"


def test_the_cue_starts_and_ends_at_silence_and_is_over_quickly() -> None:
    """Two things a generated cue has to get right, both of which were got wrong once.

    It must reach zero at both ends: an exponential decay is still at a tenth of its volume
    when a fixed-length tone stops, and a step from a tenth to zero is a click — the artefact
    the envelope exists to prevent, reintroduced at the far end of the note. And it must be
    SHORT, measured by what is audible rather than by the length of the file: the reference
    cue this was tuned against is a 180 ms file holding 47 ms of sound, and comparing file
    lengths made a cue more than twice too long look correct.
    """
    from array import array

    from stt_cli.live import status

    samples = array("h")
    samples.frombytes(status._two_tones(status._OPEN_TONES))
    peak = max(abs(value) for value in samples)

    assert abs(samples[0]) < peak * 0.02, "it fades in rather than starting on a step"
    assert abs(samples[-1]) < peak * 0.02, "and fades out rather than stopping on one"

    loud = [at for at, value in enumerate(samples) if abs(value) > peak * 0.05]
    audible_ms = (loud[-1] - loud[0]) / status._CHIME_RATE * 1000
    assert audible_ms < 70, f"a cue, not a chord: {audible_ms:.0f} ms of sound"
    assert peak < 0.12 * 32767, "heard beside what is playing, not over it"


async def test_a_connection_the_server_tidied_away_is_opened_again(monkeypatch) -> None:
    """The commonest connection failure there is, and none of the server tests covered it.

    `http.client` never reconnects on its own: once its socket exists it keeps using it, so a
    server that closed an idle connection between two sentences — an ordinary thing to do —
    made the next request fail, and that failure ended the session. A pause was enough to
    stop dictation.
    """
    from stt_cli.live.server import WhisperServer, _ConnectionGone

    class Alive:
        returncode = None

    attempts: list[int] = []

    def first_one_is_gone(self, link, body, content_type):
        del link, body, content_type
        attempts.append(1)
        if len(attempts) == 1:
            raise _ConnectionGone("the server closed it while nobody was talking")
        return '{"text": "и вот ответ"}'

    server = a_server(Alive())
    monkeypatch.setattr(WhisperServer, "_post", first_one_is_gone)
    monkeypatch.setattr(WhisperServer, "_the_link", lambda self: _ready(object()))

    assert await server.transcribe(b"\x00\x00" * 100) == "и вот ответ"
    assert len(attempts) == 2, "it asked once, was cut off, and asked again"


async def test_a_microphone_that_dies_WITH_something_to_say_says_it(monkeypatch) -> None:
    """The other half of the ffmpeg-death handling: the branch where there IS a diagnostic.

    Only the silent branch was tested, and the silent branch is the rarer one — a denied
    Microphone permission, the commonest cause of all, makes ffmpeg complain on the way out.
    That complaint is the whole value: it is the sentence that names the checkbox to tick.
    """
    from stt_cli._errors import EngineError
    from stt_cli.live import capture

    with pytest.raises(EngineError) as raised:
        async for _block in capture._blocks(
            _EndedProcess(1), deque(["[AVFoundation] Failed to open input device"])
        ):
            pass
    assert "Failed to open input device" in raised.value.why, "ffmpeg's own words, not ours"
    assert "Microphone" in raised.value.how


async def test_a_noise_dropped_mid_settle_does_not_hand_back_a_typist_the_user_took() -> None:
    """One keypress used to be enough to have a whole sentence typed twice.

    A key's own sound opens the gate on a noise while the sentence before it is still
    decoding, so the noise claims the next sentence number. The user then presses a key — the
    ordinary "stop correcting" interaction — which disowns the draft on screen. The noise is
    discarded for holding no speech, and the take-back used to `settle()` unconditionally,
    clearing the takeover. The earlier sentence was never marked taken over (the noise had
    claimed that number), so its answer was typed in full at the user's caret, underneath the
    draft it had orphaned.
    """
    keys = FakeKeyboard()
    settling = asyncio.Event()

    class SlowModel:
        async def transcribe(self, _audio: bytes, *, prompt: str = "") -> str:
            await settling.wait()
            return "первое предложение"

    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=SlowModel())

    async def a_sentence_then_a_keypress_then_a_click():
        yield tone(400, 20)  # the room
        yield SPEECH  # the sentence
        yield SILENCE  # which closes it and starts it decoding
        yield tone(100, 6000)  # a click, which claims the NEXT sentence number
        dictating.interrupt()  # the user presses a key, marking that noise's number
        yield SILENCE  # long enough for the click to be discarded as no speech
        settling.set()  # and only now does the first sentence come back from the model
        await asyncio.sleep(0.05)

    await dictating.run(a_sentence_then_a_keypress_then_a_click())

    assert keys.line == "", "the taken-over window is never written to again"


async def test_the_check_counts_speech_that_was_still_going_when_time_ran_out(monkeypatch) -> None:
    """Six seconds of steady speech used to close no utterance at all, because the window
    ended before the pause did — and the check reported "too quiet" to somebody who had been
    talking the whole time, which is the one answer that sends them looking in the wrong
    place. Driven through `meter.listen` rather than the gate, because the gate was never the
    part that was broken."""
    from stt_cli.live import meter

    @contextlib.asynccontextmanager
    async def talking_without_pausing(_device):
        async def blocks():
            yield tone(400, 20)  # the room, so the floor settles where the room is
            for _ in range(6):  # ...and then speech, with no pause anywhere in it
                yield tone(200, 6000)
            # and the listening window closes here, mid-sentence, exactly as the clock does

        yield blocks()

    monkeypatch.setattr(meter.capture, "listening", talking_without_pausing)
    device = capture_mod.Device(0, "Built-in")
    reading = await meter.listen(device)

    assert reading.utterances == 1, "the sentence in progress is finished, not discarded"
    assert meter.verdict(reading)[0], "and the verdict is that it heard you"


# ── the rewrite ───────────────────────────────────────────────────────────────


def test_the_edit_is_the_shortest_one_the_caret_can_make() -> None:
    """A caret can only walk backwards, so the longest common prefix IS the answer rather
    than an approximation of it: a word growing costs nothing, a first word changing costs
    the sentence."""
    assert plan("привет ми", "привет мир") == typist_mod.Edit(0, "р")
    assert plan("привет мир", "Привет, мир!") == typist_mod.Edit(10, "Привет, мир!")
    assert plan("раз два", "раз") == typist_mod.Edit(4, "")


def test_a_typist_only_ever_deletes_what_it_typed() -> None:
    keys = FakeKeyboard()
    typist = Typist(keys=keys)
    typist.begin()
    typist.show("привет ми")
    typist.show("привет мир")
    assert keys.line == "привет мир"
    assert keys.events == ["+привет ми", "+р"]


def test_a_disowned_sentence_is_left_exactly_where_it_is() -> None:
    """Not tidied up and not finished. The user typed something of their own into the middle
    of it; deleting the draft would delete their characters too, and typing the accurate
    version underneath would say the same sentence twice."""
    keys = FakeKeyboard()
    typist = Typist(keys=keys)
    typist.begin()
    typist.show("привет ми")
    typist.disown()

    assert typist.show("привет мир, как дела") is False
    assert keys.line == "привет ми", "nothing was added and nothing was taken away"


# ── the session ───────────────────────────────────────────────────────────────


async def test_the_draft_appears_first_and_the_accurate_pass_replaces_it() -> None:
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("привет мир"),
        settled=FakeModel("Привет, мир!"),
    )
    said = await dictating.run(feed(SPEECH, SPEECH, SILENCE))

    assert keys.line == "Привет, мир!"
    assert said == "Привет, мир!"
    assert any(event.startswith("-") for event in keys.events), "it was corrected in place"


async def test_a_draft_that_arrives_after_the_sentence_settled_is_thrown_away() -> None:
    """The failure this whole ordering exists to prevent. The draft model is asked about a
    growing sentence and the accurate model about the finished one; if a slow draft could
    still write after the accurate answer, the text would assemble itself backwards."""
    slow = asyncio.Event()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("прив", hold=slow),
        settled=FakeModel("Привет, мир!"),
    )
    running = asyncio.create_task(dictating.run(feed(SPEECH, SPEECH, SILENCE)))
    await asyncio.sleep(0.05)
    slow.set()
    assert await running == "Привет, мир!"
    assert keys.line == "Привет, мир!"


async def test_two_sentences_are_separated_by_one_space() -> None:
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=None,
        settled=FakeModel("Раз.", "Два."),
    )
    said = await dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE))

    assert said == "Раз. Два."
    assert keys.line == "Раз. Два."


async def test_a_sentence_the_model_heard_nothing_in_types_nothing() -> None:
    """The gate is the first defence against a model inventing over silence and this is the
    second: an empty answer must not become an empty separator, a stray space, or a turn of
    the sentence counter."""
    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel("", "Два."))
    said = await dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE))

    assert said == "Два."
    assert keys.line == "Два."


async def test_a_key_pressed_by_the_user_stops_the_correction_dead() -> None:
    slow = asyncio.Event()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("привет"),
        settled=FakeModel("Привет, мир!", hold=slow),
    )
    running = asyncio.create_task(dictating.run(feed(SPEECH, SPEECH, SILENCE)))
    await asyncio.sleep(0.05)
    dictating.interrupt()
    slow.set()
    await running

    assert keys.line == "привет", "the draft is untouched and the settled text was not typed"


async def test_a_recorded_misspelling_is_corrected_before_it_is_typed() -> None:
    """The dictionary reaches live dictation the same way it reaches a file: a spelling the
    user wrote down is a fact about a word, so it is fixed in what gets typed."""
    from stt_cli import dictionary

    terms = dictionary.Dictionary([dictionary.Term(term="Figma", aka=["Вигма"])])
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=None,
        settled=FakeModel("открой Вигма"),
        terms=terms,
    )
    assert await dictating.run(feed(SPEECH, SILENCE)) == "открой Figma"


async def test_a_subtitle_credit_is_never_typed() -> None:
    """The artefacts that are never anything else. A model handed something that is not
    speech reaches for whatever dominated its training data over silence, and a stranger's
    subtitle credit appearing in the middle of your own sentence is the recognisable shape of
    it. In a file that is a line to delete; here it is typed into somebody's window."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=None,
        settled=FakeModel("Субтитры сделал DimaTorzok", "Настоящая фраза."),
    )
    said = await dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE))

    assert said == "Настоящая фраза."
    assert keys.line == "Настоящая фраза."


async def test_an_ordinary_word_is_typed_even_though_a_model_invents_it_too() -> None:
    """The file pipeline drops a bare "да" when it has evidence the phrase was invented. Live
    dictation has no such evidence — the model's own `no_speech_prob` was measured useless —
    and somebody dictating into a chat window says "да" on purpose all day. So it is typed,
    and the defence against inventing it stays where it works: the gate."""
    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel("да"))
    assert await dictating.run(feed(SPEECH, SILENCE)) == "да"


async def test_a_runaway_repetition_is_collapsed_before_it_is_typed() -> None:
    """The other shape the same failure takes: not a stock phrase but one real word, over and
    over, until the window ends. Typed as it arrives, it would bury whatever was said before."""
    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel("да " * 40))
    said = await dictating.run(feed(SPEECH, SILENCE))

    assert said.split().count("да") <= live.MAX_REPEATS


async def test_a_model_that_fails_ends_the_session_instead_of_hanging_it() -> None:
    """The accurate pass runs as its own task, and a task that dies takes the queue's
    consumer with it — `run` waits on a join that is never satisfied, and one failed sentence
    becomes a dictation session frozen with the microphone still open."""
    from stt_cli._errors import EngineError

    class Broken:
        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            del pcm, prompt
            raise EngineError(what="the speech model stopped answering", why="", how="")

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=Broken())
    running = asyncio.create_task(dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE)))
    done, _ = await asyncio.wait({running}, timeout=2.0)

    assert done, "it finished rather than waiting forever"
    with pytest.raises(EngineError):
        await running


async def test_a_sentence_starting_while_the_last_one_settles_does_not_double_it() -> None:
    """The normal case, not an edge: the accurate model is still working on one sentence when
    the next is already being spoken. Telling the typist a new sentence had begun at that
    moment made it forget what was on screen, so the accurate answer was TYPED underneath a
    draft of itself instead of replacing it."""
    slow = asyncio.Event()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("привет"),
        settled=FakeModel("Привет!", hold=slow),
        draft_every=0.0,
    )
    running = asyncio.create_task(dictating.run(feed(SPEECH, SPEECH, SILENCE, SPEECH)))
    for _ in range(50):
        await asyncio.sleep(0)
    assert keys.line == "привет", "the draft is on screen before the sentence settles"

    slow.set()
    await running
    assert keys.line == "Привет!", f"one sentence, not two: {keys.line!r}"


def test_a_single_click_is_not_an_utterance() -> None:
    """Found by running it. One noise in a quiet room held the gate open for eighty
    milliseconds; the utterance that came out — a click, its pre-roll and a breath of silence
    — went to `large-v3-turbo`, which answered "Thank you." and had it typed into the window
    that had focus. Nobody had said anything."""
    gate = Gate()
    gate.feed(tone(400, 15))
    opened = gate.feed(tone(100, 6000))
    closed = gate.feed(SILENCE)

    assert opened.started, "the gate does open — it cannot tell a click from a syllable yet"
    assert closed.utterance is None, "and then throws away what turned out not to be speech"
    assert not gate.speaking


def test_a_real_word_still_gets_through() -> None:
    """The other half: the shortest thing somebody actually says must survive the rule that
    throws clicks away."""
    gate = Gate()
    gate.feed(tone(400, 15))
    gate.feed(tone(400, 6000))
    assert gate.feed(SILENCE).utterance is not None


async def test_a_draft_model_that_stops_answering_is_given_up_on() -> None:
    """A draft is a convenience — the accurate pass decodes the same audio a moment later —
    so its failure must not end the session, and must not repeat either. Left to raise, it ran
    again every tick for as long as dictation continued, the same unread traceback from a
    server that was never coming back."""
    from stt_cli._errors import EngineError

    class Broken:
        def __init__(self) -> None:
            self.asked = 0

        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            del pcm, prompt
            self.asked += 1
            raise EngineError(what="the draft model stopped answering", why="", how="")

    draft = Broken()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=draft,
        settled=FakeModel("Привет!"),
        draft_every=0.0,
    )
    said = await dictating.run(feed(SPEECH, SPEECH, SPEECH, SILENCE))

    assert said == "Привет!", "the accurate pass carried on alone"
    assert draft.asked == 1, "and the broken model was asked exactly once"


# ── the parts that talk to the outside ────────────────────────────────────────


def test_the_audio_devices_are_read_out_of_a_listing_full_of_cameras() -> None:
    """ffmpeg reports cameras and microphones in one listing, on stderr, and then exits
    non-zero because there was no input to open. Picking the audio half out of that is the
    whole of device selection, and it is the part that breaks when ffmpeg reformats."""
    from stt_cli.live.capture import _audio_inputs

    listing = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Camera
[AVFoundation indev @ 0x1] [1] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x1] [1] Headset Microphone
[in#0 @ 0x2] Error opening input: Input/output error
"""
    found = _audio_inputs(listing)
    assert [(d.index, d.name) for d in found] == [
        (0, "MacBook Pro Microphone"),
        (1, "Headset Microphone"),
    ]


def test_the_status_line_says_what_is_provisional() -> None:
    """The terminal line is what replaces the menu bar icon, so it has to distinguish a guess
    from a settled sentence — that distinction is the thing an underline would have carried."""
    from stt_cli.live import status

    assert "listening" in status.render(listening=True, text="", settled=False)
    draft = status.render(listening=True, text="привет", settled=False)
    final = status.render(listening=True, text="Привет!", settled=True)
    assert draft != final
    assert "привет" in draft and "Привет!" in final


def test_a_status_line_that_is_not_a_terminal_says_nothing_at_all() -> None:
    """Redirected into a file it must write nothing: the carriage returns that make it a live
    line make it unreadable noise anywhere else."""
    from stt_cli.live import status

    line = status.Line(enabled=False)
    line.show("● listening")
    line.clear()
    assert not line.enabled


def test_a_second_dictation_is_refused_while_one_is_running(tmp_path, monkeypatch) -> None:
    """Two sessions type into the same window and each deletes what IT believes it wrote,
    which is by then interleaved with the other's. The result is not two transcripts, it is
    one line of debris, and neither session can tell anything is wrong."""
    from stt_cli._errors import UsageError
    from stt_cli.live.dictation import _the_only_session

    monkeypatch.setenv("STT_HOME", str(tmp_path / "home"))
    with _the_only_session(), pytest.raises(UsageError) as raised, _the_only_session():
        pass
    assert "already running" in raised.value.what

    with _the_only_session():
        pass  # and the lock is gone once the first one finishes


def test_a_device_name_cannot_end_the_applescript_string_it_is_put_in() -> None:
    """The notification text reaches `osascript` as source code, and a device name is
    whatever the hardware calls itself. A quote in it would end the literal early — a syntax
    error at best, and at worst a name that decides what the rest of the script says."""
    from stt_cli.live.status import _quoted

    assert _quoted('Mic "Pro"') == '"Mic \\"Pro\\""'
    assert _quoted("back\\slash") == '"back\\\\slash"'
    assert "\n" not in _quoted("two\nlines") and "\x07" not in _quoted("bell\x07")


async def test_a_session_nobody_speaks_into_stops_by_itself() -> None:
    """A microphone left open is a room being recorded and a window that gets typed into by
    whoever speaks near the machine next. Nobody chooses that; they walk away from the
    terminal. The default is half an hour — longer than any pause inside real dictation."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("never asked"), idle_after=0.01
    )

    async def quiet_forever():
        for _ in range(200):
            yield tone(100, 15)
            await asyncio.sleep(0.001)

    assert await dictating.run(quiet_forever()) == ""
    assert keys.line == ""


async def test_a_session_told_never_to_stop_does_not() -> None:
    """`--idle-minutes 0` is somebody who knows what they are asking for."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Привет!"), idle_after=0.0
    )
    assert await dictating.run(feed(tone(600, 15), SPEECH, SILENCE)) == "Привет!"


async def test_a_noisy_empty_room_does_not_keep_a_forgotten_session_alive() -> None:
    """The gate opens for a click as readily as for a word — it needs a few frames before it
    can tell — and clicks are what an empty room produces all afternoon. Counting those as
    somebody speaking would mean a forgotten session never times out where it matters most."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel(), idle_after=0.15
    )

    async def clicking():
        for _ in range(200):
            yield tone(100, 15)
            yield tone(100, 6000)  # too short to be speech, long enough to open the gate
            await asyncio.sleep(0.001)

    assert await dictating.run(clicking()) == ""
    assert keys.line == ""


async def test_a_microphone_is_chosen_by_number_or_by_part_of_its_name(monkeypatch) -> None:
    """Device selection is the one thing that has to work before anything else can, and it
    is the first thing somebody with a headset plugged in has to do."""
    from stt_cli._errors import UsageError
    from stt_cli.live import capture

    devices = [capture.Device(0, "MacBook Pro Microphone"), capture.Device(1, "Headset Mic")]

    async def listed():
        return devices

    monkeypatch.setattr(capture, "devices", listed)
    assert (await capture.resolve(None)).index == 0
    assert (await capture.resolve("1")).index == 1
    assert (await capture.resolve("headset")).index == 1

    with pytest.raises(UsageError) as raised:
        await capture.resolve("mic")  # matches both names
    assert "Headset Mic" in raised.value.how, "and it says what the choices were"

    # A number is an index or it is nothing. It used to fall through to the name search,
    # so `--device 5` on this machine would have opened "Headset Mic" — no device is
    # numbered 5, but "5" is not in either name either, and a machine whose microphone is
    # called "EchoWhiskey536" would have been opened by exactly that mistake.
    devices.append(capture.Device(2, "EchoWhiskey536 Microphone"))
    with pytest.raises(UsageError) as missing:
        await capture.resolve("5")
    assert "index 5" in missing.value.what
    assert "EchoWhiskey536" in missing.value.how, "the real choices, not a guess at one"


class _EndedProcess:
    """An ffmpeg that has closed its pipe and exited with a status, and said nothing."""

    def __init__(self, status: int) -> None:
        self.returncode = status
        self.stdout = self

    async def read(self, _how_much: int) -> bytes:
        return b""

    async def wait(self) -> int:
        return self.returncode


async def test_a_microphone_that_dies_without_complaining_is_not_a_clean_finish() -> None:
    """The failure this hides is the worst-behaved one there is: nothing typed, nothing said,
    and an exit code of zero. A device unplugged mid-sentence closes the pipe without writing
    a diagnostic, and `stt mic` used to treat that as the recording having finished."""
    from stt_cli._errors import EngineError
    from stt_cli.live import capture

    with pytest.raises(EngineError) as raised:
        async for _block in capture._blocks(_EndedProcess(1), deque()):
            pass
    assert "status 1" in raised.value.why
    assert "Microphone" in raised.value.how, "and it points at the permission to check"


async def test_a_microphone_that_simply_ends_is_still_a_clean_finish() -> None:
    """The other half of the same judgement: an ffmpeg that exits zero of its own accord has
    not failed, and turning every end of the audio into an error would be worse than the bug
    — every ordinary session would finish by reporting a fault."""
    from stt_cli.live import capture

    blocks = [block async for block in capture._blocks(_EndedProcess(0), deque())]
    assert blocks == []


def test_typing_never_splits_an_emoji_in_half() -> None:
    """The API counts UTF-16 code units and an emoji is two of them. Cutting between them
    produces half a character, which does not decode — a chunk boundary that happened to land
    there used to drop it."""
    from stt_cli.live.quartz import _in_chunks

    text = "a" * 15 + "\N{GRINNING FACE}" + "b" * 20
    pieces = _in_chunks(text)
    assert "".join(pieces) == text
    assert all(piece.encode("utf-16-le") for piece in pieces)


def test_a_sentence_that_never_ends_is_cut_anyway() -> None:
    """Somebody who talks without pausing should still see text, and the accurate pass has to
    be handed a bounded amount of work."""
    from stt_cli.live.gate import MAX_UTTERANCE_MS, Gate

    gate = Gate()
    gate.feed(tone(400, 15))
    cut = None
    for _ in range(int(MAX_UTTERANCE_MS / 1000) + 3):
        heard = gate.feed(tone(1000, 6000))
        if heard.utterance:
            cut = heard.utterance
            break
    assert cut is not None, "it was cut without waiting for a pause"
    assert len(cut) / (RATE * 2) <= MAX_UTTERANCE_MS / 1000 + 1


def test_the_audio_reaches_the_model_as_a_wav_it_can_read() -> None:
    """The model is handed a file, not a stream of samples, and a header with the wrong rate
    in it transcribes silently and wrongly rather than failing."""
    import io
    import wave

    from stt_cli.live.server import _wav

    payload = _wav(tone(500, 4000))
    with wave.open(io.BytesIO(payload), "rb") as read_back:
        assert read_back.getframerate() == RATE
        assert read_back.getnchannels() == 1
        assert read_back.getsampwidth() == 2
        assert read_back.getnframes() == RATE // 2


def test_one_escape_lets_go_of_the_sentence_and_two_end_the_session() -> None:
    """A single Escape is something people press in other applications all day, so it cannot
    mean "stop". Two in half a second is not something anybody does by accident."""
    from stt_cli.live import tap
    from stt_cli.live.dictation import _Stopper

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel())
    dictating.typist.show("привет")
    stopper = _Stopper(dictating)

    stopper(tap.ESCAPE)
    assert dictating.typist.abandoned, "one press lets go of what is on screen"
    assert not dictating._stopped, "and does not end the session"

    stopper(tap.ESCAPE)
    assert dictating._stopped, "two in quick succession do"


def test_two_escapes_far_apart_are_two_separate_presses() -> None:
    from stt_cli.live import tap
    from stt_cli.live.dictation import DOUBLE_PRESS, _Stopper

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel())
    stopper = _Stopper(dictating)

    stopper(tap.ESCAPE)
    stopper._last_escape -= DOUBLE_PRESS * 2  # as if the second came much later
    stopper(tap.ESCAPE)
    assert not dictating._stopped


def test_any_other_key_lets_go_without_ending_anything() -> None:
    """Typing a word of your own mid-sentence means "leave that alone", not "stop"."""
    from stt_cli.live.dictation import _Stopper

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=FakeModel())
    _Stopper(dictating)(4)  # some ordinary key

    assert dictating.typist.abandoned and not dictating._stopped


async def test_a_sentence_you_took_over_is_never_typed_even_after_you_start_another() -> None:
    """Ownership is remembered per sentence, not as a single flag. With a flag: the user takes
    over sentence A, starts speaking sentence B before A has finished decoding, and beginning
    B cleared the flag — so A's answer arrived and was typed, text the user had explicitly
    rejected, at whatever caret they had moved to by then."""
    slow = asyncio.Event()
    taken_over = asyncio.Event()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=None,
        settled=FakeModel("Первое.", "Второе.", hold=slow),
    )

    async def two_sentences():
        yield SPEECH
        yield SILENCE  # sentence A is finished and handed to the model, which is holding
        await taken_over.wait()  # the user takes it over BEFORE sentence B begins
        yield SPEECH
        yield SILENCE

    running = asyncio.create_task(dictating.run(two_sentences()))
    for _ in range(30):
        await asyncio.sleep(0)
    assert dictating._utterance == 1, "still on sentence A when the user takes over"
    dictating.interrupt()
    taken_over.set()
    slow.set()
    said = await running

    assert "Первое." not in said, "the sentence the user took over was never typed"
    assert "Первое." not in keys.line
    assert "Второе." in said, "and the next one still works"


async def test_an_unexpected_failure_ends_the_session_rather_than_hanging_it() -> None:
    """With two sentences queued and the first raising something that is not a diagnosed
    error, the consumer task died with the second still in the queue — and `run` waited on a
    join that could never be satisfied, microphone open, forever."""

    class Odd:
        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            del pcm, prompt
            raise ValueError("not an SttError at all")

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=Odd())
    running = asyncio.create_task(dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE)))
    done, _ = await asyncio.wait({running}, timeout=2.0)

    assert done, "it finished rather than waiting forever"
    with pytest.raises(ValueError):
        await running


async def test_a_server_that_did_not_get_its_port_is_refused(monkeypatch) -> None:
    """The port is chosen by binding a socket, reading the number and closing it, because
    whisper-server has no "pick one and tell me" mode — and the readiness probe only asks
    whether SOMETHING answers there. A local process that grabbed the port in the gap and
    replied would otherwise be sent every recorded sentence and the glossary with it."""
    from stt_cli import proc
    from stt_cli._errors import EngineError

    class Pretend:
        pid = 4242
        returncode = None
        stderr = None

        def terminate(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return 0

    server = a_server(Pretend(), draining=True)

    async def somebody_else(argv, **kwargs):
        del argv, kwargs
        return proc.Result(code=0, stdout="9999\n", stderr="", argv=[])

    monkeypatch.setattr(proc, "which", lambda name: "/usr/sbin/lsof")
    monkeypatch.setattr(proc, "run", somebody_else)
    with pytest.raises(EngineError) as raised:
        await server._confirm_it_is_ours()
    assert "listening on the speech model's port" in raised.value.what


async def test_a_sentence_the_typist_declined_is_not_in_the_transcript_either() -> None:
    """Sentence A is settling, sentence B starts, the user presses a key during B. The
    interrupt marks B, so A is still allowed — but the typist has let go of the screen and
    declines it. Ignoring that answer put a sentence into the printed transcript that never
    reached the window: the two disagreeing about what happened."""
    slow = asyncio.Event()
    started_b = asyncio.Event()
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Первое.", hold=slow)
    )

    async def a_then_b():
        yield SPEECH
        yield SILENCE  # A is finished and handed to the model, which is holding
        yield SPEECH  # B begins
        started_b.set()
        await slow.wait()
        yield SILENCE

    running = asyncio.create_task(dictating.run(a_then_b()))
    await started_b.wait()
    dictating.interrupt()  # the user types during B; A is still in the model
    slow.set()
    said = await running

    assert keys.line == "", "nothing was typed"
    assert said == "", "and nothing was recorded as though it had been"


def test_a_keyboard_watcher_that_never_starts_is_a_failure_not_a_silence(monkeypatch) -> None:
    """Not reaching the end of the tap's setup is as much a failure as being refused by it,
    and it used to look like success — dictation beginning with nothing watching for the
    user's own keystrokes, which is when synthetic corrections start deleting text stt did
    not type."""
    from stt_cli.live import tap

    watcher = tap.Watcher(marker=tap.new_marker(), on_key=lambda code: None)
    monkeypatch.setattr(watcher._ready, "wait", lambda timeout=None: False)
    monkeypatch.setattr(watcher, "_run", lambda: None)
    watcher.start()

    assert watcher.failure is not None
    assert "did not start" in watcher.failure


def test_a_quiet_voice_is_not_cut_off_by_its_own_noise_floor() -> None:
    """The floor is a model of the ROOM, and while somebody is talking you are not observing
    the room. Learning from speech frames too made it climb four percent a second through a
    long sentence until the sentence stopped clearing its own threshold — measured, a steady
    quiet voice was cut off after thirteen seconds, and every second after that raised the
    floor further, so the rest of what they said never reached a model at all."""
    from stt_cli.live.gate import Gate

    gate = Gate()
    gate.feed(tone(1000, 15))
    for _ in range(30):
        gate.feed(tone(1000, 2500))  # thirty seconds of steady, quiet speech

    assert gate.speaking, "still hearing them after thirty seconds"
    assert gate.floor < 2500 / 6, "the floor never climbed toward the voice"
    remainder = gate.close()
    assert remainder is not None and len(remainder) / (RATE * 2) > 1.0


async def test_a_dictionary_turned_off_does_not_reach_the_models_or_the_text() -> None:
    """The file pipeline honours `dictionary` and `dict_bias`; live dictation was ignoring
    both. Somebody who turned the dictionary off to keep the spelling they dictated was
    having it corrected anyway, and their glossary sent to the models on top."""
    from stt_cli import dictionary

    terms = dictionary.Dictionary([dictionary.Term(term="Figma", aka=["Вигма"])])
    asked: list[str] = []

    class Listening(FakeModel):
        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            asked.append(prompt)
            return await super().transcribe(pcm, prompt=prompt)

    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=None,
        settled=Listening("открой Вигма"),
        terms=terms,
        bias=False,
    )
    said = await dictating.run(feed(SPEECH, SILENCE))

    assert asked == [""], "no glossary was sent to the speech model"
    assert said == "открой Figma", "and the recorded spelling is still corrected"


async def test_a_draft_nothing_replaces_is_taken_back_not_left_behind() -> None:
    """A cough gets through the gate, the fast model hears an ordinary word in it and types
    that word, and the accurate model answers with something the filters drop. The draft was
    then left on screen for good: `settle()` forgot it was ours, so no later edit could reach
    it. That is stt typing a word nobody said into somebody's window and walking away."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("так"),
        settled=FakeModel("Субтитры сделал DimaTorzok"),
        draft_every=0.0,
    )
    said = await dictating.run(feed(SPEECH, SPEECH, SILENCE))

    assert keys.line == "", f"the draft was taken back, not abandoned: {keys.line!r}"
    assert said == ""


async def test_a_speech_model_that_died_is_not_sent_the_next_sentence() -> None:
    """The port was verified as ours at startup, and that expires the moment the process
    behind it exits — the port is then free for anything on the machine to bind, and the next
    sentence plus the glossary would go to whatever answered."""
    from stt_cli._errors import EngineError

    class Dead:
        returncode = 1

    server = a_server(Dead())

    with pytest.raises(EngineError) as raised:
        await server.transcribe(b"\x00\x00" * 100)
    assert "no longer running" in raised.value.what


def test_backspaces_are_counted_the_way_a_text_field_counts_them() -> None:
    """A text field deletes what a reader calls one character. "é" as e-plus-accent, or a
    family emoji as four people joined by zero-width joiners, is several code points and one
    of those. Counting code points asked for seven backspaces where the field needed one, and
    the other six came out of whatever the user had written before stt started typing."""
    from stt_cli.live.typist import plan

    family = "\N{MAN}‍\N{WOMAN}‍\N{GIRL}‍\N{BOY}"
    assert plan(family, "x").backspaces == 1
    assert plan("é", "x").backspaces == 1  # e + combining acute
    assert (
        plan(
            "\N{REGIONAL INDICATOR SYMBOL LETTER R}\N{REGIONAL INDICATOR SYMBOL LETTER U}", "x"
        ).backspaces
        == 1
    )
    assert plan("привет", "").backspaces == 6, "ordinary text is still one each"


def test_a_rewrite_never_stops_inside_a_character() -> None:
    """A common prefix that ends halfway through a cluster would leave the caret inside one,
    where neither half of the count means anything."""
    from stt_cli.live.typist import plan

    edit = plan("да é", "да e")  # the accent is dropped
    assert edit.backspaces == 1 and edit.text == "e", "it went back to the whole character"


async def test_a_queued_sentence_still_counts_as_settling() -> None:
    """Sentence A is slow; B finishes and queues behind it; A returns and the worker picks up
    B. Clearing the settling flag between those two let the user's next sentence be drafted
    and typed — and then B came back and rewrote that draft into B's older words."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("черновик"),
        settled=FakeModel("Первое.", "Второе."),
        draft_every=0.0,
    )
    said = await dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE))

    assert said == "Первое. Второе.", f"in the order they were spoken: {said!r}"
    assert keys.line == "Первое. Второе."


def test_two_flags_next_to_each_other_are_two_characters() -> None:
    """A flag is two regional indicators, so pairing them by looking back two places was
    right for one flag and wrong for two: the fourth indicator saw two behind it, decided it
    was starting a fresh flag, and made the pair count as three. An over-count is the
    direction that deletes what the user wrote rather than what stt wrote."""
    from stt_cli.live.typist import _clusters

    us = "\N{REGIONAL INDICATOR SYMBOL LETTER U}\N{REGIONAL INDICATOR SYMBOL LETTER S}"
    ca = "\N{REGIONAL INDICATOR SYMBOL LETTER C}\N{REGIONAL INDICATOR SYMBOL LETTER A}"
    ru = "\N{REGIONAL INDICATOR SYMBOL LETTER R}\N{REGIONAL INDICATOR SYMBOL LETTER U}"

    assert _clusters(us) == 1
    assert _clusters(us + ca) == 2
    assert _clusters(us + ca + ru) == 3
    assert _clusters(us + "да") == 3, "and ordinary letters after one still count singly"


def test_an_idle_time_that_would_silently_mean_never_is_refused() -> None:
    """`--idle-minutes -1` parsed, became zero, and switched the timeout off — which is what
    0 means, but nobody types a minus sign to ask for it. A typo left the microphone open
    indefinitely, the one outcome the timeout exists to prevent."""
    import argparse

    from stt_cli.commands.mic import _minutes

    assert _minutes("12.5") == 12.5
    assert _minutes("0") == 0.0
    for refused in ("-1", "nan", "inf", "-inf", "soon"):
        with pytest.raises(argparse.ArgumentTypeError):
            _minutes(refused)


async def test_a_draft_that_fails_while_being_typed_gives_up_quietly() -> None:
    """Showing the text is where a bad dictionary term or a refused keystroke would raise, and
    that happened outside the guard: the task died with nobody to receive the exception, and
    the next tick started another draft that failed the same way, once a second, all session."""

    class Refusing:
        def type_text(self, text: str) -> None:
            raise OSError("the keyboard went away")

        def press_backspace(self, times: int) -> None:
            raise OSError("the keyboard went away")

    draft = FakeModel("черновик", "ещё", "и ещё")
    dictating = live.Session(
        typist=Typist(keys=Refusing()),
        draft=draft,
        settled=FakeModel(""),  # nothing to type, so only the draft touches the keyboard
        draft_every=0.0,
    )
    assert await dictating.run(feed(SPEECH, SPEECH, SPEECH, SILENCE)) == ""

    assert dictating.draft is None, "it gave up on drafting rather than failing every tick"
    assert draft.asked == 1


def test_doctor_asks_about_the_checkout_the_user_configured(tmp_path, monkeypatch) -> None:
    """`stt doctor` looked only on PATH while `stt mic` honours the configured checkout, so
    somebody who had installed whisper.cpp where they told stt about it was shown "NOT found"
    and an install instruction for something they already had."""
    from stt_cli import config
    from stt_cli.backends import whispercpp
    from stt_cli.commands import doctor

    monkeypatch.setenv("STT_HOME", str(tmp_path))
    config.ensure_dirs()
    config.save_setting("whispercpp_root", "/somewhere/of/my/own")

    asked: list[str | None] = []

    def remember(root=None):
        asked.append(root)
        return None

    monkeypatch.setattr(whispercpp, "server_binary", remember)
    doctor._dictation_line()

    assert asked == ["/somewhere/of/my/own"]


def test_doctor_survives_the_broken_config_it_exists_to_report(
    tmp_path, monkeypatch, capsys
) -> None:
    """`stt doctor` is the command somebody runs BECAUSE their setup is broken, and a
    hand-edited config.json is one of the ways it can be. Reading it used to raise straight
    out of this line, so the report stopped here and the storage section and the summary
    never printed — half a report from the one command whose job is to say what is wrong."""
    from stt_cli import config
    from stt_cli.commands import doctor

    monkeypatch.setenv("STT_HOME", str(tmp_path))
    config.ensure_dirs()
    config.config_path().write_text("{ this is not json", encoding="utf-8")

    doctor._dictation_line()  # must not raise

    said = capsys.readouterr().out
    assert "config unreadable" in said
    assert "fix:" in said, "and it says what to do about it"


def test_a_conjunct_or_a_syllable_block_is_one_press_of_backspace() -> None:
    """The claim that this counting is "safe because it under-counts" was false for scripts
    nobody here had tried. `क्ष` is one character in a text field and was counted as two,
    because the consonant after the virama looked like a fresh start — and the second
    backspace comes out of whatever was in front of stt's text."""
    from stt_cli.live.typist import _clusters

    assert _clusters("क्ष") == 1, "Devanagari conjunct: क + virama + ष"
    assert _clusters("각") == 1, "Hangul jamo: lead, vowel, tail"
    assert _clusters("हिन्दी") == 2, "हिन्दी is two characters"
    assert _clusters("привет") == 6, "and ordinary letters are still one each"


def test_ownership_lost_mid_edit_is_not_taken_back() -> None:
    """`disown` runs on the keyboard watcher's thread and can land between the backspaces and
    the text. The old code then wrote its own idea of the line over the empty value `disown`
    had just set, so the typist went on believing it owned characters at a caret the user had
    since moved — and the next correction backspaced over whatever was now in front of it."""
    keys = FakeKeyboard()
    typist = Typist(keys=keys)
    typist.begin()
    typist.show("привет")

    class InterruptingKeyboard(FakeKeyboard):
        def type_text(self, text: str) -> None:
            typist.disown()  # the user clicks, exactly here
            super().type_text(text)

    typist.keys = InterruptingKeyboard()
    assert typist.show("привет мир") is False
    assert typist.shown == "", "it did not take back what it had just let go of"

    typist.begin()  # the next sentence starts
    assert typist.shown == "", "and still owns nothing from before the interruption"


async def test_a_guess_typed_at_a_noise_is_taken_back_when_the_noise_is_discarded() -> None:
    """The flagship failure, by a second route. A click clears the gate's onset, the fast
    model hears an ordinary word in it and types that word — and then the gate throws the
    utterance away for holding no real speech, so no accurate pass ever runs and nothing was
    left to take the word back. It stayed in the window as if somebody had said it."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys),
        draft=FakeModel("так"),
        settled=FakeModel("never asked"),
        draft_every=0.0,
    )

    async def one_click():
        yield tone(400, 15)
        yield tone(100, 6000)  # opens the gate, too short to be speech
        for _ in range(12):
            yield tone(100, 15)
            await asyncio.sleep(0)

    said = await dictating.run(one_click())

    assert keys.line == "", f"the guess was taken back: {keys.line!r}"
    assert said == ""


def test_a_rewrite_stops_the_moment_the_user_interrupts() -> None:
    """A sentence rewrite is a couple of hundred backspaces. A click landing after the first
    of them used to be answered by posting the rest anyway, deleting whatever the user had
    just typed at their new caret."""
    typed: list[str] = []

    class CountingKeyboard:
        def __init__(self) -> None:
            self.presses = 0

        def press_backspace(self, times: int) -> None:
            self.presses += times
            if self.presses == 2:
                typist.disown()  # the user clicks, two backspaces in

        def type_text(self, text: str) -> None:
            typed.append(text)

    keys = CountingKeyboard()
    typist = Typist(keys=keys)
    typist.begin()
    typist.shown = "двенадцать"  # ten characters on screen, all ours

    assert typist.show("д") is False
    assert keys.presses == 2, f"it stopped at the interruption, not after all nine: {keys.presses}"
    assert typed == [], "and never typed the replacement"


async def test_the_connection_to_a_model_is_opened_once(monkeypatch) -> None:
    """A TCP connection, once established, cannot be handed to somebody else — which is what
    makes checking the port's owner at connect time enough. Reopening per request would put
    the gap back.

    ONE check per open, and this pins the count because the cost is on a hot path rather
    than at startup: whisper-server closes an idle connection between sentences, so every
    pause for thought is followed by a reopen, and each check is an `lsof` subprocess worth
    about a tenth of a second on the latency of the next sentence. There were two, before
    and after connecting; the earlier one bought nothing, since connecting to the wrong
    process is harmless and nothing is sent until the later check has passed.
    """
    from stt_cli.live.server import WhisperServer

    class Alive:
        pid = 1
        returncode = None

    checks = 0

    class FakeLink:
        def connect(self) -> None: ...
        def request(self, *a, **k) -> None: ...
        def getresponse(self):
            return _Reply()

    class _Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return '{"text": "привет"}'.encode()

    async def counted(self) -> None:
        nonlocal checks
        checks += 1

    server = a_server(Alive())
    monkeypatch.setattr(WhisperServer, "_confirm_it_is_ours", counted)
    monkeypatch.setattr("http.client.HTTPConnection", lambda *a, **k: FakeLink())

    for _ in range(3):
        assert await server.transcribe(b"\x00\x00" * 100) == "привет"
    assert checks == 1, "checked once while connecting, then not again"


async def test_a_refusal_is_reported_rather_than_read_as_silence() -> None:
    """A refusal comes back as a JSON object with an `error` key and no `text`. Reading only
    `text` turned that into an empty answer — indistinguishable from "the model heard
    nothing" — so dictation removed every draft and typed nothing, all session, with no reason
    given anywhere."""
    from stt_cli._errors import EngineError

    class Alive:
        pid = 1
        returncode = None

    class Refusing:
        status = 400

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return b'{"error": "failed to read audio data"}'

    class Link:
        def connect(self) -> None: ...

        def request(self, *a, **k) -> None: ...

        def getresponse(self):
            return Refusing()

        def close(self) -> None: ...

    server = a_server(Alive(), link=Link())

    with pytest.raises(EngineError) as raised:
        await server.transcribe(b"\x00\x00" * 100)
    assert "refused the audio" in raised.value.what
    assert "400" in raised.value.why and "failed to read audio data" in raised.value.why


async def test_a_fatal_failure_drops_the_sentences_still_waiting() -> None:
    """`run` waits for the queue to drain, so every sentence spoken while the failing one was
    decoding still got sent to a model that had just proved it does not answer — each able to
    wait out its timeout in turn, with the microphone open throughout."""
    from stt_cli._errors import EngineError

    holding = asyncio.Event()
    asked = 0

    class SlowThenFatal:
        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            nonlocal asked
            asked += 1
            if asked == 1:
                await holding.wait()  # long enough for the next sentences to queue behind it
                raise EngineError(what="the speech model stopped answering", why="", how="")
            return "should never be asked"

    keys = FakeKeyboard()
    dictating = live.Session(typist=Typist(keys=keys), draft=None, settled=SlowThenFatal())
    running = asyncio.create_task(
        dictating.run(feed(SPEECH, SILENCE, SPEECH, SILENCE, SPEECH, SILENCE))
    )
    for _ in range(40):
        await asyncio.sleep(0)
    assert dictating._queue.qsize() >= 1, "sentences really are waiting behind the slow one"

    holding.set()
    done, _ = await asyncio.wait({running}, timeout=2.0)

    assert done, "it finished at once rather than working through the queue"
    with pytest.raises(EngineError):
        await running
    assert asked == 1, f"the model was not asked again after it failed: asked {asked} times"
    assert keys.line == "", "and nothing was typed after the fatal error"


async def test_a_failure_while_the_next_sentence_is_being_spoken_does_not_hang() -> None:
    """The settler ends itself on a fatal failure, so it is the only thing that ever calls
    `task_done` and there is nothing left listening. Queueing the half-spoken sentence after
    that left `run` waiting on a join that could never complete, microphone open — the same
    hang the failure handling was written to prevent, one line further along."""
    from stt_cli._errors import EngineError

    holding = asyncio.Event()

    class FailsWhileTheyKeepTalking:
        async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
            await holding.wait()
            raise EngineError(what="the speech model stopped answering", why="", how="")

    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FailsWhileTheyKeepTalking()
    )

    async def one_sentence_then_still_speaking():
        yield SPEECH
        yield SILENCE  # sentence A is queued and the model takes it
        yield SPEECH  # sentence B begins and never finishes: the microphone just stops
        holding.set()  # A fails now, with B still open in the gate
        for _ in range(5):
            yield SPEECH
            await asyncio.sleep(0)

    running = asyncio.create_task(dictating.run(one_sentence_then_still_speaking()))
    done, _ = await asyncio.wait({running}, timeout=2.0)

    assert done, "it finished rather than waiting for a consumer that had gone"
    with pytest.raises(EngineError):
        await running


async def test_a_refused_connection_is_a_sentence_not_a_traceback() -> None:
    """The server can pass the exit-code check and be gone before the first connect. Nothing
    is listening, the connect is refused, and an untranslated `ConnectionRefusedError` travels
    all the way out through the error guard, which only knows how to render an `SttError` —
    so the user gets a Python traceback where they were promised a sentence."""
    from stt_cli._errors import EngineError, SttError
    from stt_cli.live.server import WhisperServer

    class Alive:
        pid = 1
        returncode = None

    async def owned(self) -> None: ...

    server = a_server(Alive(), port=1)  # nothing is listening there

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(WhisperServer, "_confirm_it_is_ours", owned)
        with pytest.raises(EngineError) as raised:
            await server.transcribe(b"\x00\x00" * 100)

    assert isinstance(raised.value, SttError), "the guard can render it"
    assert "could not reach" in raised.value.what
    assert raised.value.how


def test_typing_stops_between_pieces_when_the_user_interrupts() -> None:
    """The replacement for a settled sentence can be a hundred characters, posted in bursts.
    A click landing after the first burst was answered by posting the rest anyway, into the
    window the click had just focused."""
    posted: list[str] = []

    class InterruptingKeyboard:
        def press_backspace(self, times: int) -> None: ...

        def type_text(self, text: str) -> None:
            posted.append(text)
            if len(posted) == 1:
                typist.disown()  # the user clicks after the first piece

    typist = Typist(keys=InterruptingKeyboard())
    typist.begin()

    whole = "привет мир, как у тебя сегодня дела"
    assert typist.show(whole) is False
    assert len(posted) == 1, f"it stopped after the first piece: {posted}"
    assert posted[0] != whole, "and that piece was not the entire sentence"
    assert len(posted[0]) < len(whole) / 2, f"only a small piece went out: {posted[0]!r}"


def test_a_piece_never_ends_inside_a_character() -> None:
    """Splitting the replacement for the ownership checks must not split a character in half:
    the halves would be typed as two broken things rather than one whole one."""
    from stt_cli.live.typist import _in_pieces

    family = "\N{MAN}‍\N{WOMAN}‍\N{GIRL}‍\N{BOY}"
    for text in ("привет мир, как дела сегодня", "क्ष" * 3, family * 2, "да"):
        pieces = _in_pieces(text)
        assert "".join(pieces) == text
        assert all(pieces), "no empty piece, so the loop always makes progress"


async def test_dictation_runs_without_the_fast_model_rather_than_refusing(tmp_path) -> None:
    """`stt setup` downloads the one model it was asked for, so somebody who accepted the
    default has `large-v3-turbo` and no `base` — and the first `stt mic` they ever ran failed
    before opening the microphone, over a model nobody had told them they needed."""
    from stt_cli._errors import EngineError
    from stt_cli.backends import whispercpp
    from stt_cli.live.dictation import _the_draft_is_installed

    said: list[str] = []

    class NothingInstalled(whispercpp.WhisperCppBackend):
        def model_path(self, model: str):
            raise EngineError(what=f"model file for {model} is missing", why="", how="")

    assert _the_draft_is_installed(NothingInstalled(), "base", said.append) is False
    assert said and "stt models pull base" in said[0]
    del tmp_path


def test_every_click_that_can_move_the_caret_is_watched() -> None:
    """A right-click opens a menu somewhere else and an extra button is bound to whatever its
    owner chose; both put the caret in a different window. Only the left one was watched, so
    a correction after one of the others went wherever the click had landed."""
    from stt_cli.live import tap

    assert set(tap._MOUSE_DOWN) == {1, 3, 25}, "left, right and other button down"
    watched = (1 << tap._KEY_DOWN) | sum(1 << button for button in tap._MOUSE_DOWN)
    for event in (tap._KEY_DOWN, *tap._MOUSE_DOWN):
        assert watched & (1 << event), f"event {event} is in the mask"


def test_a_character_too_long_for_a_piece_goes_out_whole() -> None:
    """A letter carrying a dozen combining marks is one character and longer than a piece.
    Cutting it to fit produced a piece that was half a character — and if the user interrupted
    right there, that half is what they were left looking at."""
    from stt_cli.live.typist import _clusters, _in_pieces

    long_one = "a" + "\N{COMBINING ACUTE ACCENT}" * 12
    for text in (long_one, long_one + "да", "привет мир, как дела"):
        pieces = _in_pieces(text)
        assert "".join(pieces) == text
        assert all(pieces), "every piece has something in it"
        # No piece may end inside a character: the pieces' character counts must add up.
        assert sum(_clusters(piece) for piece in pieces) == _clusters(text)


async def test_a_long_sentence_does_not_trip_the_idle_timer(monkeypatch) -> None:
    """The idle timer was only reset when an utterance CLOSED, and the gate holds one open
    until a pause or twenty-five seconds. So a short `--idle-minutes` and one long sentence
    ended the session mid-word — the opposite of what "nobody has spoken" means."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Договорил."), idle_after=1.0
    )

    delivered = 0

    # A clock the test moves itself. It used to be the real one, with a fifty-millisecond
    # idle timeout and five-millisecond sleeps, and that made the test fail whenever the
    # machine was busy: a stall during the quiet lead-in — before any speech has reset the
    # timer — is indistinguishable from a silence long enough to end the session. Measuring
    # elapsed time against a clock nobody controls tests the machine's load, not the code.
    class _Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

    clock = _Clock()
    # Through the fixture, so the real clock is put back afterwards. Assigning it directly
    # would have left every later test in the file running on a clock that never moves.
    monkeypatch.setattr(live, "time", clock)

    async def one_long_sentence():
        nonlocal delivered
        yield tone(400, 15)
        delivered += 1
        for _ in range(40):  # speaking without pausing, well past the idle time
            yield tone(100, 6000)
            delivered += 1
            clock.now += 0.1  # four seconds of speech, against a one-second idle timeout
        yield SILENCE
        delivered += 1

    said = await dictating.run(one_long_sentence())

    # The transcript alone proves nothing: the trailing `gate.close()` salvages whatever was
    # in flight, so the sentence comes out either way. What tells them apart is whether the
    # microphone was still being read when the person stopped talking.
    assert delivered == 42, f"it kept listening to the end: read {delivered} of 42 blocks"
    assert said == "Договорил."


async def test_two_requests_never_share_the_connection() -> None:
    """Cancelling the task waiting on a worker thread does not stop the thread. A cancelled
    draft can still be mid-request when the next one starts, and two requests sharing one
    `HTTPConnection` is not something it supports: the replies come back interleaved, or
    matched to the wrong question."""
    from stt_cli.live.server import WhisperServer

    class Alive:
        pid = 1
        returncode = None

    overlapping = []
    inside = 0

    class Link:
        def request(self, *a, **k) -> None:
            nonlocal inside
            inside += 1
            overlapping.append(inside > 1)
            time.sleep(0.05)

        def getresponse(self):
            return _Reply()

        def close(self) -> None: ...

    class _Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            nonlocal inside
            inside -= 1
            return False

        def read(self) -> bytes:
            return '{"text": "да"}'.encode()

    server = a_server(Alive(), link=Link())

    async def owned(self) -> None: ...

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(WhisperServer, "_confirm_it_is_ours", owned)
        await asyncio.gather(*(server.transcribe(b"\x00\x00" * 10) for _ in range(4)))

    assert not any(overlapping), "no two requests were on the connection at once"


async def test_a_wrong_model_name_reaches_the_user() -> None:
    """Only a model that exists and is not downloaded is downgraded to "no fast guess". A
    name that is not a model at all was swallowed the same way, and the note told the user to
    run `stt models pull definitely-not-a-model` — worse than saying nothing, because it
    reads like the answer."""
    from stt_cli._errors import EngineError, UnknownItemError, UsageError, unknown_item
    from stt_cli.backends import whispercpp
    from stt_cli.live.dictation import _the_draft_is_installed

    said: list[str] = []

    class NotDownloaded(whispercpp.WhisperCppBackend):
        def model_path(self, model: str):
            raise EngineError(what=f"model file for {model} is missing", why="", how="")

    class NoSuchName(whispercpp.WhisperCppBackend):
        def model_path(self, model: str):
            raise unknown_item("model", model, ["base"])

    class NoBuildForIt(whispercpp.WhisperCppBackend):
        def model_path(self, model: str):
            raise UsageError(what=f"model {model!r} has no whispercpp build", why="", how="")

    assert _the_draft_is_installed(NotDownloaded(), "base", said.append) is False
    # Named types, not a tuple ending in `Exception`. It used to end in `Exception`, which
    # catches everything — so the assertion held whatever the code raised, including the
    # `EngineError` this test exists to prove is NOT raised for these two.
    with pytest.raises(UnknownItemError):
        _the_draft_is_installed(NoSuchName(), "nonsense", said.append)
    with pytest.raises(UsageError):
        _the_draft_is_installed(NoBuildForIt(), "nonsense", said.append)
    assert len(said) == 1, "only the downloadable one was quietly skipped"


def test_the_thread_count_is_the_one_the_user_configured() -> None:
    """Zero keeps the meaning it has everywhere else in stt: let the engine decide. It used to
    become four — so somebody who had set eight got four, and somebody who asked for the
    engine default got four as well, two servers between them oversubscribing the machine."""
    from pathlib import Path as _Path

    from stt_cli.live.server import Voice, _argv

    told = _argv(_Path("/bin/true"), Voice(model=_Path("m.bin"), language="ru", threads=8), 1)
    assert "-t" in told and told[told.index("-t") + 1] == "8"

    left_alone = _argv(_Path("/bin/true"), Voice(model=_Path("m.bin"), language="ru"), 1)
    assert "-t" not in left_alone, "zero passes no thread count at all"


async def test_clicking_somewhere_and_then_speaking_puts_the_text_there() -> None:
    """This is the interaction, not a bug, and it was reported as one — so it is written down.

    Click where you want the words, then say them. The click moves the caret; the sentence
    that follows belongs at the new caret and is typed there. Refusing to type it — on the
    grounds that a click happened — would break the ordinary way anybody uses dictation.

    What the click DOES protect is the sentence already on screen when it happened: that one
    is never touched again, which is `test_a_sentence_you_took_over_is_never_typed...`. The
    two rules are different halves of the same promise, and only the first has to leave the
    user free to move."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Сказанное после клика.")
    )

    async def clicked_then_spoke():
        yield tone(400, 15)
        dictating.interrupt()  # the user clicks into another window, having said nothing yet
        yield SPEECH  # and only then starts speaking
        yield SILENCE

    said = await dictating.run(clicked_then_spoke())

    assert said == "Сказанное после клика."
    assert keys.line == "Сказанное после клика.", "the new sentence goes where they clicked"


async def test_a_stale_request_does_not_close_the_healthy_connection() -> None:
    """A cancelled draft leaves its worker thread inside the request. If that stale request
    fails after a newer one has already opened a replacement, closing "the current
    connection" closes the HEALTHY one — the retry that was working is cut off and drafting
    stops for the rest of the session."""

    class Link:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            self.closed = True

    stale, healthy = Link("stale"), Link("healthy")
    server = a_server(link=healthy)

    server._forget_the_link(stale)  # the stale worker's request fails, late

    assert stale.closed, "the one that went wrong is closed"
    assert not healthy.closed, "and the one in use is left alone"
    assert server._link is healthy, "so the next request still has it"


async def test_ctrl_c_keeps_the_sentence_it_interrupted() -> None:
    """Ctrl-C is a key press like any other, so the watcher sees it and lets go of the
    sentence in flight — and the last thing said then went nowhere at all, which is not what
    "Ctrl-C finishes the session and prints the transcript" promises. It is kept when
    stopping, and still not typed: the focus is in the terminal by then, and the words belong
    in the printed transcript rather than in the shell."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Последнее слово.")
    )

    async def speaking_when_they_stop():
        yield SPEECH
        # Ctrl-C: the tap sees the key, then the signal handler asks the session to stop.
        dictating.interrupt()
        dictating.stop()
        yield SILENCE

    said = await dictating.run(speaking_when_they_stop())

    assert said == "Последнее слово.", "the last sentence survived into the transcript"
    assert keys.line == "", "and was not typed into whatever had focus"


async def test_a_stop_that_never_reached_the_tap_still_types_nothing() -> None:
    """The same ending, arriving by the other door. `kill -INT`, or a tap macOS disabled,
    reaches `stop()` without `interrupt()` ever running — and the sentence was still owned,
    so its answer was typed into the terminal the user had just come back to."""
    keys = FakeKeyboard()
    dictating = live.Session(
        typist=Typist(keys=keys), draft=None, settled=FakeModel("Последнее слово.")
    )

    async def stopped_from_outside():
        yield SPEECH
        dictating.stop()  # and no interrupt(): nothing told the tap
        yield SILENCE

    said = await dictating.run(stopped_from_outside())

    assert said == "Последнее слово.", "the transcript keeps it, exactly as with Ctrl-C"
    assert keys.line == "", "and nothing was typed into the focused window"
