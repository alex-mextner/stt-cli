"""session — the two-speed loop that turns an open microphone into typed text.

THE SHAPE OF IT
    One utterance at a time, and two answers about each. The draft model is asked about the
    sentence as it grows, every second or so, and each answer replaces the last one on
    screen — that is what makes words appear while they are still being said. When the gate
    decides the sentence is over, the accurate model is asked about the whole of it once,
    and its answer replaces the drafts. So the text is fast first and right second, and the
    correction happens under the user rather than after them.

WHY THE TWO PASSES ARE NOT ALLOWED TO OVERLAP
    They are decoding different audio and both want to write at the same caret. If the
    accurate answer for one sentence and the first draft of the next could arrive in either
    order, the text would sometimes assemble itself backwards. So drafting stops while a
    sentence is being settled, and starts again once its final text is on screen. Nothing is
    lost while that happens — the gate keeps recording — the next sentence's first draft is
    simply a little later than it could theoretically have been.

WHAT MAKES IT SAFE TO EDIT SOMEBODY ELSE'S WINDOW
    `typist.Typist` may only delete what it typed. The keyboard watcher tells this loop the
    moment a human presses a key, and the answer is to let go of the sentence in flight
    entirely: not to tidy it, not to finish it, just to stop touching it. See `Typist.disown`.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from .. import cleaning, phrases
from .. import dictionary as dict_mod
from ..models import Segment
from .gate import Gate
from .server import Transcriber
from .typist import Typist

# How often the draft model is asked about the sentence so far. Not set by what the model
# costs — `base` answers about a three-second window in forty milliseconds — but by what is
# comfortable to watch: every answer rewrites the tail of the line, and a line that rewrites
# itself twice a second reads as flicker rather than as dictation.
DRAFT_EVERY = 0.6

# How many times a phrase may repeat before it is a loop rather than emphasis. The same
# default the file pipeline uses.
MAX_REPEATS = 3

# How long a session will sit through complete silence before deciding it was forgotten. A
# microphone left open is a room being recorded and a window that gets typed into by whoever
# speaks near the machine next; nobody chooses that, they just walk away from the terminal.
# Half an hour is longer than any pause inside real dictation and shorter than an afternoon.
IDLE_MINUTES = 30.0


@dataclass
class Progress:
    """What the status line has to say. The session never prints; it reports."""

    listening: bool
    text: str = ""
    settled: bool = False


Report = Callable[[Progress], None]


def _report_nothing(progress: Progress) -> None:
    """The default `report`: a session nobody is watching still has to run."""


@dataclass
class Session:
    """One dictation, from the first block of audio to the last word typed."""

    typist: Typist
    settled: Transcriber
    draft: Transcriber | None = None
    terms: dict_mod.Dictionary = field(default_factory=dict_mod.Dictionary)
    # Whether the glossary may be fed to the speech models as a prompt. Separate from having
    # a dictionary at all, exactly as it is for a file: `dict_bias` off means "fix my
    # spellings afterwards, but do not tell the decoder what to expect".
    bias: bool = True
    report: Report = _report_nothing
    # A field rather than the constant directly, so a test can ask for a draft on every
    # block instead of waiting out a cadence chosen for human eyes.
    draft_every: float = DRAFT_EVERY
    # Seconds of unbroken silence after which the session ends itself. Zero means never.
    idle_after: float = IDLE_MINUTES * 60
    # Compiled once. See `_cleaned` for why live dictation needs them at all when the
    # gate is supposed to have made them impossible.
    refuse: list[re.Pattern[str]] = field(default_factory=lambda: phrases.compile_all([])[0])
    # The speech detector, which a caller MAY supply because its threshold is configurable.
    # Public, unlike the runtime state below it: `Session(..., _gate=...)` was a legal call
    # with an underscored keyword, which reads as a mistake wherever it appears.
    gate: Gate = field(default_factory=Gate)

    # Everything below is this session's own running state, not something to construct it
    # with. As ordinary dataclass fields they were all accepted as keyword arguments —
    # `Session(_said=["hi"], _stopped=True)` was a legal call — and every one of them had to
    # carry a default whether or not it wanted one. `init=False` says what they are.
    # Each finished sentence is queued with the number of the sentence it is, so an answer
    # that comes back can be checked against the sentence it was an answer TO.
    _queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=asyncio.Queue, init=False)
    _utterance: int = field(default=0, init=False)
    _taken_over: set[int] = field(default_factory=set, init=False)
    _said: list[str] = field(default_factory=list, init=False)
    _lead: str = field(default="", init=False)
    _settling: bool = field(default=False, init=False)
    _drafting: asyncio.Task[None] | None = field(default=None, init=False)
    _drafted_at: float = field(default=0.0, init=False)
    _heard_at: float = field(default=0.0, init=False)
    _stopped: bool = field(default=False, init=False)
    _failure: Exception | None = field(default=None, init=False)

    @property
    def transcript(self) -> str:
        """Everything the accurate pass settled on, in the order it was spoken."""
        return "".join(self._said)

    def interrupt(self) -> None:
        """A human touched the keyboard. The sentence in flight stops being ours — for good.

        Remembered by sentence number, not as a single flag. The flag alone was wrong in a
        way that took a reviewer to see: the user takes over sentence A, then starts speaking
        sentence B before A has finished decoding, and beginning B cleared the flag. A's
        answer then arrived and was typed — text the user had explicitly rejected, at whatever
        caret they had moved to by then, possibly in a different application.
        """
        self.typist.disown()
        self._taken_over.add(self._utterance)

    def stop(self) -> None:
        """Finish the session — and let go of the sentence in flight on the way out.

        Setting the flag alone was not enough, and the gap was narrow enough to need a
        reviewer to find it. Ctrl-C usually reaches the tap first, so `interrupt` has already
        disowned the sentence by the time this runs. Usually is not always: an external
        `kill -INT`, or a tap that macOS has disabled, arrives here with the sentence still
        owned, and the answer to it was typed into the terminal the user had just returned
        to. Disowning here as well makes the outcome the same however the stop arrived.

        The transcript is deliberately NOT given up: `_settle_one` keeps a sentence that
        settles after a stop, because "Ctrl-C finishes the session and prints what you said"
        is the promise, and dropping the last sentence would break it.
        """
        self._stopped = True
        self.typist.disown()
        self._taken_over.add(self._utterance)

    async def run(self, audio: AsyncIterator[bytes]) -> str:
        """Consume the microphone until it stops or someone asks to stop."""
        settler = asyncio.create_task(self._settle_each())
        try:
            await self._listen(audio)
            # And not waited on at all if something already failed: the consumer is gone by
            # then, so a join is a wait for somebody who has left. Belt as well as braces —
            # the guard in `_listen` above is what stops anything being queued, and this is
            # what stops the wait even if some later change puts something there anyway.
            if self._failure is None:
                await self._queue.join()
        finally:
            settler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await settler
            await self._cancel_draft()
        if self._failure is not None:
            raise self._failure
        return self.transcript

    async def _listen(self, audio: AsyncIterator[bytes]) -> None:
        self._heard_at = time.monotonic()
        async for block in audio:
            if self._stopped or self._forgotten():
                break
            heard = self.gate.feed(block)
            # Ending before starting, because that is the order they happened in. One block
            # of audio can carry both — a sentence that ran past the gate's own limit is cut
            # and the next one opens a few frames later — and handling the start first told
            # the typist a new sentence had begun while the old one was still being decoded.
            if heard.utterance:
                await self._end(heard.utterance)
            elif heard.dropped:
                await self._take_back_a_guess_at_nothing()
            if self.gate.confirmed:
                # Somebody is talking right now, so the session is plainly not forgotten. The
                # idle timer was only reset when an utterance CLOSED, and the gate holds one
                # open until a pause or twenty-five seconds — so a short `--idle-minutes` and
                # one long sentence ended the session mid-word, which is the opposite of what
                # "nobody has spoken" is supposed to mean. Asked of the gate rather than of
                # `heard.speaking`, because a click opens the gate too and must not count.
                self._heard_at = time.monotonic()
            if heard.started:
                self._begin()
            elif heard.speaking and not heard.utterance:
                self._maybe_draft()
        # Nothing more is queued once a decode has failed for good. The settler ends itself
        # on that failure, so it is the only thing that ever calls `task_done` and there is
        # no longer anything listening: putting the half-spoken sentence in here left `run`
        # waiting on a join that could never complete, with the microphone still open. That
        # is the same hang the failure handling was written to prevent, moved one line along.
        # Asked BEFORE closing, because closing is what clears it: `close()` finishes the
        # utterance and sets the gate back to not-speaking either way, so the answer after
        # the call is always "no" and the rollback below would never run.
        was_open = self.gate.speaking
        remainder = self.gate.close()
        if remainder and self._failure is None:
            await self._end(remainder)
        elif was_open and remainder is None:
            # The gate was open on something it then judged to be nothing, and the audio ran
            # out before the pause that would have said so. A hundred-millisecond click opens
            # the gate, the next quiet block schedules a draft, the microphone is unplugged —
            # and `close()` discards the utterance for holding no real speech, so no accurate
            # pass ever runs and nothing takes the guess back. The word the fast model
            # invented for a click is left in somebody's window, permanently, which is the
            # exact failure `_take_back_a_guess_at_nothing` exists to prevent; it simply had
            # a second way in that only opens when the stream ends mid-noise.
            await self._take_back_a_guess_at_nothing()

    def _forgotten(self) -> bool:
        """Has anybody said anything in a very long time?"""
        return bool(self.idle_after) and time.monotonic() - self._heard_at > self.idle_after

    async def _take_back_a_guess_at_nothing(self) -> None:
        """The gate opened on a noise and then threw it away. Remove whatever was typed for it.

        This is the flagship failure, and it had a second way in. A click clears the gate's
        onset, the fast model hears an ordinary word in it and types that word — and then the
        gate discards the utterance for holding no real speech, so no accurate pass ever runs
        and nothing was left to take the word back. It stayed in the window, exactly as if
        somebody had said it.
        """
        await self._cancel_draft()
        # Settled only if the retraction actually happened. `settle` means "this sentence is
        # over, the next may be owned", and it clears the takeover flag — so calling it after
        # `show` DECLINED handed ownership back that the user had taken. The sequence is
        # reachable with one keypress: a key's own sound opens the gate on a noise while the
        # sentence before it is still decoding, the user presses a key (the ordinary "stop
        # correcting" interaction), the noise is discarded — and the earlier sentence, which
        # was never in `_taken_over` because the noise had already claimed the next number,
        # was then typed in full at the user's caret, underneath the draft it had orphaned.
        if self.typist.show(""):
            self.typist.settle()

    def _begin(self) -> None:
        self._utterance += 1
        self.typist.begin()
        self._drafted_at = 0.0
        self.report(Progress(listening=True))

    async def _end(self, spoken: bytes) -> None:
        """The sentence is over. Stop guessing at it and have it decoded properly.

        This, and not the moment the gate opened, is what counts as somebody having spoken.
        The gate opens for a click as readily as for a word — it needs a few frames before it
        can tell — and the clicks are exactly what an empty room produces all afternoon.
        Resetting the idle timer on those would mean a forgotten session never times out in
        the one place it most needs to.
        """
        self._heard_at = time.monotonic()
        await self._cancel_draft()
        self._settling = True
        await self._queue.put((self._utterance, spoken))

    def _maybe_draft(self) -> None:
        """Ask the draft model about the sentence so far, if it is time and nothing is busy."""
        if self.draft is None or self._settling or self._drafting is not None:
            return
        now = time.monotonic()
        if now - self._drafted_at < self.draft_every:
            return
        self._drafted_at = now
        self._drafting = asyncio.create_task(self._one_draft(self._utterance, self.gate.pending()))

    async def _one_draft(self, which: int, spoken: bytes) -> None:
        """Ask the fast model, and give up on it for good if it stops answering.

        A draft is a convenience — the accurate pass decodes the same audio a moment later —
        so its failure must not end the session, and must not repeat either. Left to raise,
        it ran again on the next tick and every tick after that, once a second, for as long
        as dictation continued: the same unread traceback, forever, from a server that was
        never coming back. Dropping to the accurate pass alone is what the user would have
        chosen, so it is what happens.
        """
        try:
            heard = await self.draft.transcribe(spoken, prompt=self._prompt())  # type: ignore[union-attr]
            # `_settling` can have become true while this was decoding, in which case the
            # sentence has already been handed to the accurate model and this answer is
            # stale. INSIDE the guard, not after it: showing the text is where a bad
            # dictionary term or a refused keystroke raises, and out there it killed the task
            # with nobody to receive the exception — after which the next tick started
            # another draft that failed the same way, once a second, for the whole session.
            if heard and not self._settling and which not in self._taken_over:
                guess = self._cleaned(heard)
                if guess:
                    self._show(guess, settled=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.draft = None
        finally:
            self._drafting = None

    async def _settle_each(self) -> None:
        """Decode finished sentences one at a time, in the order they were spoken.

        Nothing raises out of here. This runs as its own task, and a task that dies takes the
        queue's consumer with it: `run` is waiting on `_queue.join()`, the join is never
        satisfied, and a model that failed once turns into a dictation session that hangs
        with the microphone still open. So a failed sentence ends the session, with the error
        kept for `run` to raise once everything is shut down.
        """
        while True:
            which, spoken = await self._queue.get()
            # Set again for each sentence taken off the queue, not only when one is put on
            # it. Clearing it in the `finally` below and picking the next one up in the next
            # breath left a gap where drafting was allowed while a sentence was still being
            # decoded: the user starts sentence C, C's draft is typed, and then B — queued
            # behind A and settling all along — comes back and rewrites C's draft into B's
            # older words.
            self._settling = True
            try:
                await self._settle_one(which, spoken)
            except Exception as failure:
                # Every exception, not just the diagnosed ones. The promise above is that
                # nothing raises out of here, and `except SttError` did not keep it: a
                # `re.error` out of a pathological dictionary term, or any coercion failure
                # in the reply, killed this task with a second sentence still in the queue —
                # and `run` then waited on a `join()` that could never be satisfied, with the
                # microphone still open. That is the exact hang the guard exists to prevent.
                self._failure = failure
                self._stopped = True
            finally:
                # Still settling if anything is waiting: the gap between this line and the
                # next `get()` is the same gap, one loop iteration wide.
                self._settling = not self._queue.empty()
                self.typist.settle()
                self._queue.task_done()
            if self._failure is not None:
                self._abandon_what_is_left()
                return

    async def _settle_one(self, which: int, spoken: bytes) -> None:
        """Decode one finished sentence: put it on screen, and keep it for the transcript.

        Those are two decisions, not one, and they part company exactly once. Normally the
        transcript is what went into the window — a sentence printed at the end that never
        reached the caret would be the transcript disagreeing with what the user watched
        happen, and a sentence the user took over is neither typed nor recorded.

        The exception is the sentence being spoken when they ask to stop. Ctrl-C is a key
        press like any other, so the watcher sees it and lets go of the sentence in flight —
        and the last thing said then went nowhere at all, which is not what "Ctrl-C finishes
        the session and prints the transcript" promises. It is kept when stopping, and still
        not typed: their focus is in the terminal by then, and the words belong in the
        printed transcript rather than in the shell.
        """
        heard = await self.settled.transcribe(spoken, prompt=self._prompt())
        sentence = self._cleaned(heard)
        if not sentence:
            # Nothing is going to replace the draft, so take it back. A cough gets through
            # the gate, the fast model hears an ordinary word in it and types that word, and
            # the accurate model answers with something the filters drop — leaving the draft
            # on screen for good, because `settle()` forgets we own it and no later edit can
            # reach it. `show` declines by itself if the user has taken the text over, which
            # is the one case where it must NOT be removed.
            self.typist.show("")
            return
        typed = which not in self._taken_over and self._show(sentence, settled=True)
        if typed or self._stopped:
            self._said.append(self._lead + sentence)
            # The separator turns on only when a sentence was really added. One the model
            # invented and the filters dropped used to leave the space behind: nothing on
            # screen, and the next real sentence starting with a space that belonged to a
            # sentence nobody said.
            self._lead = " "

    def _abandon_what_is_left(self) -> None:
        """Give up on the sentences still waiting, once one of them has failed for good.

        Recording the failure and carrying on looked harmless and was not: `run` waits for the
        queue to drain, so every sentence spoken while the failing one was decoding still got
        sent to a model that had just proved it does not answer — each able to wait out the
        ninety-second timeout in turn. The microphone stayed open through all of it, and
        sentences decoded after a fatal error could still be typed. They are dropped instead,
        and the queue is emptied so the join that `run` is waiting on can finish at once.
        """
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.task_done()

    def _show(self, sentence: str, *, settled: bool) -> bool:
        """Put a sentence on screen. False when the typist declined it.

        Synchronous on purpose, and this is the load-bearing decision in the whole file. The
        draft pass and the accurate pass both write here, and putting the keystrokes on a
        thread meant two of them could be posting at once — the settled sentence interleaved
        character by character with the draft it was replacing. Worse, cancelling a draft
        mid-write cancels the await and NOT the thread, so the guard against it would have
        been the one thing it could not guard.

        The thread was there because a full sentence rewrite is a few hundred synthetic
        events and that sounded expensive. Measured, it is not: two hundred backspaces cost
        three milliseconds and two hundred characters cost less than one. That is a
        thirtieth of one block of audio, so the loop can simply do it, and one write at a
        time stops being something to enforce and starts being something asyncio guarantees.
        """
        # What the typist DID, not what it was asked to do. It declines when the user has
        # taken the text over, and ignoring that answer put a sentence into the printed
        # transcript that had never reached the window — the two disagreeing about what
        # happened, which is worse than either of them being empty.
        if not self.typist.show(self._lead + sentence):
            return False
        self.report(Progress(listening=True, text=sentence, settled=settled))
        return True

    def _prompt(self) -> str:
        """The glossary the speech models are primed with, or nothing when it is turned off."""
        return self.terms.prompt() if self.bias else ""

    def _cleaned(self, heard: str) -> str:
        """What is safe to type, out of what the model said. Empty when that is nothing.

        The gate is the real defence — a model that is never given silence cannot invent over
        it — and this stands behind it, because the gate's threshold is a judgement and a
        cough or a door opens it. Measured, on this machine: two seconds of pure silence
        handed to `large-v3-turbo` came back as "Продолжение следует...". In a file that is a
        line to delete afterwards. Here it would have been TYPED, into whatever window
        happened to have focus, while nobody was speaking.

        Two filters, both the file pipeline's: the phrases that are never anything but an
        artefact, and a phrase repeating past the point of being emphasis. The file
        pipeline's third list — the ordinary-sounding ones, "продолжение следует", "да",
        "okay" — is deliberately NOT applied here. It needs corroborating evidence that the
        phrase was invented, the file pipeline gets that from silence overlap, and the
        equivalent here would have to be the model's own `no_speech_prob`, which was measured
        useless (see `server.py`). Applying the list without evidence would swallow "да",
        which somebody dictating into a chat window says on purpose all day.
        """
        if any(pattern.search(heard) for pattern in self.refuse):
            return ""
        segment = Segment(start=0.0, end=0.0, text=heard)
        cleaning.collapse_only([segment], MAX_REPEATS)
        return dict_mod.correct_text(segment.text, self.terms).strip()

    async def _cancel_draft(self) -> None:
        task, self._drafting = self._drafting, None
        if task is None:
            return
        task.cancel()
        # A draft that dies because the sentence ended, or because the server it was talking
        # to went away, is not news: the accurate pass is about to decode the same audio.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
