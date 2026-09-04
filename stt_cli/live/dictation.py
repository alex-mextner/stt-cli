"""dictation — assembling `stt mic` out of the parts, and taking it apart again afterwards.

Everything here is wiring: permissions, then two models, then the keyboard watcher, then the
microphone, then the loop in `session.py`, and then all of it closed down in the reverse
order however the run ends. The decisions are elsewhere; the ordering is the point of this
file, because half of these things hold something the user notices being held — a recording
light, a system-wide event tap, two copies of a model in memory.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import signal
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import config, dictionary, phrases
from .._errors import EXIT_OK, EngineError, MissingDependencyError, UsageError
from ..backends import whispercpp
from . import capture, quartz, status, tap
from .gate import Gate
from .server import Transcriber, Voice, WhisperServer
from .session import IDLE_MINUTES, Progress, Session
from .typist import Typist

SERVER_HINT = (
    "`stt mic` needs the whisper-server binary from a whisper.cpp build — `brew install "
    "whisper-cpp`, or clone https://github.com/ggml-org/whisper.cpp and "
    "`cmake -B build && cmake --build build -j`"
)
# Two presses of Escape this far apart end the session. One press is something people do in
# other applications all day; two in half a second is not.
DOUBLE_PRESS = 0.5


@contextlib.contextmanager
def _the_only_session() -> Iterator[None]:
    """Refuse to start a second dictation while one is already running.

    Two sessions type into the same window at the same time, and each one deletes what it
    believes it wrote — which is now interleaved with the other's. The result is not two
    transcripts, it is one line of debris, and neither session can tell that anything is
    wrong. An exclusive lock is the whole fix; the file is a marker, its contents are of no
    interest, and the lock goes away with the process however the process ends.
    """
    config.ensure_dirs()
    # Appended to rather than truncated, purely so that failing to get the lock does not
    # first reach into the file the process that HAS it is holding open.
    marker = config.app_home() / "mic.lock"
    handle = marker.open("a")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise UsageError(
                what="dictation is already running",
                why="another `stt mic` holds the microphone and the keyboard",
                how="finish that one first — Escape twice in the window it is typing into,"
                " or Ctrl-C in the terminal it was started from",
            ) from exc
        yield
    finally:
        handle.close()


async def dictate(args: argparse.Namespace) -> int:
    """`stt mic`, from the permission check to the last word typed.

    `--list-devices` and `--check` are answered by the command before this is reached, so
    everything here is a real session. The device listing used to live here as well, which is
    how `--check --list-devices` came to run a six-second listening test instead of printing
    a list: two places answered the same flag and the wrong one got there first.
    """
    quartz.require_accessibility()
    settings = config.load_settings()
    device = await capture.resolve(args.device)
    # The lock before the models, not after. Loading `large-v3-turbo` takes ten seconds and a
    # couple of gigabytes, and a second `stt mic` used to pay both of those before being told
    # that the first one already had the keyboard.
    with _the_only_session():
        async with _models(args, settings) as models:
            said = await _dictate_until_stopped(args, device, models, settings)
    if said:
        print(said)
    return EXIT_OK


@dataclass
class Models:
    """The two loaded models, and the language they were told to expect."""

    settled: Transcriber
    draft: Transcriber | None


@contextlib.asynccontextmanager
async def _models(args: argparse.Namespace, settings: config.Settings) -> AsyncIterator[Models]:
    """Load both models, and unload them however the block ends."""
    # The configured checkout, not just PATH. Someone whose whisper.cpp lives somewhere they
    # told stt about could transcribe files perfectly well and be told by `stt mic` that
    # whisper-server does not exist — the two commands disagreeing about an installation the
    # user had already configured once.
    binary = whispercpp.server_binary(settings.whispercpp_root)
    if binary is None:
        raise MissingDependencyError(
            what="whisper-server was not found",
            why="live dictation keeps a model loaded, which whisper-cli cannot do",
            how=SERVER_HINT,
        )
    language = args.language or settings.language or "auto"
    engine = whispercpp.WhisperCppBackend(settings.whispercpp_root)
    wanted = [_settled_model(args, settings)]
    drafting = args.draft and _the_draft_is_installed(
        engine, args.draft_model, lambda said: status.note(said, quiet=args.quiet)
    )
    if drafting:
        wanted.append(args.draft_model)
    # Both model files resolved before either server starts. Otherwise a missing model is
    # discovered after `large-v3-turbo` has already spent ten seconds and two gigabytes
    # loading, and the run that was never going to work took the longest to say so.
    files = [engine.model_path(name) for name in wanted]
    # Said after the files are found, not before: "loading models" followed immediately by
    # "that model has no whisper.cpp build" reads like something went wrong during a load
    # that never started.
    # In the order they are loaded, which is the order they finish in. The note used to
    # reverse them, so the model somebody was waiting on was named second.
    status.note("loading " + " and ".join(wanted), quiet=args.quiet)
    threads = settings.threads if args.threads is None else args.threads
    started = await _load_together(binary, files, language, threads)
    try:
        yield Models(settled=started[0], draft=started[1] if drafting else None)
    finally:
        await asyncio.gather(*(server.stop() for server in started))


async def _load_together(
    binary: Path, files: list[Path], language: str, threads: int
) -> list[WhisperServer]:
    """Both models at once, and neither left running if the other cannot start.

    They used to load one after the other, so the wait was the SUM: `large-v3-turbo` takes
    about ten seconds and `base` follows it, for no reason but the order of a loop. They are
    separate processes reading separate files, so overlapping them makes the wait the longer
    of the two instead — which matters because this wait is the whole of what somebody
    experiences before dictation will listen to them.

    The failure path is why this is not a one-line `gather`. If one model fails while the
    other is still loading, the one that succeeded is a process holding a gigabyte open with
    nobody left to use it, so every server that did start is stopped before the failure is
    allowed out.
    """
    loading = [asyncio.create_task(_load(binary, path, language, threads)) for path in files]
    done = await asyncio.gather(*loading, return_exceptions=True)
    started = [server for server in done if isinstance(server, WhisperServer)]
    failed = [outcome for outcome in done if isinstance(outcome, BaseException)]
    if failed:
        await asyncio.gather(*(server.stop() for server in started))
        raise failed[0]
    return started


def _the_draft_is_installed(
    engine: whispercpp.WhisperCppBackend, model: str, note: Callable[[str], None]
) -> bool:
    """Is the fast model on disk? If not, say so and carry on without it.

    `stt setup` downloads the one model it was asked for, so somebody who accepted the
    default has `large-v3-turbo` and no `base` — and the first `stt mic` they ever ran failed
    before opening the microphone, over a model they had never been told they needed. The
    draft pass is a convenience; refusing to dictate at all because the convenience is
    missing is the wrong trade. A missing ACCURATE model is still fatal, because without it
    there is nothing to dictate with.
    """
    try:
        engine.model_path(model)
    except EngineError:
        # Only a model that EXISTS and is not downloaded. A name that is not a model at all,
        # or one with no whisper.cpp build, raises something else and is left to reach the
        # user: telling somebody to `stt models pull definitely-not-a-model` is worse than
        # telling them nothing, because it reads like the answer.
        note(f"no {model} model, so no fast first guess — `stt models pull {model}` adds one")
        return False
    return True


def _settled_model(args: argparse.Namespace, settings: config.Settings) -> str:
    return str(args.model or settings.model)


async def _load(binary: Path, model: Path, language: str, threads: int) -> WhisperServer:
    """One `whisper-server`, holding one model file open for the rest of the session.

    `threads` is the user's configured value, and zero keeps the meaning it has everywhere
    else in stt: let the engine decide. It used to be turned into four — so somebody who had
    set eight got four, and somebody who asked for the engine default got four as well, with
    two servers between them oversubscribing the machine.
    """
    return await WhisperServer.start(binary, Voice(model=model, language=language, threads=threads))


async def _dictate_until_stopped(
    args: argparse.Namespace,
    device: capture.Device,
    models: Models,
    settings: config.Settings,
) -> str:
    """Open the keyboard watcher and the microphone, and run until one of them stops."""
    line = status.Line(enabled=not args.quiet)
    marker = tap.new_marker()
    dictating = Session(
        typist=Typist(keys=_Keys(marker=marker)),
        settled=models.settled,
        draft=models.draft,
        # The same two settings the file pipeline reads. Somebody who turned the dictionary
        # off to keep the spelling they dictated was having it corrected anyway, and having
        # their glossary sent to the models on top.
        terms=dictionary.load() if settings.dictionary else dictionary.Dictionary(),
        bias=bool(settings.dictionary and settings.dict_bias),
        # The user's own always-drop patterns, which file transcription has always read and
        # live dictation was not. Somebody who wrote down a phrase they never want to see
        # was watching stt type it into their window.
        refuse=phrases.compile_all(phrases.load_user_patterns(config.app_home()))[0],
        idle_after=_idle_seconds(args),
        report=lambda progress: line.show(_paint(progress)),
        gate=gate_for(settings),
    )
    watcher = _watch(dictating, marker)
    try:
        status.note(f"listening on {device.name} — press Escape twice to finish", quiet=args.quiet)
        return await _through_the_microphone(device, dictating, line, quiet=args.quiet)
    finally:
        watcher.stop()
        line.clear()
        await status.closed(said=dictating.transcript, quiet=args.quiet)


def gate_for(settings: config.Settings) -> Gate:
    """The speech detector, at this machine's threshold rather than the built-in one.

    Zero means nobody has chosen, which is not the same as somebody choosing zero — that
    would open the gate on every frame and hand the room to the model continuously.
    """
    # Anything not above zero means nobody chose, INCLUDING a negative one: a hand-edited
    # `"mic_threshold": -5` is not a quieter threshold, it is a value below every level a
    # microphone can produce, and honouring it would open the gate on the room continuously.
    # Asked here rather than in the config layer because "can this be negative" is a question
    # about the setting, and the shared coercion path knows nothing about microphones.
    if settings.mic_threshold > 0:
        return Gate(minimum=settings.mic_threshold)
    return Gate()


async def _through_the_microphone(
    device: capture.Device, dictating: Session, line: status.Line, *, quiet: bool
) -> str:
    """The microphone stays open for exactly as long as this block runs."""
    async with capture.listening(device) as audio:
        with _interruptible(dictating):
            line.show(status.render(listening=True, text="", settled=False))
            # Announced beside the session, not in front of it. The sound and the banner
            # are each a subprocess, and `afplay` alone takes a second and a half to play a
            # noise a fifth of that long — waited on, that is a second and a half of somebody
            # talking into a microphone that is open but not being read yet.
            saying = asyncio.create_task(status.opened(device.name, quiet=quiet))
            try:
                return await dictating.run(audio)
            finally:
                await saying


@contextlib.contextmanager
def _interruptible(dictating: Session) -> Iterator[None]:
    """Make Ctrl-C mean "finish", not "throw away what I just dictated".

    Left alone, SIGINT raises `KeyboardInterrupt` out of `asyncio.run` — the microphone is
    closed and the models are unloaded, correctly, and the sentences already spoken are
    discarded along with them. Handled on the loop it just stops the session, which lets the
    last sentence settle and the transcript print. Both ways out then behave the same, which
    matters because Ctrl-C is what people reach for in a terminal and Escape is what the
    terminal cannot see when the focus is somewhere else.
    """
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, dictating.stop)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - not the main thread
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(signal.SIGINT)


def _idle_seconds(args: argparse.Namespace) -> float:
    """How long the session sits through silence before deciding it was forgotten."""
    minutes = IDLE_MINUTES if args.idle_minutes is None else args.idle_minutes
    return max(0.0, float(minutes)) * 60


def _paint(progress: Progress) -> str:
    return status.render(listening=progress.listening, text=progress.text, settled=progress.settled)


class _Keys:
    """The typist's keyboard: real synthesized keystrokes, stamped so the tap ignores them."""

    def __init__(self, marker: int) -> None:
        self.marker = marker

    def type_text(self, text: str) -> None:
        quartz.type_text(text, marker=self.marker)

    def press_backspace(self, times: int) -> None:
        quartz.press_backspace(times, marker=self.marker)


class _Stopper:
    """Turns the watcher's stream of key presses into the two things the session cares about.

    Every key the user presses means "stop correcting what you typed"; two Escapes in quick
    succession mean "stop entirely". The key code goes no further than this object — see the
    note in `tap.py` about what this must never become.
    """

    def __init__(self, dictating: Session) -> None:
        self._session = dictating
        self._last_escape = 0.0

    def __call__(self, code: int) -> None:
        self._session.interrupt()
        if code != tap.ESCAPE:
            return
        now = time.monotonic()
        if now - self._last_escape < DOUBLE_PRESS:
            self._session.stop()
        self._last_escape = now


def _watch(dictating: Session, marker: int) -> tap.Watcher:
    """Start the keyboard watcher, marshalling its thread's events onto the event loop."""
    stopper = _Stopper(dictating)

    def arrived(code: int) -> None:
        """Act on the key press HERE, on the tap's own thread, and not a moment later.

        This used to hand the press to the event loop with `call_soon_threadsafe`, which
        reads as the careful thing to do and was the wrong thing. The loop is what performs
        the synthetic edits, so a click arriving while it was already about to type left the
        edit ahead of the interrupt in the queue: text going into the window the user had
        just clicked into. "Ownership ends the instant the user clicks" has to mean this
        instant, not the next time the loop looks at its queue.

        Doing it here is safe because of what it does: `_Stopper` sets two attributes and adds
        an integer to a set, each of which is a single bytecode step. It touches no loop
        machinery, so there is nothing to be woken up and nothing to raise if the loop has
        already gone away — which was the other reason the indirection was there.
        """
        stopper(code)

    watcher = tap.Watcher(marker=marker, on_key=arrived)
    watcher.start()
    if watcher.failure:
        # Stopped before the error goes up, because nothing above will ever see this object
        # again — the `finally` that normally stops it only runs once it has been returned.
        watcher.stop()
        raise EngineError(
            what="stt cannot tell when you type",
            why=watcher.failure,
            how="without it stt could delete text it did not write, so it will not start",
        )
    return watcher
