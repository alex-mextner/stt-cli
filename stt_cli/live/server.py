"""server — one whisper.cpp model, loaded once and asked over and over.

WHY A SERVER AND NOT THE COMMAND-LINE TOOL
    Everything else in stt shells out to `whisper-cli` per chunk, which is right for a file:
    the model loads once per chunk and the chunk takes minutes. Live dictation asks the same
    model a question every second or so, and loading `large-v3-turbo` takes longer than that
    on its own — the process launch would be most of the latency, and all of it would be
    spent re-reading a file that had not changed.

    `whisper-server` ships in the same build as `whisper-cli`, keeps the weights in memory,
    and answers over a local socket. That is the entire reason it is used here.

TWO OF THESE RUN AT ONCE
    A small one to answer immediately and be wrong at the edges, and a large one to answer a
    second later and be right. See `session.py` for what is done with the two answers; from
    here they are simply two servers with different models, and neither knows about the other.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import io
import json
import socket
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol

from .. import proc
from .._errors import EngineError, MissingDependencyError
from ..jsonio import as_dict, as_str
from .gate import RATE

# How long to wait for a model to load before deciding it never will. `large-v3-turbo` off a
# cold disk is the slow case; anything past this is a broken build, not a slow one.
STARTUP_SECONDS = 120.0
# A single decode of at most `gate.MAX_UTTERANCE_MS` of audio. Generous, because the machine
# may be busy; it exists so a wedged server becomes an error rather than a hung dictation.
DECODE_SECONDS = 90.0
_POLL_SECONDS = 0.2


# MEASURED, AND IT DOES NOT WORK: whisper reports a `no_speech_prob` per segment, and it is
# the obvious evidence for "was anybody actually speaking". Asked about eight seconds of this
# machine's own quiet room it answered 0.000 — no doubt at all — while inventing "Спасибо."
# out of the noise, and 0.000 again for pure digital silence decoded as "Продолжение
# следует...". So the number is not evidence of anything here, and a filter built on it would
# be a defence that never fires. What actually keeps invented text out of somebody's window
# is `gate.py` refusing to send the audio at all, tuned against recordings of this room.


class _ConnectionGone(Exception):
    """The connection broke, as opposed to the model refusing what was on it.

    Only this is worth trying again: a closed socket is answered by opening another one, and
    a refusal is answered by the same refusal however many times it is asked.
    """


class Transcriber(Protocol):
    """Something that turns a block of PCM into words. Implemented here and in the tests."""

    async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str: ...


# MEASURED, SO NOBODY TRIES IT AGAIN: whisper's encoder always processes a thirty-second
# window, padding whatever it is given with silence, so decoding one second costs very nearly
# what decoding thirty does. The obvious fix is `audio_ctx`, which shortens that window, and
# it is a trap on this machine. On a three-second window: `base` went from 0.04s to 0.02s,
# which is nothing worth having, and `large-v3-turbo` went from 0.99s to 4.07s — four times
# SLOWER, because a non-standard context size falls off whatever fast path Metal has for the
# usual one. The draft model is fast enough at full width; the accurate one is ruined by
# trimming. So neither trims, and this comment is here instead of a knob.


@dataclass(frozen=True)
class Voice:
    """What a server is being asked to be: which model, which language, primed with what."""

    model: Path
    language: str
    # Zero means "let whisper.cpp choose", the same as everywhere else in stt.
    threads: int = 0


class WhisperServer:
    """A `whisper-server` process, its port, and the one question anybody asks it."""

    def __init__(self, process: asyncio.subprocess.Process, port: int) -> None:
        self._process = process
        self._port = port
        # whisper-server writes a few lines about every request it answers. Nothing reads
        # them, and a pipe nobody reads fills up and then blocks the writer — the server
        # would answer a few hundred sentences and then simply stop, mid-dictation, with no
        # error anywhere. So they are read continuously and thrown away, except the last few,
        # which are the only thing worth saying if the process dies.
        self._last_words: deque[str] = deque(maxlen=20)
        # Started by `start`, not here. Creating the task in the constructor meant the
        # constructor could only be called with a loop running, which put every test through
        # `__new__` and a hand-written list of private fields — six of them, kept in step by
        # hand, and already drifting apart between copies. A constructor tests can call is a
        # constructor that cannot silently leave a new field unset.
        self._draining: asyncio.Task[None] | None = None
        # One connection, opened once and kept. See `_the_link` for why that is a safety
        # property and not an optimisation.
        self._link: http.client.HTTPConnection | None = None
        # Held by whichever worker thread is actually using the connection. Cancelling the
        # task that is waiting on `to_thread` does not stop the thread doing the work — so a
        # cancelled draft can still be mid-request when the next one starts, and two requests
        # sharing one `HTTPConnection` is not something it supports: the replies come back
        # interleaved, or matched to the wrong question. The waiting is invisible in practice
        # because the cancelled draft's answer is thrown away anyway.
        self._talking = threading.Lock()
        # Held while a connection is being made, so two drafts arriving at once cannot each
        # build one and leave the loser's socket open with nobody holding it.
        self._connecting = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._port}/inference"

    @classmethod
    async def start(cls, binary: Path, voice: Voice) -> WhisperServer:
        """Launch the server and wait until it will actually answer."""
        port = _free_port()
        process = await asyncio.create_subprocess_exec(
            *_argv(binary, voice, port),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        server = cls(process, port)
        server._begin_draining()
        try:
            await server._await_readiness(voice)
            await server._confirm_it_is_ours()
        except BaseException:
            # A server that never became ready still left a process and a task behind. They
            # are not attached to anything yet — the caller never got the object — so this is
            # the only place that can close them.
            await server.stop()
            raise
        return server

    def _begin_draining(self) -> None:
        """Start reading what the server says, which must happen before it says much."""
        if self._draining is None:
            self._draining = asyncio.create_task(self._drain())

    async def transcribe(self, pcm: bytes, *, prompt: str = "") -> str:
        """The words in `pcm`, or an empty string when the model heard nothing worth saying."""
        if self._process.returncode is not None:
            raise EngineError(
                what="the speech model is no longer running",
                why=f"whisper-server exited with code {self._process.returncode}",
                how="start dictation again",
            )
        body, content_type = _multipart(pcm, prompt=prompt)
        try:
            payload = await asyncio.wait_for(self._ask(body, content_type), timeout=DECODE_SECONDS)
        except TimeoutError as exc:
            raise EngineError(
                what="the speech model stopped answering",
                why=f"no reply in {DECODE_SECONDS:.0f}s from {self.endpoint}",
                how="stop dictation and start it again; try a smaller --draft-model",
            ) from exc
        try:
            answer = as_dict(json.loads(payload))
        except ValueError as exc:
            # Not JSON at all: an HTML error page, or a truncated reply. A traceback out of
            # the parser in the middle of dictation says nothing anybody can act on.
            raise EngineError(
                what="the speech model answered with something that is not a transcript",
                why=str(exc),
                how="stop dictation and start it again",
            ) from exc
        return as_str(answer.get("text")).strip()

    async def _ask(self, body: bytes, content_type: str) -> str:
        """Send one request, opening the connection again if the old one had gone.

        `http.client` does not reconnect by itself: once its socket exists it will keep using
        it, and a server that closed an idle connection — which is an ordinary thing to do
        between two sentences — makes the next request fail. That failure ends the session,
        so a pause long enough for the server to tidy up was enough to stop dictation. The
        second attempt is safe because a decode changes nothing: asking twice and asking once
        give the same answer.
        """
        broke = ""
        for attempt in range(2):
            link = await self._the_link()
            try:
                return await asyncio.to_thread(self._post, link, body, content_type)
            except _ConnectionGone as gone:
                # `_post` has already closed and forgotten the connection; the next turn of
                # this loop opens a fresh one and checks again who is on the other end. A
                # refusal is not caught here on purpose — it would be the same refusal.
                broke = str(gone)
                if attempt or self._process.returncode is not None:
                    break
        raise EngineError(
            what="the speech model stopped answering",
            why=broke or "the connection to it closed twice in a row",
            how="stop dictation and start it again",
        )

    async def _the_link(self) -> http.client.HTTPConnection:
        """The one connection to the model, opened and checked once and then kept open.

        This is the shape it is for a reason that took three rounds of review to get right.
        The port is picked by binding a socket, reading the number and letting it go, because
        `whisper-server` cannot be told to choose one and report back — so between letting go
        and the server binding, anything on the machine can take it. Asking the operating
        system who owns the listening socket answers that, and answering it once at startup
        was not enough: ownership stops being true the moment the process behind it exits.

        Asking again before every decode narrowed the gap to the microseconds between the
        answer and the send, and narrow is not closed — the reviewer who pointed that out
        twice was right. What closes it is that a TCP connection, once established, cannot be
        handed to somebody else: a process that binds the port afterwards gets new
        connections, never this one. So the check happens when the connection is made, and
        every sentence after that travels down a pipe already known to end at our own child.
        """
        async with self._connecting:
            if self._link is not None:
                return self._link
            return await self._open_one()

    async def _open_one(self) -> http.client.HTTPConnection:
        """Make the connection and check who is on the other end, under `_connecting`.

        Checked ONCE, after connecting. There used to be a check before it as well, and the
        pair cost two `lsof` subprocesses — around a tenth of a second each — on a path that
        runs far more often than "at startup": whisper-server closes an idle connection
        between sentences, so every pause for thought was followed by a reopen, and the
        latency landed on the next sentence in a feature whose whole budget is one to three
        seconds. The earlier check bought nothing the later one does not. Connecting to the
        wrong process is harmless; SENDING to it is the danger, and nothing is sent until
        after the check below.
        """
        link = http.client.HTTPConnection("127.0.0.1", self._port, timeout=DECODE_SECONDS)
        try:
            await asyncio.to_thread(link.connect)
        except BaseException as exc:
            # Diagnosed, like every other failure here. The server can pass the exit-code
            # check and then be gone before this connects — nothing is listening, the connect
            # is refused, and an untranslated `ConnectionRefusedError` travels all the way out
            # through `guard`, which only knows how to render an `SttError`. The user gets a
            # Python traceback where they were promised a sentence about what to do.
            # Closed on ANY way out, cancellation included. A draft cancelled during the
            # connect left the socket open and `self._link` still empty, so the next draft
            # made another — a slow drip of file descriptors through a long session.
            with contextlib.suppress(OSError):
                link.close()
            if isinstance(exc, (OSError, http.client.HTTPException)):
                raise EngineError(
                    what="could not reach the speech model",
                    why=str(exc),
                    how="stop dictation and start it again",
                ) from exc
            raise
        # Now the connection exists, ask who is holding the port: if the server died before
        # this, we are talking to whatever took the port after it, and this refuses before a
        # single sample is sent down the wire.
        #
        # Closed here if that check refuses. Its failure path calls `stop`, which forgets
        # `self._link` — and `self._link` was still empty, because the socket only becomes
        # ours on the line below. So the one connection this function had just opened was
        # orphaned: the same descriptor leak the comment above says was fixed for the
        # cancelled connect, through the other door.
        try:
            await self._confirm_it_is_ours()
        except BaseException:
            with contextlib.suppress(OSError):
                link.close()
            raise
        self._link = link
        return link

    def _post(self, link: http.client.HTTPConnection, body: bytes, kind: str) -> str:
        """One request down the established connection, on a worker thread."""
        try:
            with self._talking:
                link.request("POST", "/inference", body=body, headers={"Content-Type": kind})
                with link.getresponse() as reply:
                    said = reply.read().decode("utf-8", "replace")
                    status = reply.status
        except (OSError, http.client.HTTPException) as exc:
            self._forget_the_link(link)
            raise _ConnectionGone(str(exc)) from exc
        if status >= 400:
            # Read, rather than ignored. A refusal comes back as a JSON object with an
            # `error` key and no `text`, and reading only `text` turned that into an empty
            # answer — indistinguishable from "the model heard nothing". Dictation then
            # removed each draft and typed nothing, all session, with no reason given
            # anywhere. Some whisper.cpp builds refuse an uploaded WAV outright, so this is
            # not a hypothetical shape.
            self._forget_the_link(link)
            raise EngineError(
                what="the speech model refused the audio",
                why=f"it answered {status}: {said.strip()[:200] or 'with nothing'}",
                how="check the whisper.cpp build is recent enough to accept a WAV upload",
            )
        # The connection is kept only while it is known to be healthy: everything above that
        # can go wrong closes it first, so the next call opens and re-checks a fresh one
        # rather than reusing something in an unknown state.
        return said

    def _forget_the_link(self, failed: http.client.HTTPConnection | None = None) -> None:
        """Drop the connection — but only the one that actually went wrong.

        A cancelled draft leaves its worker thread inside the request. If that stale request
        fails after a newer one has already opened a replacement, closing "the current
        connection" closes the HEALTHY one: the retry that was working is cut off and
        drafting stops for the rest of the session. So the caller says WHICH connection it
        was holding, and this clears it only if that is still the one on the shelf.
        """
        if failed is not None and self._link is not failed:
            with contextlib.suppress(OSError):
                failed.close()
            return
        link, self._link = self._link, None
        if link is not None:
            with contextlib.suppress(OSError):
                link.close()

    async def stop(self) -> None:
        self._forget_the_link()
        if self._draining is not None:
            self._draining.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._draining
        await proc.end(self._process, grace=5.0)

    async def _await_readiness(self, voice: Voice) -> None:
        """Poll until the model is loaded, or say why it never will be."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STARTUP_SECONDS
        while loop.time() < deadline:
            if self._process.returncode is not None:
                await self._died(voice)
            if await asyncio.to_thread(_answers, self._port):
                return
            await asyncio.sleep(_POLL_SECONDS)
        raise EngineError(
            what=f"the speech model did not load within {STARTUP_SECONDS:.0f}s",
            why=f"{voice.model.name} never started answering on port {self._port}",
            how="check the model file is not truncated, or try a smaller one",
        )

    async def _confirm_it_is_ours(self) -> None:
        """Check that the thing answering on that port is the process we started.

        The port is chosen by binding a socket, reading the number, and closing it, because
        `whisper-server` has no "pick one and tell me" mode. Something else can bind it in the
        gap — and the readiness probe only asks whether SOMETHING answers HTTP there, so a
        local process that grabbed the port and replies to `/` would then be sent every
        recorded sentence and the terminology prompt with it. Asking the operating system who
        owns the listening socket is the check that closes that.

        Called while a connection is being OPENED — before the first request over it and
        again after it is established — and not per decode. An earlier version did check per
        decode, and this paragraph went on describing that after the design changed. What
        makes connection-time enough is in `_the_link`: a socket already connected to our
        process cannot be handed to something that binds the port later, so the window the
        check closes is exactly the one between choosing the port and connecting to it.

        A missing `lsof` is a REFUSAL, not a skip. This used to carry on without the check,
        reasoning that a missing utility should not stop a microphone from opening — which
        quietly turned the one defence against that race into something an attacker could
        remove by arranging for `lsof` not to be found, and the payload it protects is live
        microphone audio and the terminology prompt. A security check that can be skipped is
        not one. `lsof` ships with macOS, and `stt mic` is macOS-only, so the refusal is
        theoretical in practice and total when it is not.
        """
        finder = proc.which("lsof")
        if finder is None:
            await self.stop()
            raise MissingDependencyError(
                what="stt cannot confirm which process is holding the speech model's port",
                why="lsof was not found, and without it a local process that took the port "
                "first would be sent the microphone audio",
                how="lsof ships with macOS at /usr/sbin/lsof; check that it is on PATH",
            )
        found = await proc.run(
            [finder, "-nP", "-a", "-p", str(self._process.pid),
             f"-iTCP:{self._port}", "-sTCP:LISTEN", "-t"],
            timeout=15.0,
        )  # fmt: skip
        if str(self._process.pid) in found.stdout.split():
            return
        await self.stop()  # nothing more is sent to it, whatever it is
        raise EngineError(
            what="something else is listening on the speech model's port",
            why=f"port {self._port} is not held by the whisper-server stt started",
            how="another process took the port first; start dictation again",
        )

    async def _drain(self) -> None:
        """Keep the server's own output moving, and keep the last of it."""
        await proc.drain_stderr(self._process, self._last_words)

    async def _died(self, voice: Voice) -> NoReturn:
        # Wait for the drain task to reach end-of-stream rather than for one turn of the
        # loop: the process has exited, so its last words are already in the pipe, and one
        # turn is not enough to have read them. They are the entire diagnosis.
        if self._draining is not None:
            await asyncio.wait({self._draining}, timeout=2.0)
        raise EngineError(
            what="whisper-server exited before it was ready",
            why=self._last_words[-1] if self._last_words else "it wrote nothing",
            # Two different failures land here and the message has to cover both, because
            # the line above is whisper-server's and it does not always say which. Either the
            # model file is not one, or the binary is old enough not to know a flag it was
            # given, in which case what it printed is the first line of its own usage text.
            how=f"check that {voice.model} is a whisper.cpp model file, and that the"
            " whisper.cpp build is recent enough for the flags in the message above",
        )


def _argv(binary: Path, voice: Voice, port: int) -> list[str]:
    return [
        str(binary),
        "-m", str(voice.model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-l", voice.language,
        *(("-t", str(voice.threads)) if voice.threads > 0 else ()),
        # Timestamps are the one thing live dictation has no use for: the text goes into a
        # text field, not a subtitle file.
        "-nt",
        # A model told to keep quiet when it hears no speech is the second line of the same
        # defence `gate.py` is the first line of.
        "-sns",
    ]  # fmt: skip


def _free_port() -> int:
    """A port nothing is using, chosen by the kernel and handed straight over.

    There is a gap between closing this socket and whisper-server binding the same number,
    and something else could take it in between. Nothing can close that gap from out here —
    whisper-server has no "pick a port and tell me" mode — so the failure it produces is a
    refusal to start, which `_died` reports, rather than a wrong answer.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _answers(port: int) -> bool:
    """Is anything listening AND replying on that port yet?

    Through the same `http.client` the transcripts go through, deliberately. It used to use
    `urllib.request.urlopen`, which honours `http_proxy` and the system proxy settings — so
    in a shell with a corporate proxy exported and no `no_proxy` entry for `127.0.0.1` (which
    is the default), every probe was sent to a proxy that cannot reach the user's own
    machine. The model was loaded and answering, and `stt mic` waited two minutes and said it
    had not loaded. Two HTTP stacks in one module that can disagree about whether the same
    port is reachable is the bug; using one closes it.
    """
    link = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        link.request("GET", "/")
        link.getresponse().read()
    except OSError:
        return False
    else:
        # Any reply at all means the model finished loading — a 404 for `/` included.
        return True
    finally:
        link.close()


def _multipart(pcm: bytes, *, prompt: str) -> tuple[bytes, str]:
    """The request whisper-server's `/inference` endpoint expects."""
    boundary = uuid.uuid4().hex
    fields = {"response_format": "json", "temperature": "0.0", "no_timestamps": "true"}
    if prompt:
        fields["prompt"] = prompt
    parts = [_field(boundary, name, value) for name, value in fields.items()]
    parts.append(_file(boundary, _wav(pcm)))
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _field(boundary: str, name: str, value: str) -> bytes:
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
    return head.encode("utf-8") + value.encode("utf-8") + b"\r\n"


def _file(boundary: str, payload: bytes) -> bytes:
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="live.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    )
    return head.encode("utf-8") + payload + b"\r\n"


def _wav(pcm: bytes) -> bytes:
    """Wrap raw samples in the header whisper-server reads them through."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return buffer.getvalue()
