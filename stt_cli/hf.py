"""hf — Hugging Face credentials: where the token lives, and whether the gated models are
actually unlocked.

WHY THE TOKEN GOES WHERE huggingface_hub ALREADY LOOKS
    pyannote loads its pipeline through ``huggingface_hub`` — in this process and in every
    other one on the machine. Writing the token to the hub's own file therefore logs in
    every tool at once and nobody has to keep ``export HF_TOKEN=...`` in a shell profile.
    A private store under ``STT_HOME`` would look tidier and would make stt the only thing
    that works.

WHY THE GATE IS CHECKED BY FETCHING A FILE, NOT BY ASKING THE API
    ``/api/models/pyannote/speaker-diarization-3.1`` answers 200 for someone who has never
    accepted the terms: the metadata is public, only the weights are not. The one honest
    question is "can I download this", so that is the question asked — a request for
    ``config.yaml``, which is 401 before the terms are accepted and 200 after.

NO CREDENTIALS ARE EVER PRINTED
    Tokens are secrets that end up in scrollback, screenshots and pasted logs. Everything
    here reports a token as present or absent, never as a value.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ._errors import NetworkError, PermissionDeniedError

HOST = "https://huggingface.co"
TOKEN_PAGE = f"{HOST}/settings/tokens/new?tokenType=read&tokenName=stt-cli"
WHOAMI_URL = f"{HOST}/api/whoami-v2"

# Hugging Face user tokens are `hf_` + base62. The length is not contractual, so the
# pattern is deliberately loose: it exists to spot a token on the clipboard, not to
# validate one — validation is a round trip to WHOAMI_URL.
TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")

# Longest thing that could still plausibly be one token rather than a line of prose.
_MAX_TOKEN_CHARS = 200

# A token file holds one token. Anything larger is not one, and reading it is a trap:
# see _read_bounded.
_MAX_TOKEN_FILE = 64 * 1024

# The two gated repositories speaker-diarization-3.1 needs. The pipeline pulls the
# segmentation model in as a component, so accepting only the pipeline's own terms gets
# you an authorization error halfway through the first run instead of at login.
DIARIZATION_REPOS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0")

_TIMEOUT = 20.0
_GATE_PROBE = "config.yaml"


@dataclass(frozen=True, slots=True)
class Identity:
    """Who a token belongs to, as Hugging Face sees it."""

    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class Gate:
    """Whether one gated repository is downloadable with the token we hold."""

    repo: str
    granted: bool

    @property
    def page(self) -> str:
        return f"{HOST}/{self.repo}"


def token_path() -> Path:
    """The file ``huggingface_hub`` reads, honouring the hub's own environment overrides.

    All three of them, in the hub's own order. `XDG_CACHE_HOME` was missing, so on a machine
    that sets it stt wrote and reported `~/.cache/huggingface/token` while every other
    Hugging Face tool read `$XDG_CACHE_HOME/huggingface/token` — and the whole reason the
    token goes to the hub's file rather than a private one is that everything shares it.
    """
    explicit = os.environ.get("HF_TOKEN_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "token"
    cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "huggingface" / "token"


def read_token() -> str | None:
    """The token in force: an environment variable wins over the stored file, as in the hub."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value:
            return value if usable(value) else None
    return _read_bounded(token_path())


def unusable_variable() -> str:
    """The name of an environment variable holding something that cannot be a token.

    Reported rather than swallowed: `read_token` drops the value, so without this the status
    output would say "not signed in" while `$HF_TOKEN` sat there looking set, and the user
    would have no idea which of the two facts to believe.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value and not usable(value):
            return var
    return ""


def usable(token: str) -> bool:
    """Can this string be sent as a credential at all?

    A Hugging Face token is `hf_` and base62, so anything with whitespace or a non-ASCII
    character in it is not one. The check is not fussiness: such a value travels straight
    into an HTTP header, where a newline raises `ValueError` and a non-Latin-1 character
    raises `UnicodeEncodeError` — neither an `SttError`, so both came out as a raw traceback
    from `stt login --status`, the command people run when something is already wrong.
    """
    return bool(token) and token.isascii() and not re.search(r"\s", token)


def _content_of(path: Path) -> str:
    """A bounded prefix of a regular file, for the guards that decide whether it is ours.

    One character past the cap on purpose: an oversized file yields an oversized string, so
    a caller can tell "this is longer than a token" without reading the file whole. The
    guard that acts on it is `_too_big_to_be_ours`, which asks the filesystem rather than
    the content — past the cap the content itself can no longer say how long the file is.
    """
    return _read_through_a_descriptor(path, _MAX_TOKEN_FILE + 1) or ""


def _read_bounded(path: Path) -> str | None:
    """Read a token file, refusing anything that is not a small regular file.

    ``HF_TOKEN_PATH`` is caller-controlled and this is the FIRST thing that touches it, long
    before the guards that protect writing and deleting. ``HF_TOKEN_PATH=/dev/zero stt
    login --status`` would otherwise read until memory ran out, and a FIFO would block
    forever waiting for a writer — a hang with no message, on the command people run when
    something is already wrong.

    Decoded with ``errors="replace"`` for the same reason `_content_of` is: a small binary
    file passes every guard above and then a strict decode raises `UnicodeDecodeError`,
    which is not an `OSError` and so escaped as a raw traceback. What comes back is then
    required to be ASCII, because a Hugging Face token is `hf_` and base62 and nothing else:
    without that check the replacement characters travelled on into an HTTP header and
    raised `UnicodeEncodeError` one layer further down. "This file is not a token" is the
    honest answer, and it is the one the caller can act on.
    """
    stored = _read_through_a_descriptor(path, _MAX_TOKEN_FILE, cap_size=True)
    if stored is None:
        return None
    return stored if usable(stored) and len(stored) <= _MAX_TOKEN_CHARS else None


def _read_through_a_descriptor(path: Path, limit: int, *, cap_size: bool = False) -> str | None:
    """Open once, without blocking, and ask the DESCRIPTOR what it is.

    Checking the path and then opening it are two moments, and the file can change between
    them: a regular file replaced by a FIFO in that gap made the open wait for a writer that
    never came — `stt login --status` hanging with no message, which is the exact failure
    the `is_file()` check was there to prevent. `O_NONBLOCK` returns instead of waiting, and
    `fstat` then asks the file that was really opened rather than whatever the name points
    at by now. The same shape as the glossary import's read.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                return None
            if cap_size and info.st_size > _MAX_TOKEN_FILE:
                return None
            return handle.read(limit).strip()
    except OSError:
        return None


def stored_token() -> str | None:
    """What is in the token FILE, ignoring the environment that normally wins.

    `read_token` answers "what is in force", which is the question almost everything has. The
    exception is the shadowing check: to say whether a variable is overriding something
    different, it has to be able to see both.
    """
    return _read_bounded(token_path())


def token_source() -> str:
    """Where the token in force came from, for status output. Never the token itself."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var, "").strip():
            return f"${var}"
    return str(token_path()) if read_token() else ""


def store_token(token: str) -> Path:
    """Write the token where the hub looks for it, readable by this user only.

    Written to a fresh temporary file and renamed into place. Three things have to be true
    at once and only this shape gets all three: the file is never world-readable even for an
    instant (created 0600 by ``mkstemp``, not chmod'ed afterwards, because between the write
    and the chmod the token sits there at whatever the umask allows); a half-written file is
    never visible under the real name, where the next process would read it as a malformed
    credential rather than as a missing one; and two logins racing cannot delete each
    other's temporary file, which a single predictable ``token.part`` would allow.
    """
    import os
    import tempfile

    path = token_path()
    with _sole_writer(path):
        # Checked INSIDE the lock, like the delete side: between a check outside it and the
        # write, another writer can change what is at the path.
        _refuse_to_clobber(path)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".token-", suffix=".part")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(token.strip() + "\n")
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return path


@contextmanager
def _sole_writer(path: Path) -> Iterator[None]:
    """Hold the token file still while stt writes it or deletes it.

    `logout` checks that the file really holds a token and only then unlinks it. Between
    those two steps a `login` in another terminal can replace the file — and the unlink then
    removes the token that was just stored, signing the user out of a session they had just
    signed into. Reading and deleting under the same lock as writing makes the check and the
    act one step. The lock covers stt's own processes; a `huggingface-cli login` running at
    the same instant does not take it, and nothing here can make it.
    """
    lock = path.with_name(f"{path.name}.lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock.open("a+", encoding="utf-8")
    except OSError as exc:
        # `HF_TOKEN_PATH` is caller-controlled and need not be usable: pointed inside a
        # regular file (`HF_TOKEN_PATH=/tmp/notes.txt/token`) the mkdir raises, and a raw
        # traceback is not a diagnosis. It surfaced only after the token had been verified,
        # so the user had done the whole browser dance before seeing it.
        raise PermissionDeniedError(
            what=f"cannot use {path} for the token",
            why=str(exc),
            how="point HF_TOKEN_PATH at a writable location, or unset it to use the default",
        ) from exc
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _refuse_to_clobber(path: Path) -> None:
    """Never write over a file that is not already a token file.

    ``HF_TOKEN_PATH`` is caller-controlled, so ``HF_TOKEN_PATH=~/notes.md stt login`` would
    otherwise replace those notes with a credential. ``forget_token`` guards the same case
    on the way out; writing needs it just as much as deleting does.
    """
    if not path.exists():
        return
    if not path.is_file():
        raise PermissionDeniedError(
            what=f"refusing to write the token to {path}",
            why="it exists and is not a regular file",
            how="check HF_TOKEN_PATH — it should point at a token file, or at nothing yet",
        )
    content = _content_of(path)
    if not content and _is_an_override(path):
        # An empty file the caller pointed us at is still THEIR file — `touch notes.md` and
        # `HF_TOKEN_PATH=~/notes.md stt login` would have replaced it with a credential. The
        # hub's own path is different: an empty `~/.cache/huggingface/token` is a token file
        # that happens to be empty, and refusing it would make login unrecoverable.
        raise PermissionDeniedError(
            what=f"refusing to overwrite {path}",
            why="it exists and is empty, so there is nothing to say it is a token file",
            how="delete it first if it really is stt's, or point HF_TOKEN_PATH elsewhere",
        )
    if content and not _is_a_credential_file(content, path):
        raise PermissionDeniedError(
            what=f"refusing to overwrite {path}",
            why="it already holds something that is not a Hugging Face token",
            how="check HF_TOKEN_PATH, or move that file aside if you really meant this one",
        )


def _too_big_to_be_ours(path: Path) -> bool:
    """A file stt would refuse to READ as a token is not stt's to overwrite or delete.

    `_content_of` stops at a couple of hundred characters, so it cannot tell a slightly long
    token file from a hundred-kilobyte blob — and without this the blob passed the
    single-line test and `stt logout` deleted it, while `stt login --status` had already
    refused to read it. Size is the one question the content cannot answer.
    """
    try:
        return path.stat().st_size > _MAX_TOKEN_FILE
    except OSError:
        return False


def _is_a_credential_file(content: str, path: Path) -> bool:
    """Is this file stt's to overwrite or delete? The answer depends on WHOSE path it is.

    On the hub's own token file the test is loose: the hub has written more than one token
    format over the years, and refusing to log a user in or out of their own credential just
    because it does not start with ``hf_`` is the guard firing on the wrong target. What
    distinguishes a document there is whitespace, not a prefix and not length. Length was
    the third test here and it disagreed with the one `read_token` applies: a file between
    the two bounds was readable as a credential and at the same time refused replacement and
    deletion, so the user was signed in to something they could neither log out of nor
    overwrite. One long unbroken line is not a document, whatever its length; something with
    a space or a newline in it is, and that is still refused.

    On a path the caller supplied through ``HF_TOKEN_PATH`` the test is strict, because that
    value points anywhere: ``HF_TOKEN_PATH=~/important/checksum`` names a file whose single
    whitespace-free line is a checksum, not a credential, and the loose test would let
    ``stt login`` overwrite it and ``stt logout`` delete it. There, only something shaped
    like a Hugging Face token counts.
    """
    if not content or _too_big_to_be_ours(path):
        return False
    if _is_an_override(path):
        return bool(TOKEN_PATTERN.fullmatch(content))
    return usable(content)


def _is_an_override(path: Path) -> bool:
    """Was this path named by the caller rather than derived from the hub's own convention?"""
    explicit = os.environ.get("HF_TOKEN_PATH", "").strip()
    return bool(explicit) and Path(explicit).expanduser() == path


def forget_token() -> Path | None:
    """Delete the stored token. Returns the path removed, or None if there was nothing.

    The path is not ours to trust: ``HF_TOKEN_PATH`` points it anywhere the caller likes, so
    ``HF_TOKEN_PATH=~/notes.md stt logout`` would otherwise delete a file that has nothing to
    do with stt. Deleting only a file whose content is a Hugging Face token keeps ``logout``
    honest about what it claims to remove.
    """
    path = token_path()
    if not path.is_file():
        return None
    with _sole_writer(path):
        # Re-checked INSIDE the lock, not outside it: see `_sole_writer`.
        if not path.is_file():
            return None
        content = _content_of(path)
        if not _is_a_credential_file(content, path):
            raise PermissionDeniedError(
                what=f"refusing to delete {path}",
                why="it does not contain a Hugging Face token, so it is not stt's to remove",
                how="check HF_TOKEN_PATH, or delete the file yourself if you really meant to",
            )
        path.unlink()
    return path


def whoami(token: str) -> Identity | None:
    """Who the token belongs to, or None if Hugging Face rejects it."""
    code, body = _request(WHOAMI_URL, token=token)
    if code in (401, 403):
        return None
    if code != 200:
        raise NetworkError(
            what="Hugging Face did not answer the identity check",
            why=f"{WHOAMI_URL} returned HTTP {code}",
            how="try again in a moment; if it persists, check https://status.huggingface.co",
        )
    import json

    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        data = None
    if not isinstance(data, dict):
        # A 200 that is not the identity JSON is not an identity. A captive portal or a
        # corporate proxy answers 200 with an HTML login page to everything, and reading
        # that as "signed in as ?" reported a successful login, with every gate granted,
        # for a machine that could not reach Hugging Face at all — while the module is
        # otherwise careful to tell "refused" apart from "the service did not answer".
        raise NetworkError(
            what="Hugging Face did not answer the identity check",
            why=f"{WHOAMI_URL} returned 200 but not an identity — something answered for it",
            how="check whether a proxy or a captive portal is intercepting the connection",
        )
    return Identity(name=str(data.get("name") or "?"), kind=str(data.get("type") or "user"))


# What "you may not have this" actually looks like. Everything else is the service having
# a bad day, which is a different sentence to say to the user.
_REFUSED = (401, 403)


def gate(repo: str, token: str | None) -> Gate:
    """Can this token download ``repo``? Only a refusal counts as not yet granted.

    Treating every non-200 as "not granted" turned a 500 or a 429 into "you have not
    accepted the terms of these models", sending the user back to a page they had already
    agreed to. A service failure is a network problem and is reported as one.
    """
    url = f"{HOST}/{repo}/resolve/main/{_GATE_PROBE}"
    code, _ = _request(url, token=token, method="HEAD")
    if code == 200 or code in _REFUSED:
        return Gate(repo=repo, granted=code == 200)
    raise NetworkError(
        what=f"could not check whether {repo} is unlocked",
        why=f"Hugging Face answered {code}",
        how="this is a service or network problem, not a login one — try again shortly",
    )


def gates(token: str | None, repos: tuple[str, ...] = DIARIZATION_REPOS) -> list[Gate]:
    return [gate(repo, token) for repo in repos]


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Drop the bearer token when a redirect leaves https://huggingface.co.

    ``resolve/main/<file>`` answers with a redirect to a CDN, and urllib copies every
    header of the original request onto the redirected one. That would hand the user's
    Hugging Face token to whatever host the redirect names. The CDN URL is signed and
    needs no credential, so removing the header costs nothing and closes the leak.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        follow = super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]
        if follow is not None and _origin_of(newurl) != _origin_of(HOST):
            # Request.remove_header does not exist for unredirected headers on every
            # version, so both stores are cleared by hand.
            follow.headers.pop("Authorization", None)
            follow.unredirected_hdrs.pop("Authorization", None)
        return follow


def _origin_of(url: str) -> tuple[str, str]:
    """Scheme AND host. Comparing hosts alone would let an https -> http redirect on
    huggingface.co keep the bearer token, and put it on the wire in cleartext."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def _request(url: str, *, token: str | None = None, method: str = "GET") -> tuple[int, bytes]:
    """One small HTTP call.

    Synchronous ``urllib`` rather than ``proc.run``: login is interactive, sits outside the
    transcription path, and has nothing to overlap with. It also keeps the token out of any
    argument vector, which a ``curl`` call would not.
    """
    import urllib.error

    headers = {"User-Agent": "stt-cli"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    opener = urllib.request.build_opener(_SameHostAuth)
    try:
        with opener.open(request, timeout=_TIMEOUT) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except urllib.error.URLError as exc:
        raise NetworkError(
            what=f"could not reach {HOST}",
            why=str(exc.reason),
            how="check the network or your proxy, then run `stt login diarization` again",
        ) from exc
    except OSError as exc:  # a TLS or socket failure that urllib did not wrap
        raise NetworkError(
            what=f"could not reach {HOST}",
            why=str(exc),
            how="check the network or your proxy, then run `stt login diarization` again",
        ) from exc
