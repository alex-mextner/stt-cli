"""auth — `stt login <capability> --provider <name>`: credentials without the wiki page.

WHAT THIS IS FOR
    Some features need somebody else's credentials. Speaker diarization needs a Hugging
    Face token, because the pyannote pipeline is a gated model. The usual way to get one is
    a paragraph of instructions: open this page, accept those terms, open that other page,
    create a token, copy it, export it in your shell profile. Every step is a place to stop.
    So stt drives the whole thing instead — it opens the right pages in the right order,
    picks the token up the moment it lands on the clipboard, verifies it against the API,
    stores it where every tool on the machine will find it, then checks that each gated
    model is genuinely unlocked and reopens the ones that are not.

WHY A REGISTRY FOR ONE PROVIDER
    ``CAPABILITIES`` maps a feature to the providers that can serve it, first one being the
    default. Today that is one entry. It is a dict so that the second provider — a different
    diarizer, a paid transcription API — is a dict entry and a status/login pair, not a
    rewrite of the command.

INTERACTIVE ON PURPOSE
    Everything here is synchronous and talks to a terminal. It is outside the transcription
    path, so it does not use ``proc.run``: it wants a browser, a clipboard and a prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import hf
from ._errors import (
    EXIT_NETWORK,
    EXIT_OK,
    EXIT_PERMISSION,
    NetworkError,
    PermissionDeniedError,
    unknown_item,
)

# capability -> providers that can serve it; the first is the default.
CAPABILITIES: dict[str, tuple[str, ...]] = {"diarization": ("hf",)}

# What people actually type. `stt login diarize` should not be a did-you-mean.
ALIASES: dict[str, str] = {
    "diarize": "diarization",
    "speakers": "diarization",
    "diarisation": "diarization",
}

PROVIDER = "hf"
PROVIDER_TITLES: dict[str, str] = {PROVIDER: "Hugging Face"}

DEFAULT_CAPABILITY = "diarization"
CLIPBOARD_POLL = 0.5
CAPTURE_TIMEOUT = 300.0
GATE_ROUNDS = 4


@dataclass(slots=True)
class Status:
    """The answer to "am I signed in, and can I actually use the thing"."""

    capability: str
    provider: str
    ok: bool
    lines: list[str] = field(default_factory=list)
    # Why it is not ok, when that is not "the credential is wrong". A script reads the exit
    # code, and `stt login --status || stt login` must not launch an interactive login just
    # because the network blipped — the stored token was fine. The lines already say so;
    # this is what lets the code say it too.
    unreachable: bool = False

    def render(self) -> str:
        head = f"{self.capability} via {PROVIDER_TITLES.get(self.provider, self.provider)}"
        return "\n".join([head, *(f"  {line}" for line in self.lines)])

    def exit_code(self) -> int:
        """The code a script reads. Lives on the status so its two callers cannot drift.

        `report` and `stt diarize --status` both need it, and the second one had a copy —
        identical today, and one more case away from disagreeing about whether a network
        blip means "sign in again".
        """
        if self.ok:
            return EXIT_OK
        return EXIT_NETWORK if self.unreachable else EXIT_PERMISSION


def resolve(capability: str | None, provider: str | None) -> tuple[str, str]:
    """Normalize what the user typed into a (capability, provider) pair, or explain why not."""
    name = ALIASES.get(capability or DEFAULT_CAPABILITY, capability or DEFAULT_CAPABILITY)
    if name not in CAPABILITIES:
        raise unknown_item("capability", name, sorted(CAPABILITIES), plural="capabilities")
    known = CAPABILITIES[name]
    if provider is None:
        return name, known[0]
    if provider not in known:
        raise unknown_item(f"{name} provider", provider, list(known), plural="providers")
    return name, provider


def _only_hugging_face(provider: str) -> None:
    """Refuse a provider nobody implemented, loudly, at the moment the assumption breaks.

    ``CAPABILITIES`` is a registry, but there is no lookup from a provider name to a
    status/login pair — every function below IS the Hugging Face flow. Adding a second
    provider to that dict is therefore not enough on its own, and without this the day it
    happens `stt login diarization --provider assemblyai` passes `resolve()`, silently runs
    the Hugging Face flow, and reports success attributed to a provider that never ran.
    """
    if provider != PROVIDER:
        raise unknown_item("provider", provider, [PROVIDER], plural="providers")


def status(capability: str, provider: str) -> Status:
    """Report without changing anything.

    Never raises on a bad credential — a rejected token is a finding, not an exception. It
    does not raise on a missing network either: this is the command people run when
    something is wrong, and refusing to answer offline makes it useless exactly then.
    """
    _only_hugging_face(provider)
    token = hf.read_token()
    if not token:
        # A variable that is set but holds something that cannot be a token is reported as
        # such: dropped silently, the output would say "not signed in" while `$HF_TOKEN` sat
        # there looking set, and nothing would say which of the two facts to believe.
        broken = hf.unusable_variable()
        lines = ["not signed in", f"fix: stt login {capability}"]
        if broken:
            lines.insert(1, f"note: ${broken} is set but is not a Hugging Face token")
        return Status(capability, "hf", False, lines)
    try:
        return _status_online(capability, token)
    except NetworkError as exc:
        return Status(
            capability,
            "hf",
            False,
            [
                f"a token is present ({hf.token_source()}) but it could not be checked",
                f"why: {exc.why}",
                "the stored token may well be fine — this is a network problem, not a login one",
            ],
            unreachable=True,
        )


def _status_online(capability: str, token: str) -> Status:
    identity = hf.whoami(token)
    if identity is None:
        return _rejected(capability)
    return _status_from(capability, token, hf.gates(token), identity=identity)


def _rejected(capability: str) -> Status:
    """One rendering of "Hugging Face does not accept this token", not two.

    Both status paths built it, byte for byte, and this file has already been bitten once by
    a mapping that existed in two places.
    """
    return Status(
        capability,
        "hf",
        False,
        [f"the token in {hf.token_source()} is rejected", f"fix: stt login {capability} --force"],
    )


def _status_from(
    capability: str,
    token: str,
    gates: list[hf.Gate],
    *,
    identity: hf.Identity | None = None,
) -> Status:
    """Render a status from gates somebody has already fetched."""
    who = identity or hf.whoami(token)
    if who is None:
        return _rejected(capability)
    lines = [f"signed in as {who.name} ({who.kind}), token from {hf.token_source()}"]
    lines += _shadow_warning()
    lines += [f"{'ok  ' if g.granted else 'GATED'} {g.repo}" for g in gates]
    ok = all(g.granted for g in gates)
    if not ok:
        lines.append(f"fix: stt login {capability}  (reopens the pages that need accepting)")
    return Status(capability, "hf", ok, lines)


def _shadow_variable() -> str:
    """The environment variable overriding the stored token, if one is set."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var, "").strip():
            return var
    return ""


def _shadow_warning() -> list[str]:
    """Say when an exported variable is the credential actually in force.

    This is the trap that makes a successful login look like a failure: `stt login` writes a
    fresh token to the file, and then everything — this status, and pyannote at run time —
    reads the stale `HF_TOKEN` from the shell instead, because the environment wins. Saying
    so is the only fix available from inside the process: a child cannot unset a variable in
    the shell that started it.
    """
    var = _shadow_variable()
    if not var:
        return []
    if os.environ.get(var, "").strip() == (hf.stored_token() or ""):
        # The same token in both places is not a misconfiguration, and calling it one sends
        # somebody off to "fix" a setup that works. `_report_shadowing` already compared the
        # values before warning; this half did not.
        return []
    if not hf.token_path().is_file():
        # There is nothing for the variable to shadow. Telling somebody to unset the only
        # credential they have, in order to "use" a file that does not exist, is advice that
        # signs them out — so the note says what is true of this machine instead.
        return [f"note: the token comes from ${var}, not from a stored file — nothing is saved"]
    return [f"note: ${var} is set and overrides the stored file — `unset {var}` to use it"]


def login(capability: str, provider: str, *, browser: bool = True, force: bool = False) -> Status:
    """Get a working credential, accepting the model terms on the way."""
    _only_hugging_face(provider)
    stored = hf.read_token()
    token = None if force else stored
    identity = hf.whoami(token) if token else None
    if token and identity is None:
        saved = _the_saved_one_if_it_still_works(token)
        if saved is not None:
            token, identity = saved
        else:
            print("the stored Hugging Face token is no longer valid — getting a new one")
            token = None
    if token is None:
        # `stored` is on the clipboard as often as not — it is what the user copied last
        # time — and handing it straight back would make `--force` a no-op.
        token = _obtain_token(browser=browser, rejected=stored)
        identity = hf.whoami(token)
        if identity is None:
            raise PermissionDeniedError(
                what="Hugging Face rejected that token",
                why="the identity check came back unauthorized",
                how=f"create a fresh read token at {hf.TOKEN_PAGE} and try again",
            )
        path = hf.store_token(token)
        print(f"signed in as {identity.name} — token stored in {path} (mode 600)")
        _report_shadowing(token)
    else:
        _adopt_an_exported_token(token)
    gates = _unlock_gates(token, browser=browser)
    # Built from the gates the walkthrough just fetched. Calling `status()` here instead
    # would re-run `whoami` and re-fetch every gate — four more network round-trips on the
    # happy path, to learn what the walkthrough already knows.
    result = _status_from(capability, token, gates, identity=identity)
    return _unless_shadowed(result, token)


def _adopt_an_exported_token(token: str) -> None:
    """Save a working `$HF_TOKEN` that has nowhere to be saved yet.

    `read_token` prefers the environment, so a valid exported token made `login` verify it,
    print a success and store nothing at all — while the README promises the command exists
    so that "HF_TOKEN never has to live in your shell profile". The user then took the
    variable out of their profile, on the strength of that promise, and diarization stopped
    working. Writing it to the file the hub reads is what they asked for and what every
    other path of this command does.

    Only when there is no stored token. An existing file is a deliberate credential of its
    own, and overwriting it with whatever a shell happens to export — the thing
    `_report_shadowing` warns about — would be the opposite of the fix.
    """
    if hf.token_source().startswith("$") and not hf.token_path().is_file():
        path = hf.store_token(token)
        print(f"the exported token works — saved to {path} (mode 600), so it is no longer")
        print("only in your shell profile")


def _the_saved_one_if_it_still_works(rejected: str) -> tuple[str, hf.Identity] | None:
    """The token in the FILE, when the one that just failed came from the environment.

    `read_token` prefers `$HF_TOKEN`, so a revoked variable made `login` conclude that "the
    stored token" was dead, ask for a new one, and write it over the file — which was
    holding a perfectly good token the whole time. The variable still won, so the command
    failed anyway, having destroyed the credential that `unset HF_TOKEN` would have
    restored. Checking the file before asking for anything costs one request and keeps it.
    """
    saved = hf.stored_token()
    if not saved or saved == rejected:
        return None
    identity = hf.whoami(saved)
    if identity is None:
        return None
    print("the token in the environment is rejected, but the stored one still works")
    return saved, identity


def _unless_shadowed(result: Status, verified: str) -> Status:
    """Refuse to call it a success while a different token is the one actually in force.

    `login` verifies and stores the token it obtained, but `hf.read_token()` prefers
    `$HF_TOKEN`. With a stale variable exported, the command would verify the NEW token,
    report "signed in", and exit zero — while diarization went on using the stale one and
    failed. The warning was already printed; what was missing is that the exit code agreed
    with it, since that is what a script reads.
    """
    effective = hf.read_token()
    if effective == verified:
        return result
    var = _shadow_variable()
    why = (
        [
            f"${var} is exported and wins over the stored file",
            f"fix: unset {var}, then run `stt login {result.capability} --status`",
        ]
        if var
        else ["fix: check HF_TOKEN_PATH and HF_HOME — the stored file is not the one read"]
    )
    return Status(
        result.capability,
        result.provider,
        False,
        ["the token was stored, but it is NOT the one in force", *why, *result.lines],
    )


def _report_shadowing(stored: str) -> None:
    """Warn when the shell's variable will keep beating the token we just stored.

    Without this the command prints "signed in, token stored" and then, one line later, that
    the token is rejected — because `status` re-reads the environment, which still holds the
    old one. Diarization would fail at run time for the same reason.
    """
    var = _shadow_variable()
    if not var or os.environ[var].strip() == stored:
        return
    print(f"\n  WARNING: ${var} is set to a DIFFERENT token and takes precedence.")
    print(f"  The token just stored will not be used until you run:  unset {var}")


def logout(capability: str, provider: str) -> Status:
    _only_hugging_face(provider)
    removed = hf.forget_token()
    # Said plainly, because the scope is wider than the command name suggests. The token is
    # deliberately stored where `huggingface_hub` reads it — that is what makes one login
    # serve every tool on the machine — and the same fact makes this one logout take them
    # all with it, including a token that was created with `huggingface-cli login` in the
    # first place. Somebody tidying up after trying diarization should not have to work out
    # afterwards why an unrelated workflow stopped authenticating.
    lines = (
        [f"removed {removed}", "this signs every Hugging Face tool on this machine out"]
        if removed
        else ["nothing stored"]
    )
    var = _shadow_variable()
    if var:
        lines.append(f"note: ${var} is still set in this shell and overrides the file")
    return Status(capability, "hf", True, lines)


def report(result: Status) -> int:
    print(result.render())
    return result.exit_code()


# ── the interactive half ──────────────────────────────────────────────────────────────


def open_page(url: str) -> None:
    """Show a page in the default browser, and stay useful when there isn't one."""
    import webbrowser

    print(f"  opening {url}")
    try:
        opened = webbrowser.open(url)
    except Exception:  # a headless or misconfigured environment, never fatal here
        opened = False
    if not opened:
        print("  (no browser could be opened — visit that URL yourself)")


def _obtain_token(*, browser: bool, rejected: str | None = None) -> str:
    """Send the user to the token page and wait for the token to come back.

    ``rejected`` is the token we already know does not work — the one `--force` or a failed
    check just discarded. It is very likely still on the clipboard, so it has to be ignored
    there or the user is handed straight back the credential they came here to replace.
    """
    print("stt needs a Hugging Face token to download the diarization models.")
    print("A read token is enough; stt never writes to your account.")
    if browser:
        open_page(hf.TOKEN_PAGE)
    else:
        print(f"  create one at {hf.TOKEN_PAGE}")
    return _capture_token(rejected=rejected)


def _capture_token(timeout: float = CAPTURE_TIMEOUT, *, rejected: str | None = None) -> str:
    """Wait for an ``hf_…`` token from the clipboard or from the prompt, whichever lands first.

    The clipboard branch is what makes this painless: the token page has a Copy button, so
    the user clicks it and stt already has the token. Typing it is still supported, because
    a clipboard manager, a remote shell or a paste of something else all break that path.
    """
    import sys

    if not sys.stdin.isatty():
        return _token_from_stdin(_piped(timeout), rejected)
    print("\nClick 'Copy' on the new token — stt will pick it up from the clipboard.")
    print("(or paste it here and press Enter)")
    return _watch(timeout, rejected=rejected)


# How much of a pipe is read while looking for a token. Generous for the shapes people
# actually pipe in (`echo $TOKEN`, a one-line file, a small JSON blob), and small enough
# that an endless producer is a diagnosed failure rather than a hang.
PIPED_TOKEN_CHARS = 64 * 1024


def _piped(timeout: float) -> str:
    """Whatever the pipe has to say, without waiting on a writer that has stopped talking.

    Bounding the SIZE of the read does not bound the WAIT: `read(n)` returns when it has n
    bytes or when the writer closes, so `{ printf 'not a token'; sleep 3600; } | stt login`
    sat there for an hour on a payload that had already arrived and was already wrong. This
    reads what is ready, stops as soon as a token is in hand, and otherwise gives up at the
    same deadline the interactive path uses — the failure is then the diagnosed one.
    """
    import os
    import select
    import sys
    import time

    try:
        source = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        # Not a real file: a captured or substituted stream cannot be polled, and there is
        # nobody on the other end of it to wait for either. Read it whole, bounded.
        return sys.stdin.read(PIPED_TOKEN_CHARS)

    deadline, blob = time.monotonic() + timeout, ""
    while len(blob) < PIPED_TOKEN_CHARS:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([source], [], [], left)[0]:
            break
        chunk = os.read(source, 4096)
        if not chunk:  # the writer closed: this is everything there will ever be
            break
        blob += chunk.decode("utf-8", "replace")
        if hf.TOKEN_PATTERN.search(blob):
            break
    return blob


def _watch(timeout: float, *, rejected: str | None = None) -> str:
    """Poll the clipboard and the terminal together until one of them yields a token.

    The clipboard is read BEFORE the loop as well as inside it. Waiting for a change was the
    obvious way to avoid grabbing a stale token, but it makes the common order fail: copy the
    token first, then run `stt login`, and the token is sitting right there while the command
    waits five minutes for it to be copied again. What actually has to be ignored is the ONE
    token already known to be bad, which is `rejected` — a value, not a moment in time.
    """
    import select
    import sys
    import time

    seen = _clipboard()
    if (first := _token_in(seen, rejected)) is not None:
        print("  picked the token up from the clipboard")
        return first
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = _clipboard()
        if current != seen:
            seen = current
            if (found := _token_in(current, rejected)) is not None:
                print("  picked the token up from the clipboard")
                return found
        if select.select([sys.stdin], [], [], CLIPBOARD_POLL)[0]:
            typed = sys.stdin.readline()
            if not typed:
                break
            if (given := _token_in(typed, rejected)) is not None:
                return given
            if rejected and rejected in typed:
                print("  that is the token that was just refused — create a new one.")
            elif typed.strip():
                print("  that is not a token — they look like hf_xxxxxxxx. Try again.")
    raise PermissionDeniedError(
        what="no Hugging Face token arrived",
        why=f"nothing matching hf_… reached the clipboard or the prompt in {timeout / 60:.0f} min",
        how=f"create a read token at {hf.TOKEN_PAGE}, then run `stt login diarization` again",
    )


def _token_in(blob: str, rejected: str | None) -> str | None:
    """A usable token in this text — never the one we already know is refused.

    Every match, not just the first: a clipboard that holds the old token followed by the
    new one is exactly what `--force` produces, and stopping at the first match would find
    the rejected one, decline it, and wait out the timeout with the good token right there.
    """
    for found in hf.TOKEN_PATTERN.finditer(blob):
        if found.group(0) != rejected:
            return found.group(0)
    return None


def _token_from_stdin(blob: str, rejected: str | None = None) -> str:
    """Non-interactive path: ``echo $TOKEN | stt login diarization``.

    `rejected` matters here as much as on the clipboard: `echo "$OLD" | stt login --force`
    would otherwise accept, verify and re-store the very token `--force` exists to replace,
    and report success.
    """
    found = _token_in(blob, rejected)
    if found:
        return found
    if rejected and rejected in blob:
        raise PermissionDeniedError(
            what="that is the token being replaced",
            why="`--force` was asked to get a NEW token, and stdin carried the old one",
            how=f"create a fresh read token at {hf.TOKEN_PAGE} and pipe that in instead",
        )
    raise PermissionDeniedError(
        what="no Hugging Face token on stdin",
        why="stdin is not a terminal and what arrived contained nothing matching hf_…",
        how="pipe the token in, or run `stt login diarization` from a terminal",
    )


def _clipboard() -> str:
    """The clipboard as text, or empty when it cannot be read. Never an error: this is a
    convenience path and the prompt is always there as the real one."""
    import shutil
    import subprocess

    pbpaste = shutil.which("pbpaste")
    if pbpaste is None:
        return ""
    try:
        done = subprocess.run([pbpaste], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout


def _unlock_gates(token: str, *, browser: bool) -> list[hf.Gate]:
    """Walk the user through accepting the terms of every gated model that still needs it."""
    import sys

    checked = hf.gates(token)
    for _ in range(GATE_ROUNDS):
        pending = [gate for gate in checked if not gate.granted]
        if not pending:
            break
        print(f"\n{len(pending)} model(s) still need their terms accepted:")
        for gate in pending:
            # The URL is printed whether or not a browser is opened. Over a remote shell
            # `--no-browser` used to leave the user told to click a button on a page whose
            # address was never shown — the token half of this flow prints its page for
            # exactly the same reason.
            print(f"  {gate.repo} — click 'Agree and access repository'")
            print(f"    {gate.page}")
            if browser:
                open_page(gate.page)
        if not sys.stdin.isatty():
            break
        input("press Enter once you have accepted them ")
        checked = hf.gates(token)
    return checked
