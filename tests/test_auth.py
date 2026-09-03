"""`stt login` — where the token lives, and how the flow behaves when nothing cooperates.

Nothing here touches the network. The three things worth pinning down are that the token
file is the one huggingface_hub reads (get that wrong and pyannote stays logged out while
stt claims success), that a token is never printed, and that every path a non-interactive
shell can take ends in a diagnosed error rather than a five-minute wait.
"""

from __future__ import annotations

import io
import stat

import pytest

from stt_cli import auth, hf
from stt_cli._errors import (
    EXIT_OK,
    EXIT_PERMISSION,
    PermissionDeniedError,
    UnknownItemError,
    unknown_item,
)
from stt_cli.cli import main

TOKEN = "hf_" + "aB3" * 8


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """No inherited credentials, and never the real ~/.cache/huggingface."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))


# ── where the token lives ─────────────────────────────────────────────────────
def test_token_path_is_the_one_huggingface_hub_reads(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_HOME")
    assert hf.token_path().parts[-3:] == (".cache", "huggingface", "token")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    assert hf.token_path() == tmp_path / "hub" / "token"
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "elsewhere"))
    assert hf.token_path() == tmp_path / "elsewhere"


def test_stored_token_round_trips_and_is_private() -> None:
    path = hf.store_token(TOKEN)
    assert hf.read_token() == TOKEN
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert hf.forget_token() == path
    assert hf.read_token() is None
    assert hf.forget_token() is None


def test_environment_beats_the_stored_file(monkeypatch) -> None:
    hf.store_token(TOKEN)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 20)
    assert hf.read_token() == "hf_" + "z" * 20
    assert hf.token_source() == "$HF_TOKEN"


def test_token_source_is_a_location_never_the_token() -> None:
    assert hf.token_source() == ""
    hf.store_token(TOKEN)
    assert TOKEN not in hf.token_source()


# ── capability / provider resolution ──────────────────────────────────────────
@pytest.mark.parametrize("typed", [None, "diarization", "diarize", "speakers"])
def test_aliases_all_reach_diarization(typed: str | None) -> None:
    assert auth.resolve(typed, None) == ("diarization", "hf")


def test_unknown_capability_and_provider_are_diagnosed() -> None:
    with pytest.raises(UnknownItemError):
        auth.resolve("translation", None)
    with pytest.raises(UnknownItemError):
        auth.resolve("diarization", "openai")


def test_unknown_item_pluralizes_without_inventing_words() -> None:
    error = unknown_item("capability", "x", ["diarization"], plural="capabilities")
    assert "capabilitys" not in error.render()
    assert "capabilities" in error.render()


# ── status ────────────────────────────────────────────────────────────────────
def test_status_without_a_token_says_how_to_fix_it() -> None:
    result = auth.status("diarization", "hf")
    assert not result.ok
    assert "stt login diarization" in result.render()


def test_status_reports_a_rejected_token_instead_of_raising(monkeypatch) -> None:
    hf.store_token(TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _token: None)
    result = auth.status("diarization", "hf")
    assert not result.ok
    assert "rejected" in result.render()


def test_status_is_only_ok_when_every_gate_is_granted(monkeypatch) -> None:
    hf.store_token(TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity(name="alex", kind="user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True), hf.Gate("b", False)])
    assert not auth.status("diarization", "hf").ok
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True), hf.Gate("b", True)])
    assert auth.status("diarization", "hf").ok


# ── the non-interactive paths ─────────────────────────────────────────────────
def test_a_piped_token_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(f"  {TOKEN}\n"))
    assert auth._capture_token() == TOKEN


def test_a_piped_non_token_fails_fast_rather_than_waiting(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not a token\n"))
    with pytest.raises(PermissionDeniedError):
        auth._capture_token()


def test_gate_walkthrough_does_not_loop_when_nobody_can_answer(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr(auth, "open_page", lambda _url: None)
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", False)])
    assert auth._unlock_gates(TOKEN, browser=True) == [hf.Gate("a", False)]


def test_an_unreadable_clipboard_is_empty_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert auth._clipboard() == ""


# ── the commands ──────────────────────────────────────────────────────────────
def test_login_status_exits_permission_when_signed_out(capsys) -> None:
    assert main(["login", "--status"]) == EXIT_PERMISSION
    assert "not signed in" in capsys.readouterr().out


def test_logout_removes_the_file_and_warns_about_the_shell(monkeypatch, capsys) -> None:
    hf.store_token(TOKEN)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert main(["logout", "diarize"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "removed" in out
    assert "$HF_TOKEN is still set" in out
    assert TOKEN not in out


def test_login_stores_the_token_it_was_given(monkeypatch, capsys) -> None:
    monkeypatch.setattr(auth, "_obtain_token", lambda *, browser, rejected=None: TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity(name="alex", kind="user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    assert main(["login", "diarization", "--no-browser"]) == EXIT_OK
    assert hf.read_token() == TOKEN
    assert TOKEN not in capsys.readouterr().out


def test_login_refuses_a_token_the_api_rejects(monkeypatch) -> None:
    monkeypatch.setattr(auth, "_obtain_token", lambda *, browser, rejected=None: TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: None)
    assert main(["login", "--no-browser"]) == EXIT_PERMISSION
    assert hf.read_token() is None


# ── the trap a review found: the shell's variable beats the file we just wrote ─
def test_login_warns_when_an_exported_token_shadows_the_stored_one(monkeypatch, capsys) -> None:
    """Without the warning the command prints "signed in, stored" and then "rejected"."""
    stale = "hf_" + "s" * 22
    monkeypatch.setenv("HF_TOKEN", stale)
    monkeypatch.setattr(auth, "_obtain_token", lambda *, browser, rejected=None: TOKEN)
    monkeypatch.setattr(
        hf, "whoami", lambda tok: None if tok == stale else hf.Identity("a", "user")
    )
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    # Exit code, not just a warning: a script reads the code, and the stale token is the
    # one diarization will actually use, so "signed in" with exit 0 is a lie.
    assert main(["login", "--no-browser"]) == EXIT_PERMISSION
    out = capsys.readouterr().out
    assert "unset HF_TOKEN" in out
    assert "NOT the one in force" in out
    assert hf.read_token() == stale  # the shell still wins — which is exactly the warning
    assert TOKEN not in out


def test_status_reports_a_network_failure_instead_of_raising(monkeypatch) -> None:
    """`--status` is what people run when something is broken; offline it must still answer."""
    from stt_cli._errors import NetworkError

    hf.store_token(TOKEN)

    def unreachable(_token):
        raise NetworkError(what="no network", why="getaddrinfo failed", how="check the network")

    monkeypatch.setattr(hf, "whoami", unreachable)
    result = auth.status("diarization", "hf")
    assert not result.ok
    assert "could not be checked" in result.render()
    assert "network problem" in result.render()


def test_logout_refuses_to_delete_something_that_is_not_a_token(tmp_path, monkeypatch) -> None:
    """HF_TOKEN_PATH points anywhere the caller likes; logout must not honour it blindly."""
    precious = tmp_path / "notes.md"
    precious.write_text("do not delete me", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(precious))
    with pytest.raises(PermissionDeniedError):
        hf.forget_token()
    assert precious.exists()


def test_the_stored_token_file_is_never_world_readable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    path = hf.store_token(TOKEN)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob("*.part"))  # nothing left behind


def test_login_refuses_to_overwrite_something_that_is_not_a_token(tmp_path, monkeypatch) -> None:
    """The symmetric half of the logout guard: HF_TOKEN_PATH must not clobber real files."""
    precious = tmp_path / "notes.md"
    precious.write_text("do not overwrite me", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(precious))
    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    assert precious.read_text(encoding="utf-8") == "do not overwrite me"


def test_replacing_an_existing_token_file_is_still_allowed(tmp_path, monkeypatch) -> None:
    older = "hf_" + "o" * 22
    target = tmp_path / "token"
    target.write_text(older + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(target))
    hf.store_token(TOKEN)
    assert hf.read_token() == TOKEN


def test_a_redirect_off_huggingface_does_not_carry_the_token() -> None:
    """`resolve/main/<file>` redirects to a CDN. urllib copies every header onto the
    redirected request, so without the guard the user's token is handed to that host."""
    import urllib.request

    original = urllib.request.Request(
        f"{hf.HOST}/pyannote/segmentation-3.0/resolve/main/config.yaml",
        headers={"Authorization": f"Bearer {TOKEN}"},
        method="HEAD",
    )
    handler = hf._SameHostAuth()

    off_site = handler.redirect_request(
        original, None, 302, "", {}, "https://cdn-lfs.example.net/x"
    )
    assert off_site is not None
    assert off_site.get_header("Authorization") is None

    on_site = handler.redirect_request(original, None, 302, "", {}, f"{hf.HOST}/elsewhere")
    assert on_site is not None
    assert on_site.get_header("Authorization") == f"Bearer {TOKEN}"


def test_status_does_not_advise_unsetting_the_only_credential(monkeypatch, capsys) -> None:
    """With HF_TOKEN exported and no stored file there is nothing to unset TO — the old note
    told the user to drop the only token they had in order to "use" a file that isn't there."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity("a", "user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    assert not hf.token_path().is_file()

    assert main(["login", "--status"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "unset HF_TOKEN" not in out
    assert "not from a stored file" in out

    # The same token in both places is not a misconfiguration, and calling it one sends
    # somebody off to "fix" a setup that works.
    hf.store_token(TOKEN)
    assert main(["login", "--status"]) == EXIT_OK
    assert "unset HF_TOKEN" not in capsys.readouterr().out

    # A DIFFERENT stored token is the case the note is for: the file is being overridden.
    monkeypatch.delenv("HF_TOKEN")
    hf.store_token("hf_" + "eF5" * 8)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert main(["login", "--status"]) == EXIT_OK
    assert "unset HF_TOKEN" in capsys.readouterr().out


LEGACY = "api_org_ABCDEFGHIJKLMNOPQRSTUVWX"


def test_a_token_the_hub_wrote_in_an_older_format_is_still_ours(tmp_path, monkeypatch) -> None:
    """The guard exists to protect a caller-supplied path, not to audit the hub's own file.
    Requiring the `hf_` shape there refused to log a user in or out of their own credential
    just because `huggingface-cli` had written an older format into it."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    target = hf.token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(LEGACY + "\n", "utf-8")

    hf.store_token(TOKEN)
    assert target.read_text("utf-8").strip() == TOKEN

    target.write_text(LEGACY + "\n", "utf-8")
    assert hf.forget_token() == target
    assert not target.exists()


def test_a_caller_supplied_path_must_hold_something_shaped_like_a_token(
    tmp_path, monkeypatch
) -> None:
    """HF_TOKEN_PATH points anywhere, so the loose test is wrong there: a checksum file holds
    one whitespace-free line too, and `stt login` would overwrite it and `stt logout` delete
    it. On that path only a Hugging Face token shape counts."""
    checksum = tmp_path / "checksum"
    checksum.write_text("d41d8cd98f00b204e9800998ecf8427e\n", "utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(checksum))

    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    with pytest.raises(PermissionDeniedError):
        hf.forget_token()
    assert checksum.read_text("utf-8").strip() == "d41d8cd98f00b204e9800998ecf8427e"

    # A real token there is still stt's to replace and to remove.
    checksum.write_text(TOKEN + "\n", "utf-8")
    hf.store_token("hf_" + "zY9" * 8)
    assert hf.forget_token() == checksum


def test_a_document_is_still_refused_in_both_directions(tmp_path, monkeypatch) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("# my notes\n\nremember to buy milk\n", "utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(notes))

    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    with pytest.raises(PermissionDeniedError):
        hf.forget_token()
    assert "buy milk" in notes.read_text("utf-8")


def test_a_provider_nobody_implemented_fails_loudly_rather_than_running_hugging_face() -> None:
    """CAPABILITIES is a registry with no dispatch behind it — every function here IS the
    Hugging Face flow. Without this guard, the day a second provider joins that dict it
    would silently run Hugging Face and report success attributed to the other one."""
    for call in (
        lambda: auth.status("diarization", "assemblyai"),
        lambda: auth.login("diarization", "assemblyai", browser=False),
        lambda: auth.logout("diarization", "assemblyai"),
    ):
        with pytest.raises(UnknownItemError):
            call()


def test_login_does_not_re_check_the_gates_it_just_walked_through(monkeypatch, capsys) -> None:
    """The gate walkthrough already fetched every gate. Asking `status()` afterwards re-ran
    whoami and every gate probe — four more network round-trips to learn what it knew."""
    calls = {"gates": 0, "whoami": 0}

    def counted_gates(_token, repos=hf.DIARIZATION_REPOS):
        calls["gates"] += 1
        return [hf.Gate(repo, True) for repo in repos]

    def counted_whoami(_token):
        calls["whoami"] += 1
        return hf.Identity("a", "user")

    monkeypatch.setattr(hf, "gates", counted_gates)
    monkeypatch.setattr(hf, "whoami", counted_whoami)
    monkeypatch.setattr(auth, "_obtain_token", lambda *, browser, rejected=None: TOKEN)

    assert main(["login", "--no-browser"]) == EXIT_OK
    assert calls["gates"] == 1
    assert calls["whoami"] == 1
    assert "signed in as a (user)" in capsys.readouterr().out


def test_a_token_already_on_the_clipboard_is_taken_at_once(monkeypatch) -> None:
    """Copy the token, then run `stt login` — the usual order. Waiting for the clipboard to
    CHANGE meant the token was sitting right there while the command waited five minutes."""
    monkeypatch.setattr(auth, "_clipboard", lambda: f"here it is: {TOKEN}")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert auth._watch(0.0) == TOKEN


def test_the_token_that_was_just_refused_is_not_handed_straight_back(monkeypatch) -> None:
    """It is on the clipboard precisely because the user copied it last time. Returning it
    would make `stt login --force` a no-op that reports success."""
    monkeypatch.setattr(auth, "_clipboard", lambda: TOKEN)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(PermissionDeniedError):
        auth._watch(0.0, rejected=TOKEN)


def test_a_downgrade_to_plain_http_does_not_carry_the_token() -> None:
    """Comparing hosts alone let an https -> http redirect on huggingface.co keep the
    bearer header, putting the token on the wire in cleartext."""
    import urllib.request

    request = urllib.request.Request(
        f"{hf.HOST}/pyannote/segmentation-3.0/resolve/main/config.yaml",
        headers={"Authorization": f"Bearer {TOKEN}"},
        method="HEAD",
    )
    downgraded = hf._SameHostAuth().redirect_request(
        request, None, 302, "", {}, "http://huggingface.co/same/path"
    )
    assert downgraded is not None
    assert downgraded.get_header("Authorization") is None


def test_a_service_failure_is_not_reported_as_unaccepted_terms(monkeypatch) -> None:
    """Mapping every non-200 to "not granted" turned a 500 into "you have not accepted the
    terms of these models", sending the user back to a page they had already agreed to."""
    from stt_cli._errors import NetworkError

    monkeypatch.setattr(hf, "_request", lambda url, token=None, method="GET": (500, b""))
    with pytest.raises(NetworkError):
        hf.gate("pyannote/segmentation-3.0", TOKEN)

    for code, granted in ((200, True), (401, False), (403, False)):
        monkeypatch.setattr(hf, "_request", lambda url, token=None, method="GET", c=code: (c, b""))
        assert hf.gate("pyannote/segmentation-3.0", TOKEN).granted is granted


def test_a_good_token_behind_the_rejected_one_is_still_found(monkeypatch) -> None:
    """`--force` leaves the old token on the clipboard; the new one is often pasted after
    it. Stopping at the first match found the rejected one and waited out the timeout."""
    fresh = "hf_" + "zY9" * 8
    monkeypatch.setattr(auth, "_clipboard", lambda: f"{TOKEN}\n{fresh}")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert auth._watch(0.0, rejected=TOKEN) == fresh


def test_a_piped_force_login_refuses_the_token_it_was_asked_to_replace(monkeypatch) -> None:
    """`echo "$OLD" | stt login --force` accepted, verified and re-stored the very token
    --force exists to replace, then reported success."""
    monkeypatch.setattr("sys.stdin", io.StringIO(TOKEN))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(PermissionDeniedError) as caught:
        auth._capture_token(rejected=TOKEN)
    assert "being replaced" in caught.value.what

    fresh = "hf_" + "zY9" * 8
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{TOKEN}\n{fresh}"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert auth._capture_token(rejected=TOKEN) == fresh


def test_an_unreachable_service_is_not_reported_as_a_bad_credential(monkeypatch) -> None:
    """`stt login --status || stt login` in a setup script must not launch an interactive
    login because the network blipped — the stored token was fine."""
    from stt_cli._errors import EXIT_NETWORK, NetworkError

    hf.store_token(TOKEN)

    def unreachable(_token):
        raise NetworkError(what="no route", why="the network is down", how="try later")

    monkeypatch.setattr(hf, "whoami", unreachable)
    assert main(["login", "--status"]) == EXIT_NETWORK


def test_a_token_path_that_is_not_a_small_file_is_refused_not_read(tmp_path, monkeypatch) -> None:
    """HF_TOKEN_PATH is caller-controlled and this is the first thing that touches it.
    `HF_TOKEN_PATH=/dev/zero stt login --status` read until memory ran out, and a FIFO
    blocked forever waiting for a writer — a hang with no message."""
    import os

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    monkeypatch.setenv("HF_TOKEN_PATH", str(fifo))
    assert hf.read_token() is None  # must not block

    huge = tmp_path / "big"
    huge.write_text("x" * 200_000, "utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(huge))
    assert hf.read_token() is None
    # ...and it is protected, not clobbered: an oversized file is nobody's token.
    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    assert len(huge.read_text("utf-8")) == 200_000


def test_a_logout_in_flight_cannot_delete_a_login_that_just_finished(monkeypatch) -> None:
    """`logout` checks the file really holds a token and only then unlinks it. Between those
    two steps a `login` in another terminal replaced the file — and the unlink then removed
    the token that had just been stored, signing the user out of the session they had just
    signed into. Reading and deleting now happen under the same lock as writing.

    The two orders are told apart by events rather than by sleeping: the logout waits, from
    inside the check, until the login says it has finished writing. Locked, that wait times
    out because the login is still blocked on the lock, and the fresh token is written after
    the delete. Unlocked, the login really does finish first and the delete eats it.
    """
    import threading

    fresh = "hf_" + "cD4" * 8
    hf.store_token(TOKEN)
    checking, stored = threading.Event(), threading.Event()
    real_content_of = hf._content_of

    def dawdling(path):
        # Stand where the race used to open: the check has passed, the unlink has not
        # happened yet. Held under the lock, no login can land here.
        content = real_content_of(path)
        checking.set()
        stored.wait(1.0)
        return content

    def login() -> None:
        hf.store_token(fresh)
        stored.set()

    monkeypatch.setattr(hf, "_content_of", dawdling)
    logout = threading.Thread(target=hf.forget_token)
    logout.start()
    assert checking.wait(2.0), "the logout never reached the check"
    writer = threading.Thread(target=login)
    writer.start()
    logout.join(5.0)
    writer.join(5.0)
    assert not logout.is_alive() and not writer.is_alive()

    assert hf.read_token() == fresh, "the fresh login was deleted by the logout it outran"


def test_login_saves_a_working_exported_token_that_has_nowhere_to_live(monkeypatch) -> None:
    """`read_token` prefers the environment, so a valid `$HF_TOKEN` made login verify it,
    report success and store nothing at all — while the README says the command exists so
    that the token "never has to live in your shell profile". Somebody who took the variable
    out of their profile on the strength of that lost diarization."""
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity(name="alex", kind="user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    monkeypatch.setattr(
        auth, "_obtain_token", lambda **_: pytest.fail("a working token must not be replaced")
    )

    assert not hf.token_path().is_file()
    assert main(["login", "diarization", "--no-browser"]) == EXIT_OK
    assert hf.token_path().read_text("utf-8").strip() == TOKEN


def test_login_does_not_overwrite_a_stored_token_with_an_exported_one(monkeypatch) -> None:
    """The other half of the same rule. A stored token is a deliberate credential; replacing
    it with whatever a shell happens to export is what `_report_shadowing` warns about."""
    stored = "hf_" + "eF5" * 8
    hf.store_token(stored)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity(name="alex", kind="user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])

    main(["login", "diarization", "--no-browser"])
    assert hf.token_path().read_text("utf-8").strip() == stored


def test_an_unusable_token_path_is_diagnosed_rather_than_traced(monkeypatch, tmp_path) -> None:
    """`HF_TOKEN_PATH` is caller-controlled and need not be usable: pointed inside a regular
    file, the directory creation raises and a raw traceback came out. It surfaced only after
    the token had been verified, so the whole browser dance was already done."""
    blocker = tmp_path / "notes.txt"
    blocker.write_text("not a directory", "utf-8")
    monkeypatch.setenv("HF_TOKEN_PATH", str(blocker / "token"))

    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)

    monkeypatch.setattr(auth, "_obtain_token", lambda *, browser, rejected=None: TOKEN)
    monkeypatch.setattr(hf, "whoami", lambda _t: hf.Identity(name="alex", kind="user"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    assert main(["login", "--no-browser"]) == EXIT_PERMISSION


def test_logout_says_that_it_signs_every_tool_out(monkeypatch, capsys) -> None:
    """The token is stored where huggingface_hub reads it, which is what makes one login
    serve every tool on the machine — and the same fact makes one logout take them all."""
    hf.store_token(TOKEN)
    assert main(["logout"]) == EXIT_OK
    assert "every Hugging Face tool" in capsys.readouterr().out


def test_a_binary_token_file_is_refused_rather_than_crashing(monkeypatch, tmp_path) -> None:
    """A small binary file passes the regular-file and size guards, and a strict decode then
    raised `UnicodeDecodeError` — not an `OSError`, so it escaped as a raw traceback out of
    `login --status` and out of every caller that reads the token."""
    binary = tmp_path / "photo.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01")
    monkeypatch.setenv("HF_TOKEN_PATH", str(binary))

    assert hf.read_token() is None, "a token is `hf_` and base62; this file is not one"
    assert main(["login", "--status"]) == EXIT_PERMISSION
    with pytest.raises(PermissionDeniedError):
        hf.forget_token()
    assert binary.exists(), "a file that is not a token is never deleted"


def test_a_malformed_exported_token_is_reported_not_crashed(monkeypatch, capsys) -> None:
    """A value with a newline in it reaches the HTTP header and raises `ValueError`; a
    non-Latin-1 one raises `UnicodeEncodeError`. Neither is an `SttError`, so both came out
    as a raw traceback from `stt login --status` — the command people run when something is
    already wrong. And dropping the value silently would say "not signed in" while the
    variable sat there looking set."""
    for broken in ("hf_good\nbad", "hf_ключ", "hf_ token"):
        monkeypatch.setenv("HF_TOKEN", broken)
        assert hf.read_token() is None
        assert hf.unusable_variable() == "HF_TOKEN"
        assert main(["login", "--status"]) == EXIT_PERMISSION
        out = capsys.readouterr().out
        assert "$HF_TOKEN is set but is not a Hugging Face token" in out
        assert broken not in out


def test_an_empty_file_the_caller_pointed_at_is_not_overwritten(monkeypatch, tmp_path) -> None:
    """`_refuse_to_clobber` let any empty file through, so `touch notes.md` followed by
    `HF_TOKEN_PATH=~/notes.md stt login` replaced it with a credential. The hub's own path
    is deliberately different: an empty token file there is stt's to write."""
    theirs = tmp_path / "notes.md"
    theirs.touch()
    monkeypatch.setenv("HF_TOKEN_PATH", str(theirs))
    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    assert theirs.read_text("utf-8") == ""

    monkeypatch.delenv("HF_TOKEN_PATH")
    hf.token_path().parent.mkdir(parents=True, exist_ok=True)
    hf.token_path().touch()
    assert hf.store_token(TOKEN) == hf.token_path()


def test_a_token_file_that_can_be_read_can_also_be_replaced() -> None:
    """The two thresholds describing "is this a token" disagreed: anything ASCII up to 64 KiB
    was readable and usable, while overwriting or deleting refused past 200 characters. A
    file in between was a login nobody could log out of or replace."""
    from stt_cli.hf import _MAX_TOKEN_CHARS

    long_one = "hf_" + "a" * _MAX_TOKEN_CHARS
    hf.token_path().parent.mkdir(parents=True, exist_ok=True)
    hf.token_path().write_text(long_one + "\n", "utf-8")

    assert hf.read_token() is None, "too long to be a token, so it is not read as one"
    assert hf.store_token(TOKEN) == hf.token_path(), "...and it is not a file stt must not touch"
    assert hf.read_token() == TOKEN


def test_the_token_path_follows_the_hub_through_all_three_variables(monkeypatch, tmp_path) -> None:
    """`XDG_CACHE_HOME` was missing, so on a machine that sets it stt wrote and reported
    `~/.cache/huggingface/token` while every other Hugging Face tool read somewhere else —
    and sharing the file with them is the entire reason it goes there."""
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hf.token_path() == tmp_path / "xdg" / "huggingface" / "token"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hub"))
    assert hf.token_path() == tmp_path / "hub" / "token", "HF_HOME wins over XDG_CACHE_HOME"
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "elsewhere"))
    assert hf.token_path() == tmp_path / "elsewhere", "...and HF_TOKEN_PATH wins over both"


def test_a_file_too_big_to_read_is_not_ours_to_delete() -> None:
    """The symmetry has to hold both ways. A hundred-kilobyte single-line blob is refused by
    the reader, so `login --status` says "not signed in" — and `logout` used to delete it
    anyway, because the content check cannot see how long a file it only read 201 bytes of."""
    hf.token_path().parent.mkdir(parents=True, exist_ok=True)
    hf.token_path().write_text("x" * (100 * 1024), "utf-8")

    assert hf.read_token() is None
    with pytest.raises(PermissionDeniedError):
        hf.forget_token()
    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)
    assert hf.token_path().stat().st_size == 100 * 1024


def test_a_pipe_that_never_ends_does_not_hang_the_login(monkeypatch) -> None:
    """`cat /dev/zero | stt login` read until memory ran out and `yes x | stt login` never
    stopped reading — neither ever reached the code that looks for a token."""
    import io

    from stt_cli._errors import PermissionDeniedError as Denied

    class Endless(io.TextIOBase):
        def isatty(self) -> bool:
            return False

        def read(self, size: int | None = -1) -> str:
            assert size is not None and size > 0, "the read must be bounded"
            return "x" * size

    monkeypatch.setattr("sys.stdin", Endless())
    with pytest.raises(Denied):
        auth._obtain_token(browser=False)


def test_a_token_path_swapped_for_a_pipe_does_not_hang(monkeypatch, tmp_path) -> None:
    """Checking the path and opening it are two moments. A regular file replaced by a FIFO
    in that gap made the open wait for a writer that never came — `stt login --status`
    hanging with no message, the exact failure the check was there to prevent."""
    import os

    fifo = tmp_path / "token"
    os.mkfifo(fifo)
    monkeypatch.setenv("HF_TOKEN_PATH", str(fifo))

    assert hf.read_token() is None
    assert hf.stored_token() is None
    assert hf.forget_token() is None, "a pipe is not a token file to delete"
    with pytest.raises(PermissionDeniedError):
        hf.store_token(TOKEN)


def test_a_revoked_variable_does_not_destroy_the_saved_token(monkeypatch, capsys) -> None:
    """`read_token` prefers `$HF_TOKEN`, so a revoked variable made login conclude that "the
    stored token" was dead, ask for a new one, and write it over the file — which was holding
    a good token all along. The variable still won, so the command failed anyway, having
    destroyed the credential that `unset HF_TOKEN` would have restored."""
    revoked = "hf_" + "r" * 22
    hf.store_token(TOKEN)
    monkeypatch.setenv("HF_TOKEN", revoked)
    monkeypatch.setattr(hf, "whoami", lambda tok: None if tok == revoked else hf.Identity("a", "u"))
    monkeypatch.setattr(hf, "gates", lambda _t: [hf.Gate("a", True)])
    monkeypatch.setattr(
        auth, "_obtain_token", lambda **_: pytest.fail("the saved token was never tried")
    )

    assert main(["login", "--no-browser"]) == EXIT_PERMISSION, "the variable still wins"
    assert hf.stored_token() == TOKEN, "...and the token it shadows is still there"
    assert "unset HF_TOKEN" in capsys.readouterr().out


def test_a_live_pipe_with_a_bad_payload_gives_up_instead_of_waiting(monkeypatch) -> None:
    """Bounding the SIZE of the read does not bound the WAIT: `read(n)` returns when it has
    n bytes or when the writer closes, so `{ printf 'not a token'; sleep 3600; } | stt login`
    sat there for an hour on a payload that had already arrived and was already wrong."""
    import os
    import time

    read_end, write_end = os.pipe()
    os.write(write_end, b"not a token\n")
    monkeypatch.setattr("sys.stdin", os.fdopen(read_end, "r"))
    try:
        started = time.monotonic()
        with pytest.raises(PermissionDeniedError):
            auth._capture_token(timeout=0.5)
        assert time.monotonic() - started < 5.0, "it waited on a writer that had gone quiet"
    finally:
        os.close(write_end)


def test_a_live_pipe_that_does_carry_a_token_is_answered_at_once(monkeypatch) -> None:
    """...and the same path must not wait for the writer to close when the token is there."""
    import os

    read_end, write_end = os.pipe()
    os.write(write_end, f"here you go: {TOKEN}\n".encode())
    monkeypatch.setattr("sys.stdin", os.fdopen(read_end, "r"))
    try:
        assert auth._capture_token(timeout=30.0) == TOKEN
    finally:
        os.close(write_end)


def test_a_proxy_answering_200_to_everything_is_not_a_login(monkeypatch) -> None:
    """A captive portal or a corporate proxy answers 200 with an HTML page to every request.
    Read as "signed in as ?", that reported a successful login with every gate granted for a
    machine that could not reach Hugging Face at all."""
    from stt_cli._errors import EXIT_NETWORK, NetworkError

    monkeypatch.setattr(hf, "_request", lambda *a, **k: (200, b"<html>sign in to the wifi</html>"))
    hf.store_token(TOKEN)

    with pytest.raises(NetworkError):
        hf.whoami(TOKEN)
    assert main(["login", "--status"]) == EXIT_NETWORK, "a service failure, not a login one"
