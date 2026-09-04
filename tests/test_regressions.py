"""Regressions for defects a multi-model review of the first version found.

Each of these was a real bug that would have shipped, and each is silent — the tool returns
an answer, just the wrong one. That is exactly the class of failure that needs a test rather
than a careful reader, so they get one apiece with the failing scenario in the name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from stt_cli import formats
from stt_cli import proc as proc_mod
from stt_cli.archive import (
    ENRICHMENTS,
    FINGERPRINT_KEYS,
    Archive,
    fingerprint,
    run_id,
    write_atomic,
)
from stt_cli.commands.transcribe import build_parser
from stt_cli.config import Settings
from stt_cli.models import EngineInfo, MediaInfo, Segment, Transcript
from stt_cli.pipeline import _missing_enrichments, render_options
from stt_cli.timestamps import Stamper


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "home"))


def _transcript(**kwargs) -> Transcript:
    media = MediaInfo(
        path=kwargs.pop("path", "/recordings/standup.m4a"),
        sha256=kwargs.pop("sha", "a" * 64),
        size_bytes=1024,
        duration=120.0,
        recorded_at=datetime(2026, 3, 31, 13, 32, 57).astimezone(),
        recorded_at_source="filename",
    )
    return Transcript(
        media=media,
        engine=EngineInfo(backend="whispercpp", model="large-v3-turbo", **kwargs),
        segments=[Segment(0, 3, "привет", confidence=0.9, speaker="S1")],
    )


# ── every fingerprint key must be a real setting ──────────────────────────────
def test_no_fingerprint_key_is_a_typo() -> None:
    """A misspelled key hashes None forever, silently merging two distinct cache entries."""
    fields = set(Settings.__dataclass_fields__)
    assert [k for k in FINGERPRINT_KEYS if k not in fields] == []
    assert [k for k in ENRICHMENTS if k not in fields] == []


# ── diarization configuration is part of the enrichment identity ──────────────
def test_a_different_speaker_count_re_runs_diarization() -> None:
    """`--diarize --speakers 2` then `--speakers 3` must not hand back the 2-speaker answer."""
    transcript = _transcript(extra={"diarize": "speakers=2"})
    assert _missing_enrichments(transcript, Settings(diarize=True, speakers=2)) == []
    assert _missing_enrichments(transcript, Settings(diarize=True, speakers=3)) == ["speakers"]


def test_labels_from_an_unknown_configuration_are_re_run() -> None:
    transcript = _transcript()
    assert _missing_enrichments(transcript, Settings(diarize=True)) == ["speakers"]


# ── enrichments must not leak into runs that did not ask for them ─────────────
def test_subtitles_have_no_speaker_prefix_unless_diarization_was_requested() -> None:
    transcript = _transcript()
    stamper = Stamper("none", transcript.media)
    plain = formats.render("srt", transcript, stamper, render_options(Settings()))
    with_speakers = formats.render(
        "srt", transcript, stamper, render_options(Settings(diarize=True))
    )
    assert "S1:" not in plain
    assert "S1:" in with_speakers


def test_the_speakers_format_shows_speakers_whatever_the_flags_said() -> None:
    """Asking for the dialogue format IS asking to see who was talking."""
    transcript = _transcript()
    text = formats.render(
        "speakers", transcript, Stamper("none", transcript.media), render_options(Settings())
    )
    assert "S1:" in text


def test_markdown_omits_a_stored_summary_when_none_was_requested() -> None:
    from stt_cli.models import Summary

    transcript = _transcript()
    transcript.summary = Summary(headline="from an earlier run")
    stamper = Stamper("none", transcript.media)
    plain = formats.render("md", transcript, stamper, render_options(Settings()))
    asked = formats.render("md", transcript, stamper, render_options(Settings(summary=True)))
    assert "from an earlier run" not in plain
    assert "from an earlier run" in asked


# ── stored preferences must remain overridable ────────────────────────────────
@pytest.mark.parametrize("flag", ["fix", "summary", "diarize", "clean", "cache", "keep-media"])
def test_every_boolean_flag_has_a_negative_form(flag: str) -> None:
    """With `fix = true` stored in config there must be a way to skip it for one run."""
    parser = build_parser()
    dest = flag.replace("-", "_")
    assert getattr(parser.parse_args(["x.m4a"]), dest) is None  # unmentioned: config wins
    assert getattr(parser.parse_args(["x.m4a", f"--{flag}"]), dest) is True
    assert getattr(parser.parse_args(["x.m4a", f"--no-{flag}"]), dest) is False


def test_render_options_come_from_settings_not_argv() -> None:
    """`stt config set show_variants true` has to actually show variants."""
    assert render_options(Settings(show_variants=True)).show_variants
    assert render_options(Settings(text_variant="raw")).text_variant == "raw"


# ── run ids and archive integrity ─────────────────────────────────────────────
def test_run_id_keeps_the_whole_fingerprint() -> None:
    """A truncated fingerprint lets two runs the index calls distinct share one directory."""
    key = fingerprint(Settings())
    assert run_id("a" * 64, key).endswith(key)


def test_an_ambiguous_run_id_prefix_is_refused_rather_than_guessed() -> None:
    """`stt archive rm <short prefix>` must never delete an arbitrary one of two matches."""
    from stt_cli._errors import MissingTargetError

    with Archive() as store:
        store.save(_transcript(), fingerprint(Settings()))
        store.save(_transcript(), fingerprint(Settings(model="large-v3")))
        with pytest.raises(MissingTargetError, match="matches 2 archived runs") as caught:
            store.get("aaaaaaaaaa")
        assert "ambiguous" in caught.value.why


def test_an_exact_run_id_still_resolves_when_others_share_its_prefix() -> None:
    with Archive() as store:
        first = store.save(_transcript(), fingerprint(Settings()))
        store.save(_transcript(), fingerprint(Settings(model="large-v3")))
        assert store.get(first.run_id).run_id == first.run_id


def test_atomic_write_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "transcript.json"
    write_atomic(target, '{"a": 1}')
    assert target.read_text() == '{"a": 1}'
    assert not (tmp_path / "transcript.json.part").exists()


def test_atomic_write_replaces_the_previous_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    write_atomic(target, "old")
    write_atomic(target, "new")
    assert target.read_text() == "new"


# ── archived audio is reachable after the source moves ────────────────────────
def test_a_moved_source_is_found_through_the_archive(tmp_path: Path) -> None:
    """The archive's whole promise: transcribe, delete the original, still re-render."""
    original = tmp_path / "gone.m4a"
    with Archive() as store:
        store.save(_transcript(path=str(original)), fingerprint(Settings()))
        stored = store.media_path("a" * 64)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(b"not really opus, but non-empty")
        assert store.media_for_source(original) == (stored, "a" * 64)


def test_an_unknown_source_has_no_archived_copy(tmp_path: Path) -> None:
    with Archive() as store:
        assert store.media_for_source(tmp_path / "never-seen.m4a") is None


def test_the_archived_copy_carries_the_original_recordings_hash(tmp_path: Path) -> None:
    """Not the hash of the Opus re-encode: the archive is keyed by the ORIGINAL recording.

    Recomputing it from the stand-in would miss the cache, re-transcribe the whole file, and
    file the result under an identity nothing else refers to.
    """
    original = tmp_path / "gone.m4a"
    with Archive() as store:
        store.save(_transcript(path=str(original)), fingerprint(Settings()))
        stored = store.media_path("a" * 64)
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(b"a different byte stream entirely")
        found = store.media_for_source(original)
        assert found is not None
        assert found[1] == "a" * 64


def test_an_indexed_source_whose_audio_was_deleted_reports_none(tmp_path: Path) -> None:
    original = tmp_path / "gone.m4a"
    with Archive() as store:
        store.save(_transcript(path=str(original)), fingerprint(Settings()))
        assert store.media_for_source(original) is None


# ── stored settings are typed, and stay typed ─────────────────────────────────
def test_config_coerces_from_the_declared_type_not_the_current_value() -> None:
    """`language` defaults to None, so there is no current value to infer a type from."""
    from stt_cli.config import coerce

    assert coerce("language", "1") == "1"  # a str field, even though "1" looks numeric
    assert coerce("timezone", "2026") == "2026"
    assert coerce("variants", "2") == 2
    assert coerce("confidence_floor", "0.8") == 0.8
    assert coerce("clean", "no") is False
    assert coerce("variant_models", "large-v3, medium") == ["large-v3", "medium"]


@pytest.mark.parametrize("key,raw", [("variants", "abc"), ("clean", "maybe")])
def test_a_value_of_the_wrong_type_is_refused_at_write_time(key: str, raw: str) -> None:
    from stt_cli._errors import UsageError
    from stt_cli.config import coerce

    with pytest.raises(UsageError):
        coerce(key, raw)


def test_a_hand_edited_config_with_a_bad_type_is_diagnosed_on_load() -> None:
    """`{"variants": "abc"}` would otherwise flow into a comparison and the cache key."""
    import json

    from stt_cli import config
    from stt_cli._errors import UsageError

    config.ensure_dirs()
    config.config_path().write_text(json.dumps({"variants": "abc"}), "utf-8")
    with pytest.raises(UsageError, match="wrong type"):
        config.load_settings()


def test_an_extensionless_file_is_not_mistaken_for_a_command(tmp_path: Path, monkeypatch) -> None:
    """`stt Interview` must transcribe the file, not offer a did-you-mean for a command."""
    from stt_cli.cli import _looks_like_a_command

    recording = tmp_path / "Interview"
    recording.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert not _looks_like_a_command("Interview")
    assert _looks_like_a_command("Interviewww")


def test_write_atomic_leaves_no_neighbour_behind_when_the_rename_fails(tmp_path, monkeypatch):
    """The temporary neighbour is what makes the write atomic. If it survives a failure it
    is a stale fragment of somebody's transcript sitting next to the real one forever."""
    import pathlib

    target = tmp_path / "transcript.json"
    original = pathlib.Path.replace

    def refuse(self, other):
        raise OSError("no space left on device")

    monkeypatch.setattr(pathlib.Path, "replace", refuse)
    with pytest.raises(OSError):
        write_atomic(target, '{"a": 1}')
    monkeypatch.setattr(pathlib.Path, "replace", original)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "a .part file was left behind"


def test_write_atomic_neighbours_do_not_collide_between_processes(tmp_path, monkeypatch):
    """A shared temp name lets two processes writing one target interleave into the same
    file, and the "atomic" rename then publishes a blend of both. The name is gone by the
    time the test could look at it, so the rename is intercepted to record what it saw."""
    import os
    import pathlib

    used: list[str] = []
    original = pathlib.Path.replace

    def record(self, other):
        used.append(self.name)
        return original(self, other)

    monkeypatch.setattr(pathlib.Path, "replace", record)
    target = tmp_path / "transcript.json"
    write_atomic(target, "mine")

    assert target.read_text("utf-8") == "mine"
    assert used and str(os.getpid()) in used[0], f"a shared temp name is back: {used}"


def test_two_writers_in_one_process_do_not_take_each_others_temporary(tmp_path) -> None:
    """The pid alone did not separate two threads: they shared one `<target>.<pid>.part`,
    so the second `replace` found its own temporary already renamed away and raised — or
    published the other writer's half-written content."""
    import threading

    from stt_cli.archive import write_atomic

    target = tmp_path / "transcript.json"
    errors: list[BaseException] = []

    def write(payload: str) -> None:
        try:
            for _ in range(40):
                write_atomic(target, payload)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(text,)) for text in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)

    assert not errors, f"a concurrent writer failed: {errors}"
    assert target.read_text("utf-8") in {"first", "second"}, "and never a blend of the two"
    assert list(tmp_path.glob("*.part")) == [], "no temporary is left behind"


def _answers(value):
    """A stand-in for an async predicate, since `monkeypatch` cannot make a lambda awaitable."""

    async def said(*args, **kwargs):
        return value

    return said


async def test_the_hugging_face_token_never_reaches_a_command_line(monkeypatch) -> None:
    """A command line is public. Any user on the machine can read another process's argv out
    of `ps`, and diarizing an hour of audio keeps the process alive long enough to be caught
    — so the gated-model credential travels in the environment instead.

    This is a regression test in the strict sense: the token used to stay in-process, the
    split into a worker subprocess moved it onto argv, and review caught it there.
    """
    from stt_cli import diarize, proc

    seen: dict[str, object] = {}

    async def remember(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = kwargs.get("env")
        return proc.Result(code=0, stdout='{"turns": []}', stderr="", argv=list(argv))

    monkeypatch.setattr(diarize, "runner", lambda **_: ["/usr/bin/python3"])
    monkeypatch.setattr(diarize, "ready", _answers(True))
    monkeypatch.setattr(diarize, "require_token", lambda: "hf_secret_value")
    monkeypatch.setattr(proc, "run", remember)

    await diarize.diarize(Path("/tmp/whatever.wav"), speakers=2)

    assert "hf_secret_value" not in " ".join(seen["argv"]), "not on the command line"
    assert seen["env"] == {"HUGGING_FACE_HUB_TOKEN": "hf_secret_value"}, "in the environment"
    assert "--speakers" in seen["argv"], "and the harmless arguments still travel as arguments"


async def test_asking_whether_diarization_is_ready_never_downloads_anything(monkeypatch) -> None:
    """`stt doctor` and `stt diarize status` are what people run when they want no surprises.
    Checking readiness with a bare `uv run --with` would have resolved and downloaded two and
    a half gigabytes to answer the question, so the probe is forbidden the network."""
    from stt_cli import diarize, proc

    asked: list[list[str]] = []

    async def remember(argv, **kwargs):
        asked.append(list(argv))
        return proc.Result(code=0, stdout='{"ready": true}', stderr="", argv=list(argv))

    monkeypatch.setattr(
        diarize, "runner", lambda: ["/opt/uv", "run", "--quiet", "--with", "x", "python"]
    )
    monkeypatch.setattr(proc, "run", remember)

    assert await diarize.ready() is True
    assert "--offline" in asked[0], "the probe may look, not fetch"


async def test_a_status_check_that_fails_reports_rather_than_raises(monkeypatch) -> None:
    """The docstring promises it never raises, and only the diagnosed errors were caught: a
    timeout from the probe would have come out of `stt doctor` as the traceback that command
    exists to replace."""
    from stt_cli import diarize, proc

    async def time_out(argv, **kwargs):
        raise TimeoutError("the probe never came back")

    monkeypatch.setattr(diarize, "runner", lambda: ["/opt/uv", "run", "--quiet", "python"])
    monkeypatch.setattr(proc, "run", time_out)

    assert await diarize.ready() is False


async def test_the_worker_inherits_the_environment_it_needs(monkeypatch, tmp_path) -> None:
    """Passing the token in the environment only works if the environment is MERGED.

    A reviewer could not rule out that `proc.run` replaces it instead — and replacing would
    leave the worker with no HOME, so uv could not find its cache and huggingface could not
    find `~/.cache/huggingface`. The token would be delivered perfectly to a process unable
    to use it, status and doctor would stay green, and only real diarization would break.

    So this runs an actual subprocess rather than asserting on what was handed to a mock.
    """
    from stt_cli import proc

    script = tmp_path / "say.py"
    script.write_text(
        "import json, os, sys\n"
        "json.dump({'token': os.environ.get('HUGGING_FACE_HUB_TOKEN'),\n"
        "           'home': bool(os.environ.get('HOME'))}, sys.stdout)\n",
        encoding="utf-8",
    )
    result = await proc.run(
        [sys.executable, str(script)], env={"HUGGING_FACE_HUB_TOKEN": "hf_secret_value"}
    )
    said = json.loads(result.stdout)

    assert said["token"] == "hf_secret_value", "the secret arrived"
    assert said["home"] is True, "and it did not arrive INSTEAD of the environment"


def test_a_result_survives_anything_printed_after_it() -> None:
    """torch and pyannote are other people's programs and may say something on their way out.
    Taking the last line would throw away an hour of finished work over a shutdown warning."""
    from stt_cli.diarize import _the_object_among

    assert _the_object_among('{"turns": []}') == {"turns": []}
    assert _the_object_among('resolving\n{"turns": [1]}\n') == {"turns": [1]}
    assert _the_object_among('{"turns": [2]}\nWarning: leaked semaphore\n') == {"turns": [2]}
    assert _the_object_among("nothing here\nnor here\n") is None
    assert _the_object_among("") is None
    assert _the_object_among("[1, 2, 3]\n") is None, "an array is not the worker's answer"


async def test_diarization_that_runs_too_long_says_so_rather_than_crashing(
    monkeypatch, tmp_path
) -> None:
    """The timeout was introduced by the move to a worker, and was the one failure in that
    function not turned into a sentence — so the longest-running command in the tool ended an
    hour of somebody's time with a stack trace."""
    from stt_cli import diarize, proc
    from stt_cli._errors import EngineError

    async def never_finishes(argv, **kwargs):
        raise TimeoutError("still going")

    audio = tmp_path / "long.wav"
    audio.write_bytes(b"\x00" * 32_000)
    monkeypatch.setattr(diarize, "runner", lambda **_: ["/usr/bin/python3"])
    monkeypatch.setattr(diarize, "ready", _answers(True))
    monkeypatch.setattr(diarize, "require_token", lambda: "hf_x")
    monkeypatch.setattr(proc, "run", never_finishes)

    with pytest.raises(EngineError) as refused:
        await diarize.diarize(audio)
    assert "did not finish" in refused.value.what
    assert "minutes" in refused.value.why, "and it says how long it waited"


def test_the_diarization_budget_follows_the_length_of_the_recording(tmp_path) -> None:
    """One fixed hour was wrong at both ends: long enough to sit on a wedged two-minute memo,
    and short enough to cut off a three-hour recording that used to finish, because the
    in-process call it replaced had no limit at all."""
    from stt_cli.diarize import MINIMUM_TIMEOUT, _long_enough_for

    # Under ninety seconds, twenty times the audio is less than the floor, and the floor is
    # what covers the model download on a first run.
    memo = tmp_path / "memo.wav"
    memo.write_bytes(b"\x00" * (32_000 * 30))  # half a minute
    assert _long_enough_for(memo) == MINIMUM_TIMEOUT, "short audio gets the floor"

    quarter_hour = tmp_path / "meeting.wav"
    quarter_hour.write_bytes(b"\x00" * (32_000 * 900))
    assert _long_enough_for(quarter_hour) == 900 * 20, "past that, it follows the audio"

    afternoon = tmp_path / "afternoon.wav"
    afternoon.write_bytes(b"\x00" * (32_000 * 3 * 3600))  # three hours
    assert _long_enough_for(afternoon) > 3 * 3600, "and a long one gets more than its length"

    assert _long_enough_for(tmp_path / "gone.wav") == MINIMUM_TIMEOUT, (
        "a missing file is not a crash"
    )


async def test_diarizing_never_downloads_two_gigabytes_on_its_own(monkeypatch, tmp_path) -> None:
    """The status probe was made offline; the WORK path was left online, which is where it
    would have hurt. `stt recording.wav --diarize` on a cold cache would have resolved and
    downloaded two and a half gigabytes inside somebody's transcription — with both streams
    captured, so not even a progress bar reached them. The download belongs to
    `stt diarize install`, where it was asked for and can be watched."""
    from stt_cli import diarize
    from stt_cli._errors import MissingDependencyError

    ran: list[list[str]] = []

    async def refuse(argv, **kwargs):
        ran.append(list(argv))
        raise AssertionError("nothing may be launched when the wheels are not ready")

    monkeypatch.setattr(diarize, "runner", lambda **_: ["/opt/uv", "run", "--quiet", "python"])
    monkeypatch.setattr(diarize, "ready", _answers(False))
    monkeypatch.setattr(diarize, "require_token", lambda: "hf_x")
    monkeypatch.setattr(proc_mod, "run", refuse)

    with pytest.raises(MissingDependencyError) as refused:
        await diarize.diarize(tmp_path / "recording.wav")
    assert "not ready" in refused.value.what
    assert ran == [], "and it did not reach for the network to find that out"


def test_uv_is_never_allowed_to_adopt_the_directory_the_user_stood_in(monkeypatch) -> None:
    """A bare `uv run` discovers a pyproject.toml in the current directory and synchronises
    that project first. Run `stt something.wav --diarize` from inside an unrelated Python
    project and uv creates ITS .venv and fails resolving ITS dependencies, which have nothing
    to do with diarization."""
    from stt_cli import diarize, proc

    monkeypatch.setattr(proc, "which", lambda name: "/opt/uv" if name == "uv" else None)
    argv = diarize.runner(force_uv=True)

    assert argv is not None
    assert "--no-project" in argv, "stt is not a member of anybody's workspace"
    assert argv.index("--no-project") < argv.index("--with"), "before the wheels it asks for"


def test_a_broken_local_pyannote_does_not_capture_the_install(monkeypatch) -> None:
    """`find_spec` succeeds for a package that cannot actually import — a Python upgrade
    leaving an incompatible torch wheel is the ordinary way. `stt diarize install` would then
    have re-run the same failing import forever instead of preparing the environment that
    works."""
    import importlib.util

    from stt_cli import diarize, proc

    monkeypatch.setattr(proc, "which", lambda name: "/opt/uv" if name == "uv" else None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    command = diarize.install_command()

    assert command is not None
    assert command[0] == "/opt/uv", "the install prepares uv's environment, not the broken one"


async def test_a_local_pyannote_is_still_enough_on_its_own(monkeypatch, tmp_path) -> None:
    """Gating the work path on `ready()` must not shut out the setup that never needed uv.

    Somebody who installed pyannote into stt's own interpreter had diarization working before
    the gate existed. It keeps working only because `ready()` asks the SAME `runner()` — and
    because `_offline` leaves a direct interpreter alone rather than handing it a uv flag it
    would not understand. Every other test here replaces `ready()` with a constant, so this
    is the one that exercises the branch.
    """
    from stt_cli import diarize, proc

    launched: list[list[str]] = []

    async def answer(argv, **kwargs):
        launched.append(list(argv))
        payload = '{"ready": true}' if "--probe" in argv else '{"turns": []}'
        return proc.Result(code=0, stdout=payload, stderr="", argv=list(argv))

    # pyannote is importable here, and there is no uv on this machine at all.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(proc, "which", lambda name: None)
    monkeypatch.setattr(diarize, "require_token", lambda: "hf_x")
    monkeypatch.setattr(proc, "run", answer)

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"\x00" * 32_000)
    assert await diarize.diarize(audio) == []

    assert len(launched) == 2, "it probed, then it worked"
    assert launched[0][0] == sys.executable, "asking the interpreter that has pyannote"
    assert "--offline" not in launched[0], "a uv flag has no meaning for a direct interpreter"
