"""Regressions for defects a multi-model review of the first version found.

Each of these was a real bug that would have shipped, and each is silent — the tool returns
an answer, just the wrong one. That is exactly the class of failure that needs a test rather
than a careful reader, so they get one apiece with the failing scenario in the name.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from stt_cli import formats
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
