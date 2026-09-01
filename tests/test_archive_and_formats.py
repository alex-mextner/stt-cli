"""The archive's cache key, and that every renderer produces something sane.

The cache-key tests are the load-bearing ones: if a setting that changes the words is
missing from the fingerprint, a user gets a stale transcript and no indication why. If a
setting that only changes presentation is IN it, they pay for a full re-transcription to get
a different file extension. Both failures are silent, so both are pinned here.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from stt_cli import formats
from stt_cli.archive import Archive, fingerprint
from stt_cli.config import Settings
from stt_cli.models import (
    EngineInfo,
    MediaInfo,
    Segment,
    Summary,
    SummarySection,
    Transcript,
    Variant,
)
from stt_cli.timestamps import Stamper


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "home"))


def _transcript() -> Transcript:
    media = MediaInfo(
        path="/recordings/standup.m4a",
        sha256="a" * 64,
        size_bytes=1024,
        duration=120.0,
        recorded_at=datetime(2026, 3, 31, 13, 32, 57).astimezone(),
        recorded_at_source="filename",
    )
    return Transcript(
        media=media,
        engine=EngineInfo(backend="whispercpp", model="large-v3-turbo", vad="silero"),
        language="ru",
        segments=[
            Segment(0, 3, "привет", confidence=0.91, speaker="S1"),
            Segment(
                3,
                7,
                "как продвигается",
                confidence=0.42,
                speaker="S2",
                flags=["low-confidence"],
                variants=[Variant("как продвигаются", "whispercpp:large-v3@t0.4", confidence=0.31)],
            ),
        ],
        summary=Summary(
            headline="Стендап",
            sections=[SummarySection("Статус", ["всё идёт по плану"], start=0.0)],
            actions=["никому ничего"],
        ),
    )


# ── cache key ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "large-v3"),
        ("backend", "mlx"),
        ("language", "en"),
        ("vad", "none"),
        ("clean", False),
        ("strict_clean", True),
        ("variants", 2),
        ("fix", True),
        ("confidence_floor", 0.8),
    ],
)
def test_settings_that_change_the_words_change_the_key(field: str, value: object) -> None:
    assert fingerprint(Settings()) != fingerprint(Settings(**{field: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("formats", ["srt", "vtt"]),
        ("timestamps", "absolute"),
        ("timezone", "Europe/Belgrade"),
        ("output", "/tmp/out"),
        ("show_variants", True),
    ],
)
def test_presentation_settings_do_not_change_the_key(field: str, value: object) -> None:
    """Asking for a different format must re-render, never re-transcribe."""
    assert fingerprint(Settings()) == fingerprint(Settings(**{field: value}))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["summary", "diarize"])
def test_enrichments_do_not_change_the_key(field: str) -> None:
    """A summary or speaker labels ADD to a transcript; they must not cost a re-transcription."""
    assert fingerprint(Settings()) == fingerprint(Settings(**{field: True}))  # type: ignore[arg-type]


def test_a_decoding_change_invalidates_old_runs() -> None:
    """Bumping DECODE_REVISION must stop stale transcripts being served as current."""
    import stt_cli.archive as archive_mod

    before = fingerprint(Settings())
    original = archive_mod.DECODE_REVISION
    try:
        archive_mod.DECODE_REVISION = original + 1
        assert fingerprint(Settings()) != before
    finally:
        archive_mod.DECODE_REVISION = original


def test_a_cached_transcript_without_a_summary_reports_it_as_missing() -> None:
    from stt_cli.pipeline import _missing_enrichments

    transcript = _transcript()
    transcript.summary = None
    assert _missing_enrichments(transcript, Settings(summary=True)) == ["summary"]
    assert _missing_enrichments(transcript, Settings()) == []


def test_a_cached_transcript_that_already_has_a_summary_needs_nothing() -> None:
    from stt_cli.pipeline import _missing_enrichments

    assert _missing_enrichments(_transcript(), Settings(summary=True)) == []


# ── archive round trip ────────────────────────────────────────────────────────
def test_saved_run_is_found_and_loads_back_identical() -> None:
    transcript = _transcript()
    key = fingerprint(Settings())
    with Archive() as store:
        record = store.save(transcript, key)
        found = store.find(transcript.media.sha256, key)
        assert found is not None and found.run_id == record.run_id
        loaded = store.load(record.run_id)
    assert loaded.to_dict() == transcript.to_dict()


def test_a_different_key_is_a_cache_miss() -> None:
    with Archive() as store:
        store.save(_transcript(), fingerprint(Settings()))
        assert store.find("a" * 64, fingerprint(Settings(model="large-v3"))) is None


def test_an_index_row_without_files_is_treated_as_a_miss() -> None:
    """A hit that cannot be loaded is worse than no hit — it must not be reported as one."""
    import shutil

    key = fingerprint(Settings())
    with Archive() as store:
        record = store.save(_transcript(), key)
        shutil.rmtree(record.directory)
        assert store.find("a" * 64, key) is None


# ── renderers ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "fmt", ["txt", "md", "json", "srt", "vtt", "csv", "tsv", "speakers", "summary"]
)
def test_every_renderer_produces_output(fmt: str) -> None:
    transcript = _transcript()
    text = formats.render(
        fmt, transcript, Stamper("relative", transcript.media), formats.RenderOptions()
    )
    assert text.strip()


def test_json_round_trips_through_the_model() -> None:
    transcript = _transcript()
    text = formats.render(
        "json", transcript, Stamper("none", transcript.media), formats.RenderOptions()
    )
    assert Transcript.from_dict(json.loads(text)).to_dict() == transcript.to_dict()


def test_srt_numbers_and_times_are_well_formed() -> None:
    transcript = _transcript()
    text = formats.render(
        "srt", transcript, Stamper("none", transcript.media), formats.RenderOptions()
    )
    assert text.startswith("1\n00:00:00,000 --> 00:00:03,000\n")


def test_variants_are_hidden_by_default_and_shown_on_request() -> None:
    transcript = _transcript()
    stamper = Stamper("none", transcript.media)
    plain = formats.render("txt", transcript, stamper, formats.RenderOptions())
    verbose = formats.render("txt", transcript, stamper, formats.RenderOptions(show_variants=True))
    assert "как продвигаются" not in plain
    assert "как продвигаются" in verbose
    assert "(0.42)" in verbose  # showing variants shows the confidence that ranks them


def test_raw_text_recovers_the_speech_models_wording_after_correction() -> None:
    transcript = _transcript()
    segment = transcript.segments[0]
    segment.variants.insert(0, Variant("привед", "asr:whispercpp:large-v3-turbo", kind="primary"))
    segment.text = "Привет!"
    stamper = Stamper("none", transcript.media)
    raw = formats.render("txt", transcript, stamper, formats.RenderOptions(text_variant="raw"))
    assert "привед" in raw


def test_all_expands_and_unknown_formats_are_rejected() -> None:
    from stt_cli._errors import UnknownItemError

    assert "txt" in formats.expand(["all"])
    with pytest.raises(UnknownItemError):
        formats.expand(["docx"])
