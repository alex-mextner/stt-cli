"""Span grouping and the chunk-to-source timestamp map.

These two pieces are where a bug is silent and expensive: bad grouping means hundreds of
model loads, and a bad map means a transcript whose timestamps quietly do not match the
audio. Both are pure functions, so both are cheap to pin down exactly.
"""

from __future__ import annotations

from pathlib import Path

from stt_cli import chunks, vad
from stt_cli.models import SpeechSpan


# ── span normalization ────────────────────────────────────────────────────────
def test_near_touching_spans_are_merged() -> None:
    spans = vad.normalize([SpeechSpan(0, 5), SpeechSpan(5.3, 9)], duration=60, min_speech=0.25)
    assert [(s.start, s.end) for s in spans] == [(0, 9)]


def test_a_real_pause_is_not_merged() -> None:
    spans = vad.normalize(
        [SpeechSpan(0, 5), SpeechSpan(20, 25)], duration=60, min_speech=0.25, merge_gap=0.8
    )
    assert len(spans) == 2


def test_scraps_below_the_floor_are_dropped() -> None:
    spans = vad.normalize([SpeechSpan(0, 0.1), SpeechSpan(10, 15)], duration=60, min_speech=0.25)
    assert [(s.start, s.end) for s in spans] == [(10, 15)]


def test_overlong_spans_are_split() -> None:
    spans = vad.normalize([SpeechSpan(0, 1500)], duration=1500, min_speech=0.25)
    assert len(spans) == 3
    assert spans[0].duration == vad.MAX_SPAN_SECONDS
    assert spans[-1].end == 1500


def test_spans_are_clamped_to_the_recording() -> None:
    spans = vad.normalize([SpeechSpan(-2, 70)], duration=60, min_speech=0.25)
    assert (spans[0].start, spans[0].end) == (0.0, 60)


def test_silence_intervals_invert_into_speech() -> None:
    spans = vad._invert(starts=[10.0, 30.0], ends=[20.0, 40.0], duration=60.0, pad=0.0)
    assert [(s.start, s.end) for s in spans] == [(0.0, 10.0), (20.0, 30.0), (40.0, 60.0)]


# ── chunk grouping ────────────────────────────────────────────────────────────
def test_grouping_respects_the_speech_budget() -> None:
    spans = [SpeechSpan(i * 100, i * 100 + 60) for i in range(20)]  # 60 s of speech each
    batches = chunks.group(spans, limit=300)
    assert all(sum(s.duration for s in batch) <= 300 for batch in batches)
    assert sum(len(batch) for batch in batches) == 20


def test_grouping_turns_hundreds_of_spans_into_a_handful_of_chunks() -> None:
    """The whole point: one hour of conversation must not become hundreds of decodes."""
    spans = [SpeechSpan(i * 8, i * 8 + 5) for i in range(430)]  # ~36 min of speech
    batches = chunks.group(spans)
    assert len(batches) <= 8


def test_a_single_overlong_span_still_forms_one_batch() -> None:
    batches = chunks.group([SpeechSpan(0, 5000)], limit=300)
    assert len(batches) == 1


# ── chunk time -> source time ─────────────────────────────────────────────────
def _chunk() -> chunks.Chunk:
    # Two spliced spans: file 100-110 s landed at 0-10, file 300-305 s landed at 10.2-15.2.
    return chunks.Chunk(
        index=1,
        path=Path("unused.wav"),
        pieces=[chunks.Piece(0.0, 10.0, 100.0), chunks.Piece(10.2, 15.2, 300.0)],
    )


def test_timestamps_map_back_into_the_first_span() -> None:
    assert _chunk().to_source(5.0) == 105.0


def test_timestamps_map_back_across_the_splice() -> None:
    assert _chunk().to_source(12.0) == 301.8


def test_a_timestamp_inside_the_join_gap_snaps_to_real_audio() -> None:
    """The gap is silence we inserted; reporting a word inside it would be a lie."""
    assert _chunk().to_source(10.1) == 300.0


def test_a_timestamp_past_the_end_clamps_to_the_end() -> None:
    assert _chunk().to_source(99.0) == 305.0


def test_chunk_reports_its_source_range() -> None:
    chunk = _chunk()
    assert (chunk.source_start, chunk.source_end) == (100.0, 305.0)
