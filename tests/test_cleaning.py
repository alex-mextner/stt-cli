"""Cleaning must remove what the model invented and nothing else.

The interesting cases are the ones where deleting is WRONG: an ordinary phrase that happens
to be on the filler list, said clearly, in the middle of real speech. Those are what keep the
filter honest, so they get as much coverage here as the artefacts do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stt_cli.cleaning import clean
from stt_cli.models import Segment, SpeechSpan

NO_USER_PATTERNS = Path("/nonexistent-stt-home")


def run(segments: list[Segment], spans: list[SpeechSpan] | None = None, **kwargs: object):
    return clean(
        segments,
        speech_spans=spans if spans is not None else [SpeechSpan(0, 10_000)],
        home=NO_USER_PATTERNS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_subtitle_credit_is_always_dropped() -> None:
    kept, report = run([Segment(0, 3, "Субтитры сделал DimaTorzok", confidence=0.99)])
    assert kept == []
    assert report.dropped[0][1] == "hallucination"


def test_amara_credit_is_always_dropped() -> None:
    kept, _ = run([Segment(0, 3, "Subtitles by the Amara.org community", confidence=0.98)])
    assert kept == []


def test_ordinary_phrase_said_clearly_survives() -> None:
    """'Thank you' is the model's favourite silence filler AND a thing people say."""
    kept, _ = run([Segment(0, 3, "Thank you.", confidence=0.95)])
    assert [s.text for s in kept] == ["Thank you."]


def test_ordinary_phrase_over_silence_is_dropped() -> None:
    segments = [Segment(20, 24, "Продолжение следует...", confidence=0.9)]
    kept, _ = run(segments, spans=[SpeechSpan(0, 10)])
    assert kept == []


def test_ordinary_phrase_with_low_confidence_is_dropped() -> None:
    kept, _ = run([Segment(0, 3, "Thanks for watching!", confidence=0.2)])
    assert kept == []


def test_strict_mode_drops_the_ambiguous_phrase_anyway() -> None:
    kept, _ = run([Segment(0, 3, "Thank you.", confidence=0.95)], strict=True)
    assert kept == []


def test_repeated_word_loop_is_collapsed() -> None:
    kept, report = run([Segment(0, 9, "да да да да да да да да", confidence=0.9)])
    assert kept[0].text == "да"
    assert "loop" in kept[0].flags
    assert report.collapsed


def test_repeated_phrase_loop_is_collapsed() -> None:
    text = "и так далее " * 6
    kept, _ = run([Segment(0, 9, text.strip(), confidence=0.9)])
    assert kept[0].text == "и так далее"


def test_a_few_repeats_are_left_alone() -> None:
    """People do say 'no, no, no'. Three is not a decoder loop."""
    kept, _ = run([Segment(0, 3, "нет нет нет", confidence=0.9)])
    assert kept[0].text == "нет нет нет"


def test_cross_segment_echo_keeps_the_first_only() -> None:
    segments = [Segment(i * 3, i * 3 + 3, "ага, понял", confidence=0.9) for i in range(4)]
    kept, _ = run(segments)
    assert len(kept) == 1
    assert kept[0].start == 0


def test_no_clean_flags_but_keeps_everything() -> None:
    segments = [Segment(0, 3, "Субтитры сделал DimaTorzok", confidence=0.99)]
    kept, _ = run(segments, apply=False)
    assert len(kept) == 1
    assert "hallucination" in kept[0].flags


def test_low_confidence_is_flagged_not_dropped() -> None:
    kept, _ = run([Segment(0, 3, "какой-то невнятный кусок", confidence=0.1)])
    assert len(kept) == 1
    assert "low-confidence" in kept[0].flags


@pytest.mark.parametrize("text", ["", "   ", "..."])
def test_empty_and_punctuation_only_segments_go(text: str) -> None:
    kept, _ = run([Segment(0, 3, text, confidence=0.9)])
    assert kept == []
