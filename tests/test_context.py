"""The context/no-context trade-off: `--context` and `--context-compare`.

Feeding the decoder its own previous output improves casing, punctuation and proper nouns,
and is also the mechanism that lets a repetition loop feed itself. stt decodes without it by
default and can run a second pass to compare. What has to hold: the loop-safe pass stays
primary, the comparison is joined by time rather than by index, and turning none of this on
leaves every already-archived run exactly where it was.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from stt_cli import archive, chunks, pipeline, variants
from stt_cli._errors import EngineError, UnknownItemError
from stt_cli.backends.base import CONTEXT_TOKENS, DecodeRequest, context_tokens
from stt_cli.config import Settings
from stt_cli.models import Segment


def _segment(start: float, end: float, text: str, confidence: float = 0.9) -> Segment:
    return Segment(start=start, end=end, text=text, confidence=confidence)


def _chunk(index: int, source_start: float, length: float) -> chunks.Chunk:
    piece = chunks.Piece(chunk_start=0.0, chunk_end=length, source_start=source_start)
    return chunks.Chunk(index=index, path=Path(f"/dev/null/{index}"), pieces=[piece])


# ── the token budget ──────────────────────────────────────────────────────────
def test_context_modes_are_ordered_and_capped_at_whispers_own_limit() -> None:
    assert context_tokens("off") == 0
    assert 0 < CONTEXT_TOKENS["short"] < CONTEXT_TOKENS["full"]
    # whisper.cpp clamps n_max_text_ctx at n_text_ctx/2 = 224; asking for more is a lie.
    assert CONTEXT_TOKENS["full"] == 224


def test_an_unknown_context_mode_is_refused_with_the_real_ones() -> None:
    with pytest.raises(UnknownItemError) as caught:
        context_tokens("maximum")
    assert "full" in caught.value.render()


def test_the_decode_request_defaults_to_no_carried_context() -> None:
    assert DecodeRequest(wav=Path("a.wav"), model="m").max_context == 0


# ── joining two independent passes ────────────────────────────────────────────
def test_a_differing_reading_is_attached_to_the_segment_it_overlaps() -> None:
    primary = [_segment(0.0, 5.0, "мы говорили про агентов")]
    alternate = [_segment(0.2, 5.4, "Мы говорили про агенство.")]
    assert variants.merge_parallel(primary, alternate, source="asr:context=224") == 1
    variant = primary[0].variants[0]
    assert variant.kind == "context"
    assert variant.text == "Мы говорили про агенство."


def test_casing_and_punctuation_alone_do_not_count_as_disagreement() -> None:
    primary = [_segment(0.0, 5.0, "мы говорили про агентов")]
    alternate = [_segment(0.0, 5.0, "Мы говорили про агентов!")]
    assert variants.merge_parallel(primary, alternate, source="s") == 0
    assert primary[0].variants == []


def test_segments_are_paired_by_overlap_not_by_position() -> None:
    """The two passes split differently, so index N of one is not index N of the other."""
    primary = [_segment(0.0, 2.0, "first"), _segment(2.0, 4.0, "second")]
    alternate = [_segment(1.9, 4.1, "second differs")]
    assert variants.merge_parallel(primary, alternate, source="s") == 1
    assert primary[0].variants == []
    assert primary[1].variants[0].text == "second differs"


def test_a_segment_with_no_counterpart_is_left_alone() -> None:
    primary = [_segment(0.0, 2.0, "alone")]
    assert variants.merge_parallel(primary, [], source="s") == 0
    assert variants.merge_parallel(primary, [_segment(90.0, 92.0, "elsewhere")], source="s") == 0


# ── when the second pass runs ─────────────────────────────────────────────────
def test_the_llm_pass_turns_the_comparison_on_by_itself() -> None:
    assert pipeline._compare_mode(Settings()) == "off"
    assert pipeline._compare_mode(Settings(fix=True)) == "auto"


def test_an_explicit_choice_beats_the_implication() -> None:
    assert pipeline._compare_mode(Settings(fix=True, context_compare="always")) == "always"
    assert pipeline._compare_mode(Settings(context_compare="always")) == "always"


def test_an_unknown_comparison_mode_is_refused() -> None:
    with pytest.raises(UnknownItemError):
        pipeline._compare_mode(Settings(context_compare="sometimes"))


# ── which chunks are worth a second pass ──────────────────────────────────────
def test_a_chunk_opening_in_lower_case_is_suspect() -> None:
    """The signature of a decoder that did not know a sentence had started."""
    built = [_chunk(1, 0.0, 60.0)]
    segments = [_segment(1.0, 3.0, "и вот тогда мы решили"), _segment(4.0, 6.0, "Дальше всё ок.")]
    assert pipeline._suspect_chunks(built, segments, Settings()) == {1}


def test_a_confident_well_formed_chunk_is_left_alone() -> None:
    built = [_chunk(1, 0.0, 60.0)]
    segments = [_segment(1.0, 3.0, "Мы решили так."), _segment(4.0, 6.0, "Дальше всё ок.")]
    assert pipeline._suspect_chunks(built, segments, Settings()) == set()


def test_a_chunk_full_of_shaky_segments_is_suspect() -> None:
    built = [_chunk(1, 0.0, 60.0)]
    segments = [
        _segment(1.0, 3.0, "Мы решили так."),
        _segment(4.0, 6.0, "Что-то неразборчивое.", confidence=0.2),
    ]
    assert pipeline._suspect_chunks(built, segments, Settings()) == {1}


def test_chunks_are_matched_to_their_own_segments() -> None:
    built = [_chunk(1, 0.0, 60.0), _chunk(2, 60.0, 60.0)]
    segments = [_segment(5.0, 7.0, "Всё хорошо."), _segment(65.0, 67.0, "и тут сломалось")]
    assert pipeline._suspect_chunks(built, segments, Settings()) == {2}


# ── the archive must survive the new settings ─────────────────────────────────
def test_default_context_settings_do_not_invalidate_an_existing_archive() -> None:
    """Adding a setting must not silently throw away everyone's cached transcriptions."""
    payload = {key: getattr(Settings(), key) for key in archive.FINGERPRINT_KEYS}
    assert "context" not in payload
    baseline = archive.fingerprint(Settings())
    assert archive.fingerprint(Settings(context="off", context_compare="off")) == baseline


def test_a_non_default_context_is_a_different_transcript() -> None:
    baseline = archive.fingerprint(Settings())
    assert archive.fingerprint(Settings(context="full")) != baseline
    assert archive.fingerprint(Settings(context_compare="always")) != baseline
    assert archive.fingerprint(Settings(context="full")) != archive.fingerprint(
        Settings(context="short")
    )


def test_a_counterpart_split_in_two_is_reassembled_not_truncated() -> None:
    """Half a sentence is not an alternative reading of a whole one."""
    primary = [_segment(0.0, 6.0, "it can be done immediately in hyper no need for vigma")]
    alternate = [
        _segment(0.0, 3.0, "It can be done immediately in Hyper.", confidence=0.9),
        _segment(3.0, 6.0, "No need for Figma.", confidence=0.8),
    ]
    assert variants.merge_parallel(primary, alternate, source="s") == 1
    variant = primary[0].variants[0]
    assert variant.text == "It can be done immediately in Hyper. No need for Figma."
    assert variant.confidence == pytest.approx(0.85)


def test_a_pure_segmentation_difference_is_not_a_disagreement() -> None:
    """Before the counterpart was reassembled, this reported half a sentence as a variant."""
    primary = [_segment(0.0, 6.0, "it can be done immediately in hyper no need for figma")]
    alternate = [
        _segment(0.0, 3.0, "It can be done immediately in Hyper."),
        _segment(3.0, 6.0, "No need for Figma."),
    ]
    assert variants.merge_parallel(primary, alternate, source="s") == 0


def test_an_alternate_straddling_two_segments_belongs_to_neither() -> None:
    primary = [_segment(0.0, 4.0, "one"), _segment(4.0, 8.0, "two")]
    assert variants.merge_parallel(primary, [_segment(2.0, 6.0, "straddles")], source="s") == 0


def test_an_alternate_that_swallows_the_next_segment_is_not_attached_to_the_first() -> None:
    """A lopsided straddle passes the majority test and used to be attached whole, so the
    first segment was shown an "alternative reading" containing the second one's words."""
    primary = [_segment(0.0, 4.0, "first sentence"), _segment(4.0, 8.0, "second sentence")]
    alternate = [_segment(0.0, 6.0, "first differs second sentence")]
    assert variants.merge_parallel(primary, alternate, source="s") == 0
    assert primary[0].variants == []
    assert primary[1].variants == []


def test_a_boundary_that_drifted_a_little_still_counts_as_the_same_reading() -> None:
    """Two decodings never cut at the same instant. Spill that small is the boundary moving,
    not the neighbour's speech, and dropping it would silence the comparison everywhere."""
    primary = [_segment(0.0, 4.0, "first sentence"), _segment(4.0, 8.0, "second sentence")]
    alternate = [_segment(0.0, 5.0, "first differs"), _segment(5.0, 8.0, "second sentence")]
    assert variants.merge_parallel(primary, alternate, source="s") == 1
    assert primary[0].variants[0].text == "first differs"


# ── the CLI actually reaches the pipeline with usable values ──────────────────
# Every test above builds `Settings` directly, which is exactly the gap a reviewer found:
# the flags default to None, and if the merge ever stopped dropping None these would be the
# tests that caught it — `context_tokens(None)` raises and the fingerprint changes for
# everybody.
def _settings_from(argv: list[str]):
    from stt_cli.commands.transcribe import _settings, build_parser

    return _settings(build_parser().parse_args(argv))


def test_a_plain_run_reaches_the_pipeline_with_real_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STT_HOME", str(tmp_path))
    settings = _settings_from(["rec.m4a"])
    assert settings.context == "off"
    assert settings.context_compare == "off"
    assert context_tokens(settings.context) == 0
    assert pipeline._compare_mode(settings) == "off"


def test_a_plain_run_does_not_change_anybodys_fingerprint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STT_HOME", str(tmp_path))
    assert archive.fingerprint(_settings_from(["rec.m4a"])) == archive.fingerprint(Settings())


def test_the_flags_arrive_where_they_are_read(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STT_HOME", str(tmp_path))
    settings = _settings_from(["rec.m4a", "--context", "full", "--context-compare", "always"])
    assert context_tokens(settings.context) == 224
    assert pipeline._compare_mode(settings) == "always"


def test_an_old_fix_run_is_not_served_without_the_comparison(monkeypatch, tmp_path) -> None:
    """The stale-cache trap: --fix now implies the comparison, so a transcript cached before
    that — which has none of the variants — must not answer for it."""
    monkeypatch.setenv("STT_HOME", str(tmp_path))
    stale = Settings(fix=True, context_compare="off")  # what an older run stored
    fresh, _ = pipeline._resolve(Settings(fix=True))
    assert fresh.context_compare == "auto"
    assert archive.fingerprint(fresh) != archive.fingerprint(stale)


# ── the flags each engine actually emits ──────────────────────────────────────
# Both `_argv` builders were untested, and that is where the context/glossary design turns
# into a command line. The whisper.cpp one rewrites `-mc` by index, the most fragile line in
# the file; the mlx one ignored `carry_prompt` entirely, so a glossary reached the model as a
# plain initial prompt that whisper drops after the first window.
def _whispercpp(monkeypatch) -> object:
    from stt_cli.backends import whispercpp

    monkeypatch.setattr(whispercpp, "_find_binary", lambda name, root: Path(f"/bin/{name}"))
    backend = whispercpp.WhisperCppBackend()
    monkeypatch.setattr(backend, "model_path", lambda model: Path(f"/models/{model}.bin"))
    return backend


def _request(**kwargs: object) -> DecodeRequest:
    base: dict[str, object] = {"wav": Path("/tmp/a.wav"), "model": "large-v3"}
    return DecodeRequest(**{**base, **kwargs})  # type: ignore[arg-type]


def test_whispercpp_buys_the_carried_glossary_a_context_budget(monkeypatch) -> None:
    """With -mc 0 whisper.cpp ignores the initial prompt entirely, so a carried glossary has
    to raise the budget — and it must raise the value of `-mc`, not append a second one."""
    from stt_cli.backends.whispercpp import PROMPT_CONTEXT

    backend = _whispercpp(monkeypatch)
    request = _request(initial_prompt="Glossary: ConLoca.", carry_prompt=True, max_context=0)
    argv = backend._argv(request, Path("/tmp/out"), carry_prompt=True)

    assert argv.count("-mc") == 1
    assert argv[argv.index("-mc") + 1] == str(PROMPT_CONTEXT)
    assert "--carry-initial-prompt" in argv
    assert argv[argv.index("--prompt") + 1] == "Glossary: ConLoca."


def test_whispercpp_does_not_lower_a_budget_the_user_asked_for(monkeypatch) -> None:
    backend = _whispercpp(monkeypatch)
    request = _request(initial_prompt="Glossary: ConLoca.", carry_prompt=True, max_context=224)
    argv = backend._argv(request, Path("/tmp/out"), carry_prompt=True)
    assert argv[argv.index("-mc") + 1] == "224"


def test_whispercpp_omits_a_flag_an_older_binary_would_choke_on(monkeypatch) -> None:
    """An older whisper-cli exits with its usage text on an unknown flag, so one `stt dict
    add` would otherwise break every transcription on that machine."""
    backend = _whispercpp(monkeypatch)
    request = _request(initial_prompt="Glossary: ConLoca.", carry_prompt=True, max_context=0)
    argv = backend._argv(request, Path("/tmp/out"), carry_prompt=False)

    assert "--carry-initial-prompt" not in argv
    assert argv[argv.index("-mc") + 1] == "0"


def test_an_older_binary_is_not_handed_a_prompt_it_will_ignore(monkeypatch) -> None:
    """Without --carry-initial-prompt the budget is never raised, and at -mc 0 whisper.cpp
    ignores the initial prompt entirely — so `--prompt` was passed to a decoder that could
    not use it while the warning told the user the glossary reached the first window. Buying
    the budget instead is the wrong trade: nothing pins the glossary to the front, so the
    tokens would go to the decoder's own previous output — the repetition loop `-mc 0` is
    there to prevent — for a glossary that gets evicted anyway."""
    backend = _whispercpp(monkeypatch)
    dead = backend._argv(
        _request(initial_prompt="Glossary: ConLoca.", max_context=0),
        Path("/tmp/out"),
        carry_prompt=False,
    )
    assert "--prompt" not in dead
    assert dead[dead.index("-mc") + 1] == "0"

    # ...but a budget the user asked for makes it land on the first window, so it is sent.
    asked = backend._argv(
        _request(initial_prompt="Glossary: ConLoca.", max_context=64),
        Path("/tmp/out"),
        carry_prompt=False,
    )
    assert asked[asked.index("--prompt") + 1] == "Glossary: ConLoca."


def test_whispercpp_probes_for_the_flag_once_and_believes_the_answer(monkeypatch) -> None:
    from stt_cli import proc
    from stt_cli.backends import whispercpp

    backend = _whispercpp(monkeypatch)
    calls: list[list[str]] = []

    async def fake_run(argv, **kwargs):
        calls.append(argv)
        return proc.Result(code=0, stdout="  --carry-initial-prompt  [false]", stderr="", argv=argv)

    monkeypatch.setattr(whispercpp.proc, "run", fake_run)
    assert asyncio.run(backend.can_pin_prompt()) is True
    assert asyncio.run(backend.can_pin_prompt()) is True
    assert len(calls) == 1, "the help probe must run once per backend, not once per chunk"


def test_mlx_carries_the_prompt_when_it_is_asked_to(monkeypatch) -> None:
    """Without --carry-prompt the glossary reaches mlx as a plain initial prompt, which
    whisper drops after the first 30-second window — so it biases half a minute and stops."""
    from stt_cli import proc
    from stt_cli.backends import mlx

    backend = mlx.MlxBackend.__new__(mlx.MlxBackend)
    backend._runner = ("env", ["python3"])
    backend._pins_prompt = True  # the probe is exercised by its own test
    seen: list[list[str]] = []

    async def fake_run(argv, **kwargs):
        seen.append(argv)
        return proc.Result(code=0, stdout='{"language":"en","segments":[]}', stderr="", argv=argv)

    monkeypatch.setattr(mlx.proc, "run", fake_run)
    monkeypatch.setattr(mlx.registry, "require_for_engine", lambda model, name: (None, "repo/x"))

    asyncio.run(backend.decode(_request(initial_prompt="Glossary: ConLoca.", carry_prompt=True)))
    assert "--carry-prompt" in seen[-1]
    assert seen[-1][seen[-1].index("--initial-prompt") + 1] == "Glossary: ConLoca."

    asyncio.run(backend.decode(_request(initial_prompt="Glossary: ConLoca.", carry_prompt=False)))
    assert "--carry-prompt" not in seen[-1]


def test_the_mlx_probe_asks_once_and_refuses_to_guess(monkeypatch) -> None:
    """The one capability path with no coverage, and the one that only runs on a cache miss
    with a dictionary on an mlx machine — the combination local testing skips. A wrong answer
    here decodes the glossary run as if there were no glossary AND keys it that way, so a
    flaky subprocess would make a capable machine re-decode the whole recording."""
    from stt_cli import proc
    from stt_cli.backends import mlx

    def _backend() -> mlx.MlxBackend:
        backend = mlx.MlxBackend.__new__(mlx.MlxBackend)
        backend._runner = ("env", ["python3"])
        backend._pins_prompt = None
        return backend

    seen: list[list[str]] = []

    def answering(stdout: str, code: int = 0):
        async def fake_run(argv, **kwargs):
            seen.append(argv)
            return proc.Result(code=code, stdout=stdout, stderr="", argv=argv)

        return fake_run

    monkeypatch.setattr(mlx.proc, "run", answering('{"carry_initial_prompt": true}'))
    backend = _backend()
    assert asyncio.run(backend.can_pin_prompt()) is True
    assert asyncio.run(backend.can_pin_prompt()) is True
    assert len(seen) == 1, "the probe is asked once per run, not once per chunk"
    assert "--probe" in seen[0], "the probe must not load a model or touch audio"
    assert "--audio" not in seen[0] and "--model" not in seen[0]

    monkeypatch.setattr(mlx.proc, "run", answering('{"carry_initial_prompt": false}'))
    assert asyncio.run(_backend().can_pin_prompt()) is False

    # A build too old to answer, a crashed interpreter and a timeout all look the same from
    # here, and none of them is the fact "this build cannot pin the prompt".
    for stdout, code in (("", 1), ("not json at all", 0), ('{"error": "no mlx_whisper"}', 1)):
        monkeypatch.setattr(mlx.proc, "run", answering(stdout, code))
        with pytest.raises(EngineError):
            asyncio.run(_backend().can_pin_prompt())


def test_collapse_only_removes_loops_without_dropping_anything() -> None:
    """The comparison pass must keep every segment — a dropped one reads as agreement when it
    is really absence — but a runaway repetition is worthless as an alternative reading."""
    from stt_cli import cleaning

    segments = [
        _segment(0.0, 4.0, " ".join(["yes"] * 12)),
        _segment(4.0, 8.0, "and then we shipped it"),
        _segment(8.0, 9.0, "продолжение следует"),
    ]
    collapsed = cleaning.collapse_only(segments, max_repeats=3)

    assert collapsed == 1
    assert len(segments) == 3, "collapse_only must never drop a segment"
    assert segments[0].text.split().count("yes") <= 3
    assert segments[1].text == "and then we shipped it"
    # The filler line survives: dropping it is the full pass's job, not this one's.
    assert segments[2].text == "продолжение следует"


def test_the_comparison_pass_drops_the_glossary_when_it_is_the_zero_context_one() -> None:
    """The glossary buys itself a context budget, so carrying it into the pass that is meant
    to have NO context would give both passes context — and the comparison would compare a
    setting against itself while labelling the variant with a number nobody used."""

    class Recorder:
        name = "recorder"

        def __init__(self, pinning_costs_context: bool = True) -> None:
            self.requests: list[DecodeRequest] = []
            self._costs = pinning_costs_context

        def pinning_the_prompt_costs_context(self) -> bool:
            return self._costs

        async def decode(self, request: DecodeRequest) -> list[Segment]:
            self.requests.append(request)
            return [_segment(0.0, 4.0, f"pass {len(self.requests)}")]

    backend = Recorder()
    built = [_chunk(0, 0.0, 4.0)]
    primary = [_segment(0.0, 4.0, "pass one")]
    settings = Settings(context_compare="always", model="large-v3")

    # The primary pass carried context, so the comparison is the zero-context one.
    asyncio.run(
        pipeline._compare_context(
            built,
            backend,
            settings,
            CONTEXT_TOKENS["full"],
            primary,
            pipeline.Reporter(),
            "Glossary: ConLoca.",
        )
    )

    assert len(backend.requests) == 1
    comparison = backend.requests[0]
    assert comparison.max_context == 0
    assert comparison.initial_prompt is None
    assert comparison.carry_prompt is False

    # The other way round the glossary stays: that pass has a budget regardless.
    backend.requests.clear()
    asyncio.run(
        pipeline._compare_context(
            built, backend, settings, 0, primary, pipeline.Reporter(), "Glossary: ConLoca."
        )
    )
    assert backend.requests[0].max_context == CONTEXT_TOKENS["full"]
    assert backend.requests[0].initial_prompt == "Glossary: ConLoca."

    # ...and on an engine where pinning the prompt is free, the zero-context pass keeps it
    # too: dropping it there would hand the comparison the very spelling the glossary
    # exists to prevent, and then offer that to the LLM as evidence against the primary.
    free = Recorder(pinning_costs_context=False)
    asyncio.run(
        pipeline._compare_context(
            built,
            free,
            settings,
            CONTEXT_TOKENS["full"],
            primary,
            pipeline.Reporter(),
            "Glossary: ConLoca.",
        )
    )
    assert free.requests[0].max_context == 0
    assert free.requests[0].initial_prompt == "Glossary: ConLoca."


def test_the_fingerprint_defaults_match_the_real_defaults() -> None:
    """`FINGERPRINT_DEFAULTS` writes the same values down a second time, in another file.

    Nothing but this test connects them. The day somebody tunes `Settings.dict_similarity`
    and leaves the table behind, every default run starts injecting the new value into the
    payload and every stored fingerprint changes — the archive thrown away wholesale by an
    edit two files away, which is the exact failure the mechanism exists to prevent.
    """
    settings = Settings()
    for key, default in archive.FINGERPRINT_DEFAULTS.items():
        if key == "dict_digest":
            continue  # filled in by the pipeline from the dictionary's content, not a setting
        assert getattr(settings, key) == default, f"{key} drifted from its Settings default"


def test_a_run_whose_glossary_did_nothing_is_not_cached_as_one_that_worked(monkeypatch) -> None:
    """An engine that cannot pin the prompt decodes a dictionary run exactly like a run with
    no dictionary — and the warning tells the user to upgrade. If both were stored under the
    same key, the upgrade would change nothing: the un-biased transcript would come back out
    of the cache forever, under an identity claiming the glossary was applied.
    """
    from stt_cli import dictionary, pipeline

    terms = dictionary.Dictionary([dictionary.Term(term="ConLoca")])
    settings = Settings(dict_digest=terms.digest())

    class Engine:
        name = "engine"

        def __init__(self, pins: bool) -> None:
            self._pins = pins

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            return self._pins

    working, _ = asyncio.run(
        pipeline._settle_engine_limits(
            settings, terms, Engine(True), variants.CrossChecks(), pipeline.Reporter()
        )
    )
    degraded, _ = asyncio.run(
        pipeline._settle_engine_limits(
            settings, terms, Engine(False), variants.CrossChecks(), pipeline.Reporter()
        )
    )

    assert working.engine_limits == ""
    assert "cannot pin the glossary" in degraded.engine_limits
    assert archive.fingerprint(working) != archive.fingerprint(degraded)


def test_a_run_without_a_dictionary_never_pays_for_the_probe() -> None:
    """The mlx probe starts a worker process. A run with no glossary must not do that."""
    from stt_cli import dictionary, pipeline

    class Engine:
        name = "engine"

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            raise AssertionError("the engine must not be probed when no glossary is carried")

    for settings, terms in (
        (Settings(), dictionary.Dictionary()),
        (Settings(dict_bias=False), dictionary.Dictionary([dictionary.Term(term="ConLoca")])),
    ):
        settled, _ = asyncio.run(
            pipeline._settle_engine_limits(
                settings, terms, Engine(), variants.CrossChecks(), pipeline.Reporter()
            )
        )
        assert settled.engine_limits == ""


def test_a_second_opinion_is_decoded_under_the_same_conditions_as_the_first(monkeypatch) -> None:
    """A variant decoded without the context and glossary the primary had is a different
    question, not a second answer: on `--context full` with a dictionary it comes back
    missing the proper noun only the glossary made recoverable, and that absence is then
    handed to the LLM as evidence."""
    from stt_cli import media, variants
    from stt_cli.models import Segment as Seg

    seen: list[DecodeRequest] = []

    class Recorder:
        name = "recorder"

        async def decode(self, request: DecodeRequest) -> list[Seg]:
            seen.append(request)
            return [_segment(0.0, 3.0, "another reading", confidence=0.4)]

    async def fake_wav(source, target, *, start=0.0, end=None):
        target.write_bytes(b"")

    monkeypatch.setattr(media, "to_engine_wav", fake_wav)
    plan = variants.plan_from_settings(
        Settings(variants=1), max_context=CONTEXT_TOKENS["full"], glossary="Glossary: ConLoca."
    )
    shaky = [_segment(0.0, 3.0, "one reading", confidence=0.1)]

    asyncio.run(
        variants.enrich(
            shaky,
            source_wav=Path("/tmp/whole.wav"),
            backend=Recorder(),
            model="large-v3",
            language=None,
            plan=plan,
        )
    )

    assert seen, "the variant pass never decoded anything"
    assert seen[0].max_context == CONTEXT_TOKENS["full"]
    assert seen[0].initial_prompt == "Glossary: ConLoca."
    assert seen[0].carry_prompt is True


def test_settings_that_cannot_matter_stay_out_of_the_cache_key() -> None:
    """With no dictionary, `--dict-similarity` and `--no-dict-bias` change nothing about the
    words — but they are non-default, so they entered the fingerprint and cost a full
    re-decode of the recording to produce a byte-identical transcript."""
    from stt_cli import pipeline

    plain, _ = pipeline._resolve(Settings(dictionary=False))
    tuned, _ = pipeline._resolve(Settings(dictionary=False, dict_similarity=0.7, dict_bias=False))
    assert archive.fingerprint(plain) == archive.fingerprint(tuned)


def test_a_context_variant_keeps_a_spelling_the_user_recorded_as_wrong(monkeypatch) -> None:
    """The comparison pass decodes WITHOUT the glossary — that is deliberate — so it comes
    back with the raw "Vigma" while the primary pass, which had the glossary, says "Figma".
    The dictionary corrects the segment and never touched the variant, so a word the user
    explicitly settled was handed to the reader and to the LLM as a live alternative."""
    from stt_cli import dictionary, pipeline
    from stt_cli.models import Transcript, Variant

    segment = _segment(0.0, 4.0, "we use Figma here")
    segment.variants.append(
        Variant(text="We use Vigma here anyway.", source="asr:context=0", kind="context")
    )
    transcript = Transcript.__new__(Transcript)
    transcript.segments = [segment]

    terms = dictionary.Dictionary([dictionary.Term(term="Figma", aka=["Vigma"])])
    pipeline._apply_dictionary(transcript, Settings(), terms, pipeline.Reporter())

    assert [v.text for v in segment.variants if v.kind == "context"] == [
        "We use Figma here anyway."
    ]


def test_cleaning_alone_can_settle_a_disagreement_without_any_dictionary() -> None:
    """The comparison pass is loop-collapsed when it is merged; the primary is collapsed
    later, by `_clean`. With no dictionary the dedupe used to be skipped entirely, so the
    two ended up word-identical and the leftover variant was shown as a disagreement."""
    from stt_cli import cleaning, variants
    from stt_cli.models import Variant

    # The comparison pass ran through `collapse_only` when it was merged, so it arrives
    # already collapsed. The primary still carries the full loop at that point.
    collapsed = _segment(0.0, 4.0, " ".join(["yes"] * 8))
    cleaning.collapse_only([collapsed], max_repeats=3)

    segment = _segment(0.0, 4.0, " ".join(["yes"] * 12))
    segment.variants.append(Variant(text=collapsed.text, source="asr:context=224", kind="context"))
    assert variants.drop_agreeing([segment], ("context",)) == 0, "a real disagreement, for now"

    cleaning.collapse_only([segment], max_repeats=3)
    assert variants.drop_agreeing([segment], ("context",)) == 1
    assert segment.variants == []


def test_an_engine_without_a_context_budget_is_keyed_by_what_it_actually_decodes() -> None:
    """mlx takes a boolean, not a token count, so `--context short` decodes exactly like
    `--context full`. Left alone the two get different keys for one transcript: the `short`
    run stores full-context words under a key promising 64 tokens, and asking for `full`
    afterwards re-decodes the recording to produce the very same text."""
    from stt_cli import dictionary, pipeline

    class NoBudget:
        name = "mlx"

        def honours_context_budget(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            return True

    empty = dictionary.Dictionary()
    short, _ = asyncio.run(
        pipeline._settle_engine_limits(
            Settings(context="short"),
            empty,
            NoBudget(),
            variants.CrossChecks(),
            pipeline.Reporter(),
        )
    )
    full, _ = asyncio.run(
        pipeline._settle_engine_limits(
            Settings(context="full"), empty, NoBudget(), variants.CrossChecks(), pipeline.Reporter()
        )
    )
    off, _ = asyncio.run(
        pipeline._settle_engine_limits(
            Settings(context="off"), empty, NoBudget(), variants.CrossChecks(), pipeline.Reporter()
        )
    )

    assert short.context == "full"
    assert archive.fingerprint(short) == archive.fingerprint(full)
    # "off" is a real setting on every engine: it means "carry nothing", not "carry less".
    assert off.context == "off"
    assert archive.fingerprint(off) != archive.fingerprint(full)


def test_a_cross_model_engine_that_cannot_pin_the_glossary_changes_the_key() -> None:
    """`--variant-model` decodes with a second engine whose readings are archived and fed to
    the LLM. If only the primary engine were probed, upgrading the variant engine would
    change what it produces while the cache key stayed the same."""
    from stt_cli import dictionary, pipeline

    class Engine:
        def __init__(self, name: str, pins: bool) -> None:
            self.name = name
            self._pins = pins

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            return self._pins

    terms = dictionary.Dictionary([dictionary.Term(term="ConLoca")])
    settings = Settings(dict_digest=terms.digest(), variant_models=["mlx:large-v3"])

    for pins, expected in ((True, ""), (False, "mlx cannot pin the glossary")):
        settled, _ = asyncio.run(
            pipeline._settle_engine_limits(
                settings,
                terms,
                Engine("whispercpp", True),
                variants.CrossChecks(resolved=[("mlx", Engine("mlx", pins), "m")]),
                pipeline.Reporter(),
            )
        )
        assert settled.engine_limits == expected


def test_a_variant_engine_without_a_budget_is_recorded_in_the_key(monkeypatch) -> None:
    """The primary engine honours `-mc 64`, so `--context short` stays short in the key. But
    the variant engine treats any positive value as unlimited, and its readings are archived
    and fed to the LLM — so the run holds full-context variants under a key promising short."""
    from stt_cli import dictionary, pipeline

    class Engine:
        def __init__(self, name: str, budget: bool) -> None:
            self.name = name
            self._budget = budget

        def honours_context_budget(self) -> bool:
            return self._budget

        async def can_pin_prompt(self) -> bool:
            return True

    settings = Settings(context="short", variant_models=["mlx:large-v3"])
    settled, _ = asyncio.run(
        pipeline._settle_engine_limits(
            settings,
            dictionary.Dictionary(),
            Engine("whispercpp", True),
            variants.CrossChecks(resolved=[("mlx", Engine("mlx", False), "m")]),
            pipeline.Reporter(),
        )
    )
    assert settled.engine_limits == "mlx has no context budget"
    assert settled.context == "short", "the primary engine does honour it"
    assert archive.fingerprint(settled) != archive.fingerprint(Settings(context="short"))


def test_a_degraded_run_does_not_also_claim_the_glossary_was_pinned(monkeypatch, capsys) -> None:
    """The engine says the glossary reaches only the first window; the pipeline said, one
    line later, that it was pinned into every window. The guard compared against a sentinel
    string the settling step never actually stores."""
    from stt_cli import chunks, pipeline
    from stt_cli.models import Segment as Seg

    class Engine:
        name = "old-whispercpp"

        def honours_context_budget(self) -> bool:
            return True

        async def can_pin_prompt(self) -> bool:
            return False

        async def decode(self, request: DecodeRequest) -> list[Seg]:
            return []

    async def fake_build(wav, spans, workdir):
        return []

    monkeypatch.setattr(chunks, "build", fake_build)
    asyncio.run(
        pipeline._decode_spans(
            Path("/tmp/a.wav"),
            pipeline.vad.VadResult(spans=[], method="none", total_duration=0.0),
            Engine(),
            Settings(),
            Path("/tmp"),
            pipeline.Reporter(),
            glossary="Glossary: ConLoca.",
        )
    )
    assert "glossary pinned" not in capsys.readouterr().err


def test_a_probe_that_does_not_answer_is_an_error_not_a_missing_capability(monkeypatch) -> None:
    """A timed-out or crashed probe is not the same fact as "this build is too old". Treated
    as one it costs twice: the run decodes without its glossary, AND the shortfall goes into
    the cache key — so one flaky subprocess makes a capable machine re-decode everything."""
    from stt_cli import proc
    from stt_cli._errors import EngineError
    from stt_cli.backends import whispercpp

    backend = _whispercpp(monkeypatch)

    async def silent(argv, **kwargs):
        return proc.Result(code=1, stdout="", stderr="", argv=argv)

    monkeypatch.setattr(whispercpp.proc, "run", silent)
    with pytest.raises(EngineError):
        asyncio.run(backend.can_pin_prompt())


def test_an_unusable_variant_model_is_reported_rather_than_dropped(monkeypatch) -> None:
    """`--variant-model` is the one thing the user explicitly asked for here. Resolving the
    engines before the cache key moved that resolution out of the variant pass, and its
    warnings were thrown away with it — the cross-check vanished in silence."""
    from stt_cli import pipeline

    settings = Settings(variant_models=["not-a-real-model"])
    checks = asyncio.run(pipeline._variant_backends(settings))

    assert checks.resolved == []
    assert checks.skipped == ["not-a-real-model"]
    assert checks.warnings and "not-a-real-model" in checks.warnings[0]


def test_a_cached_transcript_is_served_without_touching_an_engine(monkeypatch, tmp_path) -> None:
    """What the archive holds describes the run that produced it, not what this machine can
    do today. Resolving the engine before the lookup turned "answer instantly from disk"
    into "fail to start" the moment whisper.cpp was removed, downgraded or broken."""
    from stt_cli import media, pipeline
    from stt_cli._errors import MissingDependencyError

    monkeypatch.setenv("STT_HOME", str(tmp_path / "home"))

    def no_engine_here(*args, **kwargs):
        raise MissingDependencyError(what="no engine", why="removed", how="reinstall")

    monkeypatch.setattr(pipeline, "resolve", no_engine_here)
    monkeypatch.setattr(pipeline, "_variant_backends", no_engine_here)

    class Store:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def media_for_source(self, path):
            return None

        def find(self, sha, fingerprint):
            self.asked.append(fingerprint)
            return "a stored run"

    served = {}

    async def from_cache(cached, audio, info, settings, reporter, store, fingerprint):
        served["run"] = cached
        return pipeline.RunResult(transcript=None, run_id="cached", cached=True)

    async def inspect(path, *, recorded_at=None):
        from stt_cli.models import MediaInfo

        return MediaInfo(path=str(path), sha256="a" * 64, size_bytes=1, duration=10.0)

    monkeypatch.setattr(pipeline, "_from_cache", from_cache)
    monkeypatch.setattr(media, "inspect", inspect)
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"")

    result = asyncio.run(
        pipeline.transcribe(audio, Settings(), reporter=pipeline.Reporter(), store=Store())
    )
    assert result.cached and served["run"] == "a stored run"


def test_a_cross_check_that_could_not_run_is_part_of_the_run_identity() -> None:
    """Stored without it, a run made while mlx was missing looks exactly like one where the
    readings were gathered — so installing mlx and re-running hits that entry and the
    cross-check the user asked for never happens, permanently."""
    from stt_cli import dictionary, pipeline

    class Engine:
        name = "whispercpp"

        def honours_context_budget(self) -> bool:
            return True

        async def can_pin_prompt(self) -> bool:
            return True

    settings = Settings(variant_models=["mlx:large-v3"])
    skipped, _ = asyncio.run(
        pipeline._settle_engine_limits(
            settings,
            dictionary.Dictionary(),
            Engine(),
            variants.CrossChecks(
                skipped=["mlx:large-v3"],
                warnings=["cross-check mlx:large-v3 skipped: mlx-whisper is not importable"],
            ),
            pipeline.Reporter(),
        )
    )
    gathered, _ = asyncio.run(
        pipeline._settle_engine_limits(
            settings, dictionary.Dictionary(), Engine(), variants.CrossChecks(), pipeline.Reporter()
        )
    )

    assert skipped.engine_limits == "mlx:large-v3 was unavailable"
    assert gathered.engine_limits == ""
    assert archive.fingerprint(skipped) != archive.fingerprint(gathered)


def test_an_explicit_context_compare_off_survives_fix() -> None:
    """`--fix` turning the comparison on is an implication, not an override: somebody who
    typed `--context-compare off` said what they wanted, and a flag that cannot be turned
    off is not a flag. Before this, `--fix --context-compare off` still re-decoded."""
    implied, _ = pipeline._resolve(Settings(fix=True))
    assert implied.context_compare == "auto"

    chosen, _ = pipeline._resolve(Settings(fix=True, context_compare_chosen=True))
    assert chosen.context_compare == "off"
    # ...and the two are different runs, so neither is served from the other's cache entry.
    assert archive.fingerprint(implied) != archive.fingerprint(chosen)


def test_the_cli_marks_context_compare_as_chosen_only_when_it_is_given() -> None:
    assert _settings_from(["a.m4a", "--fix"]).context_compare_chosen is False
    assert _settings_from(["a.m4a", "--fix", "--context-compare", "off"]).context_compare_chosen


def test_a_cross_engine_that_cannot_answer_is_dropped_not_fatal() -> None:
    """Found by running it: `--variant-model mlx:large-v3` with no working mlx made the
    capability probe raise, and the exception took the whole transcription with it. Losing a
    second opinion must never cost the transcript — only the primary engine is load-bearing."""
    from stt_cli import dictionary, pipeline
    from stt_cli._errors import EngineError

    class Primary:
        name = "whispercpp"

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return True

        async def can_pin_prompt(self) -> bool:
            return True

    class Mute:
        name = "mlx"

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            raise EngineError(what="could not ask mlx-whisper what it supports", why="", how="")

    terms = dictionary.Dictionary([dictionary.Term(term="ConLoca")])
    checks = variants.CrossChecks(resolved=[("mlx:large-v3", Mute(), "large-v3")])
    settled, _ = asyncio.run(
        pipeline._settle_engine_limits(
            Settings(dict_digest=terms.digest(), variant_models=["mlx:large-v3"]),
            terms,
            Primary(),
            checks,
            pipeline.Reporter(),
        )
    )

    assert settled.engine_limits == "mlx:large-v3 was unavailable"
    assert checks.resolved == [], "the dead engine must not go on to decode"
    assert checks.warnings and "mlx:large-v3" in checks.warnings[0]


def test_a_cross_engine_is_not_probed_when_the_answer_cannot_matter() -> None:
    """With no glossary the prompt-pinning answer changes nothing about this run — but the
    probe starts an mlx worker, and a blip in it dropped a perfectly usable cross-check and
    keyed the run as degraded. The primary backend was already gated this way; this is the
    same gate for the engines the user asked for by name."""
    from stt_cli import dictionary, pipeline
    from stt_cli._errors import EngineError

    class Primary:
        name = "whispercpp"

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return True

        async def can_pin_prompt(self) -> bool:
            raise AssertionError("no glossary is carried — nobody should be asked")

    class Flaky:
        name = "mlx"

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

        async def can_pin_prompt(self) -> bool:
            raise EngineError(what="the probe timed out", why="", how="")

    checks = variants.CrossChecks(resolved=[("mlx:large-v3", Flaky(), "large-v3")])
    settled, _ = asyncio.run(
        pipeline._settle_engine_limits(
            Settings(variant_models=["mlx:large-v3"]),
            dictionary.Dictionary(),
            Primary(),
            checks,
            pipeline.Reporter(),
        )
    )

    assert settled.engine_limits == ""
    assert len(checks.resolved) == 1, "a cross-check must not be lost to an irrelevant probe"


def test_what_the_engines_could_not_do_is_written_into_the_transcript() -> None:
    """`engine_limits` keys the run, which is what makes an upgrade re-transcribe. But the
    stored transcript is read again months later — `stt archive show` re-renders it — and
    without this it gives no hint that the glossary reached only the first window. The
    reporter said it once, live, to a terminal that is long gone."""
    from stt_cli.models import EngineInfo, MediaInfo, Transcript

    def _blank() -> Transcript:
        media = MediaInfo(path="a.wav", sha256="a" * 64, size_bytes=1, duration=1.0)
        return Transcript(media=media, engine=EngineInfo(backend="whispercpp", model="large-v3"))

    transcript = _blank()
    limits = ["mlx:large, v3 was unavailable", "whispercpp cannot pin the glossary"]
    pipeline._record_engine_limits(transcript, limits)
    assert transcript.warnings == limits, "the lines are carried, never re-split out of prose"

    unlimited = _blank()
    pipeline._record_engine_limits(unlimited, [])
    assert unlimited.warnings == []

    # ...and the joined form exists only to key the run, which is why the round trip that
    # would mangle a model name with a comma in it does not happen anywhere.
    assert pipeline._joined(limits) == ", ".join(sorted(limits))


def test_a_direct_enrich_caller_gets_the_model_check_too() -> None:
    """The fix for "the cross-check engine resolves but its model file is missing" lived in
    the pipeline's own call site, so `enrich`'s fallback — documented as what a direct caller
    gets — still resolved engines without it and reproduced the silent no-cross-check run."""
    import inspect

    assert "usable_cross_backends" in inspect.getsource(variants.enrich)
    assert inspect.iscoroutinefunction(variants.usable_cross_backends)


def test_the_budget_the_glossary_buys_is_settled_before_the_key(monkeypatch) -> None:
    """With a dictionary, whisper.cpp decodes `--context off` with the same `-mc 64` as
    `--context short` — the same command line, the same words — and the two were stored under
    different keys, so asking for the mode that had already been used re-transcribed the
    whole recording. What the decoder will really do belongs in the settling pass."""
    from stt_cli import dictionary, pipeline

    class Whispercpp:
        name = "whispercpp"

        def __init__(self, pins: bool = True) -> None:
            self._pins = pins

        def honours_context_budget(self) -> bool:
            return True

        def pinning_the_prompt_costs_context(self) -> bool:
            return True

        async def can_pin_prompt(self) -> bool:
            return self._pins

    terms = dictionary.Dictionary([dictionary.Term(term="ConLoca")])

    def settle(engine: object, **kwargs: object) -> Settings:
        settled, _ = asyncio.run(
            pipeline._settle_engine_limits(
                Settings(dict_digest=terms.digest(), **kwargs),  # type: ignore[arg-type]
                terms,
                engine,
                variants.CrossChecks(),
                pipeline.Reporter(),
            )
        )
        return settled

    assert settle(Whispercpp()).context == "short"
    assert archive.fingerprint(settle(Whispercpp())) == archive.fingerprint(
        settle(Whispercpp(), context="short")
    ), "one decode, one key"

    # Every condition is load-bearing, and each of these would collapse two runs that really
    # do decode differently. A binary that cannot pin never gets the prompt at `off` at all;
    # the comparison pass makes `off` and `short` differ in what they are compared against.
    assert settle(Whispercpp(pins=False)).context == "off"
    assert settle(Whispercpp(), context_compare="auto").context == "off"
    assert settle(Whispercpp(), dict_bias=False).context == "off"

    # ...and an engine that pins for free buys nothing, so it stays where it was asked to be.
    class Mlx(Whispercpp):
        name = "mlx"

        def pinning_the_prompt_costs_context(self) -> bool:
            return False

    assert settle(Mlx()).context == "off"


def test_the_bought_budget_is_the_one_whispercpp_actually_buys() -> None:
    """The pipeline names a mode; whisper.cpp buys a number. They are written down in two
    files and the cache key is only honest while they agree."""
    from stt_cli import pipeline
    from stt_cli.backends.whispercpp import PROMPT_CONTEXT

    assert CONTEXT_TOKENS[pipeline.BOUGHT_CONTEXT] == PROMPT_CONTEXT


def test_a_cross_check_that_was_dropped_does_not_cut_clips_for_nothing() -> None:
    """`cross_models` is what was asked for; `cross` is what survived resolution. Reading
    only the request meant a run whose one cross-check turned out to be unusable still cut a
    clip per shaky segment — sixty ffmpeg calls to feed nothing, and an ffmpeg failure there
    could take down a transcription that was already finished and correct."""

    def plan(**kwargs: object) -> variants.VariantPlan:
        base: dict[str, object] = {
            "extra_decodes": 0,
            "cross_models": ["mlx:large-v3"],
            "confidence_floor": 0.55,
        }
        return variants.VariantPlan(**{**base, **kwargs})  # type: ignore[arg-type]

    assert not plan(cross=[]).wanted, "nothing resolved, so there is nothing to decode"
    assert plan(cross=None).wanted, "unresolved: `enrich` resolves its own and may find one"
    assert plan(cross=[("mlx:large-v3", object(), "large-v3")]).wanted
    assert plan(extra_decodes=1, cross=[]).wanted, "the temperature pass still runs"


def test_the_cli_offers_exactly_the_modes_that_exist() -> None:
    """The argparse choices and the real vocabularies are written down in two files. Add a
    budget to `CONTEXT_TOKENS` and a hand-edited config.json would accept it while the CLI
    rejected it — with the whole suite green, because nothing tied the two together."""
    from stt_cli.commands import transcribe

    parser = transcribe.build_parser()
    choices = {
        action.dest: tuple(action.choices)
        for action in parser._actions
        if action.choices and action.dest in {"context", "context_compare"}
    }

    assert choices["context"] == tuple(CONTEXT_TOKENS)
    assert choices["context_compare"] == tuple(pipeline.COMPARE_MODES)


def test_both_engines_answer_every_capability_the_pipeline_asks_about() -> None:
    """The capability trio is structural — a Protocol, no inheritance — and by its own
    docstring an engine that omits one raises AttributeError on the cache-miss path, which
    is the path quick testing never takes. Rename a method and update one backend and the
    suite stays green until somebody with the other engine hits a cache miss."""
    from stt_cli.backends.base import Backend
    from stt_cli.backends.mlx import MlxBackend
    from stt_cli.backends.whispercpp import WhisperCppBackend

    asked = ("can_pin_prompt", "honours_context_budget", "pinning_the_prompt_costs_context")
    for name in asked:
        assert hasattr(Backend, name), f"{name} is not part of the protocol any more"
        for engine in (MlxBackend, WhisperCppBackend):
            assert callable(getattr(engine, name, None)), f"{engine.__name__} cannot answer {name}"


def test_the_mlx_full_budget_is_the_one_the_table_names() -> None:
    """`FULL_CONTEXT` says in one file what `CONTEXT_TOKENS["full"]` says in another. Every
    other pair like it in this change has a drift test; this one did not."""
    from stt_cli.backends.mlx import FULL_CONTEXT

    assert CONTEXT_TOKENS["full"] == FULL_CONTEXT


def test_every_settings_field_has_a_deliberate_place_in_the_config() -> None:
    """`configurable()` is a denylist, so a field added later is a user setting by default —
    including one the pipeline computes, which `config set` would then refuse as unknown.
    Pinning the two sets makes adding a field a decision rather than an accident."""
    from stt_cli.config import Settings, configurable

    derived = {"dict_digest", "engine_limits", "context_compare_chosen"}
    per_run = {"output", "recorded_at"}
    assert set(Settings.__dataclass_fields__) == configurable() | derived | per_run


def test_an_explicit_context_compare_in_the_config_file_survives_fix(tmp_path, monkeypatch) -> None:
    """The tri-state has two entry points and only the CLI one was pinned. Somebody who
    wrote `context_compare: off` into their config said what they wanted just as plainly as
    somebody who typed it, and `--fix` must not turn it back on."""
    import json

    from stt_cli import config

    monkeypatch.setenv("STT_HOME", str(tmp_path))
    config.ensure_dirs()
    config.config_path().write_text(json.dumps({"context_compare": "off"}), "utf-8")

    chosen = config.load_settings()
    assert chosen.context_compare_chosen is True
    assert pipeline._compare_mode(replace(chosen, fix=True)) == "off"

    # ...and a config that says nothing about it leaves the implication free to apply.
    config.config_path().write_text(json.dumps({"backend": "whispercpp"}), "utf-8")
    silent = config.load_settings()
    assert silent.context_compare_chosen is False
    assert pipeline._compare_mode(replace(silent, fix=True)) == "auto"
