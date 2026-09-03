"""pipeline — the whole transcription, in the order the stages have to happen.

THE ORDER IS THE DESIGN
    Voice-activity detection comes first, because everything downstream is cheaper and more
    accurate when the model never sees the silence. Cleaning comes next, before the second
    opinions: a segment that is going to be dropped as an invented subtitle credit should not
    first be re-decoded three times, and the language model should never be asked to "fix"
    text that ought not to exist. Variants then run over what survived, and only over the
    parts the decoder was unsure of. Diarization joins in before correction so the correction
    pass can see who was speaking. Rendering is last and is the only stage cheap enough to
    redo, which is exactly why the archive stores everything up to that point.

WHAT A RUN COSTS, AND WHAT IT NEVER COSTS TWICE
    A cache hit answers from the archive without touching the GPU. The lookup key covers the
    audio's content and the settings that change the words — not the output format and not
    the timestamp mode — so asking the same recording for subtitles after asking for text
    re-renders in milliseconds. A summary or speaker labels are additions rather than
    changes, so requesting one against an already-transcribed recording runs only that pass.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import archive as archive_mod
from . import chunks, cleaning, config, dictionary, formats, media, postprocess, vad, variants
from ._errors import SttError, unknown_item
from .backends import DecodeRequest, resolve
from .backends.base import Backend, context_tokens
from .config import Settings
from .models import EngineInfo, MediaInfo, Segment, Transcript
from .timestamps import Stamper

# What --context-compare accepts. `auto` re-decodes only the chunks that look damaged.
COMPARE_MODES = ("off", "auto", "always")


@dataclass(slots=True)
class Reporter:
    """Progress to stderr. Quiet by default in scripts, chatty when a human is watching."""

    verbose: bool = False
    quiet: bool = False

    def step(self, message: str) -> None:
        if not self.quiet:
            import sys

            print(f"stt: {message}", file=sys.stderr, flush=True)

    def detail(self, message: str) -> None:
        if self.verbose and not self.quiet:
            import sys

            print(f"stt:   {message}", file=sys.stderr, flush=True)


@dataclass(slots=True)
class RunResult:
    """One finished transcription and everything worth telling the user about it."""

    transcript: Transcript
    run_id: str
    cached: bool = False
    notes: list[str] = field(default_factory=list)


async def transcribe(
    source: Path, settings: Settings, *, reporter: Reporter, store: archive_mod.Archive
) -> RunResult:
    """Transcribe one file end to end, answering from the archive when nothing has changed."""
    settings, terms = _resolve(settings)
    audio, archived_sha = await resolve_source(source, store, reporter)
    info = await media.inspect(audio, recorded_at=settings.recorded_at)
    if archived_sha is not None:
        # The audio we are reading is the archive's re-encode, so hashing it would give the
        # hash of the Opus copy rather than of the recording — a guaranteed cache MISS, a
        # full re-transcription, and a second archive entry keyed by the wrong identity. The
        # index already knows what this recording's hash is; use it, and keep the original
        # path so the run stays attributed to the file the user asked for.
        info = replace(info, sha256=archived_sha, path=str(source))
    reporter.step(
        f"{source.name}: {info.duration / 60:.1f} min, {info.codec} "
        f"{info.sample_rate} Hz{' (+video)' if info.has_video else ''}"
    )

    # The archive is asked BEFORE any engine is touched. What is stored describes the run
    # that produced it, not what this machine can do today: a transcript made months ago is
    # still that transcript after whisper.cpp is removed, downgraded or broken, and asking
    # the engine first would turn "answer instantly from disk" into "fail to start". It also
    # keeps the hot path free of the mlx probe, which starts a worker process.
    fingerprint = archive_mod.fingerprint(settings)
    if settings.cache:
        cached = store.find(info.sha256, fingerprint)
        if cached is not None:
            return await _from_cache(cached, audio, info, settings, reporter, store, fingerprint)

    # Only now, with a decode actually in prospect, is it worth asking the engines what they
    # can do — and a shortfall makes a second, different key. A machine that stored a
    # degraded run finds it here; a machine that has since been upgraded does not, which is
    # exactly right, because the upgrade is what changes the words.
    backend = resolve(settings.backend, whispercpp_root=settings.whispercpp_root)
    checks = await _variant_backends(settings)
    settled, limits = await _settle_engine_limits(settings, terms, backend, checks, reporter)
    if settled != settings:
        settings, fingerprint = settled, archive_mod.fingerprint(settled)
        if settings.cache:
            cached = store.find(info.sha256, fingerprint)
            if cached is not None:
                return await _from_cache(
                    cached, audio, info, settings, reporter, store, fingerprint
                )

    with tempfile.TemporaryDirectory(prefix="stt-run-") as tmp:
        transcript = await _produce(
            audio,
            info,
            settings,
            Path(tmp),
            reporter,
            store,
            terms,
            backend,
            checks,
            limits,
        )

    record = store.save(transcript, fingerprint)
    return RunResult(transcript, record.run_id, notes=list(transcript.warnings))


async def resolve_source(
    path: Path, store: archive_mod.Archive, reporter: Reporter
) -> tuple[Path, str | None]:
    """The audio to read, plus the recording's identity when that audio is a stand-in.

    Recordings get renamed, moved to an external drive, or deleted once "the transcript is
    done". The archive keeps its own compressed copy of everything it has transcribed, so a
    re-run with different options still works afterwards: the original path is looked up in
    the index and its stored audio stands in.

    The second element is the ORIGINAL recording's content hash, from the index — returned
    only when a stand-in is being used. It matters because the archive is keyed by that hash,
    and the stand-in is a re-encode with a hash of its own; recomputing it would miss the
    cache and file the run under an identity nothing else refers to.
    """
    if path.exists():
        return path, None
    found = store.media_for_source(path)
    if found is None:
        from ._errors import MissingTargetError

        raise MissingTargetError(
            what=f"no such file: {path}",
            why="the path does not exist and no archived copy of it was found",
            how="check the path, or run `stt archive ls` to see what has been transcribed",
        )
    stored, media_sha = found
    reporter.step(f"source is gone — using the archived copy ({stored.name})")
    return stored, media_sha


async def _from_cache(
    cached: archive_mod.RunRecord,
    audio: Path,
    info: MediaInfo,
    settings: Settings,
    reporter: Reporter,
    store: archive_mod.Archive,
    fingerprint: str,
) -> RunResult:
    """Answer from the archive, running only the enrichments this run asks for and lacks.

    A summary and speaker labels ADD to a transcript rather than change its words, so they
    are not part of the cache key. That means ``stt rec.m4a`` today and ``stt rec.m4a
    --summary`` tomorrow costs one summary call, not another pass over the audio.
    """
    transcript = store.load(cached.run_id)
    stored_media = transcript.media
    missing = _missing_enrichments(transcript, settings)

    if missing:
        reporter.step(f"cache hit ({cached.run_id}) — adding {', '.join(missing)}")
        with tempfile.TemporaryDirectory(prefix="stt-enrich-") as tmp:
            if "speakers" in missing:
                wav = Path(tmp) / "audio.wav"
                await media.to_engine_wav(audio, wav)
                await _add_speakers(transcript, wav, settings, reporter)
            if "summary" in missing:
                await _summarize(transcript, settings, reporter)
        # Saved with the media block the run was MADE with. A one-off --recorded-at is a
        # rendering choice for this invocation, not a correction to the archived record, and
        # writing it back would silently re-anchor every future render of the run.
        store.save(transcript, fingerprint, created_at=cached.created_at)
    else:
        reporter.step(f"cache hit ({cached.run_id}) — re-rendering from the archive")

    # ...but the CURRENT invocation still renders with what it knows best: an explicit
    # --recorded-at has to take effect now, without being persisted.
    transcript.media = replace(info, sha256=stored_media.sha256)
    return RunResult(transcript, cached.run_id, cached=True, notes=list(transcript.warnings))


def _missing_enrichments(transcript: Transcript, settings: Settings) -> list[str]:
    """Which requested extras this archived transcript does not already carry.

    Speakers are compared against the *configuration* that produced them, not merely against
    "does any segment have a label". Asking for ``--speakers 3`` after a run with
    ``--speakers 2`` must re-run diarization rather than hand back the two-speaker answer.
    """
    missing = []
    if settings.summary and transcript.summary is None:
        missing.append("summary")
    if settings.diarize and transcript.engine.extra.get("diarize") != _diarize_key(settings):
        missing.append("speakers")
    return missing


def _diarize_key(settings: Settings) -> str:
    """The diarization configuration, as stored on a transcript that has speaker labels."""
    return f"speakers={settings.speakers or 'auto'}"


def _resolve(settings: Settings) -> tuple[Settings, dictionary.Dictionary]:
    """Settle everything the cache key depends on BEFORE the key is computed.

    Returns the dictionary it read along with the settings, and every later stage uses THAT
    copy. Re-reading the file in each consumer would let `stt dict add` in another terminal
    change the glossary halfway through an hour-long run: the transcript would then be
    stored under a fingerprint describing a dictionary that did not produce it, and asking
    for the old dictionary again would serve the new dictionary's words out of the cache.

    Two things here are implied rather than typed by the user, and both change the words in
    the transcript, so both have to be settled before ``archive.fingerprint`` sees the
    settings — otherwise a run cached under the implied value's absence is served as if it
    had it. ``--fix`` implying the context comparison is exactly that case: a transcript
    produced by an older ``--fix`` run carries none of the variants the comparison adds, and
    with ``context_compare`` still reading ``"off"`` its fingerprint would be unchanged and
    the stale transcript would be handed straight back.

    The dictionary digest is here rather than inside the fingerprint for a plainer reason: a
    hash function that reads a file is a hash function nobody can test.
    """
    terms = dictionary.load() if settings.dictionary else dictionary.Dictionary()
    # Fail on a bad `context` here rather than after ffmpeg, voice-activity detection and
    # chunking have already run: the CLI constrains it with `choices`, but a hand-edited
    # config.json does not go through argparse.
    context_tokens(settings.context)
    dictionary.validate_similarity(settings.dict_similarity)
    settled = replace(settings, dict_digest=terms.digest(), context_compare=_compare_mode(settings))
    return _without_dead_dictionary_knobs(settled, terms), terms


def _without_dead_dictionary_knobs(settings: Settings, terms: dictionary.Dictionary) -> Settings:
    """Normalize the settings that only mean something while there IS a dictionary.

    With no terms, `--dict-similarity 0.7` and `--no-dict-bias` cannot change a single word —
    but they are non-default, so they enter the fingerprint and make `stt rec.m4a --no-dict
    --dict-similarity 0.7` miss the run `stt rec.m4a --no-dict` just stored. The transcripts
    are byte-identical and the miss costs a full re-decode of the recording.
    """
    if terms:
        return settings
    defaults = Settings()
    return replace(settings, dict_bias=defaults.dict_bias, dict_similarity=defaults.dict_similarity)


async def _settle_engine_limits(
    settings: Settings,
    terms: dictionary.Dictionary,
    backend: Backend,
    checks: variants.CrossChecks,
    reporter: Reporter,
) -> tuple[Settings, list[str]]:
    """Reconcile what was ASKED for with what the installed engines can actually do.

    The rule this obeys is the pipeline's own: anything that changes the words is settled
    before ``archive.fingerprint`` sees the settings. An engine that quietly cannot honour a
    setting produces a transcript that does not match its own cache key, and every later run
    of the same command is answered from that key.
    """
    settings = _settle_context_budget(settings, backend, reporter)
    settings = await _settle_bought_budget(settings, terms, backend, reporter)
    return await _settle_engine_shortfalls(settings, terms, backend, checks, reporter)


async def _settle_bought_budget(
    settings: Settings, terms: dictionary.Dictionary, backend: Backend, reporter: Reporter
) -> Settings:
    """Name the budget the glossary has already bought, before the key is computed.

    On whisper.cpp a pinned glossary needs a nonzero `-mc` to have any effect, so a run with
    a dictionary and `--context off` decodes with the same budget as `--context short` — the
    same command line, the same words — and the two were nonetheless stored under different
    keys, so asking for the mode that was already effectively used re-transcribed the whole
    recording. This is the same rule `_settle_context_budget` applies to mlx, applied to the
    other engine: what the decoder will really do is settled here, not discovered later
    inside `_argv`.

    Every condition below is load-bearing. Without a carried glossary nothing buys anything.
    Where pinning is free (mlx) no budget is bought. On a binary that cannot pin, `off`
    really does decode without the prompt while `short` decodes with it — the two are
    different runs and must keep different keys. And with the comparison pass on, the second
    decode is the opposite of the first, so `off` and `short` differ in the pass they are
    compared against even when the primary pass is identical.
    """
    if settings.context != "off" or settings.context_compare != "off":
        return settings
    if not (_glossary(settings, terms) and settings.dict_bias):
        return settings
    if not backend.pinning_the_prompt_costs_context() or not await backend.can_pin_prompt():
        return settings
    reporter.step(
        f"glossary carried: {backend.name} decodes this run with the {BOUGHT_CONTEXT} context "
        f"budget, so it is stored as {BOUGHT_CONTEXT} rather than off"
    )
    return replace(settings, context=BOUGHT_CONTEXT)


# The mode an engine that cannot pin a prompt for free falls back to. Named here rather
# than imported from whisper.cpp, because the pipeline must not know one engine's constant —
# `test_the_bought_budget_is_the_one_whispercpp_actually_buys` pins the two together.
BOUGHT_CONTEXT = "short"


def _joined(shortfalls: list[str]) -> str:
    """The shortfalls as one field, for the cache key ONLY.

    Deliberately one-way. The lines embed a user-typed `--variant-model` name, so a name
    containing the separator would not survive a split — and recovering structure by
    re-parsing prose is the thing `CrossChecks` was built to avoid. Everything that needs
    the lines gets the list; only the fingerprint gets the string.
    """
    return ", ".join(sorted(set(shortfalls)))


def _settle_context_budget(settings: Settings, backend: Backend, reporter: Reporter) -> Settings:
    """Name the context mode the engine will really decode with.

    mlx takes a boolean, not a token count, so `--context short` decodes exactly like
    `--context full` there. Left alone, the two produce different keys for one transcript:
    the `short` run stores full-context words under a key promising a 64-token budget, and
    a later `--context full` re-decodes the recording to produce the same text again.
    """
    if backend.honours_context_budget() or context_tokens(settings.context) == 0:
        return settings
    if settings.context == "full":
        return settings
    reporter.step(
        f"{backend.name} has no context budget: --context {settings.context} decodes as full, "
        "and the run is stored as full"
    )
    return replace(settings, context="full")


async def _settle_engine_shortfalls(
    settings: Settings,
    terms: dictionary.Dictionary,
    backend: Backend,
    checks: variants.CrossChecks,
    reporter: Reporter,
) -> tuple[Settings, list[str]]:
    """Record, in the cache key, everything the installed engines cannot actually do.

    An engine that quietly falls short produces a transcript that does not match its own key,
    and every later run of the same command is answered from that key — so upgrading the
    engine changes nothing and the compromised transcript is served forever. Writing the
    shortfall into the key instead makes the upgrade a cache miss, which is the honest
    outcome: the recording is transcribed again, properly, once.

    "The engines" is plural on purpose. A `--variant-model` engine decodes readings that are
    archived and fed to the LLM correction pass, so its limits are part of the run's identity
    exactly as the primary engine's are.
    """
    glossary = bool(_glossary(settings, terms) and settings.dict_bias)
    budget = context_tokens(settings.context)

    # A cross-check that could not be resolved at all is the largest shortfall of the lot:
    # the transcript is stored WITHOUT the readings the user asked for. Unrecorded, it looks
    # identical to a run where they were gathered — so installing the missing engine and
    # running the same command would hit that cache entry and never gather them.
    shortfalls = [f"{name} was unavailable" for name in checks.skipped]

    # The PRIMARY engine is load-bearing: if it cannot answer, nothing can be decoded and
    # the error belongs to the user. Its probe is asked only when a glossary is carried,
    # because that is the only run whose words the answer changes.
    if glossary and not await backend.can_pin_prompt():
        shortfalls.append(f"{backend.name} cannot pin the glossary")
    # Cannot fire today, and kept deliberately. `_settle_context_budget` runs first and
    # renames `short` to `full` on a primary engine that has no budget, which is the better
    # answer where it is available: the run is keyed as what it actually decodes rather than
    # as a shortfall. This branch is the same fact spelled the other way, and it stays as the
    # backstop for anything that reaches here without that normalization — a cross engine
    # does exactly that, three lines down, because nothing renames a cross engine's context.
    if budget and settings.context != "full" and not backend.honours_context_budget():
        shortfalls.append(f"{backend.name} has no context budget")

    for label, engine, model in list(checks.resolved):
        try:
            if glossary and not await engine.can_pin_prompt():
                shortfalls.append(f"{label} cannot pin the glossary")
        except SttError as exc:
            # A cross engine that cannot even answer is dropped, not fatal: `--variant-model`
            # asks for a second opinion, and losing one must never cost the transcription.
            checks.resolved.remove((label, engine, model))
            checks.skipped.append(label)
            checks.warnings.append(f"cross-check {label} skipped: {exc.what}")
            shortfalls.append(f"{label} was unavailable")
            continue
        # A budget only matters when one was asked for. `off` means "carry nothing", which
        # every engine can do, and `full` is the maximum, which is what these engines give.
        if budget and settings.context != "full" and not engine.honours_context_budget():
            shortfalls.append(f"{label} has no context budget")
    if not shortfalls:
        return settings, []
    lines = sorted(set(shortfalls))
    for line in lines:
        reporter.step(f"{line} — the run is stored under its own cache key, so upgrading re-runs")
    return replace(settings, engine_limits=_joined(lines)), lines


def _record_engine_limits(transcript: Transcript, limits: list[str]) -> None:
    """Write what the engines could not do into the transcript itself, not only the key.

    ``engine_limits`` already keys the run, so an upgraded machine re-transcribes. But the
    stored transcript is read again months later — `stt archive show` re-renders it — and
    without this the file gives no hint that the glossary reached only the first window. The
    reporter said it once, live, to a terminal that is long gone. Not routed through
    ``_report``: the reporter has already printed these lines at settling time.

    Takes the LIST the settling pass produced, never the joined field — see ``_joined``.
    """
    transcript.warnings.extend(limits)


def _report(transcript: Transcript, warnings: list[str], reporter: Reporter) -> None:
    """Put warnings where a person and the archived transcript will both see them."""
    transcript.warnings.extend(warnings)
    for warning in warnings:
        reporter.detail(warning)


async def _variant_backends(settings: Settings) -> variants.CrossChecks:
    """Resolve the engines `--variant-model` will decode with, ONCE.

    The same objects are probed before the cache key is computed and then used for the
    actual decoding, which is what makes each backend's "probed lazily, once per run" cache
    true: resolving a second set later would run every probe again, and the capability the
    key was built from would have come from an object that never decoded anything.

    A dead cross-check is skipped rather than fatal, so the reasons come back alongside the
    engines: `--variant-model` is the one thing the user explicitly asked for here, and a
    silently dropped cross-check is the failure they would never notice.
    """
    names = list(settings.variant_models or [])
    if not names:
        return variants.CrossChecks()
    return await variants.usable_cross_backends(names)


async def _produce(
    source: Path,
    info: MediaInfo,
    settings: Settings,
    workdir: Path,
    reporter: Reporter,
    store: archive_mod.Archive,
    terms: dictionary.Dictionary,
    backend: Backend,
    checks: variants.CrossChecks,
    limits: list[str],
) -> Transcript:
    """Everything between "we have a file" and "we have a transcript"."""
    wav = workdir / "audio.wav"
    reporter.step("normalizing audio to 16 kHz mono")
    await media.to_engine_wav(source, wav)

    reporter.step(f"engine: {backend.name}, model: {settings.model}")
    await backend.ensure_model(settings.model)

    spans = await _detect_speech(wav, info, settings, backend, reporter)
    transcript = Transcript(
        media=info,
        engine=EngineInfo(
            backend=backend.name,
            model=settings.model,
            language=settings.language,
            vad=spans.method,
        ),
        language=settings.language,
        speech_spans=list(spans.spans),
    )

    transcript.segments = await _decode_spans(
        wav, spans, backend, settings, workdir, reporter, glossary=_glossary(settings, terms)
    )
    reporter.step(f"decoded {len(transcript.segments)} segment(s)")
    _record_engine_limits(transcript, limits)

    _clean(transcript, settings, reporter)
    # Cleaning can create the agreement too: a repetition loop in the primary is collapsed
    # here, while the comparison pass was already collapsed at merge time. The two then say
    # the same thing, and the leftover variant is a disagreement that no longer exists. So
    # this runs whether or not there is a dictionary.
    stale = variants.drop_agreeing(transcript.segments, ("context",))
    if stale:
        reporter.detail(f"{stale} context variant(s) no longer disagree — dropped")
    _apply_dictionary(transcript, settings, terms, reporter)
    await _add_variants(transcript, wav, backend, settings, terms, checks, reporter)
    await _add_speakers(transcript, wav, settings, reporter)
    await _correct(transcript, settings, terms, reporter)
    await _summarize(transcript, settings, reporter)

    if settings.keep_media:
        reporter.step("archiving a compressed copy of the audio")
        await store.store_media(source, info.sha256)
    return transcript


async def _detect_speech(
    wav: Path, info: MediaInfo, settings: Settings, backend: Backend, reporter: Reporter
) -> vad.VadResult:
    """Find the speech, using the engine's own detector when it provides one."""
    detector = backend.vad_provider()
    result = await vad.detect(
        wav,
        info.duration,
        mode=settings.vad,
        threshold=settings.vad_threshold,
        min_silence_ms=settings.vad_min_silence_ms,
        speech_pad_ms=settings.vad_speech_pad_ms,
        min_speech_ms=settings.vad_min_speech_ms,
        silero_binary=detector.binary if detector else None,
        silero_model=detector.model if detector else None,
    )
    reporter.step(result.describe())
    return result


async def _decode_spans(
    wav: Path,
    spans: vad.VadResult,
    backend: Backend,
    settings: Settings,
    workdir: Path,
    reporter: Reporter,
    glossary: str = "",
) -> list[Segment]:
    """Transcribe the speech — spliced into a few long chunks — and map it back onto the file.

    Decoding each detected span separately would reload the model hundreds of times for one
    hour of audio and hand the model one-second fragments with no context. Instead the speech
    is spliced into a handful of long chunks (see :mod:`stt_cli.chunks`), which keeps the
    silence out of the model's ears without paying either of those costs. Chunks run one at a
    time: the engine already saturates the GPU, so decoding two at once makes both slower.
    """
    reporter.step("splicing speech into decode chunks")
    built = await chunks.build(wav, spans.spans, workdir)
    reporter.step(f"decoding {len(built)} chunk(s) of speech")
    carried = context_tokens(settings.context)
    if glossary and carried == 0 and await backend.can_pin_prompt():
        # Said out loud, because it is a real departure from what --context off promises:
        # the glossary is pinned into every decode window, so the decoder is no longer
        # reading each window with nothing carried over. HOW that is bought differs by
        # engine — whisper.cpp needs a small context budget granted (with -mc 0 it ignores
        # the initial prompt entirely), mlx re-prepends the prompt with no budget at all —
        # so the engine says what it costs and this line does not pretend to know.
        reporter.step(
            "glossary pinned into the model's prompt for every window "
            "(--no-dict-bias keeps --context off literal)"
        )
    # A chunk WAV is 16-bit PCM and a long recording makes several of them, so holding them
    # all at once is a real disk cost. They are only needed twice when a comparison pass will
    # re-decode them; with --context-compare off — the default — each one is finished the
    # moment it has been decoded and is dropped there, restoring the old peak of one chunk.
    keep = settings.context_compare != "off"
    try:
        segments = await _decode_pass(
            built, backend, settings, carried, reporter, glossary, drop_when_done=not keep
        )
        await _compare_context(built, backend, settings, carried, segments, reporter, glossary)
    finally:
        for chunk in built:
            chunk.path.unlink(missing_ok=True)
    return segments


async def _decode_pass(
    built: list[chunks.Chunk],
    backend: Backend,
    settings: Settings,
    max_context: int,
    reporter: Reporter,
    glossary: str = "",
    *,
    only: set[int] | None = None,
    drop_when_done: bool = False,
) -> list[Segment]:
    """One decoding of every chunk — or only the listed ones — on the recording's timeline.

    ``drop_when_done`` deletes each chunk's audio as soon as it has been read, for the runs
    where nothing will want it again. The caller still unlinks everything afterwards, so an
    exception mid-pass leaves nothing behind either way.
    """
    out: list[Segment] = []
    for chunk in built:
        if only is not None and chunk.index not in only:
            continue
        reporter.detail(
            f"chunk {chunk.index}/{len(built)}: {chunk.duration / 60:.1f} min of speech "
            f"from {chunk.source_start / 60:.1f}-{chunk.source_end / 60:.1f} min"
        )
        produced = await backend.decode(
            DecodeRequest(
                wav=chunk.path,
                model=settings.model,
                language=settings.language,
                temperature=0.0,
                threads=settings.threads,
                max_context=max_context,
                initial_prompt=glossary or None,
                carry_prompt=bool(glossary),
            )
        )
        out.extend(_remap(produced, chunk))
        if drop_when_done:
            chunk.path.unlink(missing_ok=True)
    out.sort(key=lambda s: s.start)
    return out


async def _compare_context(
    built: list[chunks.Chunk],
    backend: Backend,
    settings: Settings,
    carried: int,
    segments: list[Segment],
    reporter: Reporter,
    glossary: str = "",
) -> None:
    """Decode a second time with the opposite context setting and keep the disagreements.

    Feeding the decoder its own previous output is what keeps casing, punctuation and proper
    nouns consistent across window boundaries — and it is also what lets a repetition loop
    feed itself, because a garbage phrase in the prompt makes more of the same garbage the
    likeliest continuation. Rather than picking one side of that trade-off for everybody,
    this runs both and hands the places where they disagree downstream as variants, so the
    LLM correction pass decides per segment with both readings and both confidences in front
    of it. The first pass stays primary, so a loop can never win by default.
    """
    mode = settings.context_compare
    if mode == "off" or not segments:
        return
    wanted = (
        {chunk.index for chunk in built}
        if mode == "always"
        else _suspect_chunks(built, segments, settings)
    )
    if not wanted:
        reporter.step("context comparison: nothing looked damaged, second pass skipped")
        return
    other = context_tokens("full" if carried == 0 else "off")
    # Whether the zero-context side can keep the glossary is the ENGINE's answer, not the
    # pipeline's. Where pinning the prompt forces a budget (whisper.cpp), carrying it into a
    # pass meant to have none would quietly give it one: both passes would then carry
    # context, the comparison would stop being a comparison, and the variant would be
    # labelled with a number nobody used. Where pinning is free (mlx), dropping it is the
    # worse mistake — the comparison comes back with the spelling the glossary exists to
    # prevent, and that is handed to the LLM as evidence against the primary reading.
    keeps_glossary = other > 0 or not backend.pinning_the_prompt_costs_context()
    carried_glossary = glossary if keeps_glossary else ""
    reporter.step(
        f"context comparison: re-decoding {len(wanted)} of {len(built)} chunk(s) "
        f"with max-context {other}"
    )
    alternate = await _decode_pass(
        built, backend, settings, other, reporter, carried_glossary, only=wanted
    )
    cleaning.collapse_only(alternate, settings.max_repeats)
    added = variants.merge_parallel(segments, alternate, source=f"asr:context={other}")
    reporter.step(f"context comparison: {added} segment(s) read differently")


def _compare_mode(settings: Settings) -> str:
    """``--context-compare``, with the same LLM-implies-evidence rule the variants use.

    Asking an LLM to fix a transcript while hiding from it that a second decoding read a
    sentence differently withholds the one piece of evidence it could actually act on. So
    ``--fix`` turns the comparison on, at ``auto``, which keeps the cost bounded by
    re-decoding only the chunks that look damaged instead of the whole recording.

    It is an implication, not an override: somebody who typed ``--context-compare off``
    said what they wanted, and a flag that cannot be turned off is not a flag.
    """
    chosen = settings.context_compare
    if chosen not in COMPARE_MODES:
        raise unknown_item(
            "context-compare mode", chosen, list(COMPARE_MODES), plural="context-compare modes"
        )
    if chosen == "off" and settings.fix and not settings.context_compare_chosen:
        return "auto"
    return chosen


def _suspect_chunks(
    built: list[chunks.Chunk], segments: list[Segment], settings: Settings
) -> set[int]:
    """Which chunks look like decoding without context cost them something.

    Two symptoms, both cheap, both specific to this failure. A chunk whose first segment
    opens in lower case is the signature of a decoder that did not know a sentence had
    started, which is exactly what carried context supplies. And a chunk with a lot of shaky
    segments is one where a second reading has something to offer whatever the cause.
    """
    suspect: set[int] = set()
    for chunk in built:
        inside = [s for s in segments if chunk.source_start <= s.start < chunk.source_end]
        if not inside:
            continue
        # `or 1.0` would be a bug here: 0.0 is falsy, so the LEAST confident segment there is
        # would score as fully confident and never trigger the second pass.
        shaky = sum(
            1
            for s in inside
            if (1.0 if s.confidence is None else s.confidence) < settings.confidence_floor
        )
        if _opens_lowercase(inside[0].text) or shaky >= max(1, len(inside) // 5):
            suspect.add(chunk.index)
    return suspect


def _opens_lowercase(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()


def _remap(segments: list[Segment], chunk: chunks.Chunk) -> list[Segment]:
    """Rewrite chunk-relative timings back onto the original recording's timeline."""
    for segment in segments:
        segment.start = chunk.to_source(segment.start)
        segment.end = max(segment.start, chunk.to_source(segment.end))
        segment.words = [
            replace(word, start=chunk.to_source(word.start), end=chunk.to_source(word.end))
            for word in segment.words
        ]
    return segments


def _glossary(settings: Settings, terms: dictionary.Dictionary) -> str:
    """The dictionary as the speech model's initial prompt, when it is wanted and non-empty.

    This is the only one of the dictionary's three uses that can fix a word the model got
    wrong acoustically; the other two work on text it has already committed to.
    """
    if not (settings.dictionary and settings.dict_bias):
        return ""
    return terms.prompt()


def _apply_dictionary(
    transcript: Transcript,
    settings: Settings,
    terms: dictionary.Dictionary,
    reporter: Reporter,
) -> None:
    """Correct known misspellings, and flag what merely sounds like a known term."""
    if not (settings.dictionary and terms):
        return
    report = dictionary.apply(transcript.segments, terms, similarity=settings.dict_similarity)
    reporter.step(f"dictionary: {report.summary()}")
    for line in report.detail():
        reporter.detail(line)
    _correct_variants(transcript, terms, ("context",))


def _clean(transcript: Transcript, settings: Settings, reporter: Reporter) -> None:
    kept, report = cleaning.clean(
        transcript.segments,
        speech_spans=transcript.speech_spans,
        home=config.app_home(),
        apply=settings.clean,
        strict=settings.strict_clean,
        max_repeats=settings.max_repeats,
        confidence_floor=settings.confidence_floor,
    )
    transcript.segments = kept
    reporter.step(f"cleaning: {report.summary()}")
    for line in report.detail():
        reporter.detail(line.strip())
    if report.dropped or report.collapsed:
        transcript.warnings.append(f"cleaning: {report.summary()}")


async def _add_variants(
    transcript: Transcript,
    wav: Path,
    backend: Backend,
    settings: Settings,
    terms: dictionary.Dictionary,
    checks: variants.CrossChecks,
    reporter: Reporter,
) -> None:
    # Said even when nothing else runs: a cross-check the user asked for and did not get is
    # the whole reason they passed --variant-model. Resolution happens before the cache key
    # is computed now, so these no longer come back from `enrich` on their own.
    _report(transcript, checks.warnings, reporter)
    plan = variants.plan_from_settings(
        settings,
        max_context=context_tokens(settings.context),
        glossary=_glossary(settings, terms),
        cross=checks.resolved,
    )
    if not plan.wanted:
        return
    targets = variants.candidates(transcript.segments, plan)
    if not targets:
        reporter.step("no low-confidence segments — skipping the variant pass")
        return
    reporter.step(f"gathering alternative readings for {len(targets)} shaky segment(s)")
    warnings = await variants.enrich(
        transcript.segments,
        source_wav=wav,
        backend=backend,
        model=settings.model,
        language=settings.language,
        plan=plan,
        threads=settings.threads,
    )
    _report(transcript, warnings, reporter)
    _reconcile_variants_with_dictionary(transcript, settings, terms, reporter)


# The kinds `variants.enrich` attaches. They are decoded from the audio, so they say what
# the speech model says — including the misspellings the dictionary has already settled.
GATHERED_KINDS = ("temperature", "model")


def _reconcile_variants_with_dictionary(
    transcript: Transcript,
    settings: Settings,
    terms: dictionary.Dictionary,
    reporter: Reporter,
) -> None:
    del reporter
    if not (settings.dictionary and terms):
        return
    _correct_variants(transcript, terms, GATHERED_KINDS)


def _correct_variants(
    transcript: Transcript, terms: dictionary.Dictionary, kinds: tuple[str, ...]
) -> None:
    """Apply the recorded misspellings to raw decoder readings, then drop the duplicates.

    Every kind named here is decoder output, so each carries whatever the speech model heard
    — including the spellings the user has written down as wrong. Without this, a segment
    where "Vigma" became "Figma" gets "Vigma" handed straight back as an alternative reading,
    to the reader and to the LLM, about a word the user explicitly settled. An `aka` spelling
    is a fact, and a fact holds in a variant too.

    `primary` is never in `kinds`: that slot exists to hold what the speech model actually
    said, which is exactly the reading being corrected away from.
    """
    for segment in transcript.segments:
        for variant in segment.variants:
            if variant.kind in kinds:
                variant.text = dictionary.correct_text(variant.text, terms)
    variants.drop_agreeing(transcript.segments, kinds)


async def _add_speakers(
    transcript: Transcript, wav: Path, settings: Settings, reporter: Reporter
) -> None:
    if not settings.diarize:
        return
    from . import diarize as diarize_mod

    reporter.step("identifying speakers (pyannote)")
    turns = await diarize_mod.diarize(wav, speakers=settings.speakers)
    labelled = diarize_mod.attach(transcript.segments, turns)
    voices = len({t.speaker for t in turns})
    # Record WHICH diarization produced these labels, so a later run asking for a different
    # speaker count is a cache miss for this enrichment rather than a stale hit.
    transcript.engine.extra["diarize"] = _diarize_key(settings)
    reporter.step(f"speakers: {voices} voice(s) across {labelled} segment(s)")
    transcript.warnings.append(f"diarization found {voices} speaker(s)")


async def _correct(
    transcript: Transcript,
    settings: Settings,
    terms: dictionary.Dictionary,
    reporter: Reporter,
) -> None:
    if not settings.fix:
        return
    reporter.step("correcting the transcript with an LLM")
    report = await postprocess.correct(
        transcript,
        tool_name=settings.fix_with,
        language=transcript.language,
        glossary=terms.glossary() if settings.dictionary else None,
    )
    reporter.step(report.summary())
    transcript.warnings.append(report.summary())


async def _summarize(transcript: Transcript, settings: Settings, reporter: Reporter) -> None:
    if not settings.summary:
        return
    reporter.step("summarizing")
    summary = await postprocess.summarize(transcript, tool_name=settings.fix_with)
    if summary is None:
        transcript.warnings.append("the summary pass produced no usable output")
        reporter.step("summary failed — the transcript is unaffected")
        return
    transcript.summary = summary
    reporter.step(f"summary: {len(summary.sections)} section(s), {len(summary.actions)} action(s)")


# ── rendering ─────────────────────────────────────────────────────────────────
def render_options(settings: Settings) -> formats.RenderOptions:
    """What the reader should see, derived from what this run actually asked for.

    Speakers and summaries are enrichments stored on the transcript, which means an archived
    run can carry labels from a *previous* invocation that used ``--diarize``. Gating the
    renderers on the current request keeps that from leaking: ``stt rec.m4a -f srt`` produces
    plain subtitles even when the stored transcript happens to know who was talking.
    """
    return formats.RenderOptions(
        show_variants=settings.show_variants,
        show_flags=settings.show_flags,
        text_variant=settings.text_variant,
        show_speakers=settings.diarize,
        show_summary=settings.summary,
    )


def render_all(
    transcript: Transcript, settings: Settings, options: formats.RenderOptions
) -> dict[str, str]:
    """Produce every requested output format from one transcript."""
    stamper = Stamper(settings.timestamps, transcript.media, settings.timezone)
    wanted = formats.expand(settings.formats)
    return {name: formats.render(name, transcript, stamper, options) for name in wanted}


async def gather(paths: list[Path], settings: Settings, reporter: Reporter) -> list[RunResult]:
    """Transcribe several files in sequence, never letting one failure lose the others.

    Every diagnosed failure is caught, not a hand-picked couple of them: a batch that spends
    twenty minutes on the first recording must not throw that away because the second file
    turned out to be a text file. A single-file run still raises, so the exit code and the
    error message are the real ones.
    """
    results: list[RunResult] = []
    with archive_mod.Archive() as store:
        for path in paths:
            try:
                results.append(await transcribe(path, settings, reporter=reporter, store=store))
            except SttError as exc:
                if len(paths) == 1:
                    raise
                reporter.step(f"{path.name}: FAILED — {exc.what}")
    return results


def run(coro: object) -> object:
    """Entry point for the synchronous command layer."""
    return asyncio.run(coro)  # type: ignore[arg-type]
