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
from . import chunks, cleaning, config, formats, media, postprocess, vad, variants
from ._errors import SttError
from .backends import DecodeRequest, resolve
from .backends.base import Backend
from .config import Settings
from .models import EngineInfo, MediaInfo, Segment, Transcript
from .timestamps import Stamper


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

    fingerprint = archive_mod.fingerprint(settings)
    if settings.cache:
        cached = store.find(info.sha256, fingerprint)
        if cached is not None:
            return await _from_cache(cached, audio, info, settings, reporter, store, fingerprint)

    with tempfile.TemporaryDirectory(prefix="stt-run-") as tmp:
        transcript = await _produce(audio, info, settings, Path(tmp), reporter, store)

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


async def _produce(
    source: Path,
    info: MediaInfo,
    settings: Settings,
    workdir: Path,
    reporter: Reporter,
    store: archive_mod.Archive,
) -> Transcript:
    """Everything between "we have a file" and "we have a transcript"."""
    wav = workdir / "audio.wav"
    reporter.step("normalizing audio to 16 kHz mono")
    await media.to_engine_wav(source, wav)

    backend = resolve(settings.backend, whispercpp_root=settings.whispercpp_root)
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

    transcript.segments = await _decode_spans(wav, spans, backend, settings, workdir, reporter)
    reporter.step(f"decoded {len(transcript.segments)} segment(s)")

    _clean(transcript, settings, reporter)
    await _add_variants(transcript, wav, backend, settings, reporter)
    await _add_speakers(transcript, wav, settings, reporter)
    await _correct(transcript, settings, reporter)
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

    out: list[Segment] = []
    for chunk in built:
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
            )
        )
        out.extend(_remap(produced, chunk))
        chunk.path.unlink(missing_ok=True)
    out.sort(key=lambda s: s.start)
    return out


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
    transcript: Transcript, wav: Path, backend: Backend, settings: Settings, reporter: Reporter
) -> None:
    plan = variants.plan_from_settings(settings)
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
    transcript.warnings.extend(warnings)
    for warning in warnings:
        reporter.detail(warning)


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


async def _correct(transcript: Transcript, settings: Settings, reporter: Reporter) -> None:
    if not settings.fix:
        return
    reporter.step("correcting the transcript with an LLM")
    report = await postprocess.correct(
        transcript, tool_name=settings.fix_with, language=transcript.language
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
