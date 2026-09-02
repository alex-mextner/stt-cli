"""variants — get a second opinion on the parts the model was unsure about.

WHY NOT JUST TAKE THE FIRST ANSWER
    A greedy decode returns one reading and no indication of what else was in contention.
    When the decoder was confident that is fine. When it was not — a crosstalk moment, a
    proper noun, a switch between languages — the single answer it returns is frequently
    wrong, and nothing downstream can tell. Re-decoding just those moments turns "one guess"
    into "here are the candidates and how sure the model was of each", which is what both a
    human reader and the LLM correction pass need in order to choose.

TWO KINDS OF SECOND OPINION, AND WHY THE SECOND ONE MATTERS MORE
    Raising the sampling temperature and decoding again explores what else the *same* model
    considered plausible. Running a genuinely *different* model over the same audio is
    stronger evidence: two sizes of one model share their training data and tend to make the
    same mistake in the same place, so their agreement proves little, while agreement across
    architectures is real corroboration. ``--variant-model`` is for the second kind.

BOUNDED BY CONSTRUCTION
    Only low-confidence segments are re-decoded, only the worst ``MAX_SEGMENTS`` of them,
    and only a couple at a time. An hour of clean audio therefore costs nothing extra; an
    hour of terrible audio costs a bounded amount rather than an open-ended one.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import media, registry
from ._errors import EngineError, SttError
from .backends import DecodeRequest, create
from .backends.base import Backend
from .models import Segment, Variant

# Re-decoding is a full model pass over the span, so an unbounded count on a bad recording
# would dwarf the original transcription. The worst segments are the informative ones.
MAX_SEGMENTS = 60
# Temperatures to sample at, in order, when more than one extra reading is requested. 0.0 is
# already the primary, so these start where it stopped being deterministic.
TEMPERATURES = (0.4, 0.8, 1.0)
# Concurrent re-decodes. The GPU is the bottleneck; piling more on makes everything slower.
CONCURRENCY = 2
# A little audio either side, so the model hears the word boundaries rather than a clipped
# consonant — clipping alone can change what it decodes.
PAD_SECONDS = 0.35


@dataclass(slots=True)
class VariantPlan:
    """What second opinions to gather, resolved from settings once per run."""

    extra_decodes: int
    cross_models: list[str]
    confidence_floor: float
    # The decoding conditions of the pass being second-guessed. A variant decoded WITHOUT
    # the glossary and context the primary had is not a second opinion about the same
    # question: on `--context full --fix` it would come back missing a proper noun that only
    # the glossary made recoverable, and be handed to the LLM as evidence against it.
    max_context: int = 0
    glossary: str = ""
    # The cross-model engines, already resolved. The pipeline probes their capabilities
    # before it computes the cache key, and each backend caches that probe on itself — so
    # handing the SAME objects back here is what keeps "probed once per run" true. Left as
    # None, `enrich` resolves its own (which is what a direct caller gets).
    cross: list[tuple[str, Backend, str]] | None = None

    @property
    def wanted(self) -> bool:
        """Is there any decode left to do? Asked BEFORE a single clip is cut.

        `cross_models` is what was asked for; `cross` is what survived resolution. Reading
        only the request meant a run whose one cross-check turned out to be unusable still
        cut a clip per shaky segment — up to sixty ffmpeg calls to feed nothing — and an
        ffmpeg failure in that pass could then take down a transcription that was already
        finished and correct.
        """
        if self.extra_decodes > 0:
            return True
        return bool(self.cross) if self.cross is not None else bool(self.cross_models)


def plan_from_settings(
    settings: object,
    *,
    max_context: int = 0,
    glossary: str = "",
    cross: list[tuple[str, Backend, str]] | None = None,
) -> VariantPlan:
    """Read the variant knobs off a Settings object, applying the LLM-implies-variants rule.

    When the LLM correction pass is enabled, variants and confidence are gathered whether or
    not the user asked to *see* them: correcting a transcript without knowing which parts
    are shaky, and without alternatives to choose between, is guesswork dressed up as help.

    ``max_context`` and ``glossary`` are the conditions the primary pass decoded under, and
    the caller has to supply them: a second opinion taken under different conditions is a
    different question, not a second answer. ``cross`` lets the pipeline hand back the
    engine objects it already probed, so a probe really does happen once per run.
    """
    extra = int(getattr(settings, "variants", 0))
    cross_models = list(getattr(settings, "variant_models", []) or [])
    if getattr(settings, "fix", False) and extra == 0 and not cross_models:
        extra = 1
    return VariantPlan(
        extra_decodes=extra,
        cross_models=cross_models,
        confidence_floor=float(getattr(settings, "confidence_floor", 0.55)),
        max_context=max_context,
        glossary=glossary,
        cross=cross,
    )


def candidates(segments: list[Segment], plan: VariantPlan) -> list[Segment]:
    """The segments worth a second look: least confident first, capped."""
    shaky = [
        s
        for s in segments
        if s.text.strip() and (s.confidence is None or s.confidence < plan.confidence_floor)
    ]
    shaky.sort(key=lambda s: s.confidence if s.confidence is not None else 0.0)
    return shaky[:MAX_SEGMENTS]


async def enrich(
    segments: list[Segment],
    *,
    source_wav: Path,
    backend: Backend,
    model: str,
    language: str | None,
    plan: VariantPlan,
    threads: int = 0,
) -> list[str]:
    """Attach alternative readings to the shaky segments. Returns any warnings raised."""
    if not plan.wanted:
        return []
    targets = candidates(segments, plan)
    if not targets:
        return []

    warnings: list[str] = []
    if plan.cross is not None:
        cross = plan.cross
    else:
        # `usable_cross_backends`, not the bare resolver: a direct caller must get the
        # model-availability check too, or it reproduces the silent no-cross-check run the
        # pipeline was fixed for.
        checks = await usable_cross_backends(plan.cross_models)
        cross, warnings = checks.resolved, checks.warnings
    gate = asyncio.Semaphore(CONCURRENCY)
    with tempfile.TemporaryDirectory(prefix="stt-variants-") as tmp:
        tasks = [
            _for_segment(
                index,
                segment,
                Path(tmp),
                source_wav,
                backend,
                model,
                language,
                plan,
                cross,
                gate,
                threads,
            )
            for index, segment in enumerate(targets)
        ]
        await asyncio.gather(*tasks)
    total = sum(len(s.variants) for s in targets)
    if len(targets) == MAX_SEGMENTS:
        warnings.append(
            f"variant search stopped at the {MAX_SEGMENTS} least confident segments; "
            "more remain below the confidence floor"
        )
    if total:
        warnings.append(f"gathered {total} alternative reading(s) across {len(targets)} segment(s)")
    return warnings


@dataclass(slots=True)
class CrossChecks:
    """The result of resolving ``--variant-model``: what runs, what does not, and why.

    Deliberately NOT frozen. The pipeline narrows this record in place as it learns what the
    engines can actually do — a cross-check that fails its capability probe moves from
    ``resolved`` to ``skipped`` after the object has already been handed on — and the object
    the decode pass reads has to be the same one whose ``skipped`` list went into the cache
    key. A frozen record invites ``replace()``, which would leave the caller holding the wide
    version: the run would then decode with an engine the fingerprint records as unavailable.

    ``skipped`` is a list of NAMES, kept apart from ``warnings`` on purpose. It goes into the
    run's cache identity — a transcript made while an engine was missing must not share a key
    with one where it ran — and identity must never be recovered by re-parsing prose. The
    warning text carries an exception message, which varies between machines and releases;
    keying on it would churn the fingerprint and quietly stop the cache ever hitting.
    """

    resolved: list[tuple[str, Backend, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


async def usable_cross_backends(names: list[str]) -> CrossChecks:
    """Keep only the cross-checks whose MODEL is actually there, fetching it if it is not.

    Resolving the engine is not enough: `--variant-model whispercpp:large-v3` resolves fine
    whenever whisper-cli exists, and then every variant decode raises "model file missing",
    which the variant pass swallows one segment at a time. The result was a run with no
    cross-check, no warning, and — because nothing recorded the absence — the same cache key
    as a run where the cross-check had worked. Installing the model afterwards then hit that
    entry forever. The primary model is fetched the same way, one step later.
    """
    checks = cross_backends(names)
    usable: list[tuple[str, Backend, str]] = []
    for label, engine, model in checks.resolved:
        try:
            await engine.ensure_model(model)
        except SttError as exc:
            checks.skipped.append(label)
            checks.warnings.append(f"cross-check {label} skipped: {exc.what}")
            continue
        usable.append((label, engine, model))
    # Narrowed in place, like every other place that learns a cross-check is unusable: the
    # record that reaches the decode pass has to be the one whose `skipped` list was hashed.
    checks.resolved = usable
    return checks


def cross_backends(names: list[str]) -> CrossChecks:
    """Resolve each ``engine:model`` (or bare model) cross-check into a usable backend."""
    out = CrossChecks()
    for name in names:
        engine, _, model = name.rpartition(":")
        model = model or name
        try:
            spec = registry.get(model)
            engine = engine or next(iter(spec.engine_ids))
            backend = create(engine)
            status = backend.availability()
            if not status.ok:
                out.skipped.append(name)
                out.warnings.append(f"cross-check {name} skipped: {status.detail}")
                continue
            out.resolved.append((f"{engine}:{model}", backend, model))
        except Exception as exc:  # a dead cross-check must never fail the whole run
            out.skipped.append(name)
            out.warnings.append(f"cross-check {name} skipped: {exc}")
    return out


async def _for_segment(
    index: int,
    segment: Segment,
    tmp: Path,
    source_wav: Path,
    backend: Backend,
    model: str,
    language: str | None,
    plan: VariantPlan,
    cross: list[tuple[str, Backend, str]],
    gate: asyncio.Semaphore,
    threads: int,
) -> None:
    """Cut this segment's audio once, then decode it every way the plan asks for."""
    clip = tmp / f"seg{index:04d}.wav"
    start = max(0.0, segment.start - PAD_SECONDS)
    await media.to_engine_wav(source_wav, clip, start=start, end=segment.end + PAD_SECONDS)

    passes: list[tuple[str, Backend, str, float]] = [
        (f"{backend.name}:{model}@t{temp}", backend, model, temp)
        for temp in TEMPERATURES[: plan.extra_decodes]
    ]
    passes += [(label, engine, cross_model, 0.0) for label, engine, cross_model in cross]

    async with gate:
        for label, engine, engine_model, temperature in passes:
            variant = await _decode_one(
                clip, engine, engine_model, language, temperature, label, threads, plan
            )
            if variant and _is_new(variant, segment):
                segment.variants.append(variant)
    segment.variants.sort(key=lambda v: (v.confidence is None, -(v.confidence or 0.0)))


async def _decode_one(
    clip: Path,
    backend: Backend,
    model: str,
    language: str | None,
    temperature: float,
    label: str,
    threads: int,
    plan: VariantPlan,
) -> Variant | None:
    """One extra decoding pass. A failure here is never fatal — it just yields no variant."""
    try:
        produced = await backend.decode(
            DecodeRequest(
                wav=clip,
                model=model,
                language=language,
                temperature=temperature,
                threads=threads,
                max_context=plan.max_context,
                initial_prompt=plan.glossary or None,
                carry_prompt=bool(plan.glossary),
            )
        )
    except EngineError:
        return None
    text = " ".join(s.text.strip() for s in produced if s.text.strip()).strip()
    if not text:
        return None
    scores = [s.confidence for s in produced if s.confidence is not None]
    return Variant(
        text=text,
        source=label,
        kind="model" if "@t" not in label else "temperature",
        confidence=sum(scores) / len(scores) if scores else None,
    )


def _is_new(variant: Variant, segment: Segment) -> bool:
    """Drop a variant that says the same thing as the primary or an existing alternative."""
    seen = {_norm(segment.text), *(_norm(v.text) for v in segment.variants)}
    return _norm(variant.text) not in seen


def _norm(text: str) -> str:
    import re

    return re.sub(r"\W+", " ", text.lower()).strip()


# ── the parallel-pass variant ─────────────────────────────────────────────────────────
# A whole second decoding of the same audio under a different setting, joined back onto the
# primary one. Used by `--context-compare`, where the two passes differ only in how much of
# its own output the decoder was allowed to carry.


# How much of an alternate segment has to fall inside a primary one before it counts as
# that segment's reading rather than the neighbour's. Strictly more than half, so a reading
# split exactly down the middle between two primaries belongs to neither.
MIN_OVERLAP_RATIO = 0.5
MIN_SEGMENT_SECONDS = 0.2

# Majority overlap alone is not enough: an alternate can sit mostly inside one primary and
# still swallow a large part of the next one, and its text is attached whole. Two passes
# never draw the same boundaries, so a little spill is normal and has to be tolerated — but
# once an alternate covers this much of a primary it was not assigned to, the words of that
# primary are inside it, and reporting the whole thing as the alternative reading of the
# neighbour invents a disagreement that nobody said. Such a reading is dropped instead.
MAX_NEIGHBOUR_SHARE = 0.4


def merge_parallel(primary: list[Segment], alternate: list[Segment], *, source: str) -> int:
    """Attach the second pass's reading of each segment, where the two passes disagree.

    Two independent decodings never agree about where a segment starts and ends, so the join
    has to be by overlap rather than by index — matching them positionally pairs the wrong
    sentences the first time the passes split differently, which is immediately.

    The primary reading stays primary. That is the safety property: the comparison pass is
    the one carrying context and therefore the one that can run away into a repetition loop,
    and a loop that only ever appears as an alternative cannot become the transcript.
    """
    if not alternate:
        return 0
    buckets = _assign(primary, alternate)
    added = 0
    for index, segment in enumerate(primary):
        parts = buckets[index]
        text = " ".join(part.text.strip() for part in parts if part.text.strip())
        if not text or _same_words(text, segment.text):
            continue
        segment.variants.append(
            Variant(text=text, source=source, kind="context", confidence=_pooled(parts))
        )
        added += 1
    return added


def _assign(primary: list[Segment], alternate: list[Segment]) -> dict[int, list[Segment]]:
    """Give every alternate segment to the primary segment it mostly sits inside.

    Assigning from the alternate side, rather than picking one best match per primary, is
    what keeps a counterpart whole. The second pass regularly splits one sentence in two;
    taking only the better half would report a disagreement that is really nothing but a
    difference in where the passes drew their boundaries, and would show the reader half a
    sentence as if it were the alternative reading of a whole one.
    """
    buckets: dict[int, list[Segment]] = {index: [] for index in range(len(primary))}
    for other in alternate:
        span = max(MIN_SEGMENT_SECONDS, other.end - other.start)
        overlaps = [_overlap(segment, other) for segment in primary]
        if not overlaps:
            continue
        best = max(range(len(overlaps)), key=lambda index: overlaps[index])
        if overlaps[best] <= MIN_OVERLAP_RATIO * span:
            continue
        if _swallows_a_neighbour(primary, overlaps, best):
            continue
        buckets[best].append(other)
    return buckets


def _overlap(segment: Segment, other: Segment) -> float:
    return max(0.0, min(segment.end, other.end) - max(segment.start, other.start))


def _swallows_a_neighbour(primary: list[Segment], overlaps: list[float], best: int) -> bool:
    """Does this alternate cover so much of another primary that it is speaking for it too?"""
    for index, segment in enumerate(primary):
        if index == best:
            continue
        duration = max(MIN_SEGMENT_SECONDS, segment.end - segment.start)
        if overlaps[index] > MAX_NEIGHBOUR_SHARE * duration:
            return True
    return False


def _pooled(parts: list[Segment]) -> float | None:
    """One confidence for a counterpart assembled from several segments.

    Weighted by duration, because a two-second clause the model was sure of says more about
    the reading than a stray half-second interjection it was not.
    """
    weighted = [
        (max(0.0, p.end - p.start), p.confidence) for p in parts if p.confidence is not None
    ]
    total = sum(weight for weight, _ in weighted)
    if not weighted or total <= 0:
        return None
    return sum(weight * value for weight, value in weighted) / total


def drop_agreeing(segments: list[Segment], kinds: tuple[str, ...] | str) -> int:
    """Remove variants of these kinds that now say the same thing as the segment does.

    Second opinions are gathered against the speech model's wording and the dictionary
    corrects spellings afterwards, so a reading that had "Figma" where the first pass had
    "Vigma" is a disagreement when it is attached and none at all once the dictionary has
    fixed the first. Left in place it shows the reader a variant identical to the text and
    hands the LLM a disagreement to adjudicate that no longer exists.

    ``primary`` is never a candidate: that slot exists to hold what the speech model
    actually said, which is precisely the reading the dictionary just moved away from.
    """
    if isinstance(kinds, str):
        # `kind not in "context"` is substring membership, so a bare string quietly compares
        # the wrong way and a value like "temperature,model" would drop kinds nobody named.
        raise TypeError("drop_agreeing takes a tuple of kinds, not a single string")
    removed = 0
    for segment in segments:
        keep = [
            variant
            for variant in segment.variants
            if variant.kind == "primary"
            or variant.kind not in kinds
            or not _same_words(variant.text, segment.text)
        ]
        removed += len(segment.variants) - len(keep)
        segment.variants = keep
    return removed


def _same_words(left: str, right: str) -> bool:
    """Do these two readings say the same thing?

    Casing and punctuation are exactly what carrying context changes, and a variant that
    differs only in a comma is noise in the output and a distraction in the LLM's prompt.
    So the comparison is on words alone — a difference has to be a different word to count.
    """
    return _words(left) == _words(right)


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.casefold())
