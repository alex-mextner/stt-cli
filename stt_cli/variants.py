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
from dataclasses import dataclass
from pathlib import Path

from . import media, registry
from ._errors import EngineError
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

    @property
    def wanted(self) -> bool:
        return self.extra_decodes > 0 or bool(self.cross_models)


def plan_from_settings(settings: object) -> VariantPlan:
    """Read the variant knobs off a Settings object, applying the LLM-implies-variants rule.

    When the LLM correction pass is enabled, variants and confidence are gathered whether or
    not the user asked to *see* them: correcting a transcript without knowing which parts
    are shaky, and without alternatives to choose between, is guesswork dressed up as help.
    """
    extra = int(getattr(settings, "variants", 0))
    cross = list(getattr(settings, "variant_models", []) or [])
    if getattr(settings, "fix", False) and extra == 0 and not cross:
        extra = 1
    return VariantPlan(
        extra_decodes=extra,
        cross_models=cross,
        confidence_floor=float(getattr(settings, "confidence_floor", 0.55)),
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
    cross = _cross_backends(plan.cross_models, warnings)
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


def _cross_backends(names: list[str], warnings: list[str]) -> list[tuple[str, Backend, str]]:
    """Resolve each ``engine:model`` (or bare model) cross-check into a usable backend."""
    resolved: list[tuple[str, Backend, str]] = []
    for name in names:
        engine, _, model = name.rpartition(":")
        model = model or name
        try:
            spec = registry.get(model)
            engine = engine or next(iter(spec.engine_ids))
            backend = create(engine)
            status = backend.availability()
            if not status.ok:
                warnings.append(f"cross-check {name} skipped: {status.detail}")
                continue
            resolved.append((f"{engine}:{model}", backend, model))
        except Exception as exc:  # a dead cross-check must never fail the whole run
            warnings.append(f"cross-check {name} skipped: {exc}")
    return resolved


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
                clip, engine, engine_model, language, temperature, label, threads
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
