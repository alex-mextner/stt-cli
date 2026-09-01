"""registry — the pool of models the tool knows how to run, under names a human types.

WHY A REGISTRY AND NOT RAW MODEL IDS
    The same model is called ``ggml-large-v3-turbo.bin`` by whisper.cpp and
    ``mlx-community/whisper-large-v3-turbo`` by MLX. Making the user learn both, and
    re-learn them when they switch engines, is pointless. Here one name — ``large-v3-turbo``
    — maps to whatever the chosen engine calls it, so ``--model`` means the same thing
    everywhere and ``--variant-model`` can name a genuinely *different* model to cross-check
    against.

    The size figures are not decoration: :mod:`stt_cli.resources` uses them to refuse a
    download that would fill the disk, and to warn before a model that will swap.

ADDING A MODEL
    Append an :class:`ModelSpec`. Nothing else in the codebase needs to change — engines
    look up their own id from the entry, and anything unknown to an engine is reported as
    "this model has no <engine> build" rather than failing deep inside a subprocess. That
    is the extension point for a future non-Whisper model that turns out to be better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIB = 1024**2


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """One speech model, and what each engine needs to load it."""

    name: str
    summary: str
    size_bytes: int
    # engine name -> the identifier that engine wants (a ggml filename, an HF repo id)
    engine_ids: dict[str, str] = field(default_factory=dict)
    languages: str = "multilingual"
    quality: int = 3  # 1..5, rough transcription quality for ranking suggestions
    speed: int = 3  # 1..5, higher is faster

    def id_for(self, engine: str) -> str | None:
        return self.engine_ids.get(engine)

    def row(self) -> str:
        langs = "multi" if self.languages == "multilingual" else self.languages
        engines = ",".join(sorted(self.engine_ids))
        return (
            f"{self.name:<22} {self.size_bytes / MIB:>6.0f}M  {langs:<8} "
            f"q{self.quality} s{self.speed}  {engines:<18} {self.summary}"
        )


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="large-v3-turbo",
        summary="best quality-per-second; the sane default for long recordings",
        size_bytes=1620 * MIB,
        engine_ids={
            "whispercpp": "ggml-large-v3-turbo.bin",
            "mlx": "mlx-community/whisper-large-v3-turbo",
        },
        quality=4,
        speed=5,
    ),
    ModelSpec(
        name="large-v3",
        summary="highest accuracy, roughly 4x slower than turbo",
        size_bytes=3100 * MIB,
        engine_ids={
            "whispercpp": "ggml-large-v3.bin",
            "mlx": "mlx-community/whisper-large-v3-mlx",
        },
        quality=5,
        speed=2,
    ),
    ModelSpec(
        name="large-v3-q5",
        summary="quantized large-v3: near-full accuracy at a third of the memory",
        size_bytes=1080 * MIB,
        engine_ids={"whispercpp": "ggml-large-v3-q5_0.bin"},
        quality=4,
        speed=3,
    ),
    ModelSpec(
        name="distil-large-v3",
        summary="distilled large-v3, English only, very fast",
        size_bytes=1500 * MIB,
        engine_ids={"mlx": "mlx-community/distil-whisper-large-v3"},
        languages="en",
        quality=4,
        speed=5,
    ),
    ModelSpec(
        name="medium",
        summary="a reasonable fallback when large will not fit",
        size_bytes=1530 * MIB,
        engine_ids={
            "whispercpp": "ggml-medium.bin",
            "mlx": "mlx-community/whisper-medium-mlx",
        },
        quality=3,
        speed=3,
    ),
    ModelSpec(
        name="small",
        summary="fast and small; noticeably weaker on accented or noisy speech",
        size_bytes=488 * MIB,
        engine_ids={
            "whispercpp": "ggml-small.bin",
            "mlx": "mlx-community/whisper-small-mlx",
        },
        quality=2,
        speed=4,
    ),
    ModelSpec(
        name="base",
        summary="draft quality; useful for a quick check, not for a real transcript",
        size_bytes=148 * MIB,
        engine_ids={
            "whispercpp": "ggml-base.bin",
            "mlx": "mlx-community/whisper-base-mlx",
        },
        quality=1,
        speed=5,
    ),
    ModelSpec(
        name="tiny",
        summary="smoke-test model; fine for verifying the pipeline runs",
        size_bytes=78 * MIB,
        engine_ids={
            "whispercpp": "ggml-tiny.bin",
            "mlx": "mlx-community/whisper-tiny-mlx",
        },
        quality=1,
        speed=5,
    ),
)

BY_NAME: dict[str, ModelSpec] = {m.name: m for m in MODELS}

DEFAULT_MODEL = "large-v3-turbo"
# When --variants asks for a second opinion, cross-check against a genuinely different
# model rather than a smaller copy of the same one: two sizes of the same weights tend to
# make the SAME mistake, so agreement between them proves nothing.
DEFAULT_CROSS_CHECK = "large-v3"


def get(name: str) -> ModelSpec:
    from ._errors import unknown_item

    spec = BY_NAME.get(name)
    if spec is None:
        raise unknown_item("model", name, sorted(BY_NAME))
    return spec


def for_engine(engine: str) -> list[ModelSpec]:
    return [m for m in MODELS if engine in m.engine_ids]


def require_for_engine(name: str, engine: str) -> tuple[ModelSpec, str]:
    """Resolve ``name`` to the identifier ``engine`` understands, or explain why it can't."""
    from ._errors import UsageError

    spec = get(name)
    engine_id = spec.id_for(engine)
    if engine_id is None:
        supported = ", ".join(m.name for m in for_engine(engine))
        raise UsageError(
            what=f"model {name!r} has no {engine} build",
            why=f"{name} is only published for: {', '.join(sorted(spec.engine_ids)) or 'nothing'}",
            how=f"pick one of the {engine} models: {supported}",
        )
    return spec, engine_id
