"""config — where stt-cli keeps its state, and the knobs a run is configured with.

TWO LAYERS, ONE SHAPE
    :class:`Settings` is the single struct the pipeline reads. It is built by loading the
    user's ``config.json`` over the built-in defaults, then applying command-line flags
    over that. A flag always wins; an absent flag never clobbers a stored preference.

WHERE THINGS LIVE
    Everything durable goes under one root — by default the macOS application-support
    directory, overridable with ``STT_HOME`` (tests and throwaway runs use that). Nothing
    is ever written to ``/tmp``: the whole point of the archive is that a transcript you
    paid GPU time for is still there next month. Only genuinely scratch intermediates go
    to a temporary directory, and they are deleted in the same run that made them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .dictionary import DEFAULT_SIMILARITY

APP_NAME = "stt-cli"

# Formats the renderers can produce. `all` expands to this list minus the JSON-ish debug
# ones a human never asks for by name.
FORMATS = ("txt", "md", "json", "srt", "vtt", "csv", "tsv", "speakers", "summary")
DEFAULT_ALL = ("txt", "md", "json", "srt", "vtt")

TIMESTAMP_MODES = ("none", "relative", "absolute")


def app_home() -> Path:
    """The one durable root for models, archive and config."""
    override = os.environ.get("STT_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_NAME


def archive_dir() -> Path:
    return app_home() / "archive"


def media_dir() -> Path:
    return archive_dir() / "media"


def runs_dir() -> Path:
    return archive_dir() / "runs"


def models_dir() -> Path:
    return app_home() / "models"


def config_path() -> Path:
    return app_home() / "config.json"


def index_path() -> Path:
    return archive_dir() / "index.sqlite"


def ensure_dirs() -> None:
    for path in (app_home(), archive_dir(), media_dir(), runs_dir(), models_dir()):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class Settings:
    """Everything one transcription run needs to know."""

    # engine
    backend: str = "auto"
    model: str = "large-v3-turbo"
    language: str | None = None
    threads: int = 0  # 0 -> let the engine choose
    whispercpp_root: str | None = None

    # How much of its own previous output the decoder is fed back as context. Carrying it
    # keeps casing, punctuation and proper nouns consistent across the model's 30-second
    # windows; it is also exactly what lets a repetition loop feed itself, because a garbage
    # phrase in the prompt makes the same garbage the likeliest continuation. `off` is the
    # safe default and `context_compare` is how you get the quality back without the risk.
    context: str = "off"  # off | short | full
    context_compare: str = "off"
    # Did somebody actually CHOOSE the mode above, or is it just the built-in default?
    # `--fix` turns the comparison on when nobody chose, because an LLM asked to correct a
    # transcript with the disagreements hidden from it is guessing. It must not override a
    # person who typed `--context-compare off`, which is why the two are not the same fact.
    # Never configurable and never in the cache key: what gets keyed is the mode it settles.
    context_compare_chosen: bool = False

    # voice activity detection
    vad: str = "auto"  # auto | silero | ffmpeg | none
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 400
    vad_speech_pad_ms: int = 200
    vad_min_speech_ms: int = 250

    # cleaning
    clean: bool = True
    strict_clean: bool = False
    max_repeats: int = 3
    confidence_floor: float = 0.55

    # terminology (see dictionary.py: prompt biasing, exact fixes, phonetic flags)
    dictionary: bool = True
    dict_bias: bool = True  # feed the glossary to the speech model, not just to the LLM
    # One source for the number: `dictionary.apply`/`screen` take it as their default
    # argument, so a value written down twice would let a direct caller and the
    # pipeline disagree about what the threshold is.
    dict_similarity: float = DEFAULT_SIMILARITY
    # Filled in by the pipeline from the dictionary's content, never by the user: it exists
    # so that editing a term invalidates exactly the cached runs it would have changed.
    dict_digest: str = ""
    # Also filled in by the pipeline, from the ENGINES: a sorted, comma-joined list of what
    # the installed engines could not actually do for this run ("whispercpp cannot pin the
    # glossary"). Such a run decodes differently from one where they could, so it must not
    # share that run's identity — otherwise upgrading the engine changes nothing and the
    # compromised transcript is served forever. Empty means nothing fell short.
    engine_limits: str = ""

    # variants
    variants: int = 0  # extra decodings per low-confidence segment
    variant_models: list[str] = field(default_factory=list)
    show_variants: bool = False

    # correction / summary
    fix: bool = False
    fix_with: str = "auto"  # auto | codex | claude | opencode | <binary>
    summary: bool = False
    summary_style: str = "structured"

    # diarization
    diarize: bool = False
    speakers: int | None = None

    # output
    show_flags: bool = False
    text_variant: str = "fixed"  # fixed | raw | both
    formats: list[str] = field(default_factory=lambda: ["txt"])
    timestamps: str = "none"
    timezone: str | None = None
    output: str | None = None
    # Per-run only: an explicit recording start time for absolute timestamps. It describes
    # one particular file rather than a preference, so it is never stored in config.json.
    recorded_at: datetime | None = None

    # archive
    cache: bool = True
    keep_media: bool = True

    def merged(self, **overrides: Any) -> Settings:
        """Return a copy with the non-``None`` overrides applied (flags beat config)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


# Only these keys may come from the on-disk config. An unknown key is a typo the user
# should hear about rather than a silently ignored preference.
def configurable() -> set[str]:
    """The settings a person may read and write. `config list`/`get` ask this too.

    Listing every dataclass field advertised `dict_digest` and `engine_limits` — the run's
    own identity, computed by the pipeline — as if they were preferences, and `config set`
    then refused them as unknown. One contract, asked by all three commands.
    """
    return set(_CONFIGURABLE)


_CONFIGURABLE = {
    f
    for f in Settings.__dataclass_fields__
    if f not in {"output", "recorded_at", "dict_digest", "engine_limits", "context_compare_chosen"}
}


def load_settings() -> Settings:
    """Built-in defaults with the user's ``config.json`` laid over them."""
    from ._errors import UsageError

    path = config_path()
    settings = Settings()
    if not path.is_file():
        return settings
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(
            what=f"could not read {path}",
            why=str(exc),
            how="fix the JSON, or delete the file to fall back to defaults",
        ) from exc
    unknown = sorted(set(raw) - _CONFIGURABLE)
    if unknown:
        raise UsageError(
            what=f"unknown key(s) in {path}: {', '.join(unknown)}",
            why="stt only understands the settings listed by `stt config list`",
            how="remove the key, or check `stt config list` for its real name",
        )
    _validate(raw)
    settled = replace(settings, **raw)
    if "context_compare" in raw:
        settled = replace(settled, context_compare_chosen=True)
    return settled


# The declared type of each setting, taken from the dataclass annotation rather than from
# whatever value happens to be there. Reading the type off the CURRENT value cannot work for
# a field whose default is None (`language`, `timezone`): there is nothing to read, so
# `stt config set language 1` would happily store the integer 1.
_TYPES: dict[str, str] = {
    name: str(field.type) for name, field in Settings.__dataclass_fields__.items()
}

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def coerce(key: str, raw: str) -> Any:
    """Turn a command-line string into the type ``key`` is declared as, or refuse clearly."""
    from ._errors import UsageError

    declared = _TYPES.get(key, "str")
    if "bool" in declared:
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        raise UsageError(
            what=f"{key} is a true/false setting",
            why=f"{raw!r} is not a boolean",
            how=f"use `stt config set {key} true` or `... false`",
        )
    if "list" in declared:
        return [part.strip() for part in raw.split(",") if part.strip()]
    for name, caster in (("int", int), ("float", float)):
        if name in declared:
            try:
                return caster(raw)
            except ValueError as exc:
                raise UsageError(
                    what=f"{key} is a {name} setting",
                    why=f"{raw!r} is not a {name}",
                    how=f"pass a number, e.g. `stt config set {key} 2`",
                ) from exc
    return raw


def _validate(raw: dict[str, Any]) -> None:
    """Reject a stored value whose type cannot work, before it reaches the pipeline.

    A hand-edited (or older) config.json can hold ``"variants": "abc"``. Left alone that
    string flows into a numeric comparison and into the cache fingerprint, so every run is
    subtly wrong in a way whose cause is nowhere near the symptom.
    """
    from ._errors import UsageError

    for key, value in raw.items():
        declared = _TYPES.get(key, "str")
        ok = (
            isinstance(value, bool)
            if "bool" in declared
            else isinstance(value, list)
            if "list" in declared
            else isinstance(value, int) and not isinstance(value, bool)
            if "int" in declared
            else isinstance(value, (int, float)) and not isinstance(value, bool)
            if "float" in declared
            else value is None or isinstance(value, str)
        )
        if not ok:
            raise UsageError(
                what=f"{key} in {config_path()} has the wrong type",
                why=f"expected {declared}, found {type(value).__name__} ({value!r})",
                how=f"fix it, or run `stt config set {key} <value>` to write a valid one",
            )


def save_setting(key: str, value: Any) -> None:
    """Persist one preference into ``config.json``, leaving the rest untouched."""
    from ._errors import unknown_item

    if key not in _CONFIGURABLE:
        raise unknown_item("setting", key, sorted(_CONFIGURABLE))
    ensure_dirs()
    path = config_path()
    raw: dict[str, Any] = {}
    if path.is_file():
        # Read through the same guarded loader path rather than a bare json.loads: a corrupt
        # config must produce the diagnosed error, not a raw JSONDecodeError traceback.
        try:
            raw = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            from ._errors import UsageError

            raise UsageError(
                what=f"could not read {path}",
                why=str(exc),
                how="fix the JSON, or delete the file to fall back to defaults",
            ) from exc
    raw[key] = value
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", "utf-8")
