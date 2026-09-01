"""jsonio — the boundary where another program's JSON becomes our typed data.

WHY THIS EXISTS RATHER THAN CASTS
    ffprobe, whisper.cpp, mlx-whisper and the agent CLIs all hand us JSON. Inside the
    process that JSON is genuinely ``Any``: nothing at the type level knows what an external
    binary decided to emit today. Two ways to deal with that. Sprinkle ``cast`` and
    ``# type: ignore`` at each access — which silences the checker without checking
    anything, so a changed upstream shape becomes an ``AttributeError`` deep in a parser.
    Or coerce once, at the boundary, with functions that actually look at the value.

    These are the second. Each returns the requested type or a safe default, so a malformed
    or unexpected payload degrades to "that field was missing" instead of a crash, and every
    caller downstream can be fully typed with nothing suppressed.
"""

from __future__ import annotations

from typing import Any

# A decoded JSON object. `Any` is correct and honest here: this IS untyped data from
# another process. It is confined to this module's signatures — callers get real types.
JsonDict = dict[str, Any]


def as_dict(value: Any) -> JsonDict:
    """The value as an object, or an empty one when it is anything else."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """The value as an array, or an empty one when it is anything else."""
    return value if isinstance(value, list) else []


def as_dicts(value: Any) -> list[JsonDict]:
    """The value as an array of objects, silently skipping entries that are not objects."""
    return [item for item in as_list(value) if isinstance(item, dict)]


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def as_float(value: Any, default: float = 0.0) -> float:
    """A number, or ``default`` when the field is absent or not numeric."""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return default


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(float(str(value)))
    except ValueError:
        return default


def as_opt_float(value: Any) -> float | None:
    """A number, or ``None`` — for fields whose absence is meaningful (a missing confidence
    is not the same as a confidence of zero)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
