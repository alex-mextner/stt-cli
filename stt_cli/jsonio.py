"""jsonio — the boundary where somebody else's bytes become our typed data.

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

THE OTHER KIND OF OUTSIDE
    The files the USER is invited to edit — `config.json`, `dictionary.json`, the
    hallucination list — arrive through the same door and break in more ways than a
    program's output does, because a person edits them in an editor that has opinions about
    encodings. `read_json` and `read_lines` are that door: one open, the descriptor asked
    what it really is, a size bound, and every failure turned into the diagnosed error the
    user can act on. `hf.py` and `commands/dict_cmd.py` each grew their own copy of this
    before it was worth naming; those two guard files with their own extra rules and are
    left alone, but nothing new should hand-roll a third.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Final

from ._errors import UsageError

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


MIB = 1024 * 1024
KIB = 1024


def in_units(size: int) -> str:
    """A byte count in the unit it is actually in.

    `size // MIB` reads fine for a four-megabyte dictionary and turns a two-hundred-kilobyte
    list of phrases into "larger than 0 MiB", which is not a sentence anybody can act on.
    """
    return f"{size // MIB} MiB" if size >= MIB else f"{size // KIB} KiB"


class _Absent:
    """There is no file at that path — which is not the same as a file holding `null`.

    A sentinel rather than `None`, because JSON has a `null` of its own and `read_json` has
    to be able to tell the two apart. It could not, once: a `dictionary.json` containing the
    four characters `null` decoded to `None`, was read as "there is no dictionary", and every
    run after that quietly decoded with no terminology at all. Every other malformed shape —
    `[]`, `"foo"`, a bad encoding — was diagnosed; that one shape was not.
    """

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT: Final = _Absent()


def read_json(path: Path, *, how: str, limit: int, too_big: str) -> Any:
    """Read a JSON file a person is allowed to edit. `ABSENT` when there is no such file.

    `limit` is a byte count and `too_big` says why that limit is what it is, because the
    answer differs per file and the user deserves the real reason rather than a number.
    """
    text = read_text(path, how=how, limit=limit, too_big=too_big)
    if text is None:
        return ABSENT
    try:
        return json.loads(text)
    except ValueError as exc:
        # `ValueError`, not `JSONDecodeError`, because the parser refuses valid JSON in more
        # ways than one. A number with five thousand digits is perfectly good JSON and raises
        # a plain `ValueError` from Python's own limit on converting one to an int — not a
        # decode error, not an `OSError`, and so a traceback out of an ordinary run.
        # `JSONDecodeError` is a `ValueError`, so the ordinary case is still covered.
        raise UsageError(what=f"could not read {path}", why=str(exc), how=how) from exc
    except RecursionError as exc:
        # Valid JSON, nested a couple of thousand deep, is neither a decode error nor an
        # OSError — it is the parser running out of stack, and it escaped the same way.
        raise UsageError(
            what=f"could not read {path}",
            why="the JSON is nested too deeply to parse",
            how=how,
        ) from exc


def read_lines(path: Path, *, how: str, limit: int, too_big: str) -> list[str] | None:
    """The same door, for the one user-editable file that is not JSON."""
    text = read_text(path, how=how, limit=limit, too_big=too_big)
    return None if text is None else text.splitlines()


def read_text(path: Path, *, how: str, limit: int, too_big: str) -> str | None:
    """Open once, ask the descriptor what it is, and bound what comes back.

    Three things go wrong here and each one used to reach the user as a traceback or worse.
    An editor that saved the file as Latin-1 raises `UnicodeDecodeError`, which is not an
    `OSError`. Checking the path and then opening it are two moments, and a regular file
    replaced by a FIFO in between makes the open wait for a writer that never comes — a hang
    with no message, on a file read by every single run. And a file that is not a dictionary
    at all gets parsed in full before anything notices, so the bound is checked on the
    descriptor that was actually opened rather than on whatever the name pointed at.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise UsageError(what=f"could not read {path}", why=str(exc), how=how) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                return None
            content = handle.read(limit + 1)
            # Both the size the descriptor reported and the bytes that actually arrived. The
            # first is a snapshot: another process appending a megabyte between the `fstat`
            # and the read left a file over the limit being parsed as if it were under it,
            # because reading exactly `limit` bytes can never notice that there were more.
            if info.st_size > limit or len(content) > limit:
                raise UsageError(
                    what=f"{path} is larger than {in_units(limit)}",
                    why=too_big,
                    how=how,
                )
            return content.decode("utf-8")
    except OSError as exc:
        raise UsageError(what=f"could not read {path}", why=str(exc), how=how) from exc
    except UnicodeDecodeError as exc:
        raise UsageError(what=f"could not read {path}", why=str(exc), how=how) from exc
