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

import fcntl
import json
import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .dictionary import DEFAULT_SIMILARITY
from .jsonio import ABSENT, read_json

# A settings file is a dozen keys. The bound is here for the same reason the dictionary's is:
# this file is read by every run, and anything can be sitting at that path.
MAX_CONFIG_BYTES = 1024 * 1024

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


CONFIG_FILENAME = "config.json"


def config_path() -> Path:
    return app_home() / CONFIG_FILENAME


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

    # live dictation
    # How loud a frame must be before `stt mic` treats it as speech, whatever the room is
    # doing. Zero means the detector's own default. Raise it in a room where stray noises get
    # typed; lower it if a quiet voice goes unheard. `stt mic --check` reports both numbers.
    mic_threshold: float = 0.0

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


def _settings_object(path: Path) -> dict[str, Any]:
    """The contents of `config.json`, or the reason it cannot be one.

    Both the reader and the writer go through here, and that is the point. When only the
    reader checked the shape, `stt config list` reported a `config.json` holding `[]` as
    broken while `stt config set` quietly replaced it with a fresh object — the same file,
    diagnosed by one command and destroyed by the other.
    """
    from ._errors import UsageError

    raw = read_json(
        path,
        how="fix the JSON, save it as UTF-8, or delete it for the defaults",
        limit=MAX_CONFIG_BYTES,
        too_big="a settings file is a handful of keys, not a document",
    )
    if raw is ABSENT:
        return {}
    if not isinstance(raw, dict):
        raise UsageError(
            what=f"{path} is not a settings file",
            why='expected an object of setting names and values, like {"language": "ru"}',
            how="write {} into it, or delete it to fall back to the defaults",
        )
    return raw


def stored() -> dict[str, Any]:
    """The settings actually written in `config.json`, so a caller can say which are stored."""
    return _settings_object(config_path())


def load_settings() -> Settings:
    """Built-in defaults with the user's ``config.json`` laid over them."""
    from ._errors import UsageError

    path = config_path()
    settings = Settings()
    raw = _settings_object(path)
    if not raw:
        return settings
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
                number = caster(raw)
            except ValueError as exc:
                raise UsageError(
                    what=f"{key} is a {name} setting",
                    why=f"{raw!r} is not a {name}",
                    how=f"pass a number, e.g. `stt config set {key} 2`",
                ) from exc
            _refuse_a_number_that_is_not_one(key, number)
            return number
    return raw


def _refuse_a_number_that_is_not_one(key: str, number: float) -> None:
    """`float("inf")` and `float("nan")` are numbers to Python and to nobody else.

    The same hole was found and closed once already, in `--idle-minutes`: `nan` compares
    false against everything and `inf` compares false against everything finite, so either
    one switches off whatever it is a threshold for — silently, since nothing raises. Here it
    would be `mic_threshold`: `stt config set mic_threshold inf` leaves the microphone open,
    the detector never opening, nothing typed, and `--check` reporting a perfectly loud voice
    as too quiet. Closing it in one place rather than per setting, because the next threshold
    somebody adds will have exactly the same hole.

    Only finiteness. Whether a NEGATIVE value makes sense is a question about the particular
    setting — `-1` is nonsense for a threshold and nonsense for a repetition count, but this
    function knows nothing about either, and an earlier version answered it here with a
    sentence about microphone levels that `stt config set max_repeats -1` then printed.
    """
    from ._errors import UsageError

    # An `int` is finite by construction, and asking `math.isfinite` about a very large one
    # RAISES: it converts to float first, so `stt config set threads <309 digits>` answered a
    # promised usage error with an `OverflowError` traceback. Centralising the check created
    # a way to crash that neither of the copies it replaced had.
    if isinstance(number, int):
        return
    if not math.isfinite(number):
        raise UsageError(
            what=f"{key} must be an ordinary number",
            why=f"{number!r} compares false against every real measurement, so it would "
            "switch off whatever it is a threshold for without saying so",
            how=f"pass a finite number, e.g. `stt config set {key} 1500`",
        )


def _a_real_number(value: Any) -> bool:
    """A float a measurement can be compared against — so not a bool, and not `Infinity`.

    JSON has no infinity, but Python's `json` reads the bare words `Infinity` and `NaN`
    anyway, so a hand-edited config can carry one into a threshold. This is the same question
    `_refuse_a_number_that_is_not_one` asks of a value typed at the command line, and the two
    paths have to agree: one of them accepting what the other refuses is how a setting ends
    up rejected when set and honoured when hand-edited.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


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
            else _a_real_number(value)
            if "float" in declared
            else value is None or isinstance(value, str)
        )
        if not ok:
            # A non-finite float is not a type error, and saying "expected float, found
            # float (nan)" contradicts itself. It is a real number that is not a real
            # measurement, and the sentence has to say which of the two went wrong.
            if isinstance(value, float) and not _a_real_number(value):
                what = f"{key} in {config_path()} is not a number anything can be compared to"
                why = (
                    f"{value!r} compares false against every real measurement, so it would "
                    "switch off whatever it is a threshold for without saying so"
                )
            else:
                what = f"{key} in {config_path()} has the wrong type"
                why = f"expected {declared}, found {type(value).__name__} ({value!r})"
            raise UsageError(
                what=what,
                why=why,
                how=f"fix it, or run `stt config set {key} <value>` to write a valid one",
            )


def save_setting(key: str, value: Any) -> None:
    """Persist one preference into ``config.json``, leaving the rest untouched.

    The whole of this is one transaction over ONE file, and both halves of that sentence had
    to be argued for.

    One file: `config.json` may be a symlink into a dotfiles checkout, which is an ordinary
    thing to do, so the link is resolved once at the top and everything after — the check,
    the lock, the read and the write — happens on the resolved target. Resolving it a second
    time later left a gap in between for the link to be repointed, which turns "write my
    config" into "rename a file over whatever that points at now"; and locking beside the
    link rather than beside the target meant two `STT_HOME`s pointing at one file took two
    different locks and did not exclude each other at all.

    One transaction: reading the whole file, changing one key and writing the whole file back
    is three operations, and two terminals doing it at once both read the old object — then
    whichever renames last stores a version that never saw the other's key, gone with no
    error anywhere. The lock spans the read and the write, the way `dictionary.editing` does.
    """
    from ._errors import unknown_item

    if key not in _CONFIGURABLE:
        raise unknown_item("setting", key, sorted(_CONFIGURABLE))
    ensure_dirs()
    named = config_path()
    target = Path(os.path.realpath(named))
    _refuse_a_strange_target(target)
    # Imported here, not at the top: `archive` imports `config` back, the way
    # `dictionary.save` does it for the same reason.
    from .archive import write_atomic

    with _editing_the_settings(target):
        # Through the same guarded loader as everything else: a corrupt config must produce
        # the diagnosed error, not a traceback out of json or the codec — and not, as an
        # earlier version of this line did through `as_dict`, a silent `{}` written back over
        # whatever was really in the file.
        raw: dict[str, Any] = _settings_object(target)
        raw[key] = value
        write_atomic(target, _serialized(raw, named))


@contextmanager
def _editing_the_settings(target: Path) -> Iterator[None]:
    """An exclusive lock over one read-modify-write of the settings file.

    Beside the RESOLVED target, so that two homes whose `config.json` links to the same file
    take the same lock. And opened the way every other file here is opened — non-blocking,
    with the descriptor asked what it is — because a lock file is a file like any other, and
    a named pipe sitting where it goes would have hung `stt config set` forever on the very
    line that was added to make the command safe.
    """
    from ._errors import UsageError

    lock = target.with_name(target.name + ".lock")
    try:
        # Owner only. The file holds nothing — it exists to be flocked — but there is no
        # reason for anything else on the machine to be able to open it, and a mode that
        # says "world readable" invites the question of what is in it.
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NONBLOCK, 0o600)
    except OSError as exc:
        raise UsageError(
            what=f"could not lock {target}",
            why=str(exc),
            how=f"remove {lock} if something else is sitting in its place",
        ) from exc
    # The raw descriptor, not a file object: wrapping a FIFO in one raises before the check
    # that was supposed to catch the FIFO.
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UsageError(
                what=f"could not lock {target}",
                why=f"{lock} is not a regular file",
                how="move it out of the way",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _refuse_a_strange_target(path: Path) -> None:
    """Say no to writing over something that is not a settings file.

    The reader treats anything that is not a regular file as "there is no config", which is
    right for reading and leaves the writer standing in front of whatever IS there. Opening a
    FIFO for writing blocks until somebody reads it — `stt config set` hanging with no
    message — and a directory raises out of the middle of the write. Neither is a thing to
    do to a path the user pointed `STT_HOME` at.

    Following symlinks, deliberately, because the reader does: `os.open` resolves the link
    and asks the descriptor what it is. Checking the link itself instead made a `config.json`
    symlinked into a dotfiles checkout readable by every command and writable by none.
    """
    from ._errors import UsageError

    try:
        found = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return
    except OSError as exc:
        raise UsageError(
            what=f"could not write {path}",
            why=str(exc),
            how="check the path is writable, or point STT_HOME somewhere else",
        ) from exc
    if not stat.S_ISREG(found.st_mode):
        raise UsageError(
            what=f"{path} is not a settings file",
            why="something that is not a regular file is sitting where config.json goes",
            how="move it out of the way, or point STT_HOME somewhere else",
        )


def _serialized(raw: dict[str, Any], path: Path) -> str:
    """The file, refused here if the reader would refuse it.

    A hand-edited `config.json` just under the limit loads fine, and `stt config set` then
    re-serializes it with indentation and one more key — over the line. The command reported
    success and every `stt` invocation after it failed before doing any work, on a file
    nothing would read. The same guard the dictionary has, for the same reason: the failure
    belongs at the write that causes it, not at every read that follows.
    """
    from ._errors import UsageError
    from .jsonio import in_units

    blob = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    if len(blob.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise UsageError(
            what=f"{path} would be too large to read back",
            why=f"it serializes to more than {in_units(MAX_CONFIG_BYTES)}",
            how="remove what you do not need from it, or delete it for the defaults",
        )
    return blob
