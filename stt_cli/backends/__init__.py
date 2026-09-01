"""backends — engine discovery and selection.

``auto`` is not a coin flip: it prefers whatever is genuinely ready to run on this machine,
in the order that costs the user least. A working whisper.cpp build wins, because it needs
no download beyond the model and runs on Metal today; mlx-whisper is next, because ``uv``
can conjure it on demand. If neither is present, the error names both and says how to get
each — never a bare "no backend".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._errors import MissingDependencyError, unknown_item
from .base import Availability, Backend, DecodeRequest

if TYPE_CHECKING:
    from .whispercpp import WhisperCppBackend


def _configured_root() -> str | None:
    """The whisper.cpp checkout the user configured, if any. Never fatal."""
    from ..config import load_settings

    try:
        return load_settings().whispercpp_root
    except Exception:
        return None


# Preference order for `--backend auto`.
ORDER = ("whispercpp", "mlx")


def create(name: str, *, whispercpp_root: str | None = None) -> Backend:
    """Instantiate one engine by name (does not check that it can actually run).

    ``whispercpp_root`` falls back to the stored preference when the caller does not pass
    one, so `stt doctor`, `stt models pull` and `stt setup` see the same build the pipeline
    does. Leaving each caller to remember the kwarg is how a user ends up with a working
    `stt rec.m4a` and a `stt doctor` that says no engine is installed.
    """
    if name == "whispercpp":
        from .whispercpp import WhisperCppBackend

        return WhisperCppBackend(whispercpp_root or _configured_root())
    if name == "mlx":
        from .mlx import MlxBackend

        return MlxBackend()
    raise unknown_item("backend", name, list(ORDER))


def whispercpp_backend(*, root: str | None = None) -> WhisperCppBackend:
    """The whisper.cpp engine as its CONCRETE type, for the few callers that need its extras.

    Deliberately NOT named ``whispercpp``: that is the submodule's own name, and a function
    by the same name in this package silently shadows it depending on import order — which
    presented as `'module' object is not callable` at runtime while type checking cleanly.

    Model files on disk and the Silero weights are whisper.cpp specifics that do not belong
    on the general ``Backend`` protocol, but `stt doctor` and `stt models pull-vad` genuinely
    need them. Going through here rather than constructing the class directly is what keeps
    the configured checkout honoured everywhere.
    """
    from .whispercpp import WhisperCppBackend

    return WhisperCppBackend(root or _configured_root())


def survey(*, whispercpp_root: str | None = None) -> list[tuple[str, Availability]]:
    """Every known engine with its current availability — what `stt doctor` reports."""
    rows: list[tuple[str, Availability]] = []
    for name in ORDER:
        try:
            rows.append((name, create(name, whispercpp_root=whispercpp_root).availability()))
        except Exception as exc:  # a broken engine must not hide the working ones
            rows.append((name, Availability(False, f"failed to load: {exc}")))
    return rows


def resolve(name: str, *, whispercpp_root: str | None = None) -> Backend:
    """Return a usable engine, honouring an explicit choice and diagnosing a dead one."""
    if name != "auto":
        backend = create(name, whispercpp_root=whispercpp_root)
        status = backend.availability()
        if not status.ok:
            raise MissingDependencyError(
                what=f"the {name} engine cannot run",
                why=status.detail,
                how=status.how_to_install or "run `stt doctor` for the full picture",
            )
        return backend

    problems: list[str] = []
    for candidate, status in survey(whispercpp_root=whispercpp_root):
        if status.ok:
            return create(candidate, whispercpp_root=whispercpp_root)
        problems.append(f"  {candidate}: {status.detail}\n    -> {status.how_to_install}")
    raise MissingDependencyError(
        what="no speech engine is available",
        why="none of the supported engines is installed:\n" + "\n".join(problems),
        how="run `stt setup` — it detects what you have and installs the rest",
    )


__all__ = [
    "ORDER",
    "Availability",
    "Backend",
    "DecodeRequest",
    "create",
    "resolve",
    "survey",
    "whispercpp_backend",
]
