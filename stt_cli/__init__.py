"""stt-cli — speech-to-text for any audio or video file on macOS.

The package is import-clean: nothing here (or in the dispatcher) pulls a heavy or
optional dependency, so ``stt --help`` stays instant and works on a machine where no
engine is installed yet.

WHERE THE VERSION LIVES
    In ``pyproject.toml``, and nowhere else. It used to live here as a literal, with
    pyproject pointing back at it through a ``version = { attr = ... }`` indirection. That
    is tidy, and it is invisible to the release gate, which reads pyproject looking for a
    number and finds a reference it cannot follow — so a correctly bumped release reads as
    unbumped and every ship has to be waved past a check that is wrong about this project.
    Which is how somebody learns to wave one past on the day it is right.

    Reading it back is two routes, and which one answers matters.

    A source checkout — an editable install included — answers from ``pyproject.toml``
    itself, which sits one directory up from this file. That costs 0.2 ms and is always the
    version of the tree in front of you.

    Anything else — a real wheel, with no pyproject.toml installed beside it — answers from
    the package metadata. That costs about 20 ms, which is why it is second, and it is only
    trustworthy for an installed copy: an editable install records its version once and then
    keeps answering with it, so this machine's metadata still said 0.1.0 while the source
    said 0.3.1.

    Both are computed on first use, so a command that never prints the version pays neither.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

__all__ = ["__version__"]

DISTRIBUTION = "stt-cli"
_UNKNOWN = "0+unknown"


def __getattr__(name: str) -> Any:
    """Resolve ``__version__`` when something asks, so `stt --help` never pays for it."""
    if name == "__version__":
        return _declared_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@cache
def _declared_version() -> str:
    """The version this copy of stt was checked out or built at.

    Cached, because the docstring above says "first use" and a long-running process holding
    ``stt_cli`` would otherwise re-open and re-parse the file on every read — and report a
    different number if somebody edited it meanwhile.
    """
    beside_us = Path(__file__).resolve().parent.parent / "pyproject.toml"
    from_source = _version_in(beside_us) if beside_us.is_file() else None
    return from_source or _version_installed()


def _version_in(pyproject: Path) -> str | None:
    """Our version from a pyproject.toml, or None if it is unreadable, foreign or odd."""
    import tomllib

    try:
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        if project["name"] != DISTRIBUTION:
            # Somebody else's project file. `stt_cli/` is not always in a directory of ours:
            # `pip install --target ./app` drops it beside ./app/pyproject.toml, and reading
            # that one would have reported the host application's version as stt's. The same
            # mistake `uv run` makes without `--no-project`, one directory up.
            return None
        declared = project["version"]
    except (OSError, ValueError, KeyError, TypeError):
        # ValueError rather than TOMLDecodeError: tomllib decodes the bytes itself, so a file
        # that is not valid UTF-8 comes back as a UnicodeDecodeError, which the narrower
        # spelling let through — and this function promises never to raise.
        return None
    return declared if isinstance(declared, str) else None


def _version_installed() -> str:
    """What the installed metadata says, for a copy with no source tree beside it."""
    from importlib import metadata

    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return _UNKNOWN
