"""_errors — structured errors and stable exit codes for stt-cli.

WHY THIS EXISTS
    Every failure a user can hit should say three things: WHAT went wrong, WHY it went
    wrong, and HOW to fix it. A bare traceback or a lone ``exit(1)`` says none of them.
    The exit codes match the numbers the sibling personal CLIs (rig, review, research)
    already use, so a script can branch on ``$?`` the same way across all of them.

CONTRACT
    - Stdlib-only, so the dispatcher can import it at module top with zero cost.
    - Raise a subclass; let :func:`guard` render it and return the code.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import NoReturn

EXIT_OK = 0
EXIT_INTERNAL = 1  # an unexpected failure / bug in stt-cli itself
EXIT_USAGE = 2  # invalid argument or malformed config value
EXIT_UNKNOWN_ITEM = 4  # a name that doesn't exist (unknown command, model, format)
EXIT_MISSING_TARGET = 5  # a referenced path/file is gone on disk
EXIT_NETWORK = 7  # a download or remote call failed
EXIT_PERMISSION = 8  # missing token / denied access
EXIT_RESOURCES = 9  # not enough disk space or memory to proceed safely
EXIT_ENGINE = 10  # a transcription engine ran but failed
EXIT_MISSING_DEP = 127  # a required external tool (ffmpeg, an engine) isn't installed

EXIT_NAMES: dict[int, str] = {
    EXIT_OK: "ok",
    EXIT_INTERNAL: "internal error",
    EXIT_USAGE: "usage error",
    EXIT_UNKNOWN_ITEM: "unknown item",
    EXIT_MISSING_TARGET: "missing target",
    EXIT_NETWORK: "network error",
    EXIT_PERMISSION: "permission error",
    EXIT_RESOURCES: "insufficient resources",
    EXIT_ENGINE: "engine failure",
    EXIT_MISSING_DEP: "missing dependency",
}


class SttError(Exception):
    """A diagnosed failure: WHAT happened, WHY, and HOW to fix it."""

    exit_code = EXIT_INTERNAL

    def __init__(self, what: str, why: str = "", how: str = "") -> None:
        super().__init__(what)
        self.what = what
        self.why = why
        self.how = how

    def render(self) -> str:
        lines = [f"stt: {self.what}"]
        if self.why:
            lines.append(f"  why:  {self.why}")
        if self.how:
            lines.append(f"  fix:  {self.how}")
        return "\n".join(lines)


class UsageError(SttError):
    exit_code = EXIT_USAGE


class UnknownItemError(SttError):
    exit_code = EXIT_UNKNOWN_ITEM


class MissingTargetError(SttError):
    exit_code = EXIT_MISSING_TARGET


class NetworkError(SttError):
    exit_code = EXIT_NETWORK


class PermissionDeniedError(SttError):
    exit_code = EXIT_PERMISSION


class ResourceError(SttError):
    exit_code = EXIT_RESOURCES


class EngineError(SttError):
    exit_code = EXIT_ENGINE


class MissingDependencyError(SttError):
    exit_code = EXIT_MISSING_DEP


def unknown_item(kind: str, name: str, known: list[str]) -> UnknownItemError:
    """Build an unknown-<kind> error with a did-you-mean hint drawn from ``known``."""
    import difflib

    close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    hint = f"did you mean: {', '.join(close)}?" if close else f"known {kind}s: {', '.join(known)}"
    return UnknownItemError(
        what=f"unknown {kind}: {name!r}",
        why=f"{name!r} is not one of the registered {kind}s",
        how=hint,
    )


def guard(fn: Callable[[], int]) -> int:
    """Run ``fn``, rendering any diagnosed error to stderr and returning its exit code."""
    try:
        return fn()
    except SttError as exc:
        print(exc.render(), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("stt: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # `stt ... | head` — not an error worth a traceback
        return EXIT_OK


def internal(msg: str) -> NoReturn:
    raise SttError(what=msg, why="this is a bug in stt-cli", how="please open an issue")
