"""proc — one async way to run an external program.

Every heavy thing stt-cli does is another process: ffmpeg, ffprobe, a whisper binary, an
LLM CLI. Funnelling them through one helper buys three things the pipeline depends on:
a missing binary becomes a diagnosed :class:`MissingDependencyError` instead of a bare
``FileNotFoundError``; a hung child dies on a timeout instead of wedging the run forever
(long jobs are the norm here, so no call is ever left unbounded); and because the helper
is ``async``, independent children — several output formats, several decoding variants —
actually run at the same time instead of queueing behind each other.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ._errors import EngineError, MissingDependencyError

# Generous, but never absent: a wedged ffmpeg or engine must fail loudly rather than hold
# the run open until someone notices tomorrow.
DEFAULT_TIMEOUT = 3 * 60 * 60.0


@dataclass(slots=True)
class Result:
    """What a finished child process left behind."""

    code: int
    stdout: str
    stderr: str
    argv: list[str]

    @property
    def ok(self) -> bool:
        return self.code == 0

    def tail(self, lines: int = 12) -> str:
        blob = (self.stderr or self.stdout).strip()
        return "\n".join(blob.splitlines()[-lines:])


def which(binary: str) -> str | None:
    return shutil.which(binary)


def require(binary: str, *, install_hint: str) -> str:
    """Resolve ``binary`` on PATH or explain exactly how to get it."""
    found = shutil.which(binary)
    if found:
        return found
    raise MissingDependencyError(
        what=f"required tool not found: {binary}",
        why=f"{binary} is not on PATH",
        how=install_hint,
    )


async def run(
    argv: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    check: bool = False,
) -> Result:
    """Run ``argv`` to completion, capturing both streams."""
    merged_env = {**os.environ, **(env or {})} if env else None
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
        )
    except FileNotFoundError as exc:
        raise MissingDependencyError(
            what=f"could not run {argv[0]}",
            why="the executable does not exist on PATH",
            how=f"install {argv[0]}, or point stt at it explicitly",
        ) from exc

    payload = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        out, err = await asyncio.wait_for(proc.communicate(payload), timeout=timeout)
    except TimeoutError as exc:
        _terminate(proc)
        raise EngineError(
            what=f"{Path(argv[0]).name} timed out after {timeout / 60:.0f} min",
            why="the process produced no result within the allowed time",
            how=(
                "re-run with a smaller model or split the input; raise the timeout if "
                "the job really is that long"
            ),
        ) from exc

    result = Result(
        code=proc.returncode or 0,
        stdout=out.decode("utf-8", "replace"),
        stderr=err.decode("utf-8", "replace"),
        argv=list(argv),
    )
    if check and not result.ok:
        raise EngineError(
            what=f"{Path(argv[0]).name} exited with code {result.code}",
            why=result.tail() or "the process wrote nothing to stderr",
            how="re-run with -v to see the full command and output",
        )
    return result


def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Kill a child that blew its timeout, tolerating one that already exited."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
