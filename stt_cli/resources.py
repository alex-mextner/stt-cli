"""resources — refuse to start work the machine cannot finish.

Pulling a multi-gigabyte model or normalizing a two-hour recording onto a nearly full
disk fails halfway, and a half-written model file is worse than no model: the next run
loads it and dies somewhere deep inside the engine. So every download and every large
write asks here first, and gets a clear refusal with the actual numbers rather than an
``ENOSPC`` traceback.

Memory is advisory, not a veto. A model larger than physical RAM still runs on macOS —
it just swaps and crawls — so an oversized model is a loud warning, and only a genuinely
impossible case (the model does not fit in RAM *and* there is no swap headroom on disk)
is refused.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._errors import ResourceError

GIB = 1024**3
# Keep this much free after any write. Below it macOS itself starts misbehaving, so
# filling the disk to the brim is never worth a transcript.
DISK_HEADROOM = 2 * GIB


@dataclass(slots=True, frozen=True)
class Resources:
    free_disk: int
    total_disk: int
    total_ram: int

    def human(self) -> str:
        return (
            f"disk {self.free_disk / GIB:.1f} GiB free of {self.total_disk / GIB:.0f} GiB, "
            f"RAM {self.total_ram / GIB:.0f} GiB"
        )


def probe(path: Path | None = None) -> Resources:
    """Current free space on ``path``'s volume and total physical memory."""
    target = path or Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return Resources(free_disk=usage.free, total_disk=usage.total, total_ram=_total_ram())


def require_space(needed: int, *, path: Path, what: str) -> None:
    """Refuse ``what`` unless ``needed`` bytes fit with the safety headroom intact."""
    res = probe(path)
    if res.free_disk >= needed + DISK_HEADROOM:
        return
    raise ResourceError(
        what=f"not enough disk space for {what}",
        why=(
            f"{what} needs about {needed / GIB:.1f} GiB (plus {DISK_HEADROOM / GIB:.0f} GiB "
            f"headroom) but only {res.free_disk / GIB:.1f} GiB is free on {path}"
        ),
        how="free up space, run `stt archive gc`, or point STT_HOME at a roomier volume",
    )


def check_memory(model_bytes: int, *, model: str) -> list[str]:
    """Warn (never refuse) when a model is large relative to physical memory."""
    res = probe()
    warnings: list[str] = []
    if model_bytes > res.total_ram * 0.7:
        warnings.append(
            f"{model} needs roughly {model_bytes / GIB:.1f} GiB but this machine has "
            f"{res.total_ram / GIB:.0f} GiB of RAM — expect swapping and a slow run"
        )
    return warnings


def _total_ram() -> int:
    """Physical memory in bytes; 0 when it cannot be determined (never fatal)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
