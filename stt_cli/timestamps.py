"""timestamps — three ways to say when something was said, and how to choose one.

``none`` is for text you are going to read or paste somewhere. ``relative`` is the offset
from the start of the recording, which is what you want when you are going to scrub back to
a moment in the file. ``absolute`` is wall-clock time — the recording's own start plus the
offset — which is what you want when the transcript has to line up with a calendar, a log,
or somebody else's notes about the same meeting.

Absolute time is only as good as the recording's start time, and that is a guess of varying
quality (see :func:`stt_cli.media._resolve_recorded_at`). So the renderers print how it was
determined alongside it, rather than presenting an inference as a fact.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ._errors import UsageError
from .models import MediaInfo


def resolve_zone(name: str | None) -> timezone | ZoneInfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise UsageError(
            what=f"unknown timezone: {name!r}",
            why=str(exc),
            how="use an IANA name such as Europe/Belgrade",
        ) from exc


def clock(seconds: float, *, millis: bool = False) -> str:
    """``H:MM:SS`` (or ``H:MM:SS.mmm``) — the offset from the start of the recording."""
    total = max(0.0, seconds)
    hours, rest = divmod(int(total), 3600)
    minutes, secs = divmod(rest, 60)
    if not millis:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{int((total % 1) * 1000):03d}"


def srt_time(seconds: float) -> str:
    total = max(0.0, seconds)
    hours, rest = divmod(int(total), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{int((total % 1) * 1000):03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


class Stamper:
    """Formats one offset according to the mode chosen for this run."""

    def __init__(self, mode: str, media: MediaInfo, zone_name: str | None = None) -> None:
        from .config import TIMESTAMP_MODES

        if mode not in TIMESTAMP_MODES:
            raise UsageError(
                what=f"unknown timestamp mode: {mode!r}",
                why=f"--timestamps accepts {', '.join(TIMESTAMP_MODES)}",
                how="use --timestamps relative for offsets, absolute for wall-clock time",
            )
        self.mode = mode
        self.zone = resolve_zone(zone_name)
        self.base = self._base(media)
        self.base_source = media.recorded_at_source

    def _base(self, media: MediaInfo) -> datetime | None:
        """The instant absolute timestamps count from, in the zone we are displaying in.

        A start time that carries a zone (a container tag, the filesystem) is a real instant
        and is converted. A NAIVE one — a date read out of the filename — is already the wall
        clock somebody wrote down, so it is localized, not converted: converting it would
        move a recording made at 19:51 to some other hour purely because the machine reading
        the file sits in a different zone than the one that recorded it.
        """
        if self.mode != "absolute":
            return None
        if media.recorded_at is None:
            raise UsageError(
                what="cannot use absolute timestamps for this file",
                why="no recording start time could be determined from the file",
                how="pass --recorded-at '2026-03-31 13:32:57', or use --timestamps relative",
            )
        base = media.recorded_at
        if base.tzinfo is None:
            return base.replace(tzinfo=self.zone) if self.zone else base
        return base.astimezone(self.zone) if self.zone else base

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def at(self, seconds: float) -> str:
        """The label for one offset, empty when timestamps are switched off."""
        if self.mode == "none":
            return ""
        if self.mode == "relative":
            return clock(seconds)
        assert self.base is not None
        return (self.base + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

    def describe_base(self) -> str:
        """A one-line note about what absolute times are anchored to, for the file header."""
        if self.mode != "absolute" or self.base is None:
            return ""
        zone = f" {self.base:%Z}".rstrip() if self.base.tzinfo else " (local wall clock)"
        return f"anchored to {self.base:%Y-%m-%d %H:%M:%S}{zone} (from {self.base_source})"
