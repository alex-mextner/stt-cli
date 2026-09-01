"""media — turn whatever the user dropped on the command line into audio an engine can read.

This module exists because of the exact failure this tool was built to end: hand-writing
``ffmpeg -i "$N.ogg" -ar 16000 -ac 1 -c:a pcm_s16le "$N.wav"`` for every file, getting the
quoting wrong on a name with spaces and commas, and watching a whisper binary die with
"failed to read audio data as wav". Here the input is a path object from argv — no shell,
no quoting, no extension guessing — and the engine only ever sees a normalized 16 kHz mono
WAV that it is guaranteed to accept.

Video is not a special case: ffmpeg is asked for the audio stream and the video is ignored,
so an ``.mp4`` screen recording and an ``.m4a`` voice memo take the identical path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from . import proc
from ._errors import MissingTargetError, UsageError
from .jsonio import JsonDict, as_dict, as_dicts, as_float, as_int, as_str
from .models import MediaInfo

FFMPEG_HINT = "brew install ffmpeg"

# The archive's single audio format. Opus at 24 kbps mono is transparent enough for speech,
# is decodable by every engine we drive, and turns an hour of conversation into ~11 MB —
# small enough that keeping every recording forever is a rounding error on disk.
ARCHIVE_BITRATE = "24k"
ARCHIVE_SUFFIX = ".opus"

# What engines want: 16 kHz mono signed 16-bit PCM. Every whisper family model was trained
# on exactly this, so resampling here rather than inside the engine keeps behaviour uniform.
ENGINE_RATE = 16_000


async def probe(path: Path) -> JsonDict:
    """Raw ffprobe JSON for ``path`` (format + streams)."""
    if not path.exists():
        raise MissingTargetError(
            what=f"no such file: {path}",
            why="the input path does not exist",
            how="check the path; quote it if it contains spaces",
        )
    if path.is_dir():
        raise UsageError(
            what=f"{path} is a directory",
            why="stt transcribes files, not directories",
            how="pass the media files themselves, e.g. `stt dir/*.m4a`",
        )
    ffprobe = proc.require("ffprobe", install_hint=FFMPEG_HINT)
    result = await proc.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout=120,
    )
    if not result.ok:
        raise UsageError(
            what=f"ffprobe could not read {path.name}",
            why=result.tail() or "unrecognized container or codec",
            how="confirm the file is really audio or video (`ffprobe <file>`)",
        )
    return as_dict(json.loads(result.stdout or "{}"))


async def inspect(path: Path, *, recorded_at: datetime | None = None) -> MediaInfo:
    """Everything the pipeline needs to know about the input, including when it was made."""
    raw = await probe(path)
    streams = as_dicts(raw.get("streams"))
    fmt = as_dict(raw.get("format"))
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise UsageError(
            what=f"{path.name} has no audio stream",
            why="ffprobe found only non-audio streams in the file",
            how="pass a file that actually contains audio",
        )
    stamp, source = _resolve_recorded_at(path, fmt, override=recorded_at)
    return MediaInfo(
        path=str(path.resolve()),
        sha256=await sha256(path),
        size_bytes=path.stat().st_size,
        duration=as_float(fmt.get("duration") or audio.get("duration")),
        container=as_str(fmt.get("format_name")),
        codec=as_str(audio.get("codec_name")),
        sample_rate=as_int(audio.get("sample_rate")),
        channels=as_int(audio.get("channels")),
        has_video=any(s.get("codec_type") == "video" for s in streams),
        recorded_at=stamp,
        recorded_at_source=source,
    )


async def sha256(path: Path) -> str:
    """Content hash of the source file — the archive's identity for this recording."""
    return await asyncio.to_thread(_sha256_blocking, path)


def _sha256_blocking(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def to_engine_wav(src: Path, dest: Path, *, start: float = 0.0, end: float = 0.0) -> Path:
    """Decode ``src`` (audio or video, any codec) to the 16 kHz mono WAV engines expect.

    ``start``/``end`` cut one span out of the source, which is how speech spans are fed to
    the engine individually. The seek is placed before ``-i`` so ffmpeg jumps rather than
    decodes its way there — on a two-hour file that is the difference between instant and
    a minute per span.
    """
    ffmpeg = proc.require("ffmpeg", install_hint=FFMPEG_HINT)
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        argv += ["-ss", f"{start:.3f}"]
    argv += ["-i", str(src)]
    if end > start:
        argv += ["-t", f"{end - start:.3f}"]
    argv += ["-vn", "-map", "a:0", "-ac", "1", "-ar", str(ENGINE_RATE), "-c:a", "pcm_s16le"]
    argv += [str(dest)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    await proc.run(argv, check=True)
    return dest


async def to_archive_opus(src: Path, dest: Path) -> Path:
    """Compress ``src`` into the archive's single format, so every stored recording matches.

    Encoded to a temporary neighbour and renamed into place. An interrupted encode would
    otherwise leave a truncated file that looks present forever — the archive checks only
    that the audio exists and is non-empty, so a partial one is never noticed and never
    re-made.
    """
    ffmpeg = proc.require("ffmpeg", install_hint=FFMPEG_HINT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    argv = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-map", "a:0", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", ARCHIVE_BITRATE, "-application", "voip",
        "-f", "opus", str(partial),
    ]  # fmt: skip
    try:
        await proc.run(argv, check=True)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def _resolve_recorded_at(
    path: Path, fmt: JsonDict, *, override: datetime | None
) -> tuple[datetime | None, str]:
    """Decide when this recording was made, and say which source the answer came from.

    Best evidence first: an explicit flag, then the container's own ``creation_time`` tag
    (what the recorder itself wrote), then a date embedded in the filename, then the
    filesystem's birth time, and only as a last resort the modification time — which is a
    lie the moment anyone re-encodes the file.

    A filename date comes back NAIVE, on purpose. ``recording-2026-03-22 19.51.58.ogg``
    records the wall-clock time where the recording happened, with no zone attached; the
    machine reading it later may well be somewhere else. Stamping it with the reader's
    current zone and then converting to the display zone shifts a 19:51 recording to 16:51
    for no reason. Naive means "this is already the wall clock", and
    :class:`~stt_cli.timestamps.Stamper` localizes it without moving the hands.
    """
    if override is not None:
        return override, "explicit"
    tags = as_dict(fmt.get("tags"))
    tag = tags.get("creation_time") or tags.get("date")
    if isinstance(tag, str):
        parsed = _parse_iso(tag)
        if parsed:
            return parsed.astimezone(), "container tag"
    from_name = _parse_filename_date(path.name)
    if from_name:
        return from_name, "filename"  # naive: wall clock, see the docstring
    stat = path.stat()
    birth = getattr(stat, "st_birthtime", None)
    if birth:
        return datetime.fromtimestamp(birth).astimezone(), "file creation time"
    return datetime.fromtimestamp(stat.st_mtime).astimezone(), "file modification time"


def _parse_iso(value: str) -> datetime | None:
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# Recorders and screen-capture tools stamp the date into the filename in a handful of
# shapes; these cover the ones that actually show up (`2026-03-31 13.32.57`,
# `2026-03-31_13-32-57`, `20260331-133257`).
_NAME_PATTERNS = (
    "%Y-%m-%d %H.%M.%S",
    "%Y-%m-%d %H-%M-%S",
    "%Y-%m-%d_%H-%M-%S",
    "%Y-%m-%d_%H.%M.%S",
    "%Y%m%d-%H%M%S",
    "%Y%m%d_%H%M%S",
)


def _parse_filename_date(name: str) -> datetime | None:
    import re

    stem = os.path.splitext(name)[0]
    match = re.search(r"\d{4}[-_]?\d{2}[-_]?\d{2}[ _T]?\d{2}[.\-:]?\d{2}[.\-:]?\d{2}", stem)
    if not match:
        return None
    for pattern in _NAME_PATTERNS:
        try:
            return datetime.strptime(match.group(0), pattern)
        except ValueError:
            continue
    return None
