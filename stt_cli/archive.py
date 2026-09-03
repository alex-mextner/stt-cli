"""archive — keep every recording and every transcript, so nothing is ever done twice.

WHY AN ARCHIVE AND NOT A TEMP DIRECTORY
    Transcribing an hour of audio costs real minutes of GPU time. Throwing the result into
    ``/tmp`` means the next reboot deletes it, and re-running the identical command pays the
    full cost again — which is exactly what happens when you realize you wanted subtitles
    rather than plain text. So a run is stored, keyed by the *content* of the audio and the
    options that actually affect the words, and an identical request is answered from disk.

WHAT IS STORED, AND WHY IN THAT SHAPE
    The audio is re-encoded once into a single format (mono Opus, see :mod:`stt_cli.media`)
    rather than kept in whatever the source happened to be. One format means one code path
    for everything that reads it later, and roughly eleven megabytes an hour means keeping
    everything is affordable. The transcript is stored as the full JSON — segments,
    confidences, variants, flags — not as rendered text, because every other format can be
    regenerated from it and none of them can be regenerated from each other.

THE CACHE KEY IS DELIBERATELY NARROW
    Only choices that change the *words* are in the fingerprint: engine, model, language,
    voice-activity settings, cleaning, variants, correction. Timestamp mode and output format
    are not — asking for the same transcript as VTT instead of text re-renders from the
    archive instantly rather than re-transcribing. Nor are the enrichments (a summary,
    speaker labels): those ADD to a transcript, so requesting one runs just that pass over an
    existing run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config
from ._errors import MissingTargetError
from .config import Settings
from .media import ARCHIVE_SUFFIX
from .models import Transcript

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    media_sha    TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    recorded_at  TEXT,
    duration     REAL NOT NULL DEFAULT 0,
    backend      TEXT NOT NULL DEFAULT '',
    model        TEXT NOT NULL DEFAULT '',
    language     TEXT,
    segments     INTEGER NOT NULL DEFAULT 0,
    words        INTEGER NOT NULL DEFAULT 0,
    has_summary  INTEGER NOT NULL DEFAULT 0,
    has_speakers INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_media ON runs(media_sha);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS runs_identity ON runs(media_sha, fingerprint);
"""

# Settings that change the WORDS. Everything else is either a rendering choice (format,
# timestamps, destination) or an enrichment that can be added to an existing transcript
# without re-transcribing — see ENRICHMENTS below. Getting this list wrong is expensive in
# one direction and silently wrong in the other, so both are covered by tests.
FINGERPRINT_KEYS = (
    "backend", "model", "language", "vad", "vad_threshold", "vad_min_silence_ms",
    "vad_speech_pad_ms", "vad_min_speech_ms", "clean", "strict_clean", "max_repeats",
    "confidence_floor", "variants", "variant_models", "fix", "fix_with",
)  # fmt: skip

# Settings that change the transcript but were added after the archive already had runs in
# it. They enter the fingerprint only when set away from their default, so an existing
# archive stays valid: at the default they decode exactly what the old code decoded, and
# listing them unconditionally would change every stored fingerprint and quietly throw the
# whole archive away. Anything genuinely new belongs here rather than in FINGERPRINT_KEYS.
# Each entry MUST equal the same-named field's default on `Settings` — the whole mechanism
# is "at the default, the fingerprint is unchanged", and that only holds while the two agree.
# Tuning a default in config.py and leaving this table behind silently invalidates every
# stored run on the next release. `test_the_fingerprint_defaults_match_the_real_defaults`
# is what keeps the two files honest, since nothing else connects them.
FINGERPRINT_DEFAULTS: dict[str, object] = {
    "context": "off",
    "context_compare": "off",
    # Not the settings themselves but the dictionary's CONTENT: adding a term changes
    # what comes out, and an empty dictionary must leave every existing run valid.
    "dict_digest": "",
    "dict_bias": True,
    "dict_similarity": 0.80,
    # "" = every engine did everything the settings asked for. See Settings.engine_limits.
    "engine_limits": "",
}

# Things that ADD to a finished transcript rather than change it: a summary, speaker
# labels. Asking for one of these against an already-transcribed recording must run only
# that pass, not another hour on the GPU. They are therefore deliberately NOT part of the
# cache key; the pipeline applies whichever is requested and missing, then saves the run
# back with it.
ENRICHMENTS = ("summary", "diarize")

# Bumped whenever the DECODING itself changes — a different flag passed to an engine, a
# different chunking rule — so an archived transcript produced by the old behaviour is not
# silently served as if the change had never happened. It is not the package version:
# releases that do not touch decoding must not throw away everyone's archive.
DECODE_REVISION = 2


@dataclass(slots=True, frozen=True)
class RunRecord:
    """One archived transcription, as the index knows it."""

    run_id: str
    media_sha: str
    fingerprint: str
    source_path: str
    source_name: str
    created_at: str
    recorded_at: str | None
    duration: float
    backend: str
    model: str
    language: str | None
    segments: int
    words: int
    has_summary: bool
    has_speakers: bool

    @property
    def directory(self) -> Path:
        return config.runs_dir() / self.run_id

    def row(self) -> str:
        marks = ("S" if self.has_summary else "-") + ("D" if self.has_speakers else "-")
        return (
            f"{self.run_id}  {self.created_at[:16]}  {self.duration / 60:>6.1f}m  "
            f"{marks}  {self.backend}:{self.model:<16.16} {self.words:>6}w  {self.source_name}"
        )


def fingerprint(settings: Settings) -> str:
    """Hash the transcript-affecting settings into the archive's identity for this run."""
    payload: dict[str, object] = {key: getattr(settings, key) for key in FINGERPRINT_KEYS}
    for key, default in FINGERPRINT_DEFAULTS.items():
        value = getattr(settings, key, default)
        if value != default:
            payload[key] = value
    payload["_decode_revision"] = DECODE_REVISION
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_id(media_sha: str, fp: str) -> str:
    """The run's directory name. It carries the WHOLE fingerprint on purpose: truncating it
    would let two runs the unique index considers distinct share one directory, so the index
    would claim a fingerprint whose transcript.json had been overwritten by the other."""
    return f"{media_sha[:10]}-{fp}"


class Archive:
    """The on-disk store, plus the SQLite index that makes it searchable."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._db = sqlite3.connect(config.index_path())
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Archive:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── media ─────────────────────────────────────────────────────────────────
    def media_path(self, media_sha: str) -> Path:
        return config.media_dir() / media_sha[:2] / f"{media_sha}{ARCHIVE_SUFFIX}"

    def has_media(self, media_sha: str) -> bool:
        path = self.media_path(media_sha)
        return path.is_file() and path.stat().st_size > 0

    async def store_media(self, source: Path, media_sha: str) -> Path:
        """Keep a compressed copy of the recording, so the source can move or be deleted."""
        from . import media as media_mod
        from . import resources

        target = self.media_path(media_sha)
        if self.has_media(media_sha):
            return target
        resources.require_space(
            max(source.stat().st_size // 8, 16 * 1024 * 1024),
            path=config.media_dir(),
            what="the archived audio copy",
        )
        return await media_mod.to_archive_opus(source, target)

    # ── runs ──────────────────────────────────────────────────────────────────
    def find(self, media_sha: str, fp: str) -> RunRecord | None:
        cursor = self._db.execute(
            "SELECT * FROM runs WHERE media_sha = ? AND fingerprint = ?", (media_sha, fp)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        record = _record(row)
        # An index entry whose payload was deleted by hand is worse than no entry: it makes
        # the cache claim a hit and then fail. Treat it as a miss and forget it.
        if not (record.directory / "transcript.json").is_file():
            self.forget(record.run_id)
            return None
        return record

    def save(self, transcript: Transcript, fp: str, *, created_at: str | None = None) -> RunRecord:
        """Write the transcript, any rendered outputs, and the index row — the ONE writer.

        Every file lands atomically (write a temporary neighbour, then rename), because a run
        interrupted mid-write must leave either the previous transcript or none, never a
        half-written JSON that the next cache hit will fail to parse.

        ``created_at`` preserves the ORIGINAL transcription time when a run is only being
        enriched with a summary or speaker labels. Stamping it with "now" would reset the
        clock `stt archive gc --older-than` counts from, so adding a summary to a two-year-old
        recording would make it look brand new.
        """
        identifier = run_id(transcript.media.sha256, fp)
        directory = config.runs_dir() / identifier
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)
        write_atomic(directory / "transcript.json", payload)
        record = _from_transcript(transcript, identifier, fp, created_at=created_at)
        self._upsert(record)
        return record

    def _upsert(self, record: RunRecord) -> None:
        self._db.execute(
            """INSERT INTO runs (run_id, media_sha, fingerprint, source_path, source_name,
                                 created_at, recorded_at, duration, backend, model, language,
                                 segments, words, has_summary, has_speakers)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id) DO UPDATE SET
                 created_at=excluded.created_at, segments=excluded.segments,
                 words=excluded.words, has_summary=excluded.has_summary,
                 has_speakers=excluded.has_speakers, source_path=excluded.source_path""",
            (
                record.run_id,
                record.media_sha,
                record.fingerprint,
                record.source_path,
                record.source_name,
                record.created_at,
                record.recorded_at,
                record.duration,
                record.backend,
                record.model,
                record.language,
                record.segments,
                record.words,
                int(record.has_summary),
                int(record.has_speakers),
            ),
        )
        self._db.commit()

    def load(self, identifier: str) -> Transcript:
        record = self.get(identifier)
        payload = record.directory / "transcript.json"
        if not payload.is_file():
            raise MissingTargetError(
                what=f"run {identifier} has no transcript on disk",
                why=f"{payload} is missing",
                how="the archive entry is stale; run `stt archive gc` to clean it up",
            )
        return Transcript.from_dict(json.loads(payload.read_text("utf-8")))

    def get(self, identifier: str) -> RunRecord:
        """Resolve a run id, accepting a unique prefix but never guessing between two.

        Two runs of the same recording with different settings share a long id prefix, so a
        short prefix can genuinely match both. Picking whichever row SQLite happened to
        return first would make `stt archive rm 3fa2b1` delete an arbitrary one of them.
        """
        rows = self._db.execute(
            "SELECT * FROM runs WHERE run_id = ? OR run_id LIKE ? ORDER BY created_at DESC",
            (identifier, f"{identifier}%"),
        ).fetchall()
        exact = [row for row in rows if row["run_id"] == identifier]
        if exact:
            return _record(exact[0])
        if not rows:
            raise MissingTargetError(
                what=f"no archived run matches {identifier!r}",
                why="nothing in the archive has that id",
                how="run `stt archive ls` to see what is stored",
            )
        if len(rows) > 1:
            candidates = "\n".join(f"  {row['run_id']}  {row['source_name']}" for row in rows)
            raise MissingTargetError(
                what=f"{identifier!r} matches {len(rows)} archived runs",
                why=f"the prefix is ambiguous:\n{candidates}",
                how="pass more of the run id",
            )
        return _record(rows[0])

    def media_for_source(self, path: Path) -> tuple[Path, str] | None:
        """The archived audio for a source path that no longer exists, if we kept one.

        This is what lets a recording be re-processed after it has been moved or deleted:
        the index remembers which content hash each path produced, and the stored copy is
        byte-for-byte what the engine was fed the first time.
        """
        row = self._db.execute(
            "SELECT media_sha FROM runs WHERE source_path = ? ORDER BY created_at DESC LIMIT 1",
            (str(path.expanduser().resolve() if path.is_absolute() else path.absolute()),),
        ).fetchone()
        if row is None:
            return None
        media_sha = str(row["media_sha"])
        stored = self.media_path(media_sha)
        if not (stored.is_file() and stored.stat().st_size > 0):
            return None
        return stored, media_sha

    def recent(self, *, limit: int = 30, query: str | None = None) -> list[RunRecord]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if query:
            sql += " WHERE source_name LIKE ? OR source_path LIKE ? OR run_id LIKE ?"
            params += [f"%{query}%", f"%{query}%", f"{query}%"]
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_record(row) for row in self._db.execute(sql, params)]

    def forget(self, identifier: str) -> None:
        self._db.execute("DELETE FROM runs WHERE run_id = ?", (identifier,))
        self._db.commit()

    def remove(self, identifier: str) -> None:
        record = self.get(identifier)
        shutil.rmtree(record.directory, ignore_errors=True)
        self.forget(record.run_id)

    # ── housekeeping ──────────────────────────────────────────────────────────
    def gc(self, *, older_than_days: int | None = None, dry_run: bool = True) -> list[str]:
        """Report (and optionally delete) stale index rows, orphan runs and unused audio."""
        actions: list[str] = []
        cutoff = time.time() - older_than_days * 86400 if older_than_days else None
        keep_media: set[str] = set()

        for record in self.recent(limit=1_000_000):
            expired = cutoff is not None and _epoch(record.created_at) < cutoff
            if not record.directory.is_dir() or expired:
                actions.append(
                    f"remove run {record.run_id} ({'expired' if expired else 'orphaned index row'})"
                )
                if not dry_run:
                    self.remove(record.run_id)
                continue
            keep_media.add(record.media_sha)

        for path in sorted(config.media_dir().glob("*/*")):
            if path.stem not in keep_media:
                actions.append(f"remove audio {path.stem[:12]} ({path.stat().st_size // 1024} KiB)")
                if not dry_run:
                    path.unlink(missing_ok=True)
        return actions

    def usage(self) -> tuple[int, int, int]:
        """(runs, bytes of stored audio, bytes of stored transcripts) — what this costs."""
        audio = sum(p.stat().st_size for p in config.media_dir().glob("*/*") if p.is_file())
        runs = sum(p.stat().st_size for p in config.runs_dir().rglob("*") if p.is_file())
        count = self._db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        return int(count), audio, runs


# Distinguishes concurrent writers inside one process; the pid distinguishes the processes.
_writers = itertools.count()


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary neighbour, so readers never see a partial file.

    The neighbour's name is unique to the writer — the pid, plus a counter for the writers
    inside one process. A shared name would let two writers interleave into one temporary
    file and the "atomic" rename would then publish a blend of both; with the pid alone that
    was still true of two threads in one process, where the second `replace` also found its
    own temporary already renamed away and raised.

    Unique names make concurrent writes safe, not ordered: the last rename wins and the
    other writer's content is gone. Ordering needs a lock (see ``dictionary.editing``).
    Every archive writer runs sequentially within a run, so nothing here needs one today.
    """
    partial = path.with_name(f"{path.name}.{os.getpid()}.{next(_writers)}.part")
    try:
        partial.write_text(text, "utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        media_sha=row["media_sha"],
        fingerprint=row["fingerprint"],
        source_path=row["source_path"],
        source_name=row["source_name"],
        created_at=row["created_at"],
        recorded_at=row["recorded_at"],
        duration=row["duration"],
        backend=row["backend"],
        model=row["model"],
        language=row["language"],
        segments=row["segments"],
        words=row["words"],
        has_summary=bool(row["has_summary"]),
        has_speakers=bool(row["has_speakers"]),
    )


def _from_transcript(
    transcript: Transcript, identifier: str, fp: str, *, created_at: str | None = None
) -> RunRecord:
    media = transcript.media
    return RunRecord(
        run_id=identifier,
        media_sha=media.sha256,
        fingerprint=fp,
        source_path=media.path,
        source_name=Path(media.path).name,
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        recorded_at=media.recorded_at.isoformat() if media.recorded_at else None,
        duration=media.duration,
        backend=transcript.engine.backend,
        model=transcript.engine.model,
        language=transcript.language,
        segments=len(transcript.segments),
        words=sum(len(s.text.split()) for s in transcript.segments),
        has_summary=transcript.summary is not None,
        has_speakers=any(s.speaker for s in transcript.segments),
    )


def _epoch(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0
