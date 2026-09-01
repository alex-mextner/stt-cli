"""chunks — feed the engine speech and nothing but speech, in one pass instead of hundreds.

THE PROBLEM THIS SOLVES
    Voice-activity detection on an hour of conversation finds several hundred speech spans,
    most of them a second or two long. Decoding each one as its own engine invocation is a
    disaster twice over: a 1.6 GB model is loaded and a GPU context built several hundred
    times, and each fragment is decoded with no surrounding context, which is precisely the
    condition under which these models guess badly.

THE FIX
    Concatenate the speech into a handful of continuous chunks — silence removed, a short
    marker gap between spans — and decode each chunk once. The engine sees minutes of
    context, the silence it would hallucinate over is simply not present, and an hour of
    audio costs a handful of model loads instead of hundreds.

    The cost is that decoded timestamps are in *chunk* time, which no longer matches the
    recording. :class:`Chunk` carries the piece table needed to map them back, so the
    transcript still lines up with the original file to the tenth of a second.

WHY A MARKER GAP AND NOT A HARD SPLICE
    Butting two spans directly together makes the end of one word run into the start of
    another, and the model duly hears a word that was never said. A short silence between
    them reads as a natural pause. It is small enough not to invite the model to fill it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import media, proc
from .models import SpeechSpan

# Speech per decoded chunk. Long enough that model-load overhead disappears into the noise,
# short enough to report progress and to bound the damage if one chunk goes wrong.
CHUNK_SECONDS = 480.0
# The pause inserted between two spliced spans. Long enough to read as a pause, short enough
# that the model does not treat it as an invitation to invent a subtitle credit.
JOIN_SILENCE = 0.2
# A little audio either side of a span, so words are not clipped mid-consonant.
EDGE_PAD = 0.15


@dataclass(slots=True)
class Piece:
    """One span's position in both timelines: where it sits in the chunk, and in the file."""

    chunk_start: float
    chunk_end: float
    source_start: float

    def to_source(self, chunk_time: float) -> float:
        return self.source_start + (chunk_time - self.chunk_start)


@dataclass(slots=True)
class Chunk:
    """A stretch of concatenated speech, plus the map back to the original recording."""

    index: int
    path: Path
    pieces: list[Piece] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.pieces[-1].chunk_end if self.pieces else 0.0

    @property
    def source_start(self) -> float:
        return self.pieces[0].source_start if self.pieces else 0.0

    @property
    def source_end(self) -> float:
        last = self.pieces[-1]
        return last.to_source(last.chunk_end) if self.pieces else 0.0

    def to_source(self, chunk_time: float) -> float:
        """Translate a timestamp from chunk time back to a position in the original file.

        A timestamp that lands in a join gap is snapped to the nearest real audio rather
        than reported inside silence — the engine put it there because the word straddled
        the boundary, so the honest answer is the edge of the neighbouring span.
        """
        if not self.pieces:
            return chunk_time
        for piece in self.pieces:
            if chunk_time < piece.chunk_start:
                return piece.source_start
            if chunk_time <= piece.chunk_end:
                return piece.to_source(chunk_time)
        last = self.pieces[-1]
        return last.to_source(last.chunk_end)


def group(spans: list[SpeechSpan], limit: float = CHUNK_SECONDS) -> list[list[SpeechSpan]]:
    """Batch consecutive spans so each batch holds at most ``limit`` seconds of speech."""
    batches: list[list[SpeechSpan]] = []
    current: list[SpeechSpan] = []
    total = 0.0
    for span in spans:
        if current and total + span.duration > limit:
            batches.append(current)
            current, total = [], 0.0
        current.append(span)
        total += span.duration
    if current:
        batches.append(current)
    return batches


async def build(
    source_wav: Path, spans: list[SpeechSpan], workdir: Path, *, limit: float = CHUNK_SECONDS
) -> list[Chunk]:
    """Cut, splice and write one WAV per batch, returning each with its piece table."""
    chunks: list[Chunk] = []
    for index, batch in enumerate(group(spans, limit), start=1):
        target = workdir / f"chunk{index:03d}.wav"
        chunks.append(await _build_one(source_wav, batch, target, index))
    return chunks


async def _build_one(source_wav: Path, batch: list[SpeechSpan], target: Path, index: int) -> Chunk:
    """Splice one batch of spans into a single WAV and record where each span landed."""
    chunk = Chunk(index=index, path=target)
    filters: list[str] = []
    concat_inputs: list[str] = []
    cursor = 0.0

    for position, span in enumerate(batch):
        start = max(0.0, span.start - EDGE_PAD)
        end = span.end + EDGE_PAD
        label = f"s{position}"
        filters.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{label}]")
        concat_inputs.append(f"[{label}]")
        chunk.pieces.append(
            Piece(chunk_start=cursor, chunk_end=cursor + (end - start), source_start=start)
        )
        cursor += end - start
        if position < len(batch) - 1:
            gap = f"g{position}"
            filters.append(
                f"anullsrc=r={media.ENGINE_RATE}:cl=mono,atrim=duration={JOIN_SILENCE}[{gap}]"
            )
            concat_inputs.append(f"[{gap}]")
            cursor += JOIN_SILENCE

    filters.append(f"{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=0:a=1[out]")
    await _render(source_wav, target, ";".join(filters))
    return chunk


async def _render(source_wav: Path, target: Path, filter_graph: str) -> None:
    """Run the splice as one ffmpeg filter graph — one process, no intermediate files."""
    ffmpeg = proc.require("ffmpeg", install_hint=media.FFMPEG_HINT)
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_wav),
        "-filter_complex", filter_graph, "-map", "[out]",
        "-ac", "1", "-ar", str(media.ENGINE_RATE), "-c:a", "pcm_s16le",
        str(target),
    ]  # fmt: skip
    await proc.run(argv, check=True)
