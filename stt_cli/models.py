"""models — the data the whole pipeline passes around.

One shape flows end to end: every stage (engine -> variants -> cleaning -> diarization ->
LLM correction -> rendering) reads and returns a :class:`Transcript`. Stages only ever ADD
information (a variant, a confidence, a speaker, a flag) and never rewrite the structure,
so the archived JSON of a run is a complete, replayable record — you can re-render any
format, or re-run only the LLM pass, without touching the audio again.

Serialization is hand-written rather than pulled from a library because the archive holds
these files for years: an explicit ``to_dict``/``from_dict`` pair is a schema we control,
and unknown keys from a newer version are ignored rather than fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SCHEMA_VERSION = 1

# Why a segment was marked. Flags are additive and never delete text on their own: the
# renderers decide what a flag means (drop it, dim it, annotate it), so a false positive
# in cleaning is always recoverable from the archived JSON.
Flag = Literal[
    "hallucination",  # matched a known filler phrase the model emits over silence
    "loop",  # the decoder repeated itself (a phrase or an n-gram cycle)
    "low-confidence",  # below the confidence floor; a candidate for variants / LLM review
    "silence",  # overlaps a span VAD classified as non-speech
    "empty",  # no usable text after normalization
    "term",  # sounds like a dictionary term; the candidates are in `suspected_terms`
]

VariantKind = Literal["primary", "temperature", "model", "context", "llm"]


@dataclass(slots=True, frozen=True)
class Word:
    """One decoded word with its own timing and probability."""

    start: float
    end: float
    text: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text, "p": self.probability}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Word:
        return cls(
            start=float(raw["start"]),
            end=float(raw["end"]),
            text=str(raw["text"]),
            probability=_opt_float(raw.get("p")),
        )


@dataclass(slots=True)
class Variant:
    """An alternative reading of one segment, with where it came from and how sure it is.

    ``source`` is a human-readable provenance string (``whispercpp:large-v3@t0.4``,
    ``mlx:large-v3-turbo``, ``llm:codex``) so a transcript stays self-explanatory years
    later, and so the LLM correction pass can weigh a second *model*'s reading differently
    from the same model's second-best guess.
    """

    text: str
    source: str
    kind: VariantKind = "temperature"
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "kind": self.kind,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Variant:
        return cls(
            text=str(raw["text"]),
            source=str(raw.get("source", "unknown")),
            kind=raw.get("kind", "temperature"),
            confidence=_opt_float(raw.get("confidence")),
        )


@dataclass(slots=True)
class Segment:
    """One timed chunk of speech, plus everything later stages learned about it."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None
    no_speech: float | None = None
    speaker: str | None = None
    variants: list[Variant] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    # Phrases the dictionary's phonetic screen believes are misheard terms, as
    # (heard, suspected) pairs. A field of its own rather than more strings in `flags`:
    # these carry transcript text, and `flags` is a closed vocabulary that renderers join
    # into one cell (`<a,b>`, `a|b`). Smuggling arbitrary words through it puts somebody's
    # speech into a column that nothing downstream expects to have to quote.
    suspected_terms: list[tuple[str, str]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"start": self.start, "end": self.end, "text": self.text}
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.no_speech is not None:
            out["no_speech"] = self.no_speech
        if self.speaker:
            out["speaker"] = self.speaker
        if self.words:
            out["words"] = [w.to_dict() for w in self.words]
        if self.variants:
            out["variants"] = [v.to_dict() for v in self.variants]
        if self.flags:
            out["flags"] = list(self.flags)
        if self.suspected_terms:
            out["suspected_terms"] = [[heard, term] for heard, term in self.suspected_terms]
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Segment:
        return cls(
            start=float(raw["start"]),
            end=float(raw["end"]),
            text=str(raw.get("text", "")),
            words=[Word.from_dict(w) for w in raw.get("words", [])],
            confidence=_opt_float(raw.get("confidence")),
            no_speech=_opt_float(raw.get("no_speech")),
            speaker=raw.get("speaker"),
            variants=[Variant.from_dict(v) for v in raw.get("variants", [])],
            flags=list(raw.get("flags", [])),
            suspected_terms=[
                (str(pair[0]), str(pair[1]))
                for pair in raw.get("suspected_terms", [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            ],
        )


@dataclass(slots=True, frozen=True)
class SpeechSpan:
    """A stretch of audio a voice-activity detector considers speech."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class MediaInfo:
    """What the input file is, and when it was recorded.

    ``recorded_at`` is what absolute timestamps are anchored to. ``recorded_at_source``
    records HOW it was determined (a container tag, the filesystem birth time, the
    filename, an explicit flag) because the answer is a guess of varying quality and the
    reader deserves to know which one they got.
    """

    path: str
    sha256: str
    size_bytes: int
    duration: float
    container: str = ""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    has_video: bool = False
    recorded_at: datetime | None = None
    recorded_at_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "duration": self.duration,
            "container": self.container,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "has_video": self.has_video,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "recorded_at_source": self.recorded_at_source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MediaInfo:
        stamp = raw.get("recorded_at")
        return cls(
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw.get("size_bytes", 0)),
            duration=float(raw.get("duration", 0.0)),
            container=str(raw.get("container", "")),
            codec=str(raw.get("codec", "")),
            sample_rate=int(raw.get("sample_rate", 0)),
            channels=int(raw.get("channels", 0)),
            has_video=bool(raw.get("has_video", False)),
            recorded_at=datetime.fromisoformat(stamp) if stamp else None,
            recorded_at_source=str(raw.get("recorded_at_source", "unknown")),
        )


@dataclass(slots=True)
class EngineInfo:
    """Which engine and model produced the primary reading, and how it was configured."""

    backend: str
    model: str
    language: str | None = None
    vad: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.backend}:{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "vad": self.vad,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EngineInfo:
        return cls(
            backend=str(raw["backend"]),
            model=str(raw["model"]),
            language=raw.get("language"),
            vad=str(raw.get("vad", "none")),
            extra=dict(raw.get("extra", {})),
        )


@dataclass(slots=True)
class SummarySection:
    """One block of the structured summary (a topic, a decision list, an action list)."""

    title: str
    bullets: list[str] = field(default_factory=list)
    start: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "bullets": list(self.bullets), "start": self.start}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SummarySection:
        return cls(
            title=str(raw.get("title", "")),
            bullets=[str(b) for b in raw.get("bullets", [])],
            start=_opt_float(raw.get("start")),
        )


@dataclass(slots=True)
class Summary:
    """A structured summary of a transcript, produced by the LLM pass."""

    headline: str = ""
    sections: list[SummarySection] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "sections": [s.to_dict() for s in self.sections],
            "decisions": list(self.decisions),
            "actions": list(self.actions),
            "questions": list(self.questions),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Summary:
        return cls(
            headline=str(raw.get("headline", "")),
            sections=[SummarySection.from_dict(s) for s in raw.get("sections", [])],
            decisions=[str(x) for x in raw.get("decisions", [])],
            actions=[str(x) for x in raw.get("actions", [])],
            questions=[str(x) for x in raw.get("questions", [])],
            source=str(raw.get("source", "")),
        )


@dataclass(slots=True)
class Transcript:
    """The full result of one run: timed text plus every piece of provenance."""

    media: MediaInfo
    engine: EngineInfo
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    speech_spans: list[SpeechSpan] = field(default_factory=list)
    summary: Summary | None = None
    warnings: list[str] = field(default_factory=list)
    schema: int = SCHEMA_VERSION

    def text(self, sep: str = " ") -> str:
        return sep.join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speech_duration(self) -> float:
        return sum(s.duration for s in self.speech_spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "media": self.media.to_dict(),
            "engine": self.engine.to_dict(),
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
            "speech_spans": [[s.start, s.end] for s in self.speech_spans],
            "summary": self.summary.to_dict() if self.summary else None,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Transcript:
        summary = raw.get("summary")
        return cls(
            media=MediaInfo.from_dict(raw["media"]),
            engine=EngineInfo.from_dict(raw["engine"]),
            segments=[Segment.from_dict(s) for s in raw.get("segments", [])],
            language=raw.get("language"),
            speech_spans=[SpeechSpan(float(a), float(b)) for a, b in raw.get("speech_spans", [])],
            summary=Summary.from_dict(summary) if summary else None,
            warnings=list(raw.get("warnings", [])),
            schema=int(raw.get("schema", SCHEMA_VERSION)),
        )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)
