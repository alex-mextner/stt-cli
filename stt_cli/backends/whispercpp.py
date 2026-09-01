"""backends.whispercpp — drive the whisper.cpp binaries.

WHY THIS ENGINE IS THE DEFAULT ON A MAC
    It runs on Metal, it needs no Python runtime beside ours, and it ships the two things
    this tool leans on hardest: a built-in Silero voice-activity detector, and a full JSON
    output (``-ojf``) carrying a probability for every single token. That token stream is
    where per-segment confidence comes from, and confidence is what drives variants,
    hallucination detection and the LLM correction pass.

MODEL FILES
    whisper.cpp wants a ``ggml-*.bin`` on disk. We keep them in the tool's own model
    directory rather than inside somebody's whisper.cpp checkout, so wiping and rebuilding
    the checkout never costs a 3 GB re-download, and so ``stt models`` has one place to
    look. An existing checkout's ``models/`` is still searched first — if the file is
    already there, downloading a second copy would be silly.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .. import config, proc, registry, resources
from .._errors import EngineError, NetworkError
from ..jsonio import JsonDict, as_dict, as_dicts, as_float, as_str
from ..models import Segment, Word
from .base import Availability, DecodeRequest, VadProvider, confidence_from_probs

NAME = "whispercpp"

# Where a build is likely to be, in the order worth trying: an explicit override, the
# user's own checkout, then anything already on PATH (Homebrew installs `whisper-cli`).
_CHECKOUT_HINTS = (
    "~/xp/whisper.cpp",
    "~/whisper.cpp",
    "~/src/whisper.cpp",
    "~/work/whisper.cpp",
)

_HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# The VAD weights are published separately from the transcription models.
_VAD_BASE = "https://huggingface.co/ggml-org/whisper-vad/resolve/main"
SILERO_FILE = "ggml-silero-v5.1.2.bin"
# A whisper.cpp checkout ships its own copy under a test-fixture name; it is the same
# network and works fine, so reuse it rather than downloading a duplicate.
_SILERO_ALIASES = (
    "ggml-silero-v5.1.2.bin",
    "ggml-silero-v6.2.0.bin",
    "for-tests-silero-v6.2.0-ggml.bin",
)

INSTALL_HINT = (
    "install a whisper.cpp build — `brew install whisper-cpp`, or clone "
    "https://github.com/ggml-org/whisper.cpp and `cmake -B build && cmake --build build -j`"
)


class WhisperCppBackend:
    """The whisper.cpp engine, located once and reused for every chunk of a run."""

    name = NAME

    def __init__(self, root: str | None = None) -> None:
        self._root = _resolve_root(root)
        self._cli = _find_binary("whisper-cli", self._root)
        self._vad_binary = _find_binary("whisper-vad-speech-segments", self._root)

    # ── availability ──────────────────────────────────────────────────────────
    def availability(self) -> Availability:
        if self._cli is None:
            return Availability(False, "whisper-cli binary not found", INSTALL_HINT)
        where = self._root or self._cli.parent
        return Availability(True, f"whisper-cli at {where}")

    def vad_provider(self) -> VadProvider | None:
        """whisper.cpp ships Silero — the reason this engine is the better default."""
        model = self.silero_model()
        if self._vad_binary is None or model is None:
            return None
        return VadProvider(binary=self._vad_binary, model=model)

    @property
    def vad_binary(self) -> Path | None:
        return self._vad_binary

    def silero_model(self) -> Path | None:
        """The Silero VAD weights, from the tool's model directory or a nearby checkout."""
        candidates = [p for name in _SILERO_ALIASES for p in self._model_candidates(name)]
        return _existing(candidates)

    # ── models ────────────────────────────────────────────────────────────────
    async def ensure_model(self, model: str) -> None:
        _, filename = registry.require_for_engine(model, NAME)
        if _existing(self._model_candidates(filename)):
            return
        spec = registry.get(model)
        target = config.models_dir() / filename
        resources.require_space(
            spec.size_bytes, path=config.models_dir(), what=f"the {model} model"
        )
        for warning in resources.check_memory(spec.size_bytes, model=model):
            print(f"stt: warning: {warning}")
        await _download(f"{_HF_BASE}/{filename}", target)

    async def ensure_vad_model(self) -> Path:
        found = self.silero_model()
        if found:
            return found
        target = config.models_dir() / SILERO_FILE
        resources.require_space(
            8 * resources.GIB // 1000, path=config.models_dir(), what="the Silero VAD model"
        )
        await _download(f"{_VAD_BASE}/{SILERO_FILE}", target)
        return target

    def model_path(self, model: str) -> Path:
        _, filename = registry.require_for_engine(model, NAME)
        found = _existing(self._model_candidates(filename))
        if found is None:
            raise EngineError(
                what=f"model file {filename} is missing",
                why="it was not found in the model directory or a whisper.cpp checkout",
                how=f"run `stt models pull {model}`",
            )
        return found

    def _model_candidates(self, filename: str) -> list[Path]:
        paths = [config.models_dir() / filename]
        if self._root:
            paths.append(self._root / "models" / filename)
        return paths

    # ── decoding ──────────────────────────────────────────────────────────────
    async def decode(self, request: DecodeRequest) -> list[Segment]:
        if self._cli is None:
            raise EngineError(
                what="whisper.cpp is not installed",
                why="no whisper-cli binary was found",
                how=INSTALL_HINT,
            )
        with tempfile.TemporaryDirectory(prefix="stt-whispercpp-") as tmp:
            stem = Path(tmp) / "out"
            argv = self._argv(request, stem)
            result = await proc.run(argv, timeout=proc.DEFAULT_TIMEOUT)
            payload = stem.with_suffix(".json")
            if not result.ok or not payload.is_file():
                raise EngineError(
                    what="whisper.cpp failed to transcribe a chunk",
                    why=result.tail() or "the binary produced no JSON output",
                    how=(
                        "run with -v to see the exact command; check the model file "
                        "is not truncated"
                    ),
                )
            raw = as_dict(json.loads(payload.read_text("utf-8")))
        return _parse(raw, offset=request.offset)

    def _argv(self, request: DecodeRequest, stem: Path) -> list[str]:
        """Build the whisper-cli command line for one decoding pass.

        Two flags here are doing real work rather than tuning.

        ``--suppress-nst`` suppresses the non-speech tokens the model uses to narrate sounds
        (``[music]``, ``(laughter)``). Those are a common seed for a run of hallucinated
        filler, and they do not belong in a transcript anyway.

        ``--max-context 0`` stops the decoder feeding its own previous output back in as
        context. Left on, one bad chunk becomes many: a hallucinated sentence is re-read as
        context and the model continues the fiction, and a stylistic drift (dropping
        capitalisation and punctuation, say) locks in and persists for the rest of the file.
        Both were observed on real recordings before this was set. It costs a little
        cross-sentence continuity and buys robustness, which is the trade this tool exists to
        make — and it matches ``condition_on_previous_text=False`` on the MLX side, so the
        two engines behave the same way.
        """
        assert self._cli is not None
        argv = [
            str(self._cli),
            "-m", str(self.model_path(request.model)),
            "-f", str(request.wav),
            "-oj", "-ojf", "-of", str(stem),
            "-np",
            "--suppress-nst",
            "-mc", "0",
            "-tp", f"{request.temperature:.2f}",
        ]  # fmt: skip
        argv += ["-l", request.language or "auto"]
        if request.threads:
            argv += ["-t", str(request.threads)]
        if request.initial_prompt:
            argv += ["--prompt", request.initial_prompt]
        # A greedy pass at temperature 0 is deterministic and reproducible, which is what
        # the cache key promises. Sampling only happens when a variant explicitly asks.
        if request.temperature == 0.0:
            argv += ["-bo", "1", "-bs", "1"]
        return argv


def _parse(raw: JsonDict, *, offset: float) -> list[Segment]:
    """Turn whisper.cpp's full JSON into segments, carrying token probabilities across."""
    segments: list[Segment] = []
    for item in as_dicts(raw.get("transcription")):
        offsets = as_dict(item.get("offsets"))
        start = as_float(offsets.get("from")) / 1000.0 + offset
        end = as_float(offsets.get("to")) / 1000.0 + offset
        text = as_str(item.get("text")).strip()
        tokens = as_dicts(item.get("tokens"))
        # Special tokens (``[_TT_..]``, ``<|...|>``) carry no linguistic content; folding
        # their probabilities into the average would flatter or punish the score at random.
        probs = [
            as_float(token.get("p"))
            for token in tokens
            if not as_str(token.get("text")).startswith(("[_", "<|"))
        ]
        segments.append(
            Segment(
                start=start,
                end=end,
                text=text,
                confidence=confidence_from_probs(probs),
                words=_words(tokens, offset),
            )
        )
    return segments


def _words(tokens: list[JsonDict], offset: float) -> list[Word]:
    """Per-token timings, when whisper.cpp was asked for them (``-dtw``); empty otherwise."""
    words: list[Word] = []
    for token in tokens:
        text = as_str(token.get("text"))
        offsets = as_dict(token.get("offsets"))
        if text.startswith(("[_", "<|")) or not offsets:
            continue
        words.append(
            Word(
                start=as_float(offsets.get("from")) / 1000.0 + offset,
                end=as_float(offsets.get("to")) / 1000.0 + offset,
                text=text,
                probability=as_float(token.get("p")) or None,
            )
        )
    return words


def _resolve_root(explicit: str | None) -> Path | None:
    """Find a whisper.cpp checkout, so its already-downloaded models can be reused."""
    candidates = [explicit, os.environ.get("STT_WHISPERCPP_ROOT"), *_CHECKOUT_HINTS]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "build" / "bin" / "whisper-cli").is_file() or (path / "models").is_dir():
            return path
    return None


def _find_binary(name: str, root: Path | None) -> Path | None:
    if root:
        for relative in (Path("build") / "bin" / name, Path("bin") / name, Path(name)):
            candidate = root / relative
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.is_file() and p.stat().st_size > 0), None)


async def _download(url: str, target: Path) -> None:
    """Fetch a model to a temporary name and move it into place only once complete.

    A model interrupted mid-download is the worst kind of corrupt file: it looks present,
    so the next run loads it and dies deep inside the engine with an unhelpful error.
    Writing to ``<name>.part`` and renaming makes a partial download simply not exist.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"stt: downloading {target.name} ...")
    curl = proc.which("curl")
    if curl is None:
        raise NetworkError(
            what="cannot download the model",
            why="curl is not available",
            how=f"download {url} manually into {target.parent}",
        )
    result = await proc.run(
        [curl, "-fL", "--progress-bar", "-o", str(partial), url], timeout=2 * 60 * 60
    )
    if not result.ok or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise NetworkError(
            what=f"download failed: {target.name}",
            why=result.tail() or f"curl exited with {result.code}",
            how=f"check the network, or fetch {url} manually into {target.parent}",
        )
    partial.replace(target)
