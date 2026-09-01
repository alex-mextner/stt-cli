"""backends.mlx — drive mlx-whisper, Apple's MLX runtime for the Whisper family.

WHEN THIS ENGINE IS THE RIGHT ONE
    It needs no compiler and no checkout: ``uv`` installs it, and the model downloads
    itself from Hugging Face on first use. That makes it the path of least resistance on a
    machine with no whisper.cpp build, and it gives access to model variants that have no
    ggml conversion (the distilled English models, for instance).

    It is also the natural second opinion. Cross-checking a shaky segment against a
    *different* runtime and a *different* model catches the class of error where one model
    is confidently wrong — something a second pass through the same engine cannot do.

HOW IT IS INVOKED
    Out of process, through :mod:`mlx_worker`. See that module for why. The upshot here is
    that this class never imports mlx, so ``stt --help`` on a machine without it is still
    instant, and a broken MLX install degrades to "this engine is unavailable" instead of
    an import error at startup.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .. import proc, registry
from .._errors import EngineError
from ..jsonio import JsonDict, as_dict, as_dicts, as_float, as_opt_float, as_str
from ..models import Segment, Word
from .base import Availability, DecodeRequest, VadProvider, confidence_from_logprob

NAME = "mlx"
WORKER = Path(__file__).with_name("mlx_worker.py")

INSTALL_HINT = (
    "run `stt setup`, or install it yourself: `uv tool install --with mlx-whisper stt-cli`"
)


class MlxBackend:
    """The mlx-whisper engine, run through whichever interpreter can import it."""

    name = NAME

    def __init__(self) -> None:
        self._runner = _resolve_runner()

    def availability(self) -> Availability:
        if self._runner is None:
            return Availability(
                False,
                "mlx-whisper is not importable and uv is not available to supply it",
                INSTALL_HINT,
            )
        kind, argv = self._runner
        detail = (
            "mlx-whisper installed in this environment"
            if kind == "direct"
            else f"mlx-whisper via `{' '.join(argv[:3])} ...` (uv resolves it on first use)"
        )
        return Availability(True, detail)

    def vad_provider(self) -> VadProvider | None:
        """None: mlx-whisper ships no detector, so voice activity falls back to ffmpeg."""
        return None

    async def ensure_model(self, model: str) -> None:
        """Validate the model name and check there is room, before anything downloads.

        mlx-whisper fetches the weights itself on first use, inside a worker process, where a
        disk-full failure surfaces as an unhelpful traceback partway through a download. So
        the same guard the whisper.cpp path applies to its own downloads is applied here too,
        against the Hugging Face cache, before the worker is ever started.
        """
        from pathlib import Path as _Path

        from .. import resources

        spec, _ = registry.require_for_engine(model, NAME)
        cache = _Path(os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface"))
        resources.require_space(spec.size_bytes, path=cache, what=f"the {model} model")
        for warning in resources.check_memory(spec.size_bytes, model=model):
            print(f"stt: warning: {warning}")

    async def decode(self, request: DecodeRequest) -> list[Segment]:
        if self._runner is None:
            raise EngineError(
                what="the mlx engine is not available",
                why="mlx-whisper could not be imported and uv is not installed",
                how=INSTALL_HINT,
            )
        _, repo = registry.require_for_engine(request.model, NAME)
        _, base = self._runner
        argv = [
            *base,
            str(WORKER),
            "--audio", str(request.wav),
            "--model", repo,
            "--temperature", f"{request.temperature:.2f}",
        ]  # fmt: skip
        if request.language:
            argv += ["--language", request.language]
        if request.word_timestamps:
            argv += ["--word-timestamps"]
        if request.initial_prompt:
            argv += ["--initial-prompt", request.initial_prompt]

        result = await proc.run(argv, timeout=proc.DEFAULT_TIMEOUT)
        return _parse(_payload(result), offset=request.offset)


def _payload(result: proc.Result) -> JsonDict:
    """Read the worker's JSON, turning any failure into a diagnosed engine error."""
    if not result.ok:
        raise EngineError(
            what="mlx-whisper failed",
            why=result.tail() or f"the worker exited with {result.code}",
            how="check the model name and that mlx-whisper installed correctly (`stt doctor`)",
        )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EngineError(
            what="mlx-whisper produced unreadable output",
            why=f"expected one JSON object, got: {result.stdout[:200]!r}",
            how="run `stt doctor` to check the mlx installation",
        ) from exc
    if "error" in raw:
        raise EngineError(
            what="mlx-whisper could not run",
            why=str(raw["error"]),
            how=INSTALL_HINT,
        )
    return as_dict(raw)


def _parse(raw: JsonDict, *, offset: float) -> list[Segment]:
    segments: list[Segment] = []
    for item in as_dicts(raw.get("segments")):
        words = [
            Word(
                start=as_float(word["start"]) + offset,
                end=as_float(word["end"]) + offset,
                text=as_str(word.get("word")),
                probability=as_opt_float(word.get("probability")),
            )
            for word in as_dicts(item.get("words"))
            if word.get("start") is not None and word.get("end") is not None
        ]
        segments.append(
            Segment(
                start=as_float(item.get("start")) + offset,
                end=as_float(item.get("end")) + offset,
                text=as_str(item.get("text")).strip(),
                words=words,
                confidence=confidence_from_logprob(as_opt_float(item.get("avg_logprob"))),
                no_speech=as_opt_float(item.get("no_speech_prob")),
            )
        )
    return segments


def _resolve_runner() -> tuple[str, list[str]] | None:
    """Pick the interpreter that will run the worker: ours if it can, else a uv environment."""
    import importlib.util
    import sys

    if importlib.util.find_spec("mlx_whisper") is not None:
        return "direct", [sys.executable]
    uv = shutil.which("uv")
    if uv:
        # `uv run --with` builds (and then caches) an ephemeral environment, so this costs
        # a one-time resolve rather than a permanent dependency for everyone.
        return "uv", [uv, "run", "--quiet", "--with", "mlx-whisper", "python"]
    return None
