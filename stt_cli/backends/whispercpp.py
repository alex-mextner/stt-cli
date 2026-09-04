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
from .base import Availability, DecodeRequest, VadProvider, confidence_from_probs, warn_once

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


# Context budget granted to a carried glossary: enough for the prompt itself plus a few
# tokens, not enough for a repetition loop to build on.
PROMPT_CONTEXT = 64

# `whisper-cli --help` prints and exits; anything slower than this is a broken binary.
_HELP_TIMEOUT = 20.0


class WhisperCppBackend:
    """The whisper.cpp engine, located once and reused for every chunk of a run."""

    name = NAME

    def __init__(self, root: str | None = None) -> None:
        self._root = _resolve_root(root)
        self._cli = _find_binary("whisper-cli", self._root)
        self._vad_binary = _find_binary("whisper-vad-speech-segments", self._root)
        # Probed lazily, once per run: see can_pin_prompt.
        self._carries_prompt: bool | None = None

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
        carry = request.carry_prompt and await self.can_pin_prompt()
        with tempfile.TemporaryDirectory(prefix="stt-whispercpp-") as tmp:
            stem = Path(tmp) / "out"
            argv = self._argv(request, stem, carry_prompt=carry)
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

    def honours_context_budget(self) -> bool:
        """Yes: `-mc` is a token count, so `short` and `full` really are different runs."""
        return True

    def pinning_the_prompt_costs_context(self) -> bool:
        """Yes. Measured: with `-mc 0` the initial prompt has no effect at all, so a carried
        glossary has to buy itself a budget — see PROMPT_CONTEXT and `_argv`."""
        return True

    async def can_pin_prompt(self) -> bool:
        """Does this whisper-cli have --carry-initial-prompt? Asked once per run.

        The flag is not ancient, and the binary is whatever is on PATH — a Homebrew install
        can easily predate it. Passing an unknown flag makes whisper-cli print its usage and
        exit, so a single `stt dict add` would otherwise break every transcription on that
        machine with an error that says nothing about versions.
        """
        if self._carries_prompt is None:
            if self._cli is None:
                return False
            result = await proc.run([str(self._cli), "--help"], timeout=_HELP_TIMEOUT)
            printed = result.stdout + result.stderr
            if not printed.strip():
                # A binary that printed no usage at all did not answer the question, and
                # "did not answer" is not "does not support it". Guessing here would decode
                # without the glossary AND file the run under a shortfall in the cache key.
                raise EngineError(
                    what="could not ask whisper-cli what it supports",
                    why=f"`{self._cli} --help` printed nothing",
                    how="check the binary runs at all, or reinstall whisper.cpp",
                )
            self._carries_prompt = "--carry-initial-prompt" in printed
            if not self._carries_prompt:
                warn_once(
                    "this whisper-cli has no --carry-initial-prompt, so the glossary reaches "
                    "at most the first window of each chunk, and nothing at all unless a "
                    "context budget is asked for (--context short) — upgrade whisper.cpp"
                )
        return self._carries_prompt

    def _argv(self, request: DecodeRequest, stem: Path, *, carry_prompt: bool) -> list[str]:
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
            "-mc", str(request.max_context),
            "-tp", f"{request.temperature:.2f}",
        ]  # fmt: skip
        if request.initial_prompt and carry_prompt:
            # Measured, not assumed: with -mc 0 the initial prompt has NO effect at all —
            # whisper.cpp builds the prompt only inside `if (n_max_text_ctx > 0)`, and two
            # runs over the same recording with and without --prompt came back byte for
            # byte identical. So a carried glossary has to buy itself a budget. With
            # --carry-initial-prompt the glossary is pinned as a static prefix and only
            # what is left of PROMPT_CONTEXT can hold carried-back output, which keeps the
            # loop risk bounded instead of reopening it completely.
            argv += ["--carry-initial-prompt"]
            argv[argv.index("-mc") + 1] = str(max(request.max_context, PROMPT_CONTEXT))
            if request.max_context < PROMPT_CONTEXT:
                warn_once(
                    f"glossary carried: whisper.cpp needs a context budget for it, so this "
                    f"run decodes with -mc {PROMPT_CONTEXT} instead of {request.max_context}"
                )
        argv += ["-l", request.language or "auto"]
        if request.threads:
            argv += ["-t", str(request.threads)]
        if request.initial_prompt and self._prompt_lands(request, carry_prompt=carry_prompt):
            argv += ["--prompt", request.initial_prompt]
        # A greedy pass at temperature 0 is deterministic and reproducible, which is what
        # the cache key promises. Sampling only happens when a variant explicitly asks.
        if request.temperature == 0.0:
            argv += ["-bo", "1", "-bs", "1"]
        return argv

    @staticmethod
    def _prompt_lands(request: DecodeRequest, *, carry_prompt: bool) -> bool:
        """Would `--prompt` actually change anything on this command line?

        On a whisper-cli old enough to lack ``--carry-initial-prompt`` the branch above never
        raises the budget, and the default budget is zero — so ``--prompt`` was passed to a
        decoder that, by this file's own measurement, ignores it entirely. The flag then
        cost nothing and bought nothing, while the warning told the user the glossary was
        reaching the first window. Buying the budget anyway is the wrong trade here: without
        the carry flag nothing pins the glossary to the front, so the tokens would be spent
        on the decoder's own previous output — which is the repetition loop `-mc 0` exists
        to prevent, reopened in full for a glossary that would be evicted from the window
        regardless. So the prompt is dropped, the warning says the glossary needs
        ``--context short`` on this binary, and `can_pin_prompt` has already written the
        shortfall into the run's cache key, so an upgraded whisper.cpp re-transcribes.
        """
        return carry_prompt or request.max_context > 0


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


def server_binary(root: str | None = None) -> Path | None:
    """The `whisper-server` binary from the same build as `whisper-cli`, if there is one.

    Live dictation needs a model that stays loaded between questions, which the one-shot CLI
    cannot give it. Both binaries come out of the same whisper.cpp build, so this looks in
    the same places `whisper-cli` is looked for — but they are packaged separately often
    enough that finding one says nothing about the other.
    """
    return _find_binary("whisper-server", _resolve_root(root))


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
