"""models — see the model pool, and fetch what you need before you need it.

The pool is deliberately a list you can extend (see :mod:`stt_cli.registry`): if something
better than Whisper appears with an MLX or ggml build, it becomes one more entry rather than
a rewrite. Downloads go through the resource guard, so a three-gigabyte model refuses to
start when the disk cannot hold it instead of failing at 94 percent.
"""

from __future__ import annotations

import argparse
import asyncio

from .. import backends, config, registry, resources
from .._errors import EXIT_OK, UsageError

NAME = "models"
SUMMARY = "list, download and inspect speech models"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt models", description=SUMMARY)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("ls", help="list the model pool (default)")
    pull = sub.add_parser("pull", help="download a model")
    pull.add_argument("model", help="model name from `stt models ls`")
    pull.add_argument("-b", "--backend", default=None, help="engine to fetch it for")
    sub.add_parser("pull-vad", help="download the Silero voice-activity model")
    sub.add_parser("where", help="print the model directory")
    args = parser.parse_args(argv)

    action = args.action or "ls"
    if action == "ls":
        return _list()
    if action == "pull":
        return _pull(args.model, args.backend)
    if action == "pull-vad":
        return _pull_vad()
    return _where()


def _list() -> int:
    print(f"{'name':<22} {'size':>7}  {'langs':<8} {'q/s':<5} {'engines':<18} notes")
    for spec in registry.MODELS:
        print(spec.row())
    print(
        f"\ndefault: {registry.DEFAULT_MODEL}"
        f"   cross-check default: {registry.DEFAULT_CROSS_CHECK}"
        f"\nq = transcription quality, s = speed, both 1..5 and rough."
        f"\nmodels live in {config.models_dir()}"
    )
    return EXIT_OK


def _pull(model: str, backend_name: str | None) -> int:
    spec = registry.get(model)
    name = backend_name or _default_engine(spec)
    backend = backends.resolve(name)
    resources.require_space(spec.size_bytes, path=config.models_dir(), what=f"the {model} model")
    for warning in resources.check_memory(spec.size_bytes, model=model):
        print(f"warning: {warning}")
    asyncio.run(backend.ensure_model(model))
    print(f"{model} is ready for the {backend.name} engine.")
    return EXIT_OK


def _default_engine(spec: registry.ModelSpec) -> str:
    """Prefer an engine that is both installed and able to run this particular model."""
    installed = {name for name, status in backends.survey() if status.ok}
    for candidate in backends.ORDER:
        if candidate in spec.engine_ids and candidate in installed:
            return candidate
    raise UsageError(
        what=f"no installed engine can run {spec.name}",
        why=(
            f"{spec.name} is published for {', '.join(sorted(spec.engine_ids))}; "
            f"installed: {', '.join(sorted(installed)) or 'none'}"
        ),
        how="run `stt setup`, or pick a different model with `stt models ls`",
    )


def _pull_vad() -> int:
    backend = backends.whispercpp_backend()
    status = backend.availability()
    if not status.ok:
        raise UsageError(
            what="the Silero model is only used by the whisper.cpp engine",
            why=status.detail,
            how="install whisper.cpp, or use --vad ffmpeg which needs no model",
        )
    path = asyncio.run(backend.ensure_vad_model())
    print(f"Silero voice-activity model ready: {path}")
    return EXIT_OK


def _where() -> int:
    print(config.models_dir())
    return EXIT_OK
