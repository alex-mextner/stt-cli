"""setup — look at what this machine already has, explain the choice, and install the rest.

WHY AN INTERACTIVE CHOOSER RATHER THAN A HARDCODED DEFAULT
    The right engine depends on the machine. Someone with a whisper.cpp build already
    compiled should use it — it is there, it runs on Metal, and it brings the Silero
    voice-activity detector this tool leans on. Someone starting from nothing is better
    served by mlx-whisper, which ``uv`` installs in a minute with no compiler. Picking one
    globally would be wrong for half the users, and picking silently would leave them
    wondering why transcription is slow or why a model downloaded twice.

    So this reports what it found, states each option's actual trade-off, and stores the
    answer. Non-interactive runs (CI, a pipe) take the best available option and say so
    rather than hanging on a prompt that nobody will ever answer.
"""

from __future__ import annotations

import argparse
import sys

from .. import backends, config, llm, proc, registry
from .._errors import EXIT_MISSING_DEP, EXIT_OK, MissingDependencyError

NAME = "setup"
SUMMARY = "detect what is installed, choose an engine, download a model"

# What to tell the user about each engine — the honest version, trade-offs included.
PROS_CONS = {
    "whispercpp": (
        "C++ on Metal, no Python runtime; ships the Silero voice-activity detector "
        "and per-token confidence",
        "needs a compiled binary (brew install whisper-cpp, or build the repo yourself)",
    ),
    "mlx": (
        "installs with uv in about a minute, no compiler; models download themselves; "
        "access to model variants with no ggml build",
        "Apple Silicon only; voice-activity detection falls back to ffmpeg unless "
        "whisper.cpp is also present",
    ),
}


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt setup", description=SUMMARY)
    parser.add_argument("--backend", default=None, help="skip the question and pick this engine")
    parser.add_argument(
        "--model", default=None, help="model to make available (default: %(default)s)"
    )
    parser.add_argument(
        "--yes", action="store_true", help="accept the recommended choice without asking"
    )
    args = parser.parse_args(argv)

    if not _check_ffmpeg():
        return EXIT_MISSING_DEP

    survey = backends.survey()
    _report(survey)
    chosen = args.backend or _choose(survey, assume_yes=args.yes)
    if chosen is None:
        return EXIT_MISSING_DEP

    model = args.model or registry.DEFAULT_MODEL
    config.save_setting("backend", chosen)
    config.save_setting("model", model)
    print(f"\nsaved: backend={chosen}, model={model}  ({config.config_path()})")

    _install_model(chosen, model)
    _report_llm()
    print("\nReady. Try:  stt <your recording> -f txt")
    return EXIT_OK


def _check_ffmpeg() -> bool:
    if proc.which("ffmpeg") and proc.which("ffprobe"):
        return True
    print("ffmpeg is required and was not found.\n  fix: brew install ffmpeg", file=sys.stderr)
    return False


def _report(survey: list[tuple[str, backends.Availability]]) -> None:
    print("engines on this machine:\n")
    for name, status in survey:
        pros, cons = PROS_CONS.get(name, ("", ""))
        mark = "available" if status.ok else "NOT installed"
        print(f"  {name}  [{mark}]")
        print(f"    {status.detail}")
        if pros:
            print(f"    +  {pros}")
        if cons:
            print(f"    -  {cons}")
        if not status.ok and status.how_to_install:
            print(f"    install: {status.how_to_install}")
        print()


def _choose(survey: list[tuple[str, backends.Availability]], *, assume_yes: bool) -> str | None:
    """Ask which engine to use, defaulting to the best one that actually works here."""
    usable = [name for name, status in survey if status.ok]
    if not usable:
        print(
            "No engine is installed. The quickest route from here:\n"
            "  brew install whisper-cpp      (native, brings the Silero detector)\n"
            "  or: uv tool install --with mlx-whisper stt-cli",
            file=sys.stderr,
        )
        return None
    recommended = usable[0]
    if assume_yes or not sys.stdin.isatty():
        reason = "already installed" if len(usable) == 1 else "first available in preference order"
        print(f"using {recommended} ({reason}).")
        return recommended

    options = ", ".join(usable)
    answer = input(f"which engine should stt use by default? [{options}] ({recommended}): ").strip()
    if not answer:
        return recommended
    if answer not in usable:
        print(f"{answer!r} is not available here — using {recommended}.")
        return recommended
    return answer


def _install_model(backend_name: str, model: str) -> None:
    """Make sure the chosen model is actually on disk before the first real run needs it."""
    import asyncio

    spec = registry.get(model)
    print(f"\nmodel: {model} — {spec.summary} ({spec.size_bytes / 1024**2:.0f} MiB)")
    backend = backends.create(backend_name)
    try:
        asyncio.run(backend.ensure_model(model))
        print("model is ready.")
    except MissingDependencyError as exc:
        print(f"could not prepare the model: {exc.what}\n  {exc.how}", file=sys.stderr)


def _report_llm() -> None:
    tools = llm.available()
    if tools:
        print(f"\nLLM correction and --summary will use: {tools[0].name}")
    else:
        print(
            f"\nnote: --fix and --summary need one of {', '.join(llm.ORDER)} on PATH. "
            "Transcription works without them."
        )
