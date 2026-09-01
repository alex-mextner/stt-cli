"""doctor — say what is installed, what is missing, and exactly how to fix each gap.

A speech pipeline has a lot of moving parts that live outside the package: ffmpeg, an
engine, model files, an LLM CLI, optionally pyannote. When one of them is absent the
failure surfaces halfway through a long run, in whatever language that component speaks.
This command asks all of them up front and reports in one place, so a broken setup is a
thirty-second check rather than a discovery made forty minutes into a transcription.
"""

from __future__ import annotations

import argparse

from .. import backends, config, llm, proc, resources
from .._errors import EXIT_MISSING_DEP, EXIT_OK

NAME = "doctor"
SUMMARY = "check ffmpeg, engines, models, LLM tools and disk space"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt doctor", description=SUMMARY)
    parser.parse_args(argv)

    required_ok = _media_section()
    engine_ok = _engine_section()
    _optional_section()
    _storage_section()

    if required_ok and engine_ok:
        print("\nEverything stt needs is present.")
        return EXIT_OK
    print("\nSomething required is missing — see the fix lines above, or run `stt setup`.")
    return EXIT_MISSING_DEP


def _media_section() -> bool:
    print("media (required)")
    ok = True
    for binary, hint in (("ffmpeg", "brew install ffmpeg"), ("ffprobe", "brew install ffmpeg")):
        found = proc.which(binary)
        print(f"  {_mark(bool(found))} {binary:<10} {found or 'not found'}")
        if not found:
            print(f"      fix: {hint}")
            ok = False
    return ok


def _engine_section() -> bool:
    print("\nspeech engines (at least one required)")
    any_ok = False
    for name, status in backends.survey():
        print(f"  {_mark(status.ok)} {name:<10} {status.detail}")
        if not status.ok and status.how_to_install:
            print(f"      fix: {status.how_to_install}")
        any_ok = any_ok or status.ok
    if any_ok:
        _model_section()
    return any_ok


def _model_section() -> None:
    from ..registry import MODELS

    print("\n  models on disk")
    backend = backends.whispercpp_backend()
    if not backend.availability().ok:
        print("    (whisper.cpp not installed; the mlx engine downloads models on demand)")
        return
    for spec in MODELS:
        if "whispercpp" not in spec.engine_ids:
            continue
        try:
            path = backend.model_path(spec.name)
            print(f"    ✓ {spec.name:<18} {path}")
        except Exception:
            print(f"    · {spec.name:<18} not downloaded (`stt models pull {spec.name}`)")
    silero = backend.silero_model()
    print(f"    {_mark(bool(silero))} {'silero VAD':<18} {silero or _MISSING_VAD}")


def _optional_section() -> None:
    print("\noptional features")
    tools = llm.available()
    names = ", ".join(t.name for t in tools) or "none"
    print(f"  {_mark(bool(tools))} LLM correction / --summary: {names}")
    if not tools:
        print(f"      fix: install one of {', '.join(llm.ORDER)} to enable --fix and --summary")

    from .. import diarize

    installed = diarize.is_installed()
    print(
        f"  {_mark(installed)} speaker diarization: {'installed' if installed else 'not installed'}"
    )
    if not installed:
        print(f"      fix: {diarize.INSTALL_HINT}")


def _storage_section() -> None:
    from ..archive import Archive

    res = resources.probe(config.app_home())
    print("\nstorage")
    print(f"  archive: {config.app_home()}")
    print(f"  machine: {res.human()}")
    with Archive() as store:
        runs, audio, transcripts = store.usage()
    mib = 1024 * 1024
    print(
        f"  stored:  {runs} run(s), {audio / mib:.0f} MiB audio, "
        f"{transcripts / mib:.1f} MiB transcripts"
    )


_MISSING_VAD = "not downloaded (`stt models pull-vad`)"


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"
