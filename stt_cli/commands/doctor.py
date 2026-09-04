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

    import asyncio

    from .. import diarize

    installed = asyncio.run(diarize.ready())
    print(f"  {_mark(installed)} speaker diarization: {'ready' if installed else 'not ready'}")
    if not installed:
        print(f"      fix: {diarize.INSTALL_HINT}")
    _dictation_line()


def _dictation_line() -> None:
    """Whether `stt mic` would work, without starting it.

    Its two requirements fail in ways that are hard to read at the moment they fail: a
    missing `whisper-server` looks like whisper.cpp being absent when `whisper-cli` is right
    there, and a missing Accessibility grant looks like a keyboard that types nothing at all.
    Both are worth answering here rather than by trying it and wondering.
    """
    from .. import config

    # The configured checkout, the way `stt mic` asks. Without it, somebody whose whisper.cpp
    # lives where they told stt about it was shown "NOT found" and an install instruction for
    # something they had already installed.
    #
    # Reading the config is allowed to FAIL here. `stt doctor` is the command somebody runs
    # because their setup is broken, and a hand-edited config.json is one of the ways it can
    # be; letting that diagnosed error out stopped the report at this line, so the storage
    # section and the summary never printed for the one command whose job is to say what is
    # wrong. A config that cannot be read is reported as a line, not as the end of the report.
    from .._errors import SttError
    from ..backends import whispercpp
    from ..live import quartz

    try:
        root = config.load_settings().whispercpp_root
    except SttError as unreadable:
        print(f"  {_mark(False)} live dictation (`stt mic`): config unreadable")
        print(f"      fix: {unreadable.how}")
        return
    server = whispercpp.server_binary(root)
    granted = quartz.trusted()
    print(
        f"  {_mark(bool(server) and granted)} live dictation (`stt mic`): "
        f"whisper-server {'found' if server else 'NOT found'}, "
        f"Accessibility {'granted' if granted else 'NOT granted'}"
    )
    if not server:
        # The whisper-server-specific text, not the generic whisper.cpp one. They are two
        # hints for one failure, and the generic one never names `whisper-server` — so the
        # report told somebody to install something they already had, without mentioning
        # that the packaged build may not carry the binary `stt mic` needs.
        from ..live.dictation import SERVER_HINT

        print(f"      fix: {SERVER_HINT}")
    if not granted:
        print(
            "      fix: grant Accessibility to your terminal in System Settings > "
            "Privacy & Security, then start a new terminal"
        )


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
