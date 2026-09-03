"""mlx_worker — a standalone script that runs one mlx-whisper decoding pass.

RUN, NEVER IMPORTED BY THE CLI. ``mlx-whisper`` pulls in MLX and a model runtime; making
that a dependency of stt-cli itself would turn a two-second install into a large one for
every user, including the majority who run whisper.cpp instead. So it lives in whatever
environment happens to have it — the CLI's own interpreter when someone installed the
``mlx`` extra, or a throwaway ``uv run --with mlx-whisper`` environment otherwise — and
talks to us the only way that works across both: argv in, one JSON object on stdout.

Keep this file dependency-free apart from mlx_whisper and the standard library. It is
executed by a Python we do not control.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="one mlx-whisper decode pass")
    parser.add_argument("--language", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--carry-context", action="store_true")
    parser.add_argument("--carry-prompt", action="store_true")
    # Answer what this install can do and exit, without loading a model or touching audio.
    # The pipeline asks before it computes the cache key, because an install that cannot
    # pin the prompt decodes a glossary run as if there were no glossary.
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--audio")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - depends on the host environment
        json.dump({"error": f"mlx_whisper is not importable: {exc}"}, sys.stdout)
        return 1

    if args.probe:
        json.dump(
            {"carry_initial_prompt": _accepts(mlx_whisper.transcribe, "carry_initial_prompt")},
            sys.stdout,
        )
        return 0
    if not args.audio or not args.model:
        json.dump({"error": "--audio and --model are required unless --probe is given"}, sys.stdout)
        return 1

    extra: dict[str, Any] = {}
    warnings: list[str] = []
    if args.carry_prompt:
        # Whisper drops the initial prompt after the first 30-second window unless it is
        # re-prepended, so without this the glossary biases half a minute of a two-hour
        # recording. `carry_initial_prompt` is the parameter that re-prepends it, and it is
        # newer than some installed mlx-whisper builds — hence the check rather than a
        # keyword argument that would turn an old build into a TypeError mid-run.
        if _accepts(mlx_whisper.transcribe, "carry_initial_prompt"):
            extra["carry_initial_prompt"] = True
        else:
            warnings.append(
                "this mlx-whisper is too old for carry_initial_prompt: the glossary biases "
                "only the first window of each chunk (upgrade mlx-whisper, or use whispercpp)"
            )

    result = mlx_whisper.transcribe(
        args.audio,
        path_or_hf_repo=args.model,
        language=args.language,
        temperature=args.temperature,
        word_timestamps=args.word_timestamps,
        initial_prompt=args.initial_prompt,
        # The single most important switch for long recordings: with it on, one hallucinated
        # sentence is fed back in as context and the model happily continues the fiction for
        # minutes. Off, a bad chunk stays a bad chunk. Off is the default here; the caller
        # turns it on only for the comparison pass that `--context-compare` runs.
        condition_on_previous_text=args.carry_context,
        verbose=None,
        **extra,
    )
    payload = _slim(result)
    if warnings:
        payload["warnings"] = warnings
    json.dump(payload, sys.stdout, ensure_ascii=False)
    return 0


def _accepts(function: Any, parameter: str) -> bool:
    """Does this build of mlx-whisper take that keyword argument?"""
    import inspect

    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C or wrapped callable
        return False


def _slim(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the caller parses — the raw result carries whole token arrays."""
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", ""),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "words": [
                    {
                        "start": w.get("start"),
                        "end": w.get("end"),
                        "word": w.get("word", ""),
                        "probability": w.get("probability"),
                    }
                    for w in (seg.get("words") or [])
                ],
            }
        )
    return {"language": result.get("language"), "segments": segments}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
