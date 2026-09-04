"""diarize_worker — a standalone script that runs one pyannote diarization pass.

RUN, NEVER IMPORTED BY THE CLI. `pyannote.audio` pulls in torch, which is about two and a
half gigabytes; making that a dependency of stt-cli would turn a small install into a large
one for everybody, including the majority who never diarize anything. So it lives wherever
it happens to be — the CLI's own interpreter when someone installed the extra, or a
throwaway `uv run --with pyannote.audio --with torch` environment otherwise — and talks to
us the only way that works across both: argv in, one JSON object on stdout.

This is the same shape as `backends/mlx_worker.py`, and for the same reason. Diarization
used to `pip install` into `sys.executable` instead, which on a Homebrew Python is refused
outright: PEP 668 marks it externally managed, so the install could not succeed at all.

Keep this file free of any dependency but pyannote, torch and the standard library. It is
executed by a Python we do not control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="one pyannote diarization pass")
    parser.add_argument("--audio", required=False)
    parser.add_argument("--speakers", type=int, default=None)
    # No --token. A command line is public: any user on the machine can read another's
    # argv out of `ps`, and diarization can run for an hour, so a token passed that way is
    # readable for an hour. It arrives in the environment instead, which is the same channel
    # huggingface_hub reads on its own.
    parser.add_argument("--pipeline", default="pyannote/speaker-diarization-3.1")
    # Answer whether this environment can diarize, without downloading a model or reading
    # audio. `stt diarize status` asks the environment that will do the work rather than
    # importing into the CLI's own interpreter, which may not be the one that has it.
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)

    if args.probe:
        return _say({"ready": _importable()})
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not args.audio or not token:
        return _say({"error": "--audio and a token in the environment are required"})
    try:
        return _say({"turns": _run(args, token)})
    except Exception as failure:  # the caller renders this; a traceback here helps nobody
        return _say({"error": f"{type(failure).__name__}: {failure}"})


def _importable() -> bool:
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    return True


def _run(args: argparse.Namespace, token: str) -> list[dict[str, Any]]:
    """The pyannote call itself, in the interpreter that actually has pyannote."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(args.pipeline, use_auth_token=token)
    _to_metal(pipeline)
    spoken = pipeline(args.audio, **({"num_speakers": args.speakers} if args.speakers else {}))
    return [
        {"start": float(segment.start), "end": float(segment.end), "speaker": str(label)}
        for segment, _, label in spoken.itertracks(yield_label=True)
    ]


def _to_metal(pipeline: object) -> None:
    """Move the pipeline onto Apple's GPU when the runtime supports it; CPU otherwise."""
    try:
        import torch

        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))  # type: ignore[attr-defined]
    except Exception:  # any failure here is a performance issue, never a correctness one
        pass


def _say(payload: dict[str, Any]) -> int:
    """One JSON object on stdout, which is the whole protocol."""
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
