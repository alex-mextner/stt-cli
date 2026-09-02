"""transcribe — the command that does the thing: audio or video in, text out.

WHERE THE OUTPUT GOES, AND WHY
    One file, one format, no ``-o``: straight to stdout, because that is what makes ``stt
    rec.m4a | pbcopy`` work and it is what people actually want most of the time. Ask for
    several formats, or hand it several files, and stdout stops making sense — so those
    write real files named after the source. ``-o`` overrides either way: a path that looks
    like a file is a file, anything else is a directory to fill.

    Whatever happens on stdout, the full result is also in the archive. Piping into ``head``
    never loses work.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .. import config, formats, pipeline
from .._errors import UsageError
from ..archive import Archive, write_atomic
from ..backends.base import CONTEXT_TOKENS
from ..config import FORMATS, TIMESTAMP_MODES

NAME = "transcribe"
SUMMARY = "turn audio or video into text (the default command)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stt",
        description="Transcribe audio or video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  stt meeting.m4a                       text to stdout\n"
            "  stt talk.mp4 -f srt -o subs/          subtitles for a video\n"
            "  stt call.ogg -t absolute --tz Europe/Belgrade\n"
            "  stt rec.m4a --fix --summary -f md     corrected, summarized, one document\n"
            "  stt rec.m4a --diarize -f speakers     split into speaker turns\n"
        ),
    )
    parser.add_argument("inputs", nargs="+", metavar="FILE", help="audio or video files")
    _add_output_args(parser)
    _add_engine_args(parser)
    _add_vad_args(parser)
    _add_cleaning_args(parser)
    _add_dictionary_args(parser)
    _add_variant_args(parser)
    _add_enrichment_args(parser)
    _add_misc_args(parser)
    return parser


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument(
        "-f", "--format", default=None,
        help=f"comma-separated: {', '.join(FORMATS)}, or 'all' (default: txt)",
    )  # fmt: skip
    group.add_argument("-o", "--output", default=None, help="output file or directory")
    group.add_argument(
        "-t", "--timestamps", choices=TIMESTAMP_MODES, default=None,
        help="none (default), relative offsets, or absolute wall-clock time",
    )  # fmt: skip
    group.add_argument("--tz", dest="timezone", default=None, help="IANA zone for absolute times")
    group.add_argument(
        "--recorded-at", default=None,
        help="override the recording start time (ISO 8601) used by --timestamps absolute",
    )  # fmt: skip
    group.add_argument(
        "--text", dest="text_variant", choices=("fixed", "raw", "both"), default=None,
        help="with --fix: print the corrected text, the speech model's original, or both",
    )  # fmt: skip
    _flag(group, "show-variants", "print alternative readings and the confidence that ranks them")
    _flag(group, "show-flags", "print why segments were flagged")


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("engine")
    group.add_argument("-b", "--backend", default=None, help="auto (default), whispercpp, mlx")
    group.add_argument("-m", "--model", default=None, help="model name (see `stt models`)")
    group.add_argument(
        "-l", "--language", default=None, help="spoken language, e.g. ru (default: auto-detect)"
    )
    group.add_argument(
        "--threads", type=int, default=None, help="engine threads (0 = engine default)"
    )
    group.add_argument(
        "--context", choices=tuple(CONTEXT_TOKENS), default=None,
        help="feed the decoder its own previous output back (default off: loop-safe, but "
             "costs casing and punctuation at window boundaries)",
    )  # fmt: skip


def _add_vad_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("voice activity detection")
    group.add_argument("--vad", choices=("auto", "silero", "ffmpeg", "none"), default=None)
    group.add_argument(
        "--vad-threshold", type=float, default=None, help="silero speech probability (0..1)"
    )
    group.add_argument(
        "--vad-min-silence", type=int, dest="vad_min_silence_ms", default=None, metavar="MS"
    )
    group.add_argument("--vad-pad", type=int, dest="vad_speech_pad_ms", default=None, metavar="MS")
    group.add_argument(
        "--vad-min-speech", type=int, dest="vad_min_speech_ms", default=None, metavar="MS"
    )


def _add_cleaning_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hallucination and loop scrubbing")
    _flag(group, "clean", "remove invented text (--no-clean flags it but keeps it)")
    _flag(group, "strict-clean", "drop known filler phrases regardless of context")
    group.add_argument(
        "--max-repeats", type=int, default=None, help="repeats before a phrase counts as a loop"
    )
    group.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help="below this a segment is 'shaky' (0..1)",
    )


def _add_dictionary_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("terminology (see `stt dict`)")
    _flag(group, "dict", "use the dictionary (prompt the model, fix known misspellings)")
    _flag(group, "dict-bias", "put the glossary in the speech model's prompt, not just the LLM's")
    group.add_argument(
        "--dict-similarity", type=float, default=None, metavar="N",
        help="how alike a word must sound to be flagged as a term (0..1)",
    )  # fmt: skip


def _add_variant_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("decoding variants")
    group.add_argument(
        "--variants", type=int, default=None, metavar="N",
        help="extra decodings of each shaky segment (0 = off; implied by --fix)",
    )  # fmt: skip
    group.add_argument(
        "--variant-model", action="append", default=None, metavar="MODEL",
        help="cross-check shaky segments against another model; repeatable",
    )  # fmt: skip
    group.add_argument(
        "--context-compare", choices=pipeline.COMPARE_MODES, default=None,
        help="decode a second time with the opposite --context and keep the disagreements "
             "as variants; auto (implied by --fix) re-decodes only damaged-looking chunks",
    )  # fmt: skip


def _add_enrichment_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("correction, summary and speakers")
    _flag(group, "fix", "correct the transcript with an LLM CLI")
    group.add_argument("--fix-with", default=None, help="auto (default), codex, claude, opencode")
    _flag(group, "summary", "add a structured summary")
    _flag(group, "diarize", "split into speaker turns (needs pyannote)")
    group.add_argument("--speakers", type=int, default=None, help="expected number of speakers")


def _flag(group: argparse._ArgumentGroup, name: str, help_text: str) -> None:
    """Add ``--name`` and ``--no-name``, defaulting to neither.

    A plain ``store_true`` can only ever turn something ON, which is fine until a preference
    is stored: with ``fix = true`` in the config there would be no way to skip the LLM pass
    for one run. Defaulting to ``None`` also keeps "flag beats config beats default"
    honest — an unmentioned flag must not overwrite a stored value with ``False``.
    """
    group.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_text,
    )


def _add_misc_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run")
    _flag(group, "cache", "reuse an archived run (--no-cache re-transcribes)")
    _flag(group, "keep-media", "keep a compressed copy of the audio in the archive")
    group.add_argument(
        "--json",
        action="store_true",
        dest="json_report",
        help="print a machine-readable run report",
    )
    group.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    group.add_argument("-v", "--verbose", action="store_true", help="show per-span progress")


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    paths = [Path(p).expanduser() for p in args.inputs]
    reporter = pipeline.Reporter(verbose=args.verbose, quiet=args.quiet or args.json_report)

    import asyncio

    results = asyncio.run(pipeline.gather(paths, settings, reporter))
    # Derived from the merged settings, not from argv, so a stored `show_variants` actually
    # takes effect and there is only one place that decides what the reader sees.
    return _emit(results, settings, pipeline.render_options(settings), args, reporter)


def _settings(args: argparse.Namespace) -> config.Settings:
    """Layer the flags over the user's stored config, then over the built-in defaults."""
    base = config.load_settings()
    chosen_formats = _formats(args.format)
    settings = base.merged(
        recorded_at=_parse_recorded_at(args.recorded_at),
        backend=args.backend,
        model=args.model,
        language=args.language,
        threads=args.threads,
        context=args.context,
        context_compare=args.context_compare,
        # `--fix` implies the comparison only when nobody said otherwise; the flag defaults
        # to None, so its presence is exactly the signal that somebody did.
        context_compare_chosen=True if args.context_compare is not None else None,
        dictionary=args.dict,
        dict_bias=args.dict_bias,
        dict_similarity=args.dict_similarity,
        vad=args.vad,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        clean=args.clean,
        strict_clean=args.strict_clean,
        max_repeats=args.max_repeats,
        confidence_floor=args.confidence_floor,
        variants=args.variants,
        variant_models=args.variant_model,
        show_variants=args.show_variants,
        show_flags=args.show_flags,
        text_variant=args.text_variant,
        fix=args.fix,
        fix_with=args.fix_with,
        summary=args.summary,
        diarize=args.diarize,
        speakers=args.speakers,
        formats=chosen_formats,
        timestamps=args.timestamps,
        timezone=args.timezone,
        output=args.output,
        cache=args.cache,
        keep_media=args.keep_media,
    )
    return settings


def _parse_recorded_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except ValueError as exc:
        raise UsageError(
            what=f"could not read --recorded-at {value!r}",
            why=str(exc),
            how="use ISO 8601, e.g. --recorded-at '2026-03-31 13:32:57'",
        ) from exc


def _formats(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return formats.expand([part.strip() for part in raw.split(",") if part.strip()])


# ── writing the results ───────────────────────────────────────────────────────
def _emit(
    results: list[pipeline.RunResult],
    settings: config.Settings,
    options: formats.RenderOptions,
    args: argparse.Namespace,
    reporter: pipeline.Reporter,
) -> int:
    if not results:
        return 0
    written: list[str] = []
    # Rendering happens after the pipeline has closed its own connection, so the command layer
    # opens one — ONCE for the whole batch rather than once per file.
    with Archive() as store:
        for result in results:
            rendered = pipeline.render_all(result.transcript, settings, options)
            _persist(store, result, rendered)
            written += _write(result, rendered, settings, args, reporter)
    if args.json_report:
        _print_json(results, written)
    return 0


def _persist(store: Archive, result: pipeline.RunResult, rendered: dict[str, str]) -> None:
    """Keep the rendered outputs beside the transcript, so the run folder is browsable.

    Written through the archive's own atomic writer rather than by hand, so an interrupted
    write cannot leave a truncated file behind, and the run directory's layout is decided in
    one place.
    """
    directory = store.get(result.run_id).directory
    for name, text in rendered.items():
        write_atomic(directory / f"transcript.{formats.extension(name)}", text)


def _write(
    result: pipeline.RunResult,
    rendered: dict[str, str],
    settings: config.Settings,
    args: argparse.Namespace,
    reporter: pipeline.Reporter,
) -> list[str]:
    """Send each rendered format to stdout or to a file, per the rules in the module docstring."""
    single = len(rendered) == 1 and len(args.inputs) == 1
    if single and not settings.output:
        sys.stdout.write(next(iter(rendered.values())))
        return []

    stem = Path(result.transcript.media.path).stem
    targets = _targets(rendered, settings.output, stem, single)
    written: list[str] = []
    for name, path in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered[name], "utf-8")
        written.append(str(path))
        reporter.step(f"wrote {path}")
    return written


def _targets(
    rendered: dict[str, str], output: str | None, stem: str, single: bool
) -> dict[str, Path]:
    """Decide the destination path for each format."""
    if output and single and not _is_directory(output):
        return {next(iter(rendered)): Path(output).expanduser()}
    directory = Path(output).expanduser() if output else Path.cwd()
    return {name: directory / f"{stem}.{formats.extension(name)}" for name in rendered}


def _is_directory(output: str) -> bool:
    path = Path(output).expanduser()
    return output.endswith("/") or path.is_dir()


def _print_json(results: list[pipeline.RunResult], written: list[str]) -> None:
    import json

    payload = {
        "runs": [
            {
                "run_id": r.run_id,
                "cached": r.cached,
                "source": r.transcript.media.path,
                "duration": r.transcript.media.duration,
                "language": r.transcript.language,
                "engine": r.transcript.engine.to_dict(),
                "segments": len(r.transcript.segments),
                "words": sum(len(s.text.split()) for s in r.transcript.segments),
                "has_summary": r.transcript.summary is not None,
                "warnings": r.transcript.warnings,
            }
            for r in results
        ],
        "written": written,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
