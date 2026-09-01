"""archive — browse, re-render and prune what has already been transcribed.

The archive is the reason a transcription is a one-time cost. This command is how you get
at it: find a run from last month, print it in a format you did not ask for at the time,
open the folder, or reclaim the disk it is using. Re-rendering never touches the engine — it
reads the stored transcript, so a format you forgot to request is a millisecond away rather
than another pass over the audio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .. import config, formats, pipeline
from .._errors import EXIT_OK
from ..archive import Archive
from ..config import FORMATS, TIMESTAMP_MODES
from ..timestamps import Stamper

NAME = "archive"
SUMMARY = "list, re-render, open and prune stored transcriptions"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt archive", description=SUMMARY)
    sub = parser.add_subparsers(dest="action")

    listing = sub.add_parser("ls", help="list stored runs (default)")
    listing.add_argument("query", nargs="?", help="filter by filename or run id")
    listing.add_argument("-n", "--limit", type=int, default=30)

    show = sub.add_parser("show", help="re-render a stored run")
    show.add_argument("run_id")
    show.add_argument("-f", "--format", default="txt", choices=FORMATS)
    show.add_argument("-t", "--timestamps", default=None, choices=TIMESTAMP_MODES)
    show.add_argument("--tz", dest="timezone", default=None)
    show.add_argument("--show-variants", action=argparse.BooleanOptionalAction, default=None)
    show.add_argument("--show-flags", action=argparse.BooleanOptionalAction, default=None)
    show.add_argument(
        "--speakers",
        dest="diarize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show speaker labels if the run has them",
    )
    show.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the run's summary if it has one",
    )
    show.add_argument("--text", choices=("fixed", "raw", "both"), default=None)

    path = sub.add_parser("path", help="print (or open) a run's directory")
    path.add_argument("run_id")
    path.add_argument("--open", action="store_true", help="reveal it in Finder")

    remove = sub.add_parser("rm", help="delete a stored run")
    remove.add_argument("run_id")

    collect = sub.add_parser("gc", help="remove stale runs and unreferenced audio")
    collect.add_argument("--older-than", type=int, default=None, metavar="DAYS")
    collect.add_argument(
        "--yes", action="store_true", help="actually delete (default is a dry run)"
    )

    sub.add_parser("usage", help="how much disk the archive is using")

    args = parser.parse_args(argv)
    action = args.action or "ls"
    with Archive() as store:
        return _dispatch(action, args, store)


def _dispatch(action: str, args: argparse.Namespace, store: Archive) -> int:
    handlers = {
        "ls": _ls,
        "show": _show,
        "path": _path,
        "rm": _rm,
        "gc": _gc,
        "usage": _usage,
    }
    return handlers[action](args, store)


def _ls(args: argparse.Namespace, store: Archive) -> int:
    records = store.recent(limit=getattr(args, "limit", 30), query=getattr(args, "query", None))
    if not records:
        print("archive is empty")
        return EXIT_OK
    print(f"{'run':<18} {'transcribed':<17} {'len':>7}  SD  {'engine':<28} {'words':>7}  source")
    for record in records:
        print(record.row())
    return EXIT_OK


def _show(args: argparse.Namespace, store: Archive) -> int:
    """Re-render a stored run — through the SAME settings path a live transcription uses.

    Building render options straight from these flags would fork the rules: stored
    preferences would be ignored here but honoured there, and a run that once had speaker
    labels would show them in every later render regardless of what was asked for.
    """
    transcript = store.load(args.run_id)
    settings = config.load_settings().merged(
        timestamps=args.timestamps,
        timezone=args.timezone,
        show_variants=args.show_variants,
        show_flags=args.show_flags,
        text_variant=args.text,
        diarize=args.diarize,
        summary=args.summary,
    )
    stamper = Stamper(settings.timestamps, transcript.media, settings.timezone)
    options = pipeline.render_options(settings)
    sys.stdout.write(formats.render(args.format, transcript, stamper, options))
    return EXIT_OK


def _path(args: argparse.Namespace, store: Archive) -> int:
    record = store.get(args.run_id)
    print(record.directory)
    if args.open:
        subprocess.run(["open", str(record.directory)], check=False)
    return EXIT_OK


def _rm(args: argparse.Namespace, store: Archive) -> int:
    record = store.get(args.run_id)
    store.remove(record.run_id)
    print(f"removed {record.run_id} ({record.source_name})")
    return EXIT_OK


def _gc(args: argparse.Namespace, store: Archive) -> int:
    actions = store.gc(older_than_days=args.older_than, dry_run=not args.yes)
    if not actions:
        print("nothing to clean up")
        return EXIT_OK
    for line in actions:
        print(("removed " if args.yes else "would ") + line.removeprefix("remove "))
    if not args.yes:
        print(f"\n{len(actions)} item(s) — re-run with --yes to delete them")
    return EXIT_OK


def _usage(args: argparse.Namespace, store: Archive) -> int:
    runs, audio, transcripts = store.usage()
    mib = 1024 * 1024
    print(f"runs:        {runs}")
    print(f"audio:       {audio / mib:.1f} MiB")
    print(f"transcripts: {transcripts / mib:.1f} MiB")
    return EXIT_OK
