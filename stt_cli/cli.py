"""cli — the command dispatcher.

SELF-REGISTERING COMMANDS
    Every module in ``stt_cli/commands/`` that exposes ``NAME``, ``SUMMARY`` and
    ``run(argv) -> int`` becomes a subcommand. Adding one is dropping a file in; nothing
    here changes.

THE COMMON CASE IS NOT A SUBCOMMAND
    Ninety-nine times out of a hundred the user wants ``stt some-recording.m4a``, not
    ``stt transcribe some-recording.m4a``. So a first argument that is not a known command
    and is not an option falls through to ``transcribe``. The explicit spelling still works,
    and an unknown *command-looking* word still gets a did-you-mean rather than being fed to
    ffprobe as a filename.

IMPORT-CLEAN AT THE TOP
    This module imports only the standard library and the stdlib-only error layer, and pulls
    each command in lazily. ``stt --help`` therefore costs nothing, and stays useful on a
    machine where no engine, no ffmpeg and no LLM CLI is installed yet.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable

from . import __version__, palette
from ._errors import EXIT_OK, guard, unknown_item

_RunFn = Callable[[list[str]], int]


def _discover() -> dict[str, tuple[str, str]]:
    """Map command name -> (module, one-line summary) by scanning the commands package."""
    from . import commands

    found: dict[str, tuple[str, str]] = {}
    for info in pkgutil.iter_modules(commands.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{commands.__name__}.{info.name}")
        name = getattr(module, "NAME", None)
        summary = getattr(module, "SUMMARY", "")
        if name:
            found[name] = (f"{commands.__name__}.{info.name}", summary)
    return dict(sorted(found.items()))


_HEADLINE = "stt — speech to text for any audio or video file"
_CLOSING = "run `stt <command> --help` for a command's own options."

# Where a summary starts, counted from the two-space indent. Two columns, because a
# command name is a word and an example is most of a command line.
_COMMAND_COLUMN = 15
_EXAMPLE_COLUMN = 31

_INVOCATIONS = (
    ("stt <file>... [options]", "transcribe (the common case)"),
    ("stt <command> [args]", ""),
)
_GETTING_STARTED = (
    ("stt setup", "detect what is installed and choose an engine"),
    ("stt doctor", "check the toolchain"),
    ("stt rec.m4a -f txt,srt", "transcribe to text and subtitles"),
    ("stt rec.m4a --summary --fix", "correct with an LLM and summarize"),
)


def _usage(catalog: dict[str, tuple[str, str]]) -> str:
    """The top-level help, painted the way argparse paints every subcommand's."""
    paint = palette.for_help()
    return "\n".join(
        [
            _HEADLINE,
            "",
            _heading("usage:", paint),
            *_example_rows(_INVOCATIONS, paint),
            "",
            _heading("commands:", paint),
            *_command_rows(catalog, paint),
            "",
            _heading("getting started:", paint),
            *_example_rows(_GETTING_STARTED, paint),
            "",
            _CLOSING,
        ]
    )


def _heading(text: str, paint: palette.Palette) -> str:
    return f"{paint.heading}{text}{paint.reset}"


def _command_rows(catalog: dict[str, tuple[str, str]], paint: palette.Palette) -> list[str]:
    """One row per command, its name in the colour argparse gives a positional."""
    return [
        _row(f"{paint.action}{name}{paint.reset}", name, summary, _COMMAND_COLUMN)
        for name, (_, summary) in catalog.items()
    ]


def _example_rows(examples: tuple[tuple[str, str], ...], paint: palette.Palette) -> list[str]:
    """One row per example, with only the program name coloured — argparse does the same."""
    rows = []
    for invocation, summary in examples:
        prog, _, rest = invocation.partition(" ")
        shown = f"{paint.prog}{prog}{paint.reset}"
        if rest:
            shown = f"{shown} {rest}"
        rows.append(_row(shown, invocation, summary, _EXAMPLE_COLUMN))
    return rows


def _row(shown: str, plain: str, summary: str, column: int) -> str:
    """Indent *shown*, then pad to *column* — measured on *plain*, since colour is not wide."""
    if not summary:
        return f"  {shown}"
    return f"  {shown}{' ' * max(1, column - len(plain))}{summary}"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return guard(lambda: _dispatch(args))


def _dispatch(args: list[str]) -> int:
    catalog = _discover()
    if not args:
        print(_usage(catalog))
        return EXIT_OK
    head = args[0]
    if head in {"-h", "--help", "help"}:
        print(_usage(catalog))
        return EXIT_OK
    if head in {"-V", "--version", "version"}:
        print(f"stt {__version__}")
        return EXIT_OK

    if head in catalog:
        return _load(catalog[head][0])(args[1:])
    if _looks_like_a_command(head):
        raise unknown_item("command", head, list(catalog))
    # Anything else is a file (or a flag before one) — the common case.
    return _load(catalog["transcribe"][0])(args)


def _looks_like_a_command(token: str) -> bool:
    """Was this bare word meant as a command, or is it a file?

    The heuristic is "no slash, no dot, no tilde", which is right for `stt doctorr` and wrong
    for an extensionless recording called `Interview`. Checking the filesystem first costs one
    stat and removes that false positive entirely; the heuristic only decides the cases where
    nothing by that name exists.
    """
    import os

    if token.startswith("-"):
        return False
    if os.path.exists(token):
        return False
    return "/" not in token and "." not in token and "~" not in token


def _load(module_name: str) -> _RunFn:
    module = importlib.import_module(module_name)
    run = getattr(module, "run", None)
    if run is None:  # pragma: no cover - only reachable from a malformed command module
        from ._errors import SttError

        raise SttError(
            what=f"{module_name} is not a valid command module",
            why="it does not define run(argv) -> int",
            how="this is a bug in stt-cli; please open an issue",
        )
    return run  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
