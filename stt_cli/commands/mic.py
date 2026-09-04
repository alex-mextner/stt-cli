"""mic — dictate into whatever window has focus, correcting as it goes.

``stt mic`` opens the microphone, and from then on what you say is typed where you are
already typing. Two models run at once: a small one answers immediately so words appear
while you are still speaking, and `large-v3-turbo` answers a second later and replaces them
with the version that has the proper nouns and the punctuation right. Your terminology
dictionary reaches both, so a project name comes out spelled the way you wrote it down.

Press any key yourself and stt stops correcting the sentence in flight and leaves it alone —
it will never delete a character it did not type. Escape twice ends the session.

WHAT IT NEEDS THAT NOTHING ELSE IN STT NEEDS
    Accessibility and Microphone, both granted to the terminal application rather than to
    stt, because that is the level macOS grants them at.

WHEN NOTHING HAPPENS
    `stt mic --check` is the answer, and it is the one command here that exists because of a
    real failure rather than a design. Dictation can fail by producing nothing at all — no
    text, no error, no exit code — when every frame the microphone delivers is below the bar
    the detector sets. From outside, that is indistinguishable from a denied permission, the
    wrong device being open, or nobody having spoken. So `--check` opens the microphone,
    draws what it hears against the bar it has to clear, and names which of those it was.
    It types nothing anywhere; it is safe to run in the middle of anything.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys

from .._errors import EXIT_ENGINE, EXIT_OK, EXIT_PERMISSION, SttError

NAME = "mic"
SUMMARY = "dictate into the focused window, corrected live"

# The small model exists to be fast, not right: whatever it says is replaced within a second
# or two. `base` is the smallest one that produces recognisable Russian.
DEFAULT_DRAFT = "base"


def run(argv: list[str]) -> int:
    args = _parse(argv)
    # Listing first. `--check` was tested first and `--list-devices` is handled further in,
    # so `stt mic --check --list-devices` spent six seconds listening instead of printing the
    # list — the cheaper and more certain of the two answers losing to the slower one.
    if args.list_devices:
        return asyncio.run(_list_devices())
    if args.check:
        return asyncio.run(_check(args))
    from ..live import dictation

    return asyncio.run(dictation.dictate(args))


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stt mic",
        description=SUMMARY,
        epilog=(
            "any key you press yourself stops stt correcting the sentence in flight; "
            "press Escape twice to finish"
        ),
    )
    parser.add_argument("--device", help="microphone index or part of its name")
    parser.add_argument("--model", help="the accurate model (default: your configured one)")
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT,
        help=f"the fast model typed first (default: {DEFAULT_DRAFT})",
    )
    parser.add_argument(
        "--draft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="type a fast draft before the accurate answer (default: yes)",
    )
    parser.add_argument("-l", "--language", help="spoken language (default: your configured one)")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        help="threads per model (default: your configured one; 0 lets the engine choose)",
    )
    parser.add_argument(
        "--idle-minutes",
        type=_minutes,
        default=None,
        help="stop after this long with nobody speaking (0 to never stop)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="open the microphone, show what it hears, and say whether that is enough",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="no status line, sounds or notifications"
    )
    parser.add_argument("--list-devices", action="store_true", help="show microphones and exit")
    return parser.parse_args(argv)


def _minutes(given: str) -> float:
    """A number of minutes, refusing the values that quietly mean "never".

    `--idle-minutes -1` used to parse, become zero, and switch the timeout off — which is
    what `0` means, but nobody types a minus sign to ask for that. A typo left the microphone
    open indefinitely, which is the one outcome the timeout exists to prevent. `nan` did the
    same, more quietly still: every comparison against it is false.
    """
    try:
        minutes = float(given)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{given!r} is not a number of minutes") from None
    if not math.isfinite(minutes):
        raise argparse.ArgumentTypeError("a number of minutes, or 0 to never stop")
    if minutes < 0:
        raise argparse.ArgumentTypeError(
            f"{given} is negative; use 0 if you mean never to stop on your own"
        )
    return minutes


async def _list_devices() -> int:
    """Every microphone the system offers, by the number `--device` takes."""
    from ..live import capture

    for device in await capture.devices():
        print(f"{device.index}  {device.name}")
    return EXIT_OK


async def _check(args: argparse.Namespace) -> int:
    """Answer the question somebody asks when nothing happened: is it hearing me?

    The old version of this printed two lines, one of which was "not checked here" — the
    permission it could not read being the one people actually wanted to know about. It could
    not distinguish a granted microphone from a denied one, a room from a voice, or the right
    device from the wrong device, which is to say it could not diagnose a single one of the
    ways `stt mic` fails silently. So it opens the microphone and reports what arrives.
    """
    from ..live import quartz

    granted = quartz.trusted()
    print(f"accessibility: {'granted' if granted else 'NOT granted'}")
    if not granted:
        print(f"  {quartz.ACCESSIBILITY_HINT}")
    heard = await _listen_and_report(args)
    if heard is None:
        return EXIT_ENGINE
    working, said = heard
    print(f"\n{said}")
    if not granted:
        return EXIT_PERMISSION
    return EXIT_OK if working else EXIT_ENGINE


async def _listen_and_report(args: argparse.Namespace) -> tuple[bool, str] | None:
    """Open the microphone, draw the meter while it listens, and judge what it heard."""
    from .. import config
    from ..live import capture, dictation, meter, status

    try:
        device = await capture.resolve(args.device)
    except SttError as refused:
        print(f"microphone:    {refused.what}", file=sys.stderr)
        return None
    print(f"microphone:    [{device.index}] {device.name}")
    print(
        f"\n  say something for {meter.LISTEN_SECONDS:.0f} seconds — "
        "the bar passes the mark when stt can hear you\n"
    )
    line = status.Line(enabled=not args.quiet)
    reading = await meter.listen(
        device,
        # The gate a session would build, so the check answers the question the session
        # would answer rather than a similar-looking one.
        gate=dictation.gate_for(config.load_settings()),
        on_sample=lambda level, threshold: line.show(
            f"  {meter.bar(level, threshold)}  {level:5.0f}"
        ),
    )
    line.clear()
    return meter.verdict(reading)
