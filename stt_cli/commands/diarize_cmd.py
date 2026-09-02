"""diarize — manage the optional speaker-identification extra.

Kept as its own command so the cost is visible and voluntary: nothing here is installed
until you ask, and the install prints what it is about to download and how large it is
before it starts. See :mod:`stt_cli.diarize` for why it is not a dependency.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .. import diarize as diarize_mod
from .. import resources
from .._errors import EXIT_MISSING_DEP, EXIT_OK, EXIT_PERMISSION

NAME = "diarize"
SUMMARY = "install or check speaker diarization (optional, large)"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt diarize", description=SUMMARY)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("status", help="is it installed and usable? (default)")
    install = sub.add_parser("install", help="download pyannote.audio and torch")
    install.add_argument("--yes", action="store_true", help="do not ask before downloading")
    args = parser.parse_args(argv)

    if (args.action or "status") == "status":
        return _status()
    return _install(assume_yes=args.yes)


def _status() -> int:
    """Both halves of "can I diarize right now": the wheels, and the credential."""
    from .. import auth

    installed = diarize_mod.is_installed()
    print(f"pyannote.audio: {'installed' if installed else 'not installed'}")
    if not installed:
        print(f"  fix: {diarize_mod.INSTALL_HINT}")
        return EXIT_MISSING_DEP
    signed_in = auth.status("diarization", "hf")
    print(signed_in.render())
    if not signed_in.ok:
        # The status decides its own code. This caller used to throw the distinction away —
        # a network blip while a perfectly good token sits on disk is not a credential
        # failure, and automation retrying login on EXIT_PERMISSION would do it for nothing
        # — and then used to answer it with a copy of `auth.report`'s mapping.
        return signed_in.exit_code()
    print(f"\npipeline: {diarize_mod.PIPELINE}")
    print("ready — add --diarize to a transcription, or -f speakers for a dialogue transcript")
    return EXIT_OK


def _install(*, assume_yes: bool) -> int:
    if diarize_mod.is_installed():
        print("already installed")
        return _status()

    command = diarize_mod.install_command()
    gib = int(diarize_mod.INSTALL_SIZE_GIB * 1024**3)
    print(f"about to install speaker diarization (~{diarize_mod.INSTALL_SIZE_GIB} GiB):")
    print(f"  {' '.join(command)}")
    resources.require_space(
        gib, path=__import__("pathlib").Path.home(), what="the diarization extra"
    )

    should_ask = not assume_yes and sys.stdin.isatty()
    if should_ask and input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("cancelled")
        return EXIT_OK

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("install failed — see the output above", file=sys.stderr)
        return EXIT_MISSING_DEP
    print("\ninstalled. One step left: the models are gated, so stt needs a token.")
    return _login_now()


def _login_now() -> int:
    """Chain straight into the browser flow rather than printing homework.

    Only when there is somebody there to click. `stt diarize install -y` in a script must
    not open a browser and then block for five minutes waiting for a token that nobody is
    going to copy, so a non-interactive install says what is left to do and stops.
    """
    from .. import auth

    if not sys.stdin.isatty():
        print("run `stt login diarization` from a terminal to finish setting it up")
        return EXIT_PERMISSION
    return auth.report(auth.login("diarization", "hf"))
