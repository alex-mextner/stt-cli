"""install-skill — register the ``stt`` agent skill with every harness on this machine.

Run by ``install.sh`` at the end of an install, and safe to run by hand at any time. See
:mod:`stt_cli.install` for what it writes and why each layer exists.
"""

from __future__ import annotations

import argparse

from ..install import install_skill

NAME = "install-skill"
SUMMARY = "register the stt skill so coding agents know this tool exists"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt install-skill", description=SUMMARY)
    parser.parse_args(argv)
    return install_skill()
