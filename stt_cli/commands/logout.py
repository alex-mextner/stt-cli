"""logout — drop a stored credential.

Only the file stt wrote is removed; an ``HF_TOKEN`` exported in the shell belongs to the
shell and is reported rather than pretended away.
"""

from __future__ import annotations

import argparse

NAME = "logout"
SUMMARY = "forget a stored provider credential"


def run(argv: list[str]) -> int:
    from .. import auth

    parser = argparse.ArgumentParser(prog="stt logout", description=SUMMARY)
    parser.add_argument("capability", nargs="?", help=f"default: {auth.DEFAULT_CAPABILITY}")
    parser.add_argument("--provider")
    args = parser.parse_args(argv)

    capability, provider = auth.resolve(args.capability, args.provider)
    return auth.report(auth.logout(capability, provider))
