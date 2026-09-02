"""login — sign in to whatever a feature needs.

``stt login diarization --provider hf`` is the whole interface. The capability is what the
user wants to do; the provider is who happens to gate it. Both have defaults, so in practice
this is ``stt login diarization``, or just ``stt login``.
"""

from __future__ import annotations

import argparse

NAME = "login"
SUMMARY = "sign in to a provider a feature needs (diarization -> Hugging Face)"


def run(argv: list[str]) -> int:
    from .. import auth

    parser = argparse.ArgumentParser(
        prog="stt login",
        description=SUMMARY,
        epilog=(
            "stt opens the token page, picks the token up from your clipboard, verifies it, "
            "stores it where huggingface_hub looks, and walks you through the model terms."
        ),
    )
    parser.add_argument(
        "capability",
        nargs="?",
        help=f"what needs credentials (default: {auth.DEFAULT_CAPABILITY})",
    )
    parser.add_argument("--provider", help="who provides it (default: the capability's first)")
    parser.add_argument("--status", action="store_true", help="report only, change nothing")
    parser.add_argument(
        "--browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the pages in a browser (default: yes)",
    )
    parser.add_argument("--force", action="store_true", help="get a new token even if one works")
    args = parser.parse_args(argv)

    capability, provider = auth.resolve(args.capability, args.provider)
    if args.status:
        return auth.report(auth.status(capability, provider))
    return auth.report(auth.login(capability, provider, browser=args.browser, force=args.force))
