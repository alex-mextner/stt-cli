"""config — read and write the stored defaults, so common flags stop being flags.

If every run of yours is ``-l ru -t absolute --tz Europe/Belgrade``, those are not options,
they are your defaults. Store them once and the command line goes back to being the file
name. A flag always still wins over a stored value for a single run.
"""

from __future__ import annotations

import argparse
import json

from .. import config
from .._errors import EXIT_OK
from ..jsonio import JsonDict, as_dict

NAME = "config"
SUMMARY = "show or change stored defaults"


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="stt config", description=SUMMARY)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("list", help="show every setting and its current value (default)")
    sub.add_parser("path", help="print the config file location")
    get = sub.add_parser("get", help="print one setting")
    get.add_argument("key")
    put = sub.add_parser("set", help="store one setting")
    put.add_argument("key")
    put.add_argument("value")
    args = parser.parse_args(argv)

    action = args.action or "list"
    if action == "list":
        return _list()
    if action == "path":
        print(config.config_path())
        return EXIT_OK
    if action == "get":
        return _get(args.key)
    return _set(args.key, args.value)


def _list() -> int:
    settings = config.load_settings()
    stored = _stored()
    print(f"{'setting':<24} {'value':<28} source")
    for name in sorted(settings.__dataclass_fields__):
        if name in {"output", "recorded_at"}:
            continue
        value = getattr(settings, name)
        source = "config file" if name in stored else "default"
        print(f"{name:<24} {_show(value):<28} {source}")
    print(f"\nfile: {config.config_path()}")
    return EXIT_OK


def _stored() -> JsonDict:
    """What is actually in config.json, so `config list` can say which values are stored."""
    path = config.config_path()
    if not path.is_file():
        return {}
    try:
        return as_dict(json.loads(path.read_text("utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def _get(key: str) -> int:
    settings = config.load_settings()
    if key not in settings.__dataclass_fields__:
        from .._errors import unknown_item

        raise unknown_item("setting", key, sorted(settings.__dataclass_fields__))
    print(_show(getattr(settings, key)))
    return EXIT_OK


def _set(key: str, raw: str) -> int:
    """Store one setting, coercing the argv string to the field's DECLARED type."""
    settings = config.load_settings()
    if key not in settings.__dataclass_fields__:
        from .._errors import unknown_item

        raise unknown_item("setting", key, sorted(settings.__dataclass_fields__))
    value = config.coerce(key, raw)
    config.save_setting(key, value)
    print(f"{key} = {_show(value)}   ({config.config_path()})")
    return EXIT_OK


def _show(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value) or "(empty)"
    return "(unset)" if value is None else str(value)
