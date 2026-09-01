"""llm — talk to whichever coding-agent CLI is installed, and get JSON back.

WHY SHELL OUT INSTEAD OF CALLING AN API
    The machines this runs on already have an authenticated agent CLI — codex, claude,
    opencode — with a paid plan behind it. Asking the user for a second set of API keys, and
    then billing them twice for the same model, would be rude. So the correction and summary
    passes borrow whatever is already logged in, and the tool needs no key of its own.

THE HARD PART IS NOT THE CALL, IT IS THE PARSING
    These CLIs are built for humans: they print progress, banners and reasoning around the
    answer. Asking for JSON is necessary but not sufficient, so :func:`extract_json` pulls
    the outermost balanced object out of whatever came back, tolerating fenced code blocks
    and surrounding chatter. A response that still cannot be parsed is a failed pass, not a
    crash — the transcript survives uncorrected, which is exactly what should happen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import proc
from ._errors import MissingDependencyError
from .jsonio import JsonDict, as_dict

# Preference order for `--fix-with auto`. Codex leads because its `exec` subcommand is a
# first-class non-interactive mode with a structured-output flag; the others follow.
ORDER = ("codex", "claude", "opencode")

# One correction or summary call over a long transcript is a big prompt and a slow model.
# Well beyond what it should take, but bounded.
CALL_TIMEOUT = 15 * 60.0


@dataclass(slots=True, frozen=True)
class Tool:
    """One agent CLI, and how to hand it a prompt non-interactively."""

    name: str
    argv: list[str]

    def label(self) -> str:
        return f"{self.name} ({' '.join(self.argv)})"


def _spec(name: str) -> list[str] | None:
    """The non-interactive invocation for each supported CLI, prompt arriving on stdin."""
    return {
        # `-` makes codex read the prompt from stdin; the git check is irrelevant here and
        # would otherwise refuse to run outside a repository.
        "codex": ["exec", "--skip-git-repo-check", "-"],
        # Claude Code's print mode: no TTY, no session, answer to stdout.
        "claude": ["-p"],
        "opencode": ["run"],
    }.get(name)


def available() -> list[Tool]:
    """Every supported CLI that is actually installed, in preference order."""
    tools = []
    for name in ORDER:
        binary = proc.which(name)
        spec = _spec(name)
        if binary and spec is not None:
            tools.append(Tool(name=name, argv=[binary, *spec]))
    return tools


def resolve(preference: str) -> Tool:
    """Pick the CLI to use, honouring an explicit choice and diagnosing an absent one."""
    found = available()
    if preference == "auto":
        if not found:
            raise MissingDependencyError(
                what="no LLM CLI is available for --fix / --summary",
                why=f"none of {', '.join(ORDER)} is on PATH",
                how="install one of them, or drop --fix/--summary",
            )
        return found[0]

    for tool in found:
        if tool.name == preference:
            return tool
    binary = proc.which(preference)
    if binary:
        # An unrecognized name that exists on PATH is taken at face value: the user knows
        # their own tooling better than this list does.
        return Tool(name=preference, argv=[binary])
    raise MissingDependencyError(
        what=f"LLM tool {preference!r} is not available",
        why=f"{preference} is not on PATH; found: {', '.join(t.name for t in found) or 'nothing'}",
        how=f"use --fix-with with one of: {', '.join(ORDER)}",
    )


async def ask_json(tool: Tool, prompt: str, *, timeout: float = CALL_TIMEOUT) -> JsonDict | None:
    """Send one prompt and return the JSON object in the reply, or ``None`` if there wasn't one."""
    result = await proc.run(tool.argv, stdin_text=prompt, timeout=timeout)
    if not result.ok:
        return None
    return extract_json(result.stdout)


def extract_json(text: str) -> JsonDict | None:
    """Pull the outermost balanced JSON object out of an agent CLI's chatty output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for candidate in filter(None, (fenced.group(1) if fenced else None, _balanced(text))):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return as_dict(parsed)
    return None


def _balanced(text: str) -> str | None:
    """The widest ``{...}`` span whose braces balance, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
