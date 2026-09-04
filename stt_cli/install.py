"""install — register the ``stt`` agent skill so harnesses discover the tool by themselves.

Three layers, matching the sibling personal CLIs so every tool is discoverable the same way:

1. ``~/.agents/skills/stt/SKILL.md`` — the Agent Skills standard file, read by Claude Code,
   Codex, opencode, Gemini and Cursor.
2. A one-line blurb appended (between markers, so re-running replaces rather than
   duplicates) into each *detected* harness's global instruction file.
3. A ``SessionStart`` hook that prints every installed agent CLI at the top of a session, so
   the awareness survives even when skills are not loaded.

Everything here is idempotent, and nothing rewrites config it did not write. A settings file
that cannot be parsed is left completely alone rather than clobbered.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from pathlib import Path

SKILL_NAME = "stt"

SKILL_MD = """\
---
name: stt
description: >-
  Transcribe any audio or video file to text on macOS. Use when a task involves a
  recording, voice memo, meeting, call, podcast or video that needs to become text —
  including subtitles (srt/vtt), speaker-separated dialogue, or a structured summary of
  what was said. Handles any container/codec, skips silence so the model cannot hallucinate
  over it, scrubs known subtitle-credit artefacts and decoder loops, and caches every run.
metadata:
  author: alex-mextner
  repo: https://github.com/alex-mextner/stt-cli
---

# stt — speech to text for any audio or video file

## Invocation
```
stt recording.m4a                        # plain text to stdout
stt talk.mp4 -f srt -o subs/             # subtitles from a video
stt call.ogg -t absolute --tz Europe/Belgrade   # wall-clock timestamps
stt meeting.m4a --summary --fix -f md    # LLM-corrected transcript + structured summary
stt meeting.m4a --diarize -f speakers    # split into speaker turns
stt dict add ConLoca --aka ConLog        # terms the model does not know
stt mic                                  # live dictation into the focused window
stt archive ls                           # everything transcribed so far
stt archive show <run-id> -f vtt         # re-render an old run, no GPU needed
```

## When to use
- Any file the user refers to as a recording, memo, call, meeting, podcast or video that
  they want as text, subtitles, a dialogue, or a summary.
- Prefer this over hand-writing `ffmpeg ... | whisper-cli`: it handles containers, filenames
  with spaces, silence, hallucinated subtitle credits and repetition loops already.

## Formats
`txt` `md` `json` `srt` `vtt` `csv` `tsv` `speakers` `summary`, or `-f all`.

## Notes
- `--summary` produces headline, topics, decisions, action items and open questions.
- If a recording is full of product names the model mangles, add them with `stt dict add`
  BEFORE transcribing: the glossary goes into the speech model's prompt, so it fixes
  words at the acoustic level rather than guessing at them afterwards.
- `--diarize` needs a one-off `stt diarize install` and `stt login diarization`; the latter
  opens the browser, takes the Hugging Face token off the clipboard and accepts the model
  terms. Never tell the user to export `HF_TOKEN` by hand.
- `--fix` and `--summary` borrow an installed agent CLI (codex / claude / opencode); no API
  key of their own.
- Runs are cached in a local archive, so re-rendering a different format is instant.
- `stt mic` is for the HUMAN to run, not for you. It types into whatever window has focus
  and runs until they stop it, so starting one on their behalf puts text somewhere neither of
  you can see. Tell them the command; do not run it.
- `stt doctor` reports every dependency and how to install the missing ones.
"""

SKILL_BLURB = (
    "`stt` — transcribe any audio/video to text on macOS: `stt rec.m4a`, "
    "`stt talk.mp4 -f srt`, `stt meeting.m4a --summary --fix -f md`, "
    "`stt rec.m4a --diarize -f speakers`. Silence-gated (no hallucinated subtitle credits), "
    "cached in a local archive (`stt archive ls|show`). Use INSTEAD of hand-rolled "
    "ffmpeg + whisper pipelines."
)

_HOOK_MARKER = "# agent-tools-awareness"
_HOOK_COMMAND = (
    'sh -c \'d="$HOME/.agents/skills/.blurbs"; ls "$d"/*.md >/dev/null 2>&1 && '
    '{ printf "Agent CLI tools installed on this machine (prefer them):\\n"; '
    'cat "$d"/*.md; }\' ' + _HOOK_MARKER
)

_HARNESSES = (
    ("claude", Path(".claude") / "CLAUDE.md", ("~/.claude",)),
    ("codex", Path(".codex") / "AGENTS.md", ("~/.codex",)),
    ("opencode", Path(".config") / "opencode" / "AGENTS.md", ("~/.config/opencode",)),
    ("gemini", Path(".gemini") / "GEMINI.md", ("~/.gemini",)),
)


def install_skill() -> int:
    """Write every layer, reporting each target. Safe to run repeatedly."""
    home = Path.home()
    written: list[str] = []

    written.append(_write_skill_file(home))
    _write_blurb_file(home)
    _link_for_claude(home)
    written += _write_harness_blurbs(home)
    if _ensure_sessionstart_hook(home):
        written.append("SessionStart hook -> ~/.claude/settings.json")

    for target in written:
        print(f"  ✓ {target}")
    print(
        f"{SKILL_NAME}: install-skill done ({len(written)} target(s)). Idempotent — re-run anytime."
    )
    return 0


def _write_skill_file(home: Path) -> str:
    skill_dir = home / ".agents" / "skills" / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(SKILL_MD, encoding="utf-8")
    return str(path)


def _write_blurb_file(home: Path) -> None:
    """The line the SessionStart hook prints; one file per tool, concatenated at runtime."""
    blurbs = home / ".agents" / "skills" / ".blurbs"
    blurbs.mkdir(parents=True, exist_ok=True)
    (blurbs / f"{SKILL_NAME}.md").write_text(f"- {SKILL_BLURB}\n", encoding="utf-8")


def _link_for_claude(home: Path) -> None:
    """Claude Code also scans ~/.claude/skills; a symlink keeps one source of truth."""
    skills = home / ".claude" / "skills"
    if not skills.is_dir():
        return
    link = skills / SKILL_NAME
    if link.exists() or link.is_symlink():
        return
    with contextlib.suppress(OSError):
        link.symlink_to(Path("..") / ".." / ".agents" / "skills" / SKILL_NAME)


def _write_harness_blurbs(home: Path) -> list[str]:
    written = []
    for command, relative, hints in _HARNESSES:
        if _detected(command, *hints):
            path = home / relative
            _append_marked(path, SKILL_NAME, SKILL_BLURB)
            written.append(str(path))
    return written


def _detected(command: str, *dirs: str) -> bool:
    if shutil.which(command):
        return True
    return any(Path(os.path.expanduser(d)).is_dir() for d in dirs)


def _append_marked(path: Path, tool: str, blurb: str) -> None:
    """Replace this tool's marked block, leaving everything else in the file untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    start, end = f"<!-- skill:{tool} -->", f"<!-- /skill:{tool} -->"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n?", "", existing, flags=re.S)
    block = f"{start}\n{blurb}\n{end}\n"
    body = (existing.rstrip() + "\n\n" + block) if existing.strip() else block
    path.write_text(body, encoding="utf-8")


def _ensure_sessionstart_hook(home: Path) -> bool:
    """Add the shared awareness hook to Claude Code's settings, or leave them alone entirely.

    Deliberately conservative: unparseable or unexpectedly shaped settings mean "do nothing",
    never "rewrite". A backup is taken before any write.
    """
    settings = home / ".claude" / "settings.json"
    if not settings.parent.is_dir():
        return False
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    hooks = data.setdefault("hooks", {})
    session_start = hooks.setdefault("SessionStart", []) if isinstance(hooks, dict) else None
    if not isinstance(session_start, list):
        return False
    if _hook_present(session_start):
        return False
    session_start.append({"hooks": [{"type": "command", "command": _HOOK_COMMAND}]})
    if settings.exists():
        settings.with_suffix(".json.bak").write_text(settings.read_text("utf-8"), encoding="utf-8")
    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def _hook_present(session_start: list[object]) -> bool:
    for group in session_start:
        entries = group.get("hooks", []) if isinstance(group, dict) else []
        for entry in entries:
            if isinstance(entry, dict) and _HOOK_MARKER in str(entry.get("command", "")):
                return True
    return False
