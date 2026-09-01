#!/usr/bin/env bash
# install.sh — install the `stt` CLI (macOS, Python 3.11+)
#
# Works from a local clone (./install.sh) and piped from curl:
#   curl -fsSL https://raw.githubusercontent.com/alex-mextner/stt-cli/main/install.sh | bash
#
# stt has ZERO required Python dependencies — the engine is a native binary and every media
# operation shells out to ffmpeg — so the install is fast and cannot be broken by a wheel
# that will not build. What it DOES need is ffmpeg and a speech engine, and this script
# installs both rather than leaving you to discover them mid-transcription.
set -euo pipefail

TOOL="stt"
REPO="stt-cli"
GITHUB_USER="alex-mextner"
ENTRY="bin/stt"
CLONE_BASE="${XDG_DATA_HOME:-$HOME/.local/share}"

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }

# ── platform ──────────────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "stt targets macOS. Nothing here is deliberately portable — the engines, the"
  warn "Metal acceleration and the file-creation-time handling are all Apple-specific."
  warn "Continuing anyway; expect rough edges."
fi

# ── locate the source ─────────────────────────────────────────────────────────
_script_dir=""
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
  _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -n "$_script_dir" && -f "$_script_dir/$ENTRY" ]]; then
  SRC="$_script_dir"
  say "stt: using local clone at $SRC"
else
  mkdir -p "$CLONE_BASE"
  CLONE_DIR="$CLONE_BASE/$REPO"
  EXPECT_URL="https://github.com/$GITHUB_USER/$REPO.git"
  if [[ -d "$CLONE_DIR/.git" ]]; then
    actual_url="$(git -C "$CLONE_DIR" remote get-url origin 2>/dev/null || echo "")"
    if [[ "$actual_url" != "$EXPECT_URL" ]]; then
      warn "ERROR: $CLONE_DIR exists but its origin is '$actual_url', not $EXPECT_URL."
      warn "       Remove that directory or fix its remote, then re-run."
      exit 1
    fi
    say "stt: updating existing clone at $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only
  else
    say "stt: cloning $EXPECT_URL into $CLONE_DIR"
    git clone "$EXPECT_URL" "$CLONE_DIR"
  fi
  SRC="$CLONE_DIR"
fi

# ── install the CLI ───────────────────────────────────────────────────────────
BIN="${PIPX_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN"

STT_BIN=""
INSTALL_MODE=""
if command -v uv >/dev/null 2>&1; then
  INSTALL_MODE="uv"
  say "stt: installing via uv tool (isolated environment)"
  uv tool install --force "$SRC"
  STT_BIN="$(command -v "$TOOL" 2>/dev/null || echo "$HOME/.local/bin/$TOOL")"
elif command -v pipx >/dev/null 2>&1; then
  INSTALL_MODE="pipx"
  say "stt: installing via pipx (isolated environment)"
  pipx install --force "$SRC"
  STT_BIN="$(command -v "$TOOL" 2>/dev/null || echo "$BIN/$TOOL")"
else
  INSTALL_MODE="symlink"
  say "stt: neither uv nor pipx found — falling back to a symlink."
  say "     (stt needs no Python packages, so this works fine; uv is just tidier.)"
  chmod +x "$SRC/$ENTRY"
  ln -sfn "$SRC/$ENTRY" "$BIN/$TOOL"
  STT_BIN="$BIN/$TOOL"
fi

if [[ -z "$STT_BIN" || ! -x "$STT_BIN" ]]; then
  warn "ERROR: install reported success but '$TOOL' is not executable at '$STT_BIN'."
  exit 1
fi
say "stt: installed $STT_BIN (via $INSTALL_MODE)"

# ── ffmpeg: required, and the one thing people always miss ────────────────────
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    say "stt: installing ffmpeg (required for every input format)"
    brew install ffmpeg
  else
    warn ""
    warn "  ffmpeg is REQUIRED and was not found, and Homebrew is not installed either."
    warn "  Install it, then re-run:  https://brew.sh  then  brew install ffmpeg"
    warn ""
  fi
fi

# ── a speech engine ───────────────────────────────────────────────────────────
# Only offered when nothing usable is present: someone who already built whisper.cpp does
# not want a second engine installed behind their back.
if ! "$STT_BIN" doctor >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1 && ! command -v whisper-cli >/dev/null 2>&1; then
    say "stt: no speech engine found — installing whisper.cpp"
    brew install whisper-cpp || warn "  whisper-cpp install failed; run 'stt setup' to choose another engine"
  fi
fi

# ── PATH sanity ───────────────────────────────────────────────────────────────
if [[ ":$PATH:" != *":$BIN:"* ]]; then
  warn ""
  warn "  NOTE: $BIN is not on your PATH. Add this to ~/.zshrc and restart your shell:"
  warn "    export PATH=\"$BIN:\$PATH\""
  warn ""
fi

RESOLVED="$(command -v "$TOOL" 2>/dev/null || true)"
if [[ -n "$RESOLVED" && "$RESOLVED" != "$STT_BIN" ]]; then
  warn ""
  warn "  WARNING: another '$TOOL' shadows this install on PATH:"
  warn "      installed:   $STT_BIN"
  warn "      resolves to: $RESOLVED"
  warn ""
fi

# ── register the agent skill ──────────────────────────────────────────────────
# This writes OUTSIDE the project: a SKILL.md under ~/.agents/skills, a line in each detected
# harness's global instruction file, and a SessionStart hook in ~/.claude/settings.json. That
# is a real change to the user's environment, so it is reported rather than done quietly, and
# STT_NO_SKILL=1 skips it. Absolute path on purpose: a PATH shadow would otherwise run a
# different binary.
if [[ -n "${STT_NO_SKILL:-}" ]]; then
  say "stt: skipping agent-skill registration (STT_NO_SKILL is set)."
  say "     Run 'stt install-skill' later if you want coding agents to discover stt."
else
  say ""
  say "stt: registering the agent skill so coding agents can discover this tool."
  say "     This writes to ~/.agents/skills and your harness config. Skip with STT_NO_SKILL=1."
  if ! "$STT_BIN" install-skill; then
    warn "  WARNING: 'stt install-skill' failed — stt works, but coding agents may not"
    warn "           discover it. Re-run 'stt install-skill' to fix."
  fi
fi

# ── done ──────────────────────────────────────────────────────────────────────
if [[ -z "$RESOLVED" ]]; then
  warn ""
  warn "  stt is installed at $STT_BIN but does NOT resolve by name (PATH)."
  warn "  Until you fix PATH, run it by full path: $STT_BIN"
  exit 1
fi

say ""
say "  stt is installed (via $INSTALL_MODE)."
say ""
say "  Next:   stt setup            choose an engine and download a model"
say "          stt doctor           check every dependency"
say "          stt rec.m4a          transcribe something"
say ""
