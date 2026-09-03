"""phrases — the filler a Whisper-family model emits when it has nothing to transcribe.

WHERE THIS TEXT COMES FROM
    These models were trained on a very large pile of subtitle files, and subtitle files
    end with credits. Feed the model near-silence and it reaches for the most probable
    thing that follows quiet in its training data: the caption author's signature, a
    channel plug, a sign-off. That is why the same handful of strings — a particular
    subtitle author's name, "Amara.org", "Продолжение следует" — appear in transcripts of
    recordings that never contained them.

TWO CONFIDENCE TIERS, BECAUSE DELETION IS NOT FREE
    ``ALWAYS`` holds strings that cannot plausibly be something the speaker said in a
    recording you made yourself; those go regardless of context. ``CONTEXTUAL`` holds
    phrases that are perfectly ordinary speech ("Thank you", "Спасибо за просмотр") and are
    only suspicious when they arrive with low confidence, or over a stretch VAD called
    silence, or alone at the very end of a file. Deleting those unconditionally would
    quietly eat real words, which is a worse failure than leaving one artefact in.

EXTENDING
    A user's own patterns go in ``<STT_HOME>/hallucinations.txt`` (one regular expression
    per line, ``#`` comments allowed) and are treated as ``ALWAYS``. Nothing here needs
    editing to add one.
"""

from __future__ import annotations

import re
from pathlib import Path

from .jsonio import read_lines

# A few dozen phrases, one per line.
MAX_PATTERN_BYTES = 256 * 1024

# Unmistakable artefacts: subtitle credits, transcription-service plugs, channel calls to
# action. None of these belong in a recording of your own meeting.
ALWAYS: tuple[str, ...] = (
    r"субтитры\s+(с)?делал",
    r"субтитры\s+создавал",
    r"субтитры\s+и\s+перевод",
    r"редактор\s+субтитров",
    r"корректор\s+[А-ЯЁ]\.",
    r"dimatorzok",
    r"субтитр(ы|ов)\s+от\b",
    r"подписывайтесь\s+на\s+(канал|наш)",
    r"ставьте\s+лайк",
    r"жмите\s+колокольчик",
    r"subtitles?\s+by\s+",
    r"subs?\s+by\s+",
    r"amara\.org",
    r"zeoranger\.co\.uk",
    r"mooji\.org",
    r"transcription\s+by\s+eso",
    r"translated\s+by\s+\S+$",
    r"please\s+subscribe\s+to\s+(my|our|the)\s+channel",
    r"like\s+and\s+subscribe",
    r"^\W*(музыка|music|аплодисменты|applause|laughter|смех)\W*$",
    r"^[\s♪♫»«\-—…\.]+$",
)

# Ordinary phrases that are ALSO the model's favourite thing to say over silence. Dropped
# only when the surrounding evidence says the audio was not speech.
CONTEXTUAL: tuple[str, ...] = (
    r"^\W*продолжение\s+следует\W*$",
    r"^\W*спасибо\s+за\s+просмотр\W*$",
    r"^\W*спасибо\s+за\s+внимание\W*$",
    r"^\W*всем\s+пока\W*$",
    r"^\W*до\s+новых\s+встреч\W*$",
    r"^\W*thanks?\s+(you\s+)?for\s+watching\W*$",
    r"^\W*thank\s+you\W*$",
    r"^\W*thanks\W*$",
    r"^\W*you\W*$",
    r"^\W*bye\W*$",
    r"^\W*see\s+you\W*$",
    r"^\W*okay\W*$",
    r"^\W*продолжение\W*$",
    r"^\W*ну\W*$",
    r"^\W*да\W*$",
    r"^\W*так\W*$",
)


def user_patterns_path(home: Path) -> Path:
    return home / "hallucinations.txt"


def load_user_patterns(home: Path) -> list[str]:
    """Read the user's own always-drop patterns, ignoring blanks and ``#`` comments.

    Through the same guarded door as the two JSON files, and for the same reasons: this is
    read on the default path, so an editor that saved it as Latin-1, or a permission that
    changed, has to answer with the diagnosed error rather than a traceback — or, worse,
    with an empty list. Silently returning nothing would drop the user's own patterns from
    a run that then looks like it worked.
    """
    lines = read_lines(
        user_patterns_path(home),
        how="save it as UTF-8, or delete it to use the built-in patterns only",
        limit=MAX_PATTERN_BYTES,
        too_big="this is a short list of phrases, one per line",
    )
    if lines is None:
        return []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def compile_all(extra_always: list[str]) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    """Compile both tiers once per run; an invalid user pattern is skipped with a warning."""
    always = [re.compile(p, re.IGNORECASE) for p in ALWAYS]
    for pattern in extra_always:
        try:
            always.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            print(f"stt: warning: ignoring invalid hallucination pattern {pattern!r}: {exc}")
    contextual = [re.compile(p, re.IGNORECASE) for p in CONTEXTUAL]
    return always, contextual
