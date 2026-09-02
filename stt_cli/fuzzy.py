"""fuzzy — "that word sounds like something in your dictionary".

THE PROBLEM THIS SOLVES
    A speech model that has never seen your product name writes down whatever it sounded
    like: ConLoca becomes Conloca, ConLog, Coloca, ConLoka, Colocka. Exact matching finds
    none of those. Neither does plain edit distance on its own — "ConLog" and "ConLoca"
    differ by three characters out of seven, which is also roughly how far apart two
    genuinely different words are.

WHY PHONETIC, AND WHY HOME-GROWN
    All those spellings sound nearly identical, so comparing what they SOUND like rather
    than how they are spelled separates them from real words cleanly. The standard
    algorithms are no use here: Soundex and Metaphone are built for English orthography and
    do not accept Cyrillic at all, and this tool transcribes both, frequently in the same
    sentence. So the reduction below is deliberately small and explicit — transliterate,
    fold the distinctions that survive neither a microphone nor a decoder, and compare what
    is left. It is a screening filter, not a linguistic claim: it decides which words are
    worth showing to a person or an LLM, and it never rewrites anything on its own.

WHAT IT DELIBERATELY DOES NOT DO
    It does not replace words. A phonetic near-match is evidence, not proof — "Colin" also
    sounds like "ConLoca" if you squint — so the pipeline flags near matches and leaves the
    decision to the reader or the correction pass. Only an exact known misspelling, one the
    user has written into the dictionary themselves, is ever substituted automatically.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Cyrillic to a Latin skeleton. Not a transliteration standard: the target is the phonetic
# alphabet used below, so several Cyrillic letters deliberately land on the same Latin one.
_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "u",
    "я": "a",
}  # fmt: skip

# Distinctions a microphone and a decoder routinely lose. Voiced and voiceless pairs are
# the big one: a final -d and a final -t are the same sound in most recordings.
_FOLD = {
    "p": "b", "t": "d", "k": "g", "c": "g", "q": "g", "f": "v", "s": "z",
    "sh": "zh", "ch": "zh", "y": "i", "x": "gz",
}  # fmt: skip

# Vowels survive as one class rather than being dropped entirely: dropping them merges far
# too much ("ConLoca" and "Klinik" both reduce to nothing useful), while keeping them apart
# defeats the point, since an unstressed vowel is exactly what a decoder guesses at.
_VOWELS = set("aeiou")
_VOWEL_CLASS = "a"


@lru_cache(maxsize=8192)
def phonetic(word: str) -> str:
    """The sound-skeleton of one word: lower case, transliterated, folded, de-duplicated."""
    latin = _to_latin(word)
    folded = _fold(latin)
    return _dedupe(folded)


def similarity(left: str, right: str) -> float:
    """How alike two words sound, from 0 (nothing in common) to 1 (indistinguishable).

    The score is the higher of the two readings — spelling and sound — because either kind
    of match is worth surfacing: "Conloca" is a spelling hit, "Colocka" is a sound hit, and
    a filter that demanded both would miss half of what it exists to catch.
    """
    if not left or not right:
        return 0.0
    written = ratio(left.casefold(), right.casefold())
    heard = ratio(phonetic(left), phonetic(right))
    return max(written, heard)


def ratio(left: str, right: str) -> float:
    """Levenshtein distance normalized by the longer string: 1.0 is identical."""
    if left == right:
        return 1.0
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return max(0.0, 1.0 - distance(left, right) / longest)


def distance(left: str, right: str) -> int:
    """Levenshtein edit distance, one row at a time.

    The full matrix is not needed — nothing here wants the alignment, only the number — and
    a single row keeps this linear in memory over the thousands of comparisons a transcript
    full of dictionary candidates produces.
    """
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (left_char != right_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


def _to_latin(word: str) -> str:
    """Lower case, strip accents, map Cyrillic across, and drop everything else."""
    lowered = unicodedata.normalize("NFKD", word.casefold())
    stripped = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    out = [_CYRILLIC.get(ch, ch) for ch in stripped]
    return re.sub(r"[^a-z]", "", "".join(out))


def _fold(latin: str) -> str:
    """Collapse the distinctions a recording does not preserve."""
    out: list[str] = []
    index = 0
    while index < len(latin):
        pair = latin[index : index + 2]
        if pair in _FOLD:
            out.append(_FOLD[pair])
            index += 2
            continue
        char = latin[index]
        out.append(_VOWEL_CLASS if char in _VOWELS else _FOLD.get(char, char))
        index += 1
    return "".join(out)


def _dedupe(text: str) -> str:
    """Squeeze runs of the same sound: nobody hears the difference between -nn- and -n-."""
    out: list[str] = []
    for char in text:
        if not out or out[-1] != char:
            out.append(char)
    return "".join(out)
