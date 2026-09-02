"""dictionary — the words a speech model has never heard of.

WHY A TOOL LIKE THIS NEEDS ONE
    Whisper knows the language; it does not know your product, your colleagues or your
    codebase. Every recording of a real working conversation is therefore full of terms it
    guesses at, and it guesses differently each time: one five-minute clip produced ConLoca,
    Conloca, ConLog and ConLoka for a single project name. No amount of cleaning fixes that,
    because nothing in the audio is wrong — the model simply does not know the word.

THREE PLACES THE DICTIONARY IS USED, ON PURPOSE
    1. Before decoding, as the speech model's initial prompt. Whisper conditions on it, and
       measurably: with the glossary carried, "Vigma" came back as "Figma" and three of four
       ConLoca mentions came back spelled correctly. This is the only one of the three that
       can fix a word the model got wrong at the acoustic level.
    2. After decoding, as an exact substitution for spellings the user has recorded as wrong
       (``aka``) and as a FLAG for words that merely sound like a dictionary entry. The
       distinction matters: a spelling the user wrote down is a fact, a phonetic near-match
       is a suspicion, and suspicions must not silently rewrite somebody's transcript.
    3. In the LLM correction prompt, as a glossary with the flagged candidates attached, so
       the pass that can actually read the sentence decides what the near-matches were.

    Doing only the last one would be simpler and worse: by then the model has already
    committed to a word, and an LLM asked to guess between "ConLog" and "ConLoca" from text
    alone has strictly less to go on than the decoder had.
"""

from __future__ import annotations

import fcntl
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import fuzzy
from ._errors import UsageError
from .models import Segment, Variant

FILENAME = "dictionary.json"

# The glossary is carried in the decoder's prompt, whose budget is shared with the context
# it would otherwise use. Whisper truncates an over-long prompt from the FRONT, so an
# unbounded dictionary would silently drop its own first entries; capping it here means the
# tool decides what gets left out, and says so, instead of the engine doing it quietly.
PROMPT_TERMS = 24
PROMPT_CHARS = 220

# The LLM's copy of the glossary. It goes into EVERY correction window, so an unbounded
# dictionary multiplies straight into the prompt: an oversized one makes every window fail
# and the transcript comes back uncorrected with nothing naming the cause.
LLM_GLOSSARY_CHARS = 4000

# How alike a word has to sound before it is worth flagging. Measured against the real
# failure rather than guessed: every spelling the model actually produced for ConLoca —
# ConLog, Conloca, Coloca, ConLoka, Colocka, and the Cyrillic "конлока" — scores 0.857 or
# better, while ordinary words that merely share a shape ("replace", "content", "customer")
# top out at 0.714. Anything in between is a coin flip, so the line goes above it.
DEFAULT_SIMILARITY = 0.80

# Extra similarity demanded of a phrase that spans more words than the term does. Joining
# two words drops the boundary between them, and enough ordinary pairs ("can once", "can
# raise") then resemble a short term to clear a threshold calibrated for single words. The
# wider window exists for one specific case — a term the model wrote as two words — so it
# has to pay for itself rather than filling the transcript with flags.
WIDE_PENALTY = 0.15
# ... and the joined phrase has to be about the same length as the term, for the same reason.
WIDE_LENGTH_SLACK = 2

# Terms shorter than this are not screened phonetically. Three-letter words collide with
# everything, and a flag on every other word is the same as no flags at all.
MIN_TERM_LENGTH = 4

# ... and terms longer than this are not screened either, nor accepted in the first place.
# The screen compares every word run in the text up to one word wider than the widest term,
# so the cost of a single entry is paid by every segment of every recording: a pasted
# paragraph of a term turns a transcript into millions of phonetic comparisons and looks,
# from the outside, exactly like a hang. A term is a name — a product, a person, a project —
# and eight words is already generous for one. The limit is enforced where entries are
# accepted (`Dictionary.add`, so both `dict add` and `dict import` pay it) and enforced
# again where they are read (`_screened`), because a dictionary file can also be hand-edited
# and refusing to LOAD it would lock the user out of a file the error tells them to fix.
MAX_TERM_WORDS = 8
MAX_TERM_CHARS = 80

# ...and there is a limit on how MANY terms are screened, for the same reason. The screen
# costs one comparison per (word run in the segment × term), so the price of the dictionary
# is paid again by every segment of every recording: a hundred thousand imported names turn
# a long transcript into billions of comparisons, which again looks exactly like a hang.
# Five hundred is far past any real glossary — the decoder's own prompt carries 24 — and
# leaves the worst case in seconds rather than hours. Entries beyond it are refused on the
# way in; a file that already has more is screened against the first five hundred, which are
# the ones the user put first (`Dictionary` documents order as priority) and the same ones
# the prompt would carry. Correction by `aka` and the LLM glossary are unaffected: neither
# is quadratic.
MAX_SCREENED_TERMS = 500

# ...and a limit on how many spellings one term may claim. Every alias becomes a branch of
# the single substitution pattern, so a hand-edited entry with fifty thousand of them
# compiles a half-megabyte regex that every segment of every recording is then matched
# against. Two dozen recorded misspellings of one word is already an unusual amount of
# patience. Enforced on the way in, and again in `_alias_index`, because the file can be
# written by hand and refusing to LOAD it would lock the user out of the file the error
# tells them to fix.
MAX_ALIASES = 24

# The most alias branches one pattern may hold, whatever the shape of the file that produced
# them: the product of the two caps above, so a dictionary built entirely through the
# commands can never hit it, and a hand-edited one is bounded without exact correction
# quietly switching off for the entries near the end.
MAX_ALIAS_BRANCHES = MAX_SCREENED_TERMS * MAX_ALIASES

# Room for the largest dictionary the caps above allow, several times over, and no room for
# a file that is not a dictionary at all. Every transcription reads this file.
MAX_DICTIONARY_BYTES = 4 * 1024 * 1024


# A note is free text the user wrote for the LLM correction pass to read, and `stt dict
# import` will take it from a file somebody else wrote. It is framed as data in the prompt
# (JSON, inside a marked region, with a rule saying so), but framing is not a wall an LLM
# cannot climb: a note long enough to hold a paragraph of instructions is worth more to an
# attacker than to the reader it was written for. One line, one sentence's worth.
NOTE_CHARS = 120


def _as_a_note(text: str) -> str:
    """Keep a note to the shape of a note: one line, and short."""
    single = " ".join(text.split())
    return single if len(single) <= NOTE_CHARS else single[: NOTE_CHARS - 1].rstrip() + "…"


@dataclass(slots=True)
class Term:
    """One thing the user knows and the speech model does not."""

    term: str
    aka: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"term": self.term}
        if self.aka:
            payload["aka"] = list(self.aka)
        if self.note:
            payload["note"] = self.note
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Term:
        aka = raw.get("aka")
        return cls(
            term=str(raw.get("term", "")).strip(),
            aka=[str(item).strip() for item in aka if str(item).strip()]
            if isinstance(aka, list)
            else [],
            note=str(raw.get("note", "")),
        )

    def describe(self) -> str:
        parts = [self.term]
        if self.aka:
            parts.append(f"(heard as: {', '.join(self.aka)})")
        if self.note:
            parts.append(f"— {_as_a_note(self.note)}")
        return " ".join(parts)


def normalized(term: Term) -> Term:
    """Strip the separator characters out of a term and its aliases.

    The comma is the glossary's own separator: `Dictionary.prompt()` joins terms with it, so
    `stt dict add "Foo, Inc"` reaches the speech model as two terms, and the budget warning —
    which counts commas to work out how many were carried — miscounts by one for every such
    entry. Removing the character is the cheapest of the three options; refusing the term
    would reject a real company name, and quoting it would have to be understood by both the
    prompt builder and the counter.

    ``|`` and ``~`` are here for the same reason one step further out: `formats._delimited`
    joins the flagged term pairs with them into one CSV cell, and a term carrying either
    would make that cell ambiguous to whatever splits it. Not a security question — this is
    the user's own dictionary, not transcript text — just the same separator problem.
    """
    aka: list[str] = []
    seen: set[str] = set()
    for alias in term.aka:
        squeezed = _squeeze(alias)
        if squeezed and squeezed.casefold() not in seen:
            seen.add(squeezed.casefold())
            aka.append(squeezed)
    return Term(term=_squeeze(term.term), aka=aka, note=_printable(term.note))


def _squeeze(text: str) -> str:
    """Drop the separator characters and collapse the whitespace they leave behind."""
    for separator in ",|~":
        text = text.replace(separator, " ")
    return " ".join(_printable(text).split())


def _printable(text: str) -> str:
    """Remove control characters, ESC included.

    A glossary can be imported from a file somebody else wrote, and it is stored and then
    printed back verbatim by `stt dict list`. An entry carrying an ANSI escape — an OSC-8
    hyperlink, a cursor move, a colour that never gets reset — is executed by the terminal
    that prints it, so the imported file gets to decide what the user sees. Stripping them
    at the point of normalization means the stored file is clean too, not just this one
    rendering of it.
    """
    return "".join(ch for ch in text if ch == "\t" or not unicodedata.category(ch).startswith("C"))


@dataclass(slots=True)
class Dictionary:
    """Every term, in the order the user added them. Order is priority: the prompt is
    capped, so the entries at the top are the ones that survive the cap."""

    terms: list[Term] = field(default_factory=list)
    # Built on demand and kept, because `correct_text` is called once per variant per shaky
    # segment and used to rebuild the whole index — a counter over every alias, a sort and a
    # regex source — each time. The index belongs to this snapshot of the dictionary, which
    # the pipeline already pins for the run; it is dropped whenever the terms change.
    _index: tuple[re.Pattern[str] | None, dict[str, str]] | None = field(
        default=None, repr=False, compare=False
    )

    def __bool__(self) -> bool:
        return bool(self.terms)

    def alias_index(self) -> tuple[re.Pattern[str] | None, dict[str, str]]:
        """The one pattern that rewrites every recorded misspelling, and what each maps to."""
        if self._index is None:
            self._index = _alias_index(self.terms)
        return self._index

    def find(self, name: str) -> Term | None:
        wanted = name.casefold()
        return next((t for t in self.terms if t.term.casefold() == wanted), None)

    def add(self, term: Term) -> bool:
        """Add or merge. Returns True when the entry was new."""
        term = normalized(term)
        # Checked here, not only in the `dict add` command: `stt dict import` writes through
        # this method too, and a line reading `, = foo` produced a term that normalized to
        # nothing, was appended, and was SAVED — after which `load()`'s own guard refused the
        # file and every later `stt dict` command and every transcription failed until the
        # JSON was hand-edited. One stray line in an imported glossary should not be able to
        # do that, and the format is documented as the one people already have lying around.
        if not term.term:
            raise UsageError(
                what="a term cannot be empty",
                why="nothing was left of it once `,`, `|` and `~` were stripped",
                how="stt dict add ConLoca --aka ConLog",
            )
        _within_limits(term.term)
        for alias in term.aka:
            _within_limits(alias)
        found = self.find(term.term)
        known = {alias.casefold() for alias in found.aka} if found else set()
        # The UNION, not the sum: `stt dict add Figma --aka Vigma` on an entry that already
        # records Vigma stores nothing new, and refusing it at the cap would make an
        # idempotent update fail for adding nothing.
        if len(known | {alias.casefold() for alias in term.aka}) > MAX_ALIASES:
            raise UsageError(
                what=f"{term.term!r} would have more than {MAX_ALIASES} spellings",
                why="every alias is one branch of the pattern matched against every segment",
                how="keep the spellings that actually turn up, and drop the rest",
            )
        self._refuse_an_ambiguous_alias(term)
        if len(self.terms) >= MAX_SCREENED_TERMS and self.find(term.term) is None:
            raise UsageError(
                what=f"the dictionary already holds {MAX_SCREENED_TERMS} terms",
                why="every term is compared against every word run of every segment, so the "
                "dictionary is paid for again by each recording",
                how="remove the entries you no longer need (`stt dict rm <term>`)",
            )
        existing = self.find(term.term)
        self._index = None
        if existing is None:
            self.terms.append(term)
            return True
        # `known` grows as we go: computed once, `--aka Vigma --aka vigma` passed both
        # through, and the duplicate then reached the glossary and the cache key.
        known = {alias.casefold() for alias in existing.aka}
        for alias in term.aka:
            if alias.casefold() not in known:
                known.add(alias.casefold())
                existing.aka.append(alias)
        existing.note = term.note or existing.note
        return False

    def _refuse_an_ambiguous_alias(self, term: Term) -> None:
        """An alias may not be another entry's name, nor another entry's alias.

        `stt dict add Figma --aka Sketch` followed by `stt dict add Sketch` used to be
        accepted, and every honest mention of Sketch was then rewritten to Figma — silently,
        because the substitution keeps the first alias it finds and says nothing about the
        collision. A dictionary that quietly renames a real product is worse than no
        dictionary, and there is no reading of two conflicting entries that is safe to guess
        at, so the second one is refused with both sides named.
        """
        wanted = term.term.casefold()
        aliases = {alias.casefold() for alias in term.aka}
        for other in self.terms:
            if other.term.casefold() == wanted:
                continue  # merging into itself; aliases are deduplicated by `add`
            clash = next(
                (alias for alias in other.aka if alias.casefold() in aliases | {wanted}), ""
            )
            if other.term.casefold() in aliases:
                clash = other.term
            if clash:
                raise UsageError(
                    what=f"{clash!r} already belongs to {other.term!r}",
                    why="one spelling cannot be corrected to two different terms, and the "
                    "one that wins would be decided by the order of the file",
                    how=f"drop it from one of them (`stt dict rm {other.term}` to start over)",
                )

    def remove(self, name: str) -> bool:
        term = self.find(name)
        if term is None:
            return False
        self.terms.remove(term)
        self._index = None
        return True

    def prompt(self) -> str:
        """The glossary as the speech model's initial prompt, inside its budget.

        Phrased as a sentence rather than a bare list because whisper conditions on it as
        text: a comma-separated run of proper nouns with no verb reads to the model like the
        start of a list, and it will happily continue producing one.
        """
        chosen: list[str] = []
        used = 0
        for term in self.terms[:PROMPT_TERMS]:
            # `continue`, not `break`: one long entry at the top used to end the scan, so a
            # single 221-character term silenced the glossary for every short, perfectly
            # valid term after it — `Figma` never reached the decoder at all.
            if used + len(term.term) + 2 > PROMPT_CHARS:
                continue
            chosen.append(term.term)
            used += len(term.term) + 2
        return f"Glossary: {', '.join(chosen)}." if chosen else ""

    def digest(self) -> str:
        """A short hash of the content, for the cache key.

        Order is part of it, deliberately: the prompt budget is finite and takes terms from
        the top, so moving a term up or down changes which ones reach the speech model and
        therefore changes the transcript. Reformatting the file does not, which is why this
        hashes the parsed terms rather than the bytes on disk.
        """
        import hashlib

        if not self.terms:
            return ""
        blob = json.dumps(
            [term.to_dict() for term in self.terms], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def glossary(self, limit: int = LLM_GLOSSARY_CHARS) -> list[str]:
        """The dictionary as lines for the LLM prompt, inside a budget of its own.

        Far larger than the speech model's, because the LLM has room — but not unbounded.
        An imported glossary can be any size, and the whole of it is injected into EVERY
        correction window: a big enough import pushes each window past the model's context,
        the agent rejects the call, and `correct` records a failed window and moves on. The
        transcript comes back uncorrected with nothing pointing at the dictionary as the
        cause. Truncating instead loses the tail of the glossary, which is visible in the
        prompt and costs one term's spelling rather than the entire pass.
        """
        lines: list[str] = []
        used = 0
        for term in self.terms:
            line = term.describe()
            if used + len(line) + 1 > limit:
                lines.append(f"... and {len(self.terms) - len(lines)} more term(s), omitted")
                break
            lines.append(line)
            used += len(line) + 1
        return lines


@dataclass(slots=True, frozen=True)
class Hit:
    """One phrase the phonetic screen believes is a misheard term, and how sure it is."""

    phrase: str
    term: str
    score: float


@dataclass(slots=True)
class DictReport:
    """What the dictionary changed, and what it merely wants a second opinion on."""

    replaced: list[tuple[str, str]] = field(default_factory=list)
    flagged: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        if not self.replaced and not self.flagged:
            return "nothing matched"
        parts = []
        if self.replaced:
            parts.append(f"{len(self.replaced)} known misspelling(s) corrected")
        if self.flagged:
            parts.append(f"{len(self.flagged)} word(s) flagged as possible terms")
        return ", ".join(parts)

    def detail(self) -> list[str]:
        return [f"{heard} -> {canonical}" for heard, canonical in self.replaced] + [
            f"{heard} ~ {suspected}?" for heard, suspected in self.flagged
        ]


# ── storage ───────────────────────────────────────────────────────────────────────────


def path() -> Path:
    from . import config

    return config.app_home() / FILENAME


def load() -> Dictionary:
    """The user's dictionary, or an empty one. A malformed file is an error, not a shrug."""
    target = path()
    if not target.is_file():
        return Dictionary()
    try:
        # Bounded, because this file is read by EVERY transcription and can be replaced by
        # anything: a few hundred megabytes of JSON was parsed, normalized and serialized
        # again for the cache digest before a second of audio was touched. The import path
        # was already bounded; the file it writes to was not.
        if target.stat().st_size > MAX_DICTIONARY_BYTES:
            raise UsageError(
                what=f"{target} is larger than {MAX_DICTIONARY_BYTES // (1024 * 1024)} MiB",
                why=f"a dictionary holds at most {MAX_SCREENED_TERMS} names, not a corpus",
                how="trim it, or delete it to start with an empty dictionary",
            )
        raw = json.loads(target.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(
            what=f"could not read {target}",
            why=str(exc),
            how="fix the JSON, or delete the file to start with an empty dictionary",
        ) from exc
    entries = raw.get("terms") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise UsageError(
            what=f"{target} is not a dictionary",
            why='expected an object with a "terms" list',
            how='write {"terms": []} into it, or delete it',
        )
    # A malformed entry is an error, never a shrug. Skipping it looked harmless and was the
    # opposite: `{"terms": ["Figma"]}` loaded as an EMPTY dictionary, and the next
    # `stt dict add` wrote that empty list back over the file — the entry gone for good, and
    # every run in between silently decoded with no glossary at all.
    for position, item in enumerate(entries):
        if not isinstance(item, dict) or not str(item.get("term", "")).strip():
            raise UsageError(
                what=f"entry {position + 1} in {target} is not a term",
                why='every entry must be an object with a non-empty "term", like'
                ' {"term": "ConLoca", "aka": ["ConLog"]}',
                how="fix that line, or delete the file to start with an empty dictionary",
            )
        # The fields as well as the entry, and for the same reason. `"aka": "Vigma"` — one
        # spelling written without the brackets — was read as no aliases at all: the term
        # loaded looking fine, decoded without its misspelling, and the next unrelated
        # `stt dict add` saved that stripped version over the file. The alias was then gone
        # for good, having never once been reported as a problem.
        if not isinstance(item.get("aka", []), list) or not all(
            isinstance(alias, str) for alias in item.get("aka", [])
        ):
            raise UsageError(
                what=f'the "aka" of entry {position + 1} in {target} is not a list of spellings',
                why='aliases are written as a list, even when there is one: {"aka": ["Vigma"]}',
                how="put the brackets back, or remove the field",
            )
        if not isinstance(item.get("note", ""), str):
            raise UsageError(
                what=f'the "note" of entry {position + 1} in {target} is not text',
                why="a note is a short line of free text for the correction pass to read",
                how="quote it, or remove the field",
            )
    # Normalized on the way IN, not only where `stt dict add` writes. Editing the file by
    # hand is a supported path — the error above says "fix the JSON" — and a comma typed
    # there would otherwise reach the speech model as two glossary terms, miscount the
    # budget warning, and give the same dictionary a different cache key than `dict add`.
    terms = [normalized(Term.from_dict(item)) for item in entries]
    for position, term in enumerate(terms):
        # After normalizing, not before: `{"term": ","}` passes the check above — the raw
        # string is not empty — and comes out of `normalized` as nothing at all. Filtered
        # away silently, the file loaded as a smaller dictionary and the next `stt dict add`
        # wrote that smaller version back, taking the hand-edited entry with it. That is the
        # same permanent loss the check above exists to prevent, one step later.
        if not term.term:
            raise UsageError(
                what=f"entry {position + 1} in {target} is nothing but separators",
                why="a term is stripped of `,`, `|` and `~`, and this one has nothing left",
                how="give it a name, or delete the entry",
            )
    return Dictionary(terms=terms)


def save(dictionary: Dictionary) -> Path:
    from . import config
    from .archive import write_atomic

    config.ensure_dirs()
    target = path()
    payload = {"terms": [term.to_dict() for term in dictionary.terms]}
    write_atomic(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return target


@contextmanager
def editing() -> Iterator[Dictionary]:
    """Load the dictionary, hand it over to be changed, and write it back — under a lock.

    Every command that changes the dictionary reads the whole file, edits the object and
    writes the whole file back. The write itself is atomic, which is not the same as safe:
    two terminals running ``stt dict add`` at the same time both read the old list, and
    whichever finishes last stores a list that never saw the other one's term. The term is
    gone, with no error anywhere. An exclusive lock held across both the read and the write
    is what makes the pair one operation. Nothing is written if the body raises.
    """
    from . import config

    config.ensure_dirs()
    lock = path().with_name(FILENAME + ".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            dictionary = load()
            yield dictionary
            save(dictionary)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ── applying it to a transcript ───────────────────────────────────────────────────────


def apply(
    segments: list[Segment], dictionary: Dictionary, *, similarity: float = DEFAULT_SIMILARITY
) -> DictReport:
    """Correct the spellings the user recorded, and flag the words that merely sound right."""
    report = DictReport()
    if not dictionary:
        return report
    pattern, canonical = dictionary.alias_index()
    screened = _screened(dictionary.terms)
    width = _width(dictionary.terms)
    for segment in segments:
        _substitute(segment, pattern, canonical, report)
        _screen(segment, screened, similarity, width, report)
    return report


def correct_text(text: str, dictionary: Dictionary) -> str:
    """Apply the recorded misspellings to a bare string — no flags, no suspicions.

    For text that arrives AFTER `apply` has run: a second-opinion decoding of a shaky
    segment, gathered once the transcript was already corrected. An `aka` spelling is a fact
    about a word, so it is as wrong in a variant as it was in the transcript; the phonetic
    screen is deliberately not run here, because a suspicion belongs on the segment the
    reader is looking at, not duplicated onto every alternative reading of it.
    """
    pattern, canonical = dictionary.alias_index()
    if pattern is None:
        return text
    return pattern.sub(_Swap(canonical, DictReport()), text)


def _alias_index(terms: list[Term]) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """One whole-word pattern over EVERY alias in the dictionary, longest alias first.

    One pattern across all terms, not one per term: with "ACME" recorded on one entry and
    "ACME Corp" on another, running the entries in order rewrites "ACME Corp shipped" into
    "First Corp shipped" and the longer alias can never match again. Sorting globally by
    length makes the longest alias win regardless of which entry it came from — Python's
    alternation takes the first branch that matches at a position, so the order IS the rule.
    """
    canonical: dict[str, str] = {}
    # A word that is somebody's NAME is never rewritten into somebody else's, however the
    # file came to say so. `add` refuses that collision, but the file can be hand-edited,
    # and the failure it produces is the worst kind: every honest mention of a real product
    # silently becomes a different product. Skipping the alias leaves the word alone, which
    # is the only reading of two conflicting entries that cannot be wrong.
    names = {term.term.casefold() for term in terms}
    claimed = Counter(
        alias.casefold() for term in terms for alias in term.aka[:MAX_ALIASES] if alias.strip()
    )
    for term in terms:
        for alias in term.aka[:MAX_ALIASES]:
            # `_fits` here as well as in `_screened`: the count of aliases was bounded but
            # not their LENGTH, so one hand-edited entry holding a fifty-megabyte string
            # compiled a pattern of the same size before a single second of audio was
            # decoded. `add` refuses such an alias; the file can still be written by hand.
            if not alias.strip() or not _fits(alias) or alias.casefold() in canonical:
                continue
            # Claimed by two entries, or by one entry while being another's name: either way
            # which correction wins would be decided by the order of the file.
            if claimed[alias.casefold()] > 1:
                continue
            if alias.casefold() in names and alias.casefold() != term.term.casefold():
                continue
            canonical[alias.casefold()] = term.term
            if len(canonical) >= MAX_ALIAS_BRANCHES:
                # Bounded by BRANCHES, not by which entry an alias sits on. Truncating the
                # term list here silently switched off exact correction for everything past
                # entry 500 — the opposite of what the caps promise, which is that only the
                # phonetic screen is capped and a spelling the user wrote down is always
                # applied.
                return _compiled(canonical)
    return _compiled(canonical)


def _compiled(canonical: dict[str, str]) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """One alternation over every alias, longest first — see `_alias_index`."""
    if not canonical:
        return None, {}
    ordered = sorted(canonical, key=len, reverse=True)
    joined = "|".join(re.escape(alias) for alias in ordered)
    return re.compile(rf"(?<!\w)(?:{joined})(?!\w)", re.IGNORECASE), canonical


def _substitute(
    segment: Segment, pattern: re.Pattern[str] | None, canonical: dict[str, str], report: DictReport
) -> None:
    """Rewrite recorded misspellings, keeping what the speech model actually said.

    The original is kept as the ``primary`` variant — the same slot the LLM pass uses — so
    ``--text raw`` still returns the speech model's own wording. Without it the dictionary
    would quietly become the only record of what was said, and "the original is always
    kept" would be false for exactly the words most worth checking.
    """
    if pattern is None:
        return
    before = segment.text
    after = pattern.sub(_Swap(canonical, report), before)
    if after == before:
        return
    segment.text = after
    if not any(variant.kind == "primary" for variant in segment.variants):
        segment.variants.insert(
            0,
            Variant(text=before, source="asr", kind="primary", confidence=segment.confidence),
        )


@dataclass(slots=True)
class _Swap:
    """A replacement that records what it replaced, resolving each alias to its own term."""

    canonical: dict[str, str]
    report: DictReport

    def __call__(self, match: re.Match[str]) -> str:
        heard = match.group(0)
        replacement = self.canonical.get(heard.casefold(), heard)
        if heard != replacement:
            self.report.replaced.append((heard, replacement))
        return replacement


def validate_similarity(value: float) -> None:
    """A similarity outside 0..1 is not a preference, it is a typo with consequences.

    Below zero every comparable phrase is flagged; above one nothing ever is, silently
    turning the phonetic screen off. `argparse` only checks that it is a float, and a config
    file does not even do that.

    Lives here rather than in the pipeline because `stt dict check` needs the same answer:
    with the check reading the setting and only the pipeline validating it, `dict_similarity
    1.5` made the command report "nothing matched" while a real transcription refused to
    start — the command that exists to explain the threshold disagreeing with it.
    """
    if not 0.0 <= value <= 1.0 or value != value:  # the second test catches NaN
        raise UsageError(
            what=f"dict_similarity must be between 0 and 1, got {value}",
            why="below 0 flags every word; above 1 flags none, which disables the screen",
            how="try `--dict-similarity 0.8`, or `stt config set dict_similarity 0.8`",
        )


def screen(
    text: str, dictionary: Dictionary, *, similarity: float = DEFAULT_SIMILARITY
) -> list[Hit]:
    """Every phrase in ``text`` that sounds like a dictionary term without being one.

    Public because ``stt dict check`` has to answer with what the pipeline would actually
    do. Reimplementing the scoring there — one similarity call per whitespace-split word —
    gave a different answer in both directions: it flagged the term itself, which the
    pipeline deliberately never does, and it stayed silent on "hyper ide", which the
    pipeline flags. A command whose job is to explain a decision must run the decision.
    """
    return _screen_text(text, _screened(dictionary.terms), similarity, _width(dictionary.terms))


def _screened(terms: list[Term]) -> list[Term]:
    keep = [term for term in terms if len(term.term) >= MIN_TERM_LENGTH and _fits(term.term)]
    return keep[:MAX_SCREENED_TERMS]


def _fits(text: str) -> bool:
    """Is this short enough to screen phonetically? See ``MAX_TERM_WORDS``."""
    return len(text) <= MAX_TERM_CHARS and len(text.split()) <= MAX_TERM_WORDS


def _within_limits(text: str) -> None:
    """Refuse an entry the screen could not afford to look for."""
    if _fits(text):
        return
    raise UsageError(
        what=f"the term {text[:40]!r}... is too long to be a term",
        why=(
            f"a dictionary entry is a name, capped at {MAX_TERM_WORDS} words and "
            f"{MAX_TERM_CHARS} characters; longer ones make every transcript slow to screen"
        ),
        how="add the name on its own, and put the explanation in the note (`# ...`)",
    )


def _width(terms: list[Term]) -> int:
    # One past the longest term, because the error runs both ways: a two-word term can
    # be heard as one word, and a one-word term the model has never seen ("HyperIDE") is
    # routinely written down as two ("hyper ide").
    return max((len(term.term.split()) for term in _screened(terms)), default=1) + 1


def _screen_text(text: str, terms: list[Term], threshold: float, width: int) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    for phrase in _ngrams(text, width):
        best, score = _best_term(phrase, terms, threshold)
        if not best or (phrase, best) in seen:
            continue
        seen.add((phrase, best))
        hits.append(Hit(phrase=phrase, term=best, score=score))
    return hits


def _screen(
    segment: Segment, terms: list[Term], threshold: float, width: int, report: DictReport
) -> None:
    """Flag anything in this segment that sounds like a dictionary entry but is not one."""
    for hit in _screen_text(segment.text, terms, threshold, width):
        pair = (hit.phrase, hit.term)
        if pair in segment.suspected_terms:
            continue
        segment.suspected_terms.append(pair)
        segment.flag("term")
        report.flagged.append(pair)


def _best_term(phrase: str, terms: list[Term], threshold: float) -> tuple[str, float]:
    """The term this phrase most resembles, among those it resembles ENOUGH.

    Each term's own threshold is applied during selection, not afterwards. Applied
    afterwards, a wide match that outscores a narrow one and then fails the wider bar would
    take the narrow one down with it — the narrow term would have passed on its own and is
    never reconsidered.
    """
    best, score = "", 0.0
    for term in terms:
        # A phrase that already CONTAINS the term is not a near-miss of it: the wider window
        # that catches "hyper ide" also produces "ConLoca is", which scores high for the
        # obvious reason and would flag a word that is already spelled correctly.
        if term.term.casefold() in phrase.casefold() or not _comparable(phrase, term.term):
            continue
        needed = threshold + (WIDE_PENALTY if _is_wide(phrase, term.term) else 0.0)
        value = fuzzy.similarity(phrase, term.term)
        if value >= needed and value > score:
            best, score = term.term, value
    return best, score


def _is_wide(phrase: str, term: str) -> bool:
    """Does this phrase span more words than the term it is being compared against?"""
    return len(phrase.split()) > len(term.split())


def _comparable(phrase: str, term: str) -> bool:
    """Cheap length gate before the expensive comparison, and a hard one for wide phrases."""
    if not _is_wide(phrase, term):
        return True
    joined = len(phrase.replace(" ", ""))
    return abs(joined - len(term.replace(" ", ""))) <= WIDE_LENGTH_SLACK


def _ngrams(text: str, width: int) -> list[str]:
    """Word runs of 1..width, so a multi-word term has something to be compared against."""
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    out: list[str] = []
    for size in range(1, max(1, width) + 1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            if len(phrase) >= MIN_TERM_LENGTH:
                out.append(phrase)
    return out
