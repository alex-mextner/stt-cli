"""typist — put text in the window that has focus, and take it back when it improves.

THE ONE RULE
    stt may delete only characters it typed itself and has not given up. Everything here
    exists to keep that true. The draft pass produces a new guess at the same sentence three
    or four times before the accurate pass replaces it, so the text on screen has to be
    edited in place; and the moment it is not certain the caret still sits after our own
    characters, the right move is to stop touching them, not to guess.

WHY IT IS EDITED RATHER THAN WAITED FOR
    The alternative is typing nothing until the accurate pass is done, which is a second or
    two of an empty field after every sentence. Dictation that lags that far behind the voice
    feels broken even when it is right, so the draft goes in immediately and is corrected
    underneath the user. `--no-draft` is there for anyone who would rather wait.

WHAT REPLACES THE UNDERLINE
    Apple's own dictation marks provisional text with an underline. Only an input method can
    do that — marked text is a thing an application's text system draws, not a thing anyone
    can type — and an input method is a separate installed app bundle, not a CLI. So the
    provisional text here looks exactly like typed text, and the place that shows what is
    provisional and what has settled is the terminal `stt mic` is running in.
"""

from __future__ import annotations

import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class Keyboard(Protocol):
    """The two things a typist does to the outside world."""

    def type_text(self, text: str) -> None: ...

    def press_backspace(self, times: int) -> None: ...


@dataclass
class Edit:
    """One in-place correction: delete this many characters, then type this text."""

    backspaces: int
    text: str

    @property
    def empty(self) -> bool:
        return self.backspaces == 0 and not self.text


def plan(shown: str, wanted: str) -> Edit:
    """The smallest edit that turns `shown` into `wanted` at the end of a line.

    A caret can only walk backwards from where it is, so the only edit available is "delete
    a suffix, type a different one" — which makes the longest common prefix the answer, not
    an approximation of it. Whisper redeciding the last word of a growing sentence therefore
    costs those few characters, and a rewrite that changes the first word costs the sentence.

    The number of backspaces is a count of GRAPHEME CLUSTERS, not of characters, and the
    difference is not academic: a text field deletes what a reader would call one character,
    and "é" written as e plus a combining accent, or a family emoji written as four people
    joined by zero-width joiners, is several characters and one of those. Counting characters
    asked for seven backspaces where the field needed one, and the other six came out of
    whatever the user had written before stt started typing. That is the one thing this
    module exists to make impossible.
    """
    keep = 0
    for mine, theirs in zip(shown, wanted, strict=False):
        if mine != theirs:
            break
        keep += 1
    # Only ever back to a cluster boundary: a prefix that matches half of a cluster would
    # otherwise leave the caret inside one, and neither half of the count means anything then.
    keep = _at_a_boundary(shown, keep)
    return Edit(backspaces=_clusters(shown[keep:]), text=wanted[keep:])


# Code points that continue the cluster they follow rather than starting one of their own.
# This is a deliberate, conservative subset of the Unicode rules — combining marks, the
# zero-width joiner and what follows it, variation selectors, skin tones and flag pairs —
# chosen because it covers what actually turns up in transcribed text and in a user's
# terminology list. Where it is wrong it UNDER-counts, and that direction is the safe one:
# too few backspaces leaves stt's own characters on screen, which is untidy, while too many
# deletes the user's, which is the failure this whole module is written against.
#
# NOT VERIFIED AGAINST A LIVE TEXT FIELD. AppKit's `deleteBackward:` works on composed
# character sequences, which is where the cluster rule comes from, and the tests pin the
# counting — but nobody has yet watched a real window delete a family emoji with one press.
# The attempt to check it aborted rather than typing into the wrong window: TextEdit would
# not come to the front. A field that deletes by code point instead (a terminal, an editor
# with its own handling) leaves a few of stt's own characters behind, the harmless direction.
_ZERO_WIDTH_JOINER = "\u200d"
_VARIATION_SELECTORS = ("\ufe0e", "\ufe0f")
_SKIN_TONES = tuple(chr(point) for point in range(0x1F3FB, 0x1F400))
_REGIONAL = range(0x1F1E6, 0x1F200)
# The canonical combining class of a virama, which is how a conjunct is written.
_VIRAMA = 9
# Hangul jamo: a lead consonant, a vowel, and an optional tail, drawn as one syllable block.
_HANGUL_L = range(0x1100, 0x1160)
_HANGUL_V = range(0x1160, 0x11A8)
_HANGUL_T = range(0x11A8, 0x1200)


# U+E0020..U+E007F: the tag characters, which spell out a subdivision flag's region code
# after a black flag and are drawn and deleted as part of it.
_TAGS = range(0xE0020, 0xE0080)


def _joins_the_one_before(text: str, at: int) -> bool:
    """Does the code point at `at` continue the cluster that precedes it?

    Every rule here is one-directional on purpose: each says "these two are ONE character",
    never "these two are two", so being wrong can only merge things that should have been
    separate. That produces too FEW backspaces, which leaves a few of stt's own characters
    on screen. Being wrong the other way produces too many, and the extra ones come out of
    what the user wrote. The list is therefore allowed to be incomplete and is not allowed
    to be eager.

    It was eager once, and a reviewer found it: `क्ष` is one character in a Cocoa text field
    and this counted it as two, because the consonant after the virama looked like a fresh
    start. Two backspaces where the field wanted one is exactly the deletion this module
    exists to prevent, in a script nobody here had thought to try.
    """
    here, before = text[at], text[at - 1]
    if unicodedata.combining(here) or unicodedata.category(here) in ("Mn", "Mc", "Me"):
        return True
    if here in _VARIATION_SELECTORS or here in _SKIN_TONES or here == _ZERO_WIDTH_JOINER:
        return True
    # A tag character continues whatever it is tagging. The subdivision flags are spelled
    # this way — England is a black flag followed by five tag letters and a terminator — and
    # the rule below about "anything after a formatting character" does not reach the FIRST
    # of them, because what precedes it is the flag itself. So the England flag counted as
    # two characters, and a text field that deletes it with one press was sent two: the
    # second came out of whatever the user had written in front of it.
    if ord(here) in _TAGS:
        return True
    # Anything after a formatting character — the zero-width joiner among them, and the
    # prepending marks that several scripts put in front of a word. Tag characters are
    # excluded: they are formatting characters too, but a tag sequence ENDS at its
    # terminator, so this rule would have glued whatever came next onto the flag. Two flags
    # in a row counted as one character, and so did a flag followed by a letter.
    if unicodedata.category(before) == "Cf" and ord(before) not in _TAGS:
        return True
    # A virama asks for the consonant after it: Devanagari, Bengali, Tamil and the rest write
    # a conjunct that way, and it is drawn and deleted as one thing.
    if unicodedata.combining(before) == _VIRAMA:
        return True
    if _hangul_syllable_continues(here, before):
        return True
    return ord(here) in _REGIONAL and _odd_run_of_flags_before(text, at)


def _hangul_syllable_continues(here: str, before: str) -> bool:
    """Is this jamo part of the syllable the one before it started?

    Hangul written as separate jamo — a lead, a vowel, optionally a tail — is one syllable
    block on screen and one press of Backspace, however many code points it took to say.
    """
    lead, vowel, tail = ord(here) in _HANGUL_V, ord(before), ord(here) in _HANGUL_T
    if lead and (vowel in _HANGUL_L or vowel in _HANGUL_V):
        return True
    return tail and (vowel in _HANGUL_V or vowel in _HANGUL_T)


def _odd_run_of_flags_before(text: str, at: int) -> bool:
    """Is this regional indicator the SECOND half of a flag?

    Counted across the whole run, not by looking back two places. A flag is exactly two
    regional indicators, so the one at `at` completes a flag when an odd number of them sit
    immediately before it. Looking back only two was right for one flag and wrong for two
    adjacent ones: in "🇺🇸🇨🇦" the fourth indicator saw two indicators behind it, decided it
    was starting a fresh flag, and made the pair count as three characters instead of two.
    That is an over-count, and an over-count is the direction that deletes what the user
    wrote rather than what stt wrote.
    """
    run = 0
    back = at - 1
    while back >= 0 and ord(text[back]) in _REGIONAL:
        run += 1
        back -= 1
    return run % 2 == 1


def _clusters(text: str) -> int:
    """How many times Backspace has to be pressed to remove `text`."""
    return sum(1 for at in range(len(text)) if at == 0 or not _joins_the_one_before(text, at))


def _at_a_boundary(text: str, keep: int) -> int:
    """Walk `keep` back until it sits between two clusters rather than inside one."""
    while 0 < keep < len(text) and _joins_the_one_before(text, keep):
        keep -= 1
    return keep


# How much text may be posted between two ownership checks. Small enough that a click is
# never more than a few characters late, large enough that a sentence is not a hundred
# separate calls. It is not the keyboard's own burst size and does not need to match it.
_PIECE = 12


def _in_pieces(text: str) -> list[str]:
    """Split at cluster boundaries, so a piece never ends inside a character.

    A single character longer than a piece — a letter carrying a dozen combining marks, a
    long emoji sequence — goes out whole rather than being cut to fit. Cutting it produced a
    piece that was half a character, and if the user interrupted right there they were left
    looking at the half.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = _at_a_boundary(text, min(start + _PIECE, len(text)))
        if end <= start:
            end = _the_next_boundary(text, start)
        pieces.append(text[start:end])
        start = end
    return pieces


def _the_next_boundary(text: str, start: int) -> int:
    """The end of the character beginning at `start`, however long it turns out to be."""
    end = start + 1
    while end < len(text) and _joins_the_one_before(text, end):
        end += 1
    return end


@dataclass
class Typist:
    """Owns a tail of text in someone else's window, and knows when it stops owning it."""

    keys: Keyboard
    shown: str = ""
    abandoned: bool = False
    # Held for the length of ONE keystroke, and by `disown` for the length of setting a flag.
    # See `_still_ours` for what it closes and why it is safe to make the tap's thread wait.
    _keyboard: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin(self) -> None:
        """A new sentence has started being spoken. Deliberately does NOT let go of the text.

        The obvious thing to do here is clear `shown` too, and it is wrong. The previous
        sentence can still be with the accurate model when the next one starts — that is the
        normal case, not an edge — and forgetting what is on screen at that moment means the
        accurate answer gets TYPED rather than swapped in, underneath a draft of itself. What
        starts a new sentence is not this; it is `settle`, when the previous one is finished
        with. All this does is make the new sentence eligible to be owned.
        """
        self.abandoned = False

    def show(self, text: str) -> bool:
        """Make the owned tail read `text`. False if ownership was lost and nothing was done.

        Ownership is checked twice, and the second time is the one that matters. `disown` runs
        on the keyboard watcher's thread and can land in the middle of this — between the
        backspaces and the text, or between the text and the bookkeeping. The old code then
        wrote `self.shown = text` over the empty value `disown` had just set, so the typist
        went on believing it owned characters at a caret the user had since moved. The next
        sentence cleared `abandoned`, and the correction after that backspaced over whatever
        was now in front of it.

        Losing ownership mid-edit therefore leaves `shown` exactly as `disown` left it —
        empty. What has already been typed stays on screen and is simply never touched again,
        which is the same answer `disown` gives everywhere else.
        """
        # An early exit, not a guarantee: `disown` can land on the next line just as easily.
        # What makes each keystroke safe is `_still_ours`, which asks and posts as one step.
        if self.abandoned:
            return False
        edit = plan(self.shown, text)
        if edit.empty:
            return True
        # One at a time, asking again between each. A rewrite of a sentence is a couple of
        # hundred backspaces and takes a few milliseconds, and a click landing after the first
        # of them used to be answered by posting the rest anyway — deleting whatever the user
        # had just typed at their new caret. Whatever has been removed by then is stt's own
        # text; stopping mid-way leaves the line short, which is the harmless half.
        for _ in range(edit.backspaces):
            if not self._still_ours(self.keys.press_backspace, 1):
                return False
        # In small pieces, asking again between each, for the same reason as the backspaces
        # above. The replacement for a settled sentence can be a hundred characters, and the
        # keyboard posts them in bursts — a click landing after the first burst was answered
        # by posting the rest anyway, into the window the click had just focused.
        for piece in _in_pieces(edit.text):
            if not self._still_ours(self.keys.type_text, piece):
                return False
        return self._claim(text)

    def _claim(self, text: str) -> bool:
        """Record what is now on screen, unless ownership was lost while it was going out.

        A method rather than the same `with` block written inline, for the reason the old
        `_ownership_is_gone` existed: mypy narrows `self.abandoned` to False after the check
        at the top of `show` and calls every later test of it dead code. That is true of a
        value only one thread can write, and false of this one.
        """
        with self._keyboard:
            if self.abandoned:
                return False
            self.shown = text
            return True

    def _still_ours(self, post: Callable[[Any], None], what: Any) -> bool:
        """Post one keystroke, but only while the text is still ours — as ONE step.

        Asking and then posting were two steps, and a key pressed in between was answered by
        posting anyway. Rechecking before every keystroke narrowed that window to a single
        press and could not close it: `disown` runs on the event tap's thread, so it can land
        between the check and the post, and macOS then delivers the user's own key — moving
        the caret, or focusing another window — before this backspace goes out. The backspace
        then deletes a character stt did not type, which is the one thing this module exists
        to prevent.

        What the lock guarantees is that `disown` never lands BETWEEN the question and the
        answer. It does not guarantee how long the tap's thread waits: `threading.Lock` is not
        fair and permits barging, so on macOS — where CPython uses the condvar implementation
        — the loop below can reacquire the lock before a woken `disown` gets it, and the tap's
        thread then waits out the rest of the burst. That is still microseconds per event and
        far inside what a tap callback may take, and the keystrokes it waits through only ever
        remove characters stt itself typed, because they go out before the tap callback
        returns and therefore before macOS delivers the user's key at all.
        """
        with self._keyboard:
            if self.abandoned:
                return False
            post(what)
            return True

    def disown(self) -> None:
        """The user typed or clicked. What is on screen is theirs now, whoever wrote it.

        Note what this does NOT do: it does not tidy up, and it does not finish the sentence.
        A half-finished draft is left exactly as it is. Deleting it would be deleting text
        the user may have already edited, and appending the accurate version underneath
        would say the same sentence twice.

        The one window this does not close: `show` runs off the event loop, so a key pressed
        while it is halfway through an edit — after the backspaces, before the text — cannot
        stop that edit finishing. The user's character lands in the middle of ours rather
        than after them. Messy, and bounded: the backspaces only ever removed our own
        characters, and `abandoned` refuses every edit after that one.

        What it DOES close is the smaller window inside that one. Taking the keyboard lock
        means this cannot land between `show` asking whether the text is still ours and
        posting the keystroke it asked about — the case where the user's own key reaches
        macOS first and the keystroke that follows it deletes THEIR character.
        """
        with self._keyboard:
            self.shown = ""
            self.abandoned = True

    def settle(self) -> None:
        """The sentence is final. It stays on screen; it is simply no longer ours to edit.

        This, and not `begin`, is where one sentence ends and the next may be owned.
        """
        self.shown = ""
        self.abandoned = False
