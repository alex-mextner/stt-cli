"""quartz — the three macOS calls live dictation needs, through ``ctypes``.

WHY NOT PYOBJC
    Typing into whatever window has focus is a Quartz call, and the obvious way to reach it
    is `pyobjc`, which is 40 MB of bridge and a build step. This module reaches the same
    three C functions through `ctypes` against the framework the system already ships, so
    `stt mic` stays inside the rule the rest of the tool keeps: nothing is required at
    install time that the machine does not already have.

WHAT THE SYSTEM STILL DEMANDS
    Accessibility. Synthesizing a keystroke and watching for one are both privileged, and
    macOS grants them per application — the terminal running `stt`, not `stt` itself. There
    is no way to ask for it from inside a process; the checkbox is the user's to tick. So
    the only honest thing to do is detect the state and say precisely which application to
    grant it to, which is what :func:`trusted` and its caller do.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

from .._errors import PermissionDeniedError

# Where a synthesized event enters the system. The HID tap is the lowest of the three, so
# the event reaches every observer above it exactly as a real key press would.
_HID_TAP = 0
# `kCGEventSourceUserData`: 64 bits of our own on every event, which is how the watcher in
# `tap.py` tells a key stt typed from a key the user pressed. Without it, every character
# this module sends would come back through the tap as "the user is typing" and the session
# would let go of the text it had just written.
_SOURCE_USER_DATA = 42
# Virtual key code for Backspace (`kVK_Delete`). Rewriting a draft is backspaces then text.
_BACKSPACE = 51
# `CGEventKeyboardSetUnicodeString` is documented for short strings, and long ones are
# unreliable rather than refused. Sending a paragraph in one event dropped characters
# silently; in chunks of this size it does not.
_CHUNK = 16


class _Frameworks:
    """The framework handles and prototypes, loaded once and only when first needed."""

    def __init__(self) -> None:
        self.core = _load("CoreFoundation")
        self.services = _load("ApplicationServices")
        self._declare()

    def _declare(self) -> None:
        services, core = self.services, self.core
        services.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        services.CGEventCreateKeyboardEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_bool,
        ]
        services.CGEventKeyboardSetUnicodeString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_uint16),
        ]
        services.CGEventSetIntegerValueField.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int64,
        ]
        services.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        services.AXIsProcessTrusted.restype = ctypes.c_bool
        core.CFRelease.argtypes = [ctypes.c_void_p]


def _load(name: str) -> Any:
    found = ctypes.util.find_library(name)
    if found is None:  # pragma: no cover - only on a machine that is not macOS
        raise OSError(f"{name} framework not found")
    return ctypes.CDLL(found)


_LOADED: _Frameworks | None = None


def frameworks() -> _Frameworks:
    global _LOADED
    if _LOADED is None:
        _LOADED = _Frameworks()
    return _LOADED


def trusted() -> bool:
    """Has the user granted Accessibility to the application running stt?

    False, rather than an exception, when the frameworks are not there at all. `stt doctor`
    asks this among a dozen other questions, and a machine that is not macOS should get a
    "no" on the line about typing into windows rather than a crash on the whole report.
    """
    try:
        return bool(frameworks().services.AXIsProcessTrusted())
    except OSError:
        return False


ACCESSIBILITY_HINT = (
    "grant Accessibility to the terminal application running stt in System Settings > "
    "Privacy & Security > Accessibility, then start a new terminal"
)


def require_accessibility() -> None:
    """Refuse to start rather than typing into a window and having nothing appear."""
    if not trusted():
        raise PermissionDeniedError(
            what="stt cannot type into other windows",
            why="the application running stt has not been granted Accessibility",
            how=ACCESSIBILITY_HINT,
        )


def type_text(text: str, *, marker: int) -> None:
    """Type `text` into whatever has focus, stamped so our own tap can recognise it.

    A WARNING FOR WHOEVER MEASURES THIS NEXT
        The marker is for the tap in `tap.py` and nothing else. It does not make an event
        private, quiet or reversible: every character here goes into the frontmost window of
        whoever is sitting at the machine, exactly as if they had typed it. A benchmark of
        "how long does posting three hundred backspaces take" is three hundred backspaces
        into whatever they were reading. Point a throwaway TextEdit document at the keyboard
        before calling this outside of `stt mic` — `tests/test_live.py` never calls it at all,
        which is why the whole rewrite algorithm lives in `typist.py` behind a fake keyboard.
    """
    for chunk in _in_chunks(text):
        units = chunk.encode("utf-16-le")
        buffer = (ctypes.c_uint16 * (len(units) // 2)).from_buffer_copy(units)
        _post(keycode=0, marker=marker, unicode=(buffer, len(buffer)))


def press_backspace(times: int, *, marker: int) -> None:
    """Delete `times` characters to the left of the caret."""
    for _ in range(max(0, times)):
        _post(keycode=_BACKSPACE, marker=marker, unicode=None)


def _in_chunks(text: str) -> list[str]:
    """Split on UTF-16 code units, never inside a surrogate pair.

    The cut has to be made in code units because that is what the API counts, but an emoji
    is two of them and half an emoji does not decode. Landing on a high surrogate therefore
    moves the cut one unit along rather than dropping the character.
    """
    units = text.encode("utf-16-le")
    pieces: list[str] = []
    start = 0
    while start < len(units):
        end = min(start + _CHUNK * 2, len(units))
        if end < len(units) and _is_a_high_surrogate(units, end - 2):
            end += 2
        pieces.append(units[start:end].decode("utf-16-le"))
        start = end
    return pieces


def _is_a_high_surrogate(units: bytes, at: int) -> bool:
    return 0xD800 <= int.from_bytes(units[at : at + 2], "little") <= 0xDBFF


def _post(*, keycode: int, marker: int, unicode: tuple[Any, int] | None) -> None:
    """One key down and its matching key up, both carrying our marker."""
    api = frameworks()
    for down in (True, False):
        event = api.services.CGEventCreateKeyboardEvent(None, keycode, down)
        if not event:  # pragma: no cover - allocation failure
            return
        try:
            if unicode is not None:
                buffer, length = unicode
                api.services.CGEventKeyboardSetUnicodeString(event, length, buffer)
            api.services.CGEventSetIntegerValueField(event, _SOURCE_USER_DATA, marker)
            api.services.CGEventPost(_HID_TAP, event)
        finally:
            api.core.CFRelease(event)
