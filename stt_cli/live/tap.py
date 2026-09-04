"""tap — watch the real keyboard, so live dictation knows when to stop rewriting.

THE PROBLEM THIS SOLVES
    Live dictation improves what it already typed by deleting it and typing it again. That
    is safe exactly as long as the characters to the left of the caret are still the ones it
    put there. The moment the user types a word themselves, or clicks somewhere else, those
    backspaces would eat something that was never ours.

    So the session needs to know the instant a human touches the keyboard. A listen-only
    event tap is the only way macOS offers to find that out, and it costs nothing: the tap
    observes, it never modifies or swallows an event.

TELLING OUR OWN KEYSTROKES APART
    Everything `quartz.type_text` sends comes straight back through this tap — it is a real
    system-wide keyboard event, which is the whole point of it. Each one carries the marker
    stamped into `kCGEventSourceUserData`, and this module ignores anything wearing it. A
    marker that matched by accident would make stt deaf to the user, so it is a random 63-bit
    number chosen per session rather than a constant.

WHAT THIS TAP IS NOT ALLOWED TO BECOME
    A system-wide key-down tap is a keylogger with a different name. The defence that
    actually holds is that a key code never leaves this file for anything but a comparison:
    the session is told THAT a key was pressed and, for the one hotkey it listens for, WHICH.
    Nothing here writes a key code to a log, a file, or the archive, and nothing may start
    doing so. There is no reason live dictation needs the content of what the user typed by
    hand, and every reason it must not have it.
"""

from __future__ import annotations

import ctypes
import secrets
import threading
from collections.abc import Callable

from .quartz import _Frameworks, frameworks

_SESSION_TAP = 1
_HEAD_INSERT = 0
_LISTEN_ONLY = 1
_KEY_DOWN = 10
# Every button that can move the focus, not just the left one. A right-click opens a menu
# somewhere else and an extra button is bound to whatever its owner chose; both put the caret
# in a different window, and only the left one was being watched — so a correction after one
# of those went wherever the click had landed.
_MOUSE_DOWN = (1, 3, 25)
_KEYCODE_FIELD = 9
_SOURCE_USER_DATA = 42
# A tap that stops responding is switched off by the system rather than removed. Both
# reasons are reported through the callback as a pseudo-event, and both are recoverable by
# switching it back on — a tap left disabled is silently blind for the rest of the session.
_DISABLED_BY_TIMEOUT = 0xFFFFFFFE
_DISABLED_BY_USER_INPUT = 0xFFFFFFFF

ESCAPE = 53

_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p
)


def new_marker() -> int:
    """A per-session id for our own synthetic events. Never zero: zero is the default."""
    return secrets.randbits(62) + 1


class Watcher:
    """Reports the user's own key presses and clicks on a background run loop.

    `on_key(keycode)` is called ON THE TAP'S OWN THREAD and must do its work there. It must
    NOT hand the press to the event loop: that was the first implementation, via
    `call_soon_threadsafe`, and it was wrong in a way that took a real failure to see — the
    press waited its turn in the loop's queue behind an edit that was already scheduled, so
    the edit was typed into the window the user had just clicked away from. See `arrived` in
    `dictation.py` for the whole account. The cost is that the callback runs somewhere no
    loop machinery may be touched, which is why it only ever sets flags.
    """

    def __init__(self, *, marker: int, on_key: Callable[[int], None]) -> None:
        self._marker = marker
        self._on_key = on_key
        self._loop: int | None = None
        self._ready = threading.Event()
        self._failed: str | None = None
        self._thread: threading.Thread | None = None
        self._trampoline = _CALLBACK(self._handle)
        self._port: int | None = None
        # Set when the caller has already given up on this watcher; checked by `_run` before
        # it commits the thread to a run loop.
        self._unwanted = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stt-mic-tap", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0) and self._failed is None:
            # Not reaching the end of `_run` is as much a failure as being refused by it, and
            # it used to look like success: `failure` stayed None, the caller carried on, and
            # dictation began with nothing watching for the user's own keystrokes. Synthetic
            # corrections would then delete text stt had not typed, which is the one thing
            # the whole design promises never to do.
            self._failed = "the event tap did not start within five seconds"
            # And told to stop. The thread is only late, not dead: it can reach
            # `CFRunLoopRun` a moment after this and then watch every key the user presses
            # for the rest of the process's life, with nobody holding a reference to switch
            # it off. A system-wide watcher nothing owns is the last thing to leave behind.
            self._unwanted = True

    @property
    def failure(self) -> str | None:
        """Why the tap is not watching, or None. Set before `start` returns."""
        return self._failed

    def stop(self) -> None:
        api = frameworks()
        if self._loop:
            api.core.CFRunLoopStop(ctypes.c_void_p(self._loop))
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _handle(self, proxy: int, kind: int, event: int, _info: int) -> int:
        """The tap callback. Listen-only, so the event is always returned untouched."""
        del proxy
        if kind in (_DISABLED_BY_TIMEOUT, _DISABLED_BY_USER_INPUT):
            if self._port:
                frameworks().services.CGEventTapEnable(ctypes.c_void_p(self._port), True)
            return event
        api = frameworks()
        stamp = api.services.CGEventGetIntegerValueField(ctypes.c_void_p(event), _SOURCE_USER_DATA)
        if stamp != self._marker:
            code = api.services.CGEventGetIntegerValueField(ctypes.c_void_p(event), _KEYCODE_FIELD)
            self._on_key(int(code) if kind == _KEY_DOWN else -1)
        return event

    def _run(self) -> None:
        api = _declare(frameworks())
        mask = 1 << _KEY_DOWN
        for button in _MOUSE_DOWN:
            mask |= 1 << button
        port = api.services.CGEventTapCreate(
            _SESSION_TAP, _HEAD_INSERT, _LISTEN_ONLY, mask, self._trampoline, None
        )
        if not port:
            self._failed = "the event tap was refused (Accessibility not granted)"
            self._ready.set()
            return
        self._port = port
        source = api.core.CFMachPortCreateRunLoopSource(None, ctypes.c_void_p(port), 0)
        if not source:
            # Nothing here can survive a null source: handing it to `CFRunLoopAddSource` is a
            # crash inside CoreFoundation, in a thread with no traceback to show for it.
            self._failed = "the event tap could not be attached to a run loop"
            api.core.CFRelease(ctypes.c_void_p(port))
            self._port = None
            self._ready.set()
            return
        loop = api.core.CFRunLoopGetCurrent()
        modes = ctypes.c_void_p.in_dll(api.core, "kCFRunLoopCommonModes")
        api.core.CFRunLoopAddSource(ctypes.c_void_p(loop), ctypes.c_void_p(source), modes)
        api.services.CGEventTapEnable(ctypes.c_void_p(port), True)
        self._loop = loop
        self._ready.set()
        if self._unwanted:
            # Late, and no longer wanted. Enter no run loop; unwind instead.
            api.core.CFRelease(ctypes.c_void_p(source))
            api.core.CFRelease(ctypes.c_void_p(port))
            self._port = None
            return
        api.core.CFRunLoopRun()
        api.core.CFRelease(ctypes.c_void_p(source))
        api.core.CFRelease(ctypes.c_void_p(port))


def _declare(api: _Frameworks) -> _Frameworks:
    """Argument and return types for the calls only the tap makes."""
    services, core = api.services, api.core
    services.CGEventTapCreate.restype = ctypes.c_void_p
    services.CGEventTapCreate.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint64,
        _CALLBACK,
        ctypes.c_void_p,
    ]
    services.CGEventTapEnable.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    services.CGEventGetIntegerValueField.restype = ctypes.c_int64
    services.CGEventGetIntegerValueField.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    core.CFMachPortCreateRunLoopSource.restype = ctypes.c_void_p
    core.CFMachPortCreateRunLoopSource.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_long,
    ]
    core.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    core.CFRunLoopAddSource.argtypes = [ctypes.c_void_p] * 3
    core.CFRunLoopStop.argtypes = [ctypes.c_void_p]
    return api
