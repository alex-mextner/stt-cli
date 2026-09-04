"""stt-cli — speech-to-text for any audio or video file on macOS.

The package is import-clean: nothing here (or in the dispatcher) pulls a heavy or
optional dependency, so ``stt --help`` stays instant and works on a machine where no
engine is installed yet.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
