"""commands — one module per subcommand, discovered automatically.

A command module declares ``NAME`` (what the user types), ``SUMMARY`` (the one line shown
in ``stt --help``) and ``run(argv) -> int``. The dispatcher finds it by scanning this
package, so a new command is a new file and nothing else.
"""

from __future__ import annotations
