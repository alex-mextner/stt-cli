"""Every command must at least RUN — the check that would have caught a name collision.

`backends.whispercpp` shadowed the submodule of the same name and blew up with
`'module' object is not callable` the first time `stt doctor` was invoked, while both mypy
and the whole unit suite stayed green. These smoke tests exercise each command's real entry
point so an import-time or attribute-time mistake cannot reach a user again.

They must not need a model, a GPU or a network: `doctor` reports what is missing rather than
requiring it, and the rest work on an empty archive.
"""

from __future__ import annotations

import pytest

from stt_cli._errors import EXIT_MISSING_DEP, EXIT_OK, EXIT_PERMISSION
from stt_cli.cli import _discover, main


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "home"))


def test_every_command_module_exposes_the_contract() -> None:
    catalog = _discover()
    assert {"transcribe", "doctor", "archive", "models", "config", "setup"} <= set(catalog)


@pytest.mark.parametrize(
    "argv",
    [
        ["doctor"],
        ["models"],
        ["models", "ls"],
        ["models", "where"],
        ["archive"],
        ["archive", "ls"],
        ["archive", "usage"],
        ["archive", "gc"],
        ["config"],
        ["config", "list"],
        ["config", "path"],
        ["diarize"],
        ["diarize", "status"],
    ],
)
def test_read_only_commands_run(argv: list[str]) -> None:
    """A missing optional dependency is a reported exit code, never a traceback.

    Three codes, not two, and which one comes back is a property of the MACHINE rather than
    of the command: `stt diarize status` answers "not ready" where the wheels are absent and
    "ready, but no token" where they are cached, and both are correct reports of a real
    state. The claim being made here is that none of them is an exception.
    """
    # The third code is allowed for diarization ALONE. Widening it for every command would
    # have made this quietly accept "you are not signed in" from `stt config list`, which
    # would be a defect rather than a state.
    allowed = {EXIT_OK, EXIT_MISSING_DEP}
    if argv[0] == "diarize":
        allowed.add(EXIT_PERMISSION)
    assert main(argv) in allowed


@pytest.mark.parametrize(
    "argv", [["transcribe", "--help"], ["archive", "--help"], ["doctor", "--help"]]
)
def test_help_for_each_command(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 0


def test_config_round_trip_through_the_cli(capsys) -> None:
    assert main(["config", "set", "language", "ru"]) == EXIT_OK
    capsys.readouterr()
    assert main(["config", "get", "language"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == "ru"
