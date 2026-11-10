"""Tests for RHDS release branch discovery and version filtering."""

import json
import runpy
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "list_rhds_release_branches.py"
with patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    MODULE = runpy.run_path(str(SCRIPT))
select_branches = MODULE["select_branches"]
main = MODULE["main"]


def test_only_release_branches_from_3_6_ea_2_onward() -> None:
    branches = [
        "main",
        "rhoai-3.5",
        "rhoai-3.5-ea.2",
        "rhoai-3.6",
        "rhoai-3.6-ea.1",
        "rhoai-3.6-ea.2",
        "rhoai-3.7",
        "rhoai-3.7-ea.1",
        "rhoai-3.10",
        "rhoai-03.6",
        "rhoai-3.6.1",
        "other-rhoai-4.0",
    ]
    assert select_branches(branches) == [
        "rhoai-3.6-ea.2",
        "rhoai-3.6",
        "rhoai-3.6.1",
        "rhoai-3.7-ea.1",
        "rhoai-3.7",
        "rhoai-3.10",
    ]


def test_branches_are_unique_and_numerically_sorted() -> None:
    assert select_branches(["rhoai-3.10", "rhoai-4.0", "rhoai-3.10", "rhoai-3.9"]) == [
        "rhoai-3.9",
        "rhoai-3.10",
        "rhoai-4.0",
    ]


def test_cli_reads_branch_names_from_stdin(capsys) -> None:
    with patch("sys.stdin", StringIO("main\nrhoai-3.5\nrhoai-3.6-ea.2\nrhoai-3.6\n")):
        assert main(["--required-releases", "[]"]) == 0
    assert json.loads(capsys.readouterr().out) == ["rhoai-3.6-ea.2", "rhoai-3.6"]


def test_no_matching_branches(capsys) -> None:
    with patch("sys.stdin", StringIO("main\nrhoai-3.5\nrhoai-3.6-ea.1\n")):
        assert main(["--required-releases", "[]"]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_required_releases_argument_is_required(capsys) -> None:
    with patch("sys.stdin") as stdin, pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    stdin.read.assert_not_called()
    assert "--required-releases" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("not-json", "Invalid required releases"),
        ('"rhoai-3.6"', "expected a JSON array of strings"),
        ('["rhoai-3.6", 3]', "expected a JSON array of strings"),
    ],
)
def test_invalid_required_releases(value, error, capsys) -> None:
    with patch("sys.stdin") as stdin, pytest.raises(SystemExit) as exc:
        main(["--required-releases", value])
    assert exc.value.code == 2
    stdin.read.assert_not_called()
    assert error in capsys.readouterr().err


def test_required_releases_present(capsys) -> None:
    with patch("sys.stdin", StringIO("rhoai-3.6-ea.2\nrhoai-3.6\nrhoai-3.7\n")):
        assert main(["--required-releases", '["rhoai-3.6-ea.2", "rhoai-3.6"]']) == 0
    assert json.loads(capsys.readouterr().out) == ["rhoai-3.6-ea.2", "rhoai-3.6", "rhoai-3.7"]


def test_missing_required_releases_fail(capsys) -> None:
    with patch("sys.stdin", StringIO("rhoai-3.6\n")):
        assert main(["--required-releases", '["rhoai-3.6", "rhoai-3.7", "rhoai-3.6-ea.2"]']) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Quay releases missing RHDS branches: rhoai-3.6-ea.2, rhoai-3.7" in captured.err
