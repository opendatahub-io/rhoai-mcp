"""Tests for RHDS branch discovery and semantic version filtering."""

import json
import runpy
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "list_rhoai_branches.py"
MODULE = runpy.run_path(str(SCRIPT))
select_branches = MODULE["select_branches"]
version_key = MODULE["version_key"]
main = MODULE["main"]


def test_current_release_branches() -> None:
    branches = [
        "main",
        "rhoai-3.5",
        "rhoai-3.5-ea.2",
        "rhoai-3.6-ea.1",
        "rhoai-3.6",
        "rhoai-3.6-ea.2",
        "konflux/mintmaker/rhoai-3.6-ea.2/all",
        "rhoai-invalid",
        "other-rhoai-4.0",
    ]
    assert select_branches(branches) == ["rhoai-3.6-ea.2", "rhoai-3.6"]


def test_future_versions_and_numeric_ordering() -> None:
    versions = ["4.0", "3.10", "3.6.1", "3.6", "3.6-rc.1", "3.6-ea.10", "3.6-ea.2"]
    assert select_branches([f"rhoai-{v}" for v in versions]) == [
        f"rhoai-{v}" for v in reversed(versions)
    ]


def test_semver_prerelease_precedence() -> None:
    versions = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    for lower, higher in zip(versions, versions[1:], strict=False):
        assert version_key(lower) < version_key(higher)
    assert version_key("3.6+build.001") == version_key("3.6.0")
    assert version_key("3.6-ea.2") == version_key("3.6.0-ea.2")


@pytest.mark.parametrize(
    "version", ["3", "03.6", "3.06", "3.6.00", "3.6-ea.02", "3.6-", "3.6/feature"]
)
def test_invalid_versions(version: str) -> None:
    with pytest.raises(ValueError, match="Invalid semantic version"):
        version_key(version)
    assert select_branches([f"rhoai-{version}"]) == []


def test_cli_uses_paginated_gh_and_prints_selected_branches(capsys) -> None:
    with patch("subprocess.run") as run:
        run.return_value.stdout = "main\nrhoai-3.6\nrhoai-3.6-ea.1\nrhoai-3.6-ea.2\n"
        assert main([]) == 0
    run.assert_called_once_with(
        [
            "gh",
            "api",
            "--paginate",
            "repos/red-hat-data-services/rhoai-mcp/branches?per_page=100",
            "--jq",
            ".[].name",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(capsys.readouterr().out) == ["rhoai-3.6-ea.2", "rhoai-3.6"]


def test_custom_minimum(capsys) -> None:
    with patch("subprocess.run") as run:
        run.return_value.stdout = "rhoai-3.6-ea.2\nrhoai-3.6\nrhoai-3.7\n"
        assert main(["--min-version", "3.6"]) == 0
    assert json.loads(capsys.readouterr().out) == ["rhoai-3.6", "rhoai-3.7"]


def test_no_matching_branches(capsys) -> None:
    with patch("subprocess.run") as run:
        run.return_value.stdout = "main\nrhoai-3.5\n"
        assert main([]) == 0
    assert capsys.readouterr().out == "[]\n"


@pytest.mark.parametrize(
    "error, message",
    [
        (FileNotFoundError(), "GitHub CLI (gh) is required"),
        (
            subprocess.CalledProcessError(1, ["gh"], stderr="authentication failed"),
            "authentication failed",
        ),
    ],
)
def test_gh_errors_fail_without_branch_output(error: Exception, message: str, capsys) -> None:
    with patch("subprocess.run", side_effect=error):
        assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_invalid_minimum_fails_before_network_access(capsys) -> None:
    with patch("subprocess.run") as run, pytest.raises(SystemExit) as exc:
        main(["--min-version", "invalid"])
    assert exc.value.code == 2
    run.assert_not_called()
    assert "Invalid semantic version" in capsys.readouterr().err
