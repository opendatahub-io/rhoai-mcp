"""Tests for Quay release discovery and semantic version filtering."""

import json
import runpy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "list_rhoai_releases.py"
MODULE = runpy.run_path(str(SCRIPT))
select_releases = MODULE["select_releases"]
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
    assert select_releases(branches) == [
        "rhoai-3.5-ea.2",
        "rhoai-3.5",
        "rhoai-3.6-ea.1",
        "rhoai-3.6-ea.2",
        "rhoai-3.6",
    ]


def test_future_versions_and_numeric_ordering() -> None:
    versions = ["4.0", "3.10", "3.6.1", "3.6", "3.6-rc.1", "3.6-ea.10", "3.6-ea.2"]
    assert select_releases([f"rhoai-{v}" for v in versions]) == [
        f"rhoai-{v}" for v in reversed(versions)
    ]


def test_build_and_architecture_tags_are_excluded() -> None:
    release = "rhoai-3.7-ea.1"
    tags = [
        release,
        release,
        "rhoai-3.7",
        "rhoai-3.7-054a5c7a37e2d699ef6ded32885c909f7795747c",
        "rhoai-3.7-deadbee",
        "rhoai-3.7-1234567",
        *[
            f"{release}{suffix}"
            for suffix in [
                "-054a5c7a37e2d699ef6ded32885c909f7795747c",
                ".git",
                ".prefetch",
                "-linux-m2xlarge-arm64",
                "-linux-x86-64",
            ]
        ],
        "rhoai-3.7.git",
        "rhoai-3.7.prefetch",
    ]
    assert select_releases(tags) == [release, "rhoai-3.7"]


def test_latest_release_uses_semver_instead_of_input_order() -> None:
    assert (
        select_releases(["rhoai-3.10-ea.1", "rhoai-3.7", "rhoai-3.6", "rhoai-3.10-beta.2"])[-1]
        == "rhoai-3.10-ea.1"
    )


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
    assert select_releases([f"rhoai-{version}"]) == []


def test_cli_reads_oras_tags_from_stdin(capsys) -> None:
    with patch("sys.stdin", StringIO("main\nrhoai-3.6\nrhoai-3.6-ea.1\nrhoai-3.6-ea.2\n")):
        assert main([]) == 0
    assert json.loads(capsys.readouterr().out) == ["rhoai-3.6-ea.2", "rhoai-3.6"]


@pytest.mark.parametrize(
    ("minimum", "expected"),
    [
        ("3.6", ["rhoai-3.6", "rhoai-3.7"]),
        ("3.6-ea.1", ["rhoai-3.6-ea.1", "rhoai-3.6-ea.2", "rhoai-3.6", "rhoai-3.7"]),
    ],
)
def test_custom_minimum(capsys, minimum: str, expected: list[str]) -> None:
    with patch("sys.stdin", StringIO("rhoai-3.6-ea.1\nrhoai-3.6-ea.2\nrhoai-3.6\nrhoai-3.7\n")):
        assert main(["--min-version", minimum]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_no_matching_releases(capsys) -> None:
    with patch("sys.stdin", StringIO("main\nrhoai-3.6.git\n")):
        assert main([]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_empty_tag_input(capsys) -> None:
    with patch("sys.stdin", StringIO("")):
        assert main([]) == 0
    assert capsys.readouterr().out == "[]\n"


def test_invalid_minimum_fails_before_reading_stdin(capsys) -> None:
    with patch("sys.stdin") as stdin, pytest.raises(SystemExit) as exc:
        main(["--min-version", "invalid"])
    assert exc.value.code == 2
    stdin.read.assert_not_called()
    assert "Invalid semantic version" in capsys.readouterr().err
