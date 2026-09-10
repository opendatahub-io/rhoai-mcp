"""Behavioral tests for the CI requirements drift checker.

The script under test lives at `.github/scripts/check_requirements_drift.py`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "check_requirements_drift.py"

BASE = """\
--index-url https://pypi.org/simple
requests==2.31.0
protobuf==4.25.3
semver==0.10.1
"""


def _run(committed: str, regenerated: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    committed_path = tmp_path / "committed.txt"
    regenerated_path = tmp_path / "regenerated.txt"
    committed_path.write_text(committed)
    regenerated_path.write_text(regenerated)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(committed_path), str(regenerated_path)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestDriftTolerance:
    def test_identical_files_pass(self, tmp_path: Path) -> None:
        assert _run(BASE, BASE, tmp_path).returncode == 0

    def test_patch_drift_passes(self, tmp_path: Path) -> None:
        regenerated = BASE.replace("requests==2.31.0", "requests==2.31.9")
        assert _run(BASE, regenerated, tmp_path).returncode == 0

    def test_zero_x_any_drift_passes(self, tmp_path: Path) -> None:
        regenerated = BASE.replace("semver==0.10.1", "semver==0.11.0")
        assert _run(BASE, regenerated, tmp_path).returncode == 0


@pytest.mark.parametrize(
    ("regenerated", "expected_error"),
    [
        (BASE.replace("requests==2.31.0", "requests==3.0.0"), "major version change"),
        (BASE.replace("requests==2.31.0", "requests==2.32.0"), "minor version change"),
        (BASE + "httpx==0.27.0\n", "packages added: httpx"),
        (BASE.replace("protobuf==4.25.3\n", ""), "packages removed: protobuf"),
        (
            BASE.replace(
                "--index-url https://pypi.org/simple",
                "--index-url https://internal.registry/simple",
            ),
            "index-url changed",
        ),
    ],
)
def test_disallowed_drift_fails(regenerated: str, expected_error: str, tmp_path: Path) -> None:
    result = _run(BASE, regenerated, tmp_path)
    assert result.returncode == 1
    assert expected_error in result.stderr


def test_wrong_argument_count_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
