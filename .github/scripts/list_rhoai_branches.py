"""List RHDS rhoai- branches at or above a minimum semantic version.

Usage: python .github/scripts/list_rhoai_branches.py [--min-version 3.6-ea.2]

Requires an authenticated GitHub CLI (set GH_TOKEN in GitHub Actions).
Prints a JSON array of full branch names, oldest version first. Missing patch
numbers are treated as zero; otherwise SemVer precedence applies, including
prerelease identifiers and ignoring build metadata. Non-version branches are
ignored. An empty result succeeds and prints [].
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable

REPOSITORY = "red-hat-data-services/rhoai-mcp"
PREFIX = "rhoai-"
DEFAULT_MIN_VERSION = "3.6-ea.2"

_NUMBER = r"(?:0|[1-9][0-9]*)"
_IDENTIFIERS = r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*"
_VERSION = re.compile(
    rf"({_NUMBER})\.({_NUMBER})(?:\.({_NUMBER}))?"
    rf"(?:-({_IDENTIFIERS}))?(?:\+({_IDENTIFIERS}))?"
)
# Compared lexicographically as (major, minor, patch, is_release, prerelease).
# An omitted patch is zero. is_release is True for a final release, so it sorts
# after prereleases of the same major.minor.patch (False < True).
# Each prerelease identifier is (0, int_value) for numeric identifiers or
# (1, str_value) for text: numbers sort numerically and before text, while text
# sorts lexically. The tag also prevents comparing int and str values directly.
# Equal identifier prefixes sort shorter first, e.g. ea < ea.2. Build metadata
# is excluded because it does not affect SemVer precedence.
# Example: 3.6-ea.2 -> (3, 6, 0, False, ((1, "ea"), (0, 2)))
#          3.6      -> (3, 6, 0, True, ())
VersionKey = tuple[int, int, int, bool, tuple[tuple[int, int | str], ...]]


def version_key(version: str) -> VersionKey:
    """Build a SemVer ordering key, allowing the branch shorthand major.minor."""
    match = _VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"Invalid semantic version: {version!r}")
    major, minor, patch, prerelease, _build = match.groups()
    identifiers: list[tuple[int, int | str]] = []
    if prerelease is not None:
        for identifier in prerelease.split("."):
            if identifier.isdigit():
                if len(identifier) > 1 and identifier.startswith("0"):
                    raise ValueError(f"Invalid semantic version: {version!r}")
                identifiers.append((0, int(identifier)))
            else:
                identifiers.append((1, identifier))
    return int(major), int(minor), int(patch or 0), prerelease is None, tuple(identifiers)


def select_branches(branches: Iterable[str], min_version: str = DEFAULT_MIN_VERSION) -> list[str]:
    """Filter exact-prefix release branches and sort by semantic precedence."""
    minimum = version_key(min_version)
    selected = []
    for branch in branches:
        if not branch.startswith(PREFIX):
            continue
        try:
            key = version_key(branch.removeprefix(PREFIX))
        except ValueError:
            continue
        if key >= minimum:
            selected.append((key, branch))
    return [branch for _, branch in sorted(selected)]


def list_branches() -> list[str]:
    """Read all pages of branch names without cloning or checking out the fork."""
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{REPOSITORY}/branches?per_page=100",
            "--jq",
            ".[].name",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-version", default=DEFAULT_MIN_VERSION)
    args = parser.parse_args(argv)
    try:
        version_key(args.min_version)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        branches = select_branches(list_branches(), args.min_version)
    except FileNotFoundError:
        print("GitHub CLI (gh) is required but was not found.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Failed to list branches in {REPOSITORY}: {exc.stderr.strip()}", file=sys.stderr)
        return 1

    print(json.dumps(branches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
