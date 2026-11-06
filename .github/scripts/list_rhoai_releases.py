"""List release tags published to the Quay image repository.

Usage: oras repo tags REPOSITORY | python .github/scripts/list_rhoai_releases.py

Reads one tag per line from standard input, as emitted by oras repo tags.
Prints a JSON array of release tags, oldest version first. Missing patch
numbers are treated as zero; otherwise SemVer precedence applies, including
prerelease identifiers. Only final releases and ea/alpha/beta/rc prereleases
with an optional numeric identifier qualify; build and architecture tags are
ignored. By default, releases start at rhoai-3.6-ea.2, when Technology Preview
(TP) began; --min-version overrides this minimum. An empty result succeeds
and prints [].
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable

PREFIX = "rhoai-"

_NUMBER = r"(?:0|[1-9][0-9]*)"
_IDENTIFIERS = r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*"
_VERSION = re.compile(
    rf"({_NUMBER})\.({_NUMBER})(?:\.({_NUMBER}))?"
    rf"(?:-({_IDENTIFIERS}))?(?:\+({_IDENTIFIERS}))?"
)
_RELEASE = re.compile(
    rf"{_NUMBER}\.{_NUMBER}(?:\.{_NUMBER})?"
    rf"(?:-(?:ea|alpha|beta|rc)(?:\.{_NUMBER})?)?"
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


def select_releases(tags: Iterable[str], min_version: str | None = None) -> list[str]:
    """Exclude build tags and sort release tags by semantic precedence."""
    minimum = version_key(min_version) if min_version is not None else None
    selected = []
    for tag in set(tags):
        if not tag.startswith(PREFIX):
            continue
        version = tag.removeprefix(PREFIX)
        if _RELEASE.fullmatch(version) is None:
            continue
        key = version_key(version)
        if minimum is None or key >= minimum:
            selected.append((key, tag))
    return [tag for _, tag in sorted(selected)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-version",
        default="3.6-ea.2",
        help=(
            "Minimum version without the rhoai- prefix (default: %(default)s, "
            "when Technology Preview (TP) began); specify a version to override this minimum"
        ),
    )
    args = parser.parse_args(argv)
    try:
        version_key(args.min_version)
    except ValueError as exc:
        parser.error(str(exc))

    releases = select_releases(sys.stdin.read().splitlines(), args.min_version)
    print(json.dumps(releases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
