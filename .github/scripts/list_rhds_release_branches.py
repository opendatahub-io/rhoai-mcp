"""Print RHDS rhoai release branches from 3.6-ea.2 onward as a JSON array."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from list_rhoai_releases import DEFAULT_MIN_VERSION, select_releases


def select_branches(names: Iterable[str]) -> list[str]:
    """Select release branches using the same minimum as Quay releases."""
    return select_releases(names, DEFAULT_MIN_VERSION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--required-releases", required=True, help="JSON array of required release tags"
    )
    args = parser.parse_args(argv)
    try:
        required = json.loads(args.required_releases)
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("expected a JSON array of strings")
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Invalid required releases: {exc}")

    branches = select_branches(sys.stdin.read().splitlines())
    missing = set(required) - set(branches)
    if missing:
        print(
            f"::error::Quay releases missing RHDS branches: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(branches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
