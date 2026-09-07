"""Compare committed vs freshly-generated requirements-cpu.txt.

Tolerates version drift that stays within the SemVer ranges used by
``make generate-requirements-cpu``::

    non-0.x packages  →  same major.minor, any patch  (>=M.m,<M.m+1)
    0.x packages      →  any 0.x version              (>=0,<1)

Fails on:
  - packages added or removed
  - major version change on any package
  - minor version change on non-0.x packages
  - index-url line changed (different registry source)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PKG_RE = re.compile(r"^([a-zA-Z0-9_-]+)==(\d+)\.(\d+)(?:\.(\d+))?")
_INDEX_RE = re.compile(r"^--index-url\s+(\S+)")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class Requirements:
    packages: dict[str, Version]
    index_url: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> Requirements:
        packages: dict[str, Version] = {}
        index_url: str | None = None
        for line in path.read_text().splitlines():
            if m := _PKG_RE.match(line):
                name = m.group(1).lower().replace("_", "-")
                packages[name] = Version(
                    major=int(m.group(2)),
                    minor=int(m.group(3)),
                    patch=int(m.group(4)) if m.group(4) else 0,
                )
            elif m := _INDEX_RE.match(line):
                index_url = m.group(1)
        return cls(packages=packages, index_url=index_url)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <committed> <regenerated>", file=sys.stderr)
        return 2

    old = Requirements.from_file(Path(sys.argv[1]))
    new = Requirements.from_file(Path(sys.argv[2]))

    errors: list[str] = []

    if old.index_url != new.index_url:
        errors.append(f"index-url changed: {old.index_url!r} → {new.index_url!r}")

    added = sorted(new.packages.keys() - old.packages.keys())
    removed = sorted(old.packages.keys() - new.packages.keys())
    if added:
        errors.append(f"packages added: {', '.join(added)}")
    if removed:
        errors.append(f"packages removed: {', '.join(removed)}")

    for name in sorted(old.packages.keys() & new.packages.keys()):
        old_ver = old.packages[name]
        new_ver = new.packages[name]
        if old_ver == new_ver:
            continue

        if old_ver.major != new_ver.major:
            errors.append(f"{name}: major version change {old_ver} → {new_ver}")
        elif old_ver.major == 0:
            print(f"  {name}: 0.x drift {old_ver} → {new_ver}", file=sys.stderr)
        elif old_ver.minor != new_ver.minor:
            errors.append(f"{name}: minor version change {old_ver} → {new_ver}")
        else:
            print(f"  {name}: patch drift {old_ver} → {new_ver}", file=sys.stderr)

    if errors:
        print("requirements-cpu.txt drift exceeds tolerance:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    print("requirements-cpu.txt: all differences within SemVer tolerance", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
