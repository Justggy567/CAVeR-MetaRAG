#!/usr/bin/env python3
"""Build hashes for immutable package inputs and code.

Generated verification reports are excluded so that running the verifier does not
invalidate the manifest it just checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "SHA256SUMS.txt"
GENERATED = {
    ROOT / "reports/verification_results.json",
    ROOT / "audit/replication140/rq3_replication140_human_audit.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and path not in GENERATED
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    )
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
