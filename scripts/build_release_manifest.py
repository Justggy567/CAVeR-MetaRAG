#!/usr/bin/env python3
"""Build or verify the final release-wide SHA-256 manifest.

Run this only after every intended release file is final. The manifest excludes
itself, checkout metadata and offline reproduction outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FILE_MANIFEST_SHA256.json"
PROHIBITED_PARTS = {".vscode", "__pycache__"}
EXCLUDED_PREFIXES = ((".git",), ("artifacts", "reproduced"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if any(relative.parts[:len(prefix)] == prefix for prefix in EXCLUDED_PREFIXES):
            continue
        if any(part in PROHIBITED_PARTS for part in relative.parts) or path.suffix == ".pyc":
            violations.append(relative.as_posix())
            continue
        files.append(path)
    if violations:
        raise RuntimeError(
            "Release contains prohibited cache/editor files:\n" + "\n".join(violations)
        )
    return files


def build_manifest() -> dict[str, Any]:
    rows = []
    total_bytes = 0
    for path in release_files():
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": size,
            }
        )
    return {
        "schema_version": "release-file-manifest-v1",
        "repository_root": ".",
        "hash_algorithm": "SHA-256",
        "self_excluded": OUTPUT.name,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_manifest()
    if args.check:
        if not OUTPUT.is_file():
            raise FileNotFoundError(f"Missing release manifest: {OUTPUT}")
        actual = json.loads(OUTPUT.read_text(encoding="utf-8-sig"))
        if actual != expected:
            raise ValueError("Release manifest is stale or does not match the current tree.")
        print(json.dumps({"status": "PASS", "files": expected["file_count"]}))
        return 0
    OUTPUT.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "BUILT", "files": expected["file_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
