"""Build the ASQA top-5 candidate pool used by the cleaning pipeline.

The main output intentionally preserves the schema of the previous top-3 file:

    question, docs_context, standard_answer, id

By default the script downloads the pinned ``valid`` split of
``Self-GRIT/asqa_eval``.  For an offline/reproducibility run, the same raw
top-100 JSON can be supplied with ``--input-json``.

Examples (run from the project root in PyCharm's terminal):

    pip install datasets
    python scripts/build_asqa_top5.py

    python provenance/asqa/build_asqa_top5.py --input-json provenance/asqa/input_source.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_ID = "Self-GRIT/asqa_eval"
DATASET_REVISION = "3900e8bc34198ac2783660c37902d40a185599bc"

# Split name used by the pinned Hugging Face repository.
DATASET_REPOSITORY_SPLIT = "valid"

# Different datasets versions/environments may expose the same source
# split as either "valid" or canonicalized "validation".
DATASET_LOAD_SPLIT_CANDIDATES = ("valid", "validation")

TOP_K = 5
EXPECTED_RECORDS = 948
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "asqa_eval_data_top5.json"

# print(PROJECT_ROOT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the full ASQA candidate pool with the first five "
            "retrieved documents as context."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Main four-field JSON output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Optional local raw ASQA top-100 JSON. If omitted, the pinned "
            "Self-GRIT dataset is downloaded from Hugging Face."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--allow-unexpected-count",
        action="store_true",
        help=(
            "Permit a source count other than 948. Do not use this for the "
            "reported experiment unless the deviation is documented."
        ),
    )
    return parser.parse_args()


def load_source_rows(
    input_json: Path | None,
    cache_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if input_json is not None:
        source_path = input_json.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Raw input JSON does not exist: {source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError("The raw ASQA JSON must contain a top-level list.")
        rows = [_as_plain_dict(row, index) for index, row in enumerate(payload)]
        source = {
            "mode": "local_json",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
        return rows, source

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for downloading the source. "
            "Install it in the PyCharm interpreter with: pip install datasets"
        ) from exc

    dataset = None
    loaded_split = None

    for split_name in DATASET_LOAD_SPLIT_CANDIDATES:
        try:
            dataset = load_dataset(
                DATASET_ID,
                split=split_name,
                revision=DATASET_REVISION,
                cache_dir=str(cache_dir.expanduser().resolve()) if cache_dir else None,
            )
            loaded_split = split_name
            break
        except ValueError:
            continue

    if dataset is None:
        raise ValueError(
            f"Could not load any expected split: {DATASET_LOAD_SPLIT_CANDIDATES}"
        )

    rows = [_as_plain_dict(row, index) for index, row in enumerate(dataset)]

    source = {
        "mode": "huggingface_dataset",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "repository_split": DATASET_REPOSITORY_SPLIT,
        "loaded_split": loaded_split,
    }
    return rows, source


def _as_plain_dict(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"Source record {index} is not an object.")
    return dict(row)


def require_nonempty_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {index} has an empty or non-string '{field}'.")
    # Preserve the source string byte-for-byte. Stripping here would introduce
    # a second preprocessing change in addition to top-3 -> top-5.
    return value


def extract_standard_answer(row: Mapping[str, Any], index: int) -> str:
    annotations = row.get("annotations")
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise ValueError(f"Record {index} has no valid 'annotations' list.")
    if not annotations or not isinstance(annotations[0], Mapping):
        raise ValueError(f"Record {index} has no first annotation object.")
    return require_nonempty_text(
        annotations[0].get("long_answer"),
        "annotations[0].long_answer",
        index,
    )


def select_top_docs(
    row: Mapping[str, Any],
    index: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    docs = row.get("docs")
    if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes)):
        raise ValueError(f"Record {index} has no valid 'docs' list.")
    if len(docs) < TOP_K:
        raise ValueError(
            f"Record {index} contains only {len(docs)} documents; "
            f"{TOP_K} are required."
        )

    texts: list[str] = []
    trace: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs[:TOP_K], start=1):
        if not isinstance(doc, Mapping):
            raise TypeError(f"Record {index}, document rank {rank} is not an object.")
        text = require_nonempty_text(doc.get("text"), f"docs[{rank - 1}].text", index)
        texts.append(text)
        trace.append(
            {
                "rank": rank,
                "doc_id": doc.get("id"),
                "title": doc.get("title"),
                "score": doc.get("score"),
                "text_sha256": sha256_text(text),
            }
        )
    return texts, trace


def build_records(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        question = require_nonempty_text(row.get("question"), "question", index)
        standard_answer = extract_standard_answer(row, index)
        doc_texts, selected_docs = select_top_docs(row, index)
        docs_context = "\n\n".join(doc_texts)

        # Keep integer IDs identical to the previous top-3 construction so that
        # old and new contexts can be compared by ID without a fuzzy join.
        record = {
            "question": question,
            "docs_context": docs_context,
            "standard_answer": standard_answer,
            "id": index,
        }
        records.append(record)
        provenance.append(
            {
                "id": index,
                "source_sample_id": row.get("sample_id"),
                "question_sha256": sha256_text(question),
                "standard_answer_sha256": sha256_text(standard_answer),
                "docs_context_sha256": sha256_text(docs_context),
                "selected_docs": selected_docs,
            }
        )

    return records, provenance


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def companion_paths(output: Path) -> tuple[Path, Path]:
    stem = output.with_suffix("")
    return (
        stem.with_name(f"{stem.name}.provenance.json"),
        stem.with_name(f"{stem.name}.manifest.json"),
    )


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    provenance_path, manifest_path = companion_paths(output)

    rows, source = load_source_rows(args.input_json, args.cache_dir)
    if len(rows) != EXPECTED_RECORDS and not args.allow_unexpected_count:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} source records, found {len(rows)}. "
            "The output was not written. Verify the source revision, or use "
            "--allow-unexpected-count only when the change is intentional and documented."
        )

    records, provenance = build_records(rows)
    atomic_write_json(output, records)
    atomic_write_json(provenance_path, provenance)

    manifest = {
        "builder": Path(__file__).name,
        "source": source,
        "construction": {
            "top_k": TOP_K,
            "document_order": "source retrieval order",
            "context_expression": "\\n\\n.join(item['docs'][:5][*]['text'])",
            "standard_answer_expression": "item['annotations'][0]['long_answer']",
            "output_id_expression": "zero-based source row index",
            "sampling": "none",
        },
        "record_count": len(records),
        "files": {
            "data": {"path": str(output), "sha256": sha256_file(output)},
            "provenance": {
                "path": str(provenance_path),
                "sha256": sha256_file(provenance_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest)

    print(f"Built {len(records)} ASQA records with fixed top-{TOP_K} contexts.")
    print(f"Data:       {output}")
    print(f"Provenance: {provenance_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Data SHA256: {manifest['files']['data']['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
