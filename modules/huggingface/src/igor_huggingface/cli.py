from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import DatasetSelection, DocumentSelection, HuggingFaceAdapter, ImageSelection, RowSelection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igor-huggingface")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--rows", action="store_true")
    parser.add_argument("--document", action="store_true")
    parser.add_argument("--image", action="store_true")
    args = parser.parse_args(argv)
    if args.rows:
        selection = RowSelection(**json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        result = HuggingFaceAdapter().acquire_rows(selection)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"repository": result.repository, "revision": result.revision,
                          "matched_values": result.matched_value_count,
                          "rows": result.row_count, "bytes": result.source_bytes}, sort_keys=True))
        return 0
    if args.image:
        selection = ImageSelection(**json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        result = HuggingFaceAdapter().acquire_images(selection, args.output)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "acquisition.json").write_text(json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output / "judgments.json").write_text(json.dumps({"judgments": list(result.judgments), "sealed_from_runtime": True}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"repository": result.repository, "revision": result.revision, "images": len(result.records), "bytes": result.file_bytes}, sort_keys=True))
        return 0
    if args.document:
        selection = DocumentSelection(**json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        result = HuggingFaceAdapter().acquire_document(selection, args.output)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "acquisition.json").write_text(json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output / "judgments.json").write_text(json.dumps({"judgments": list(result.judgments), "sealed_from_runtime": True}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"repository": result.repository, "revision": result.revision, "pages": result.page_count, "questions": len(result.records), "bytes": result.source_bytes}, sort_keys=True))
        return 0
    selection = DatasetSelection(**json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    result = HuggingFaceAdapter().acquire(selection)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repository": result.repository, "revision": result.revision, "rows": len(result.records), "bytes": result.file_bytes}, sort_keys=True))
    return 0
