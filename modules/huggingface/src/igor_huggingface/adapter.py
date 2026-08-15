from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping
from pathlib import Path
from urllib.parse import quote


class AcquisitionError(ValueError):
    pass


def _row_windows(row_numbers: tuple[int, ...], *, max_length: int = 100) -> tuple[tuple[int, int], ...]:
    if not row_numbers:
        return ()
    windows: list[tuple[int, int]] = []
    start = row_numbers[0]
    end = row_numbers[0]
    for row_number in row_numbers[1:]:
        length = row_number - start + 1
        if length <= max_length:
            end = row_number
            continue
        windows.append((start, end - start + 1))
        start = row_number
        end = row_number
    windows.append((start, end - start + 1))
    return tuple(windows)


@dataclass(frozen=True)
class DatasetSelection:
    repository: str
    revision: str
    configuration: str
    split: str
    file: str
    row_numbers: tuple[int, ...]
    fields: tuple[str, ...]
    license: str
    expected_file_sha256: str | None = None
    expected_row_sha256: tuple[str, ...] = ()
    max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_numbers", tuple(self.row_numbers))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "expected_row_sha256", tuple(self.expected_row_sha256))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise AcquisitionError("revision must be an immutable 40-character commit SHA")
        if not self.repository or self.repository.count("/") != 1:
            raise AcquisitionError("repository must be owner/name")
        if not self.file or self.file.startswith("/") or ".." in self.file.split("/"):
            raise AcquisitionError("file must be a repository-relative path")
        if not self.row_numbers or any(row < 0 for row in self.row_numbers):
            raise AcquisitionError("row_numbers must be non-empty and zero-based")
        if tuple(sorted(set(self.row_numbers))) != self.row_numbers:
            raise AcquisitionError("row_numbers must be unique and deterministically ordered")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise AcquisitionError("fields must be non-empty and unique")
        if not self.license.strip() or self.max_bytes <= 0:
            raise AcquisitionError("license and max_bytes are required")
        if self.expected_row_sha256 and len(self.expected_row_sha256) != len(self.row_numbers):
            raise AcquisitionError("expected row hashes must match selected rows")


@dataclass(frozen=True)
class AcquisitionResult:
    repository: str
    revision: str
    configuration: str
    split: str
    file: str
    file_sha256: str
    file_bytes: int
    row_numbers: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    row_sha256: tuple[str, ...]
    license: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository, "revision": self.revision,
            "configuration": self.configuration, "split": self.split,
            "file": self.file, "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes, "row_numbers": list(self.row_numbers),
            "row_sha256": list(self.row_sha256), "license": self.license,
            "records": list(self.records),
        }


@dataclass(frozen=True)
class ImageSelection:
    """A bounded, immutable selection from an image archive."""
    repository: str
    revision: str
    configuration: str
    split: str
    file: str
    members: tuple[str, ...]
    license: str
    expected_file_sha256: str | None = None
    max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise AcquisitionError("image revision must be an immutable 40-character commit SHA")
        if not self.repository or self.repository.count("/") != 1:
            raise AcquisitionError("repository must be owner/name")
        if not self.file or self.file.startswith("/") or ".." in self.file.split("/"):
            raise AcquisitionError("image archive must be repository-relative")
        if not self.members or len(self.members) > 12 or len(set(self.members)) != len(self.members):
            raise AcquisitionError("image selection must contain 1..12 unique members")
        if tuple(sorted(self.members)) != self.members:
            raise AcquisitionError("image members must be deterministically ordered")
        if not self.license.strip() or self.max_bytes <= 0:
            raise AcquisitionError("license and max_bytes are required")


@dataclass(frozen=True)
class ImageAcquisitionResult:
    repository: str
    revision: str
    configuration: str
    split: str
    file: str
    file_sha256: str
    file_bytes: int
    records: tuple[dict[str, Any], ...]
    judgments: tuple[dict[str, Any], ...]
    license: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository, "revision": self.revision,
            "configuration": self.configuration, "split": self.split,
            "file": self.file, "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes, "records": list(self.records),
            "license": self.license,
        }


@dataclass(frozen=True)
class DocumentSelection:
    """Bounded DocVQA selection; evaluator answers are never source records."""
    repository: str
    revision: str
    config: str
    split: str
    row_numbers: tuple[int, ...]
    expected_license: str = "mit"
    max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise AcquisitionError("document revision must be an immutable commit SHA")
        if len(self.row_numbers) == 0 or len(self.row_numbers) > 32:
            raise AcquisitionError("document selection must contain 1..32 questions")
        if tuple(sorted(set(self.row_numbers))) != tuple(self.row_numbers):
            raise AcquisitionError("document row numbers must be unique and ordered")
        if self.max_bytes <= 0:
            raise AcquisitionError("document transfer ceiling must be positive")


@dataclass(frozen=True)
class DocumentAcquisitionResult:
    repository: str
    revision: str
    config: str
    split: str
    records: tuple[dict[str, Any], ...]
    judgments: tuple[dict[str, Any], ...]
    source_bytes: int
    page_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"repository": self.repository, "revision": self.revision,
                "config": self.config, "split": self.split,
                "records": list(self.records), "source_bytes": self.source_bytes,
                "page_count": self.page_count}


@dataclass(frozen=True)
class RowSelection:
    """Bounded field/value selection from a Dataset Viewer-backed HF split."""
    repository: str
    revision: str
    config: str
    split: str
    selector_field: str
    selector_values: tuple[str, ...]
    fields: tuple[str, ...]
    license: str
    row_numbers: tuple[int, ...] = ()
    max_bytes: int = 10 * 1024 * 1024
    max_rows: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "selector_values", tuple(self.selector_values))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "row_numbers", tuple(self.row_numbers))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise AcquisitionError("row selection revision must be an immutable 40-character commit SHA")
        if not self.repository or self.repository.count("/") != 1:
            raise AcquisitionError("repository must be owner/name")
        if not self.selector_field.strip():
            raise AcquisitionError("selector_field is required")
        if not self.selector_values or len(set(self.selector_values)) != len(self.selector_values):
            raise AcquisitionError("selector_values must be non-empty and unique")
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise AcquisitionError("fields must be non-empty and unique")
        if self.selector_field not in self.fields:
            raise AcquisitionError("selector_field must be included in fields")
        if not self.license.strip() or self.max_bytes <= 0:
            raise AcquisitionError("license and max_bytes are required")
        if self.max_rows <= 0:
            raise AcquisitionError("max_rows must be positive")
        if self.row_numbers:
            if any(row < 0 for row in self.row_numbers):
                raise AcquisitionError("row_numbers must be zero-based")
            if tuple(sorted(set(self.row_numbers))) != self.row_numbers:
                raise AcquisitionError("row_numbers must be unique and deterministically ordered")
            if len(self.row_numbers) > self.max_rows:
                raise AcquisitionError("row_numbers exceed max_rows")


@dataclass(frozen=True)
class RowAcquisitionResult:
    repository: str
    revision: str
    config: str
    split: str
    records: tuple[dict[str, Any], ...]
    row_sha256: tuple[str, ...]
    selector_field: str
    row_numbers: tuple[int, ...]
    matched_value_count: int
    row_count: int
    source_bytes: int
    license: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository, "revision": self.revision,
            "config": self.config, "split": self.split,
            "records": list(self.records), "row_sha256": list(self.row_sha256),
            "selector_field": self.selector_field,
            "row_numbers": list(self.row_numbers),
            "matched_value_count": self.matched_value_count,
            "row_count": self.row_count,
            "source_bytes": self.source_bytes, "license": self.license,
        }


class HuggingFaceAdapter:
    def __init__(self, *, token: str | None = None, opener=None, sleep=None) -> None:
        self._token = token if token is not None else os.environ.get("HF_TOKEN")
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def _get(self, url: str) -> bytes:
        request = urllib.request.Request(url)
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        attempts = 6
        for attempt in range(1, attempts + 1):
            try:
                with self._opener(request, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                retryable = error.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == attempts:
                    raise AcquisitionError(f"Hugging Face acquisition failed: {error}") from error
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    delay = float(retry_after) if retry_after is not None else min(10.0, 2.0 ** (attempt - 1))
                except ValueError:
                    delay = min(10.0, 2.0 ** (attempt - 1))
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == attempts:
                    raise AcquisitionError(f"Hugging Face acquisition failed: {error}") from error
                delay = min(10.0, 2.0 ** (attempt - 1))
            self._sleep(delay)
        raise AcquisitionError("Hugging Face acquisition failed after retries")

    def acquire(self, selection: DatasetSelection) -> AcquisitionResult:
        api = json.loads(self._get(f"https://huggingface.co/api/datasets/{selection.repository}"))
        if api.get("private") or api.get("gated") or api.get("disabled"):
            raise AcquisitionError("repository is private, gated, or disabled")
        if api.get("sha") != selection.revision:
            # The revision is immutable, but API metadata at another revision must be inspected.
            api = json.loads(self._get(f"https://huggingface.co/api/datasets/{selection.repository}/tree/{selection.revision}?recursive=true"))
        declared = api.get("cardData", {}).get("license", api.get("license", ""))
        declared_values = declared if isinstance(declared, list) else [declared]
        if selection.license.lower() not in {str(value).lower() for value in declared_values}:
            raise AcquisitionError(f"license mismatch: expected {selection.license}, got {declared_values}")
        url = f"https://huggingface.co/datasets/{selection.repository}/resolve/{selection.revision}/{selection.file}"
        content = self._get(url)
        if len(content) > selection.max_bytes:
            raise AcquisitionError("source transfer ceiling exceeded before parsing")
        file_hash = hashlib.sha256(content).hexdigest()
        if selection.expected_file_sha256 and file_hash != selection.expected_file_sha256:
            raise AcquisitionError("source file hash mismatch")
        rows = self._parse(selection.file, content)
        selected: list[dict[str, Any]] = []
        hashes: list[str] = []
        for row_number in selection.row_numbers:
            if row_number >= len(rows):
                raise AcquisitionError(f"selected row {row_number} is missing")
            row = rows[row_number]
            if any(field not in row for field in selection.fields):
                raise AcquisitionError("schema drift: a selected field is missing")
            projected = {field: row[field] for field in selection.fields}
            encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            row_hash = hashlib.sha256(encoded.encode()).hexdigest()
            selected.append(projected)
            hashes.append(row_hash)
        if selection.expected_row_sha256 and tuple(hashes) != selection.expected_row_sha256:
            raise AcquisitionError("selected row hash mismatch")
        return AcquisitionResult(selection.repository, selection.revision, selection.configuration,
                                 selection.split, selection.file, file_hash, len(content),
                                 selection.row_numbers, tuple(selected), tuple(hashes), selection.license)

    def acquire_images(self, selection: ImageSelection, output_dir: str | Path) -> ImageAcquisitionResult:
        """Acquire only selected image members from a pinned ZIP archive.

        The archive is bounded and hash-verified before extraction. Labels are
        returned in a separate judgment sidecar and never enter source records.
        """
        api = json.loads(self._get(f"https://huggingface.co/api/datasets/{selection.repository}").decode())
        if api.get("private") or api.get("gated") or api.get("disabled"):
            raise AcquisitionError("image repository is private, gated, or disabled")
        if api.get("sha") != selection.revision:
            raise AcquisitionError("pinned image revision is not the inspected repository revision")
        declared = api.get("cardData", {}).get("license", api.get("license", ""))
        values = declared if isinstance(declared, list) else [declared]
        if selection.license.lower() not in {str(value).lower() for value in values}:
            raise AcquisitionError(f"license mismatch: expected {selection.license}, got {values}")
        url = f"https://huggingface.co/datasets/{selection.repository}/resolve/{selection.revision}/{selection.file}"
        content = self._get(url)
        if len(content) > selection.max_bytes:
            raise AcquisitionError("image archive transfer ceiling exceeded")
        file_hash = hashlib.sha256(content).hexdigest()
        if selection.expected_file_sha256 and file_hash != selection.expected_file_sha256:
            raise AcquisitionError("image archive hash mismatch")
        output = Path(output_dir)
        images = output / "images"
        images.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        judgments: list[dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
            for member in selection.members:
                if member not in names or not member.lower().endswith((".jpg", ".jpeg", ".png")):
                    raise AcquisitionError(f"selected image member is missing or unsupported: {member}")
                image_bytes = archive.read(member)
                image_hash = "sha256:" + hashlib.sha256(image_bytes).hexdigest()
                asset_id = f"{selection.repository}@{selection.revision}:{member}"
                target = images / (hashlib.sha256(asset_id.encode()).hexdigest()[:24] + Path(member).suffix.lower())
                target.write_bytes(image_bytes)
                parts = member.split("/")
                label = parts[-2] if len(parts) > 1 else "unknown"
                records.append({
                    "asset_id": asset_id, "member": member, "image_ref": target.as_uri(),
                    "image_sha256": image_hash, "media_type": "image/jpeg",
                    "byte_count": len(image_bytes),
                })
                judgments.append({"asset_id": asset_id, "label": label, "member": member})
        return ImageAcquisitionResult(selection.repository, selection.revision, selection.configuration,
                                      selection.split, selection.file, file_hash, len(content),
                                      tuple(records), tuple(judgments), selection.license)

    def acquire_document(self, selection: DocumentSelection, output_dir: str | Path) -> DocumentAcquisitionResult:
        """Acquire selected Hub rows and inline page images without semantic leakage.

        The Dataset Viewer is used only as a bounded transport for the pinned Hub
        revision. Answers and supplied embeddings are emitted to a separate judgments
        sidecar for the evaluator and are absent from ``records``.
        """
        output = Path(output_dir)
        images = output / "images"
        images.mkdir(parents=True, exist_ok=True)
        api = json.loads(self._get(f"https://huggingface.co/api/datasets/{selection.repository}").decode())
        if api.get("private") or api.get("gated") or api.get("disabled"):
            raise AcquisitionError("document repository is private, gated, or disabled")
        if api.get("sha") != selection.revision:
            raise AcquisitionError("pinned document revision is not the repository head inspected")
        declared = api.get("cardData", {}).get("license", api.get("license", ""))
        values = declared if isinstance(declared, list) else [declared]
        if selection.expected_license.lower() not in {str(value).lower() for value in values}:
            raise AcquisitionError(f"license mismatch: expected {selection.expected_license}, got {values}")
        rows: list[Mapping[str, Any]] = []
        for row_number in selection.row_numbers:
            endpoint = ("https://datasets-server.huggingface.co/rows?dataset="
                        f"{quote(selection.repository, safe='')}&config={quote(selection.config)}"
                        f"&split={quote(selection.split)}&offset={row_number}&length=1")
            payload = json.loads(self._get(endpoint).decode())
            returned = payload.get("rows", [])
            if len(returned) != 1 or returned[0].get("row_idx") != row_number:
                raise AcquisitionError(f"dataset row {row_number} could not be pinned")
            rows.append(returned[0]["row"])
        records: list[dict[str, Any]] = []
        judgments: list[dict[str, Any]] = []
        page_paths: dict[str, Path] = {}
        source_bytes = 0
        for row in rows:
            required = ("id", "image", "image_id", "question_id", "question", "doc_id",
                        "ucsf_document_id", "ucsf_document_page_no", "question_types", "answer", "answers")
            if any(key not in row for key in required):
                raise AcquisitionError("DocVQA schema drift: required field missing")
            image = row["image"]
            image_url = image.get("src") if isinstance(image, Mapping) else None
            if not image_url:
                raise AcquisitionError("DocVQA row has no inline image reference")
            page_id = f"doc:{row['doc_id']}:page:{row['ucsf_document_page_no']}"
            if page_id not in page_paths:
                image_bytes = self._get(str(image_url))
                source_bytes += len(image_bytes)
                if source_bytes > selection.max_bytes:
                    raise AcquisitionError("document source transfer ceiling exceeded")
                image_path = images / (hashlib.sha256(page_id.encode()).hexdigest()[:20] + ".jpg")
                image_path.write_bytes(image_bytes)
                page_paths[page_id] = image_path
            image_path = page_paths[page_id]
            image_hash = "sha256:" + hashlib.sha256(image_path.read_bytes()).hexdigest()
            records.append({"row_number": row["id"], "page_id": page_id, "document_id": str(row["doc_id"]),
                            "question_id": str(row["question_id"]), "question": str(row["question"]),
                            "image_id": str(row["image_id"]), "ucsf_document_id": str(row["ucsf_document_id"]),
                            "page_number": str(row["ucsf_document_page_no"]),
                            "question_types": list(row["question_types"]),
                            "image_ref": image_path.as_uri(), "image_sha256": image_hash,
                            "image_width": image.get("width"), "image_height": image.get("height")})
            judgments.append({"question_id": str(row["question_id"]), "page_id": page_id,
                              "acceptable_answers": [str(answer) for answer in row["answers"]],
                              "answerability": "answerable"})
        return DocumentAcquisitionResult(selection.repository, selection.revision, selection.config,
                                         selection.split, tuple(records), tuple(judgments), source_bytes,
                                         len(page_paths))

    def acquire_rows(self, selection: RowSelection) -> RowAcquisitionResult:
        """Acquire a bounded field/value selection through the Dataset Viewer API."""
        api = json.loads(self._get(f"https://huggingface.co/api/datasets/{selection.repository}").decode())
        if api.get("private") or api.get("gated") or api.get("disabled"):
            raise AcquisitionError("row-selection repository is private, gated, or disabled")
        if api.get("sha") != selection.revision:
            raise AcquisitionError("pinned row-selection revision is not the repository head inspected")
        declared = api.get("cardData", {}).get("license", api.get("license", ""))
        values = declared if isinstance(declared, list) else [declared]
        if selection.license.lower() not in {str(value).lower() for value in values}:
            raise AcquisitionError(f"license mismatch: expected {selection.license}, got {values}")
        selected_values = set(selection.selector_values)
        collected: list[dict[str, Any]] = []
        hashes: list[str] = []
        source_bytes = 0
        windows = _row_windows(selection.row_numbers) or tuple(
            (offset, min(100, selection.max_rows - offset))
            for offset in range(0, selection.max_rows, 100)
        )
        requested_rows = set(selection.row_numbers)
        acquired_rows: list[int] = []
        for offset, length in windows:
            endpoint = ("https://datasets-server.huggingface.co/rows?dataset="
                        f"{quote(selection.repository, safe='')}&config={quote(selection.config)}"
                        f"&split={quote(selection.split)}&offset={offset}&length={length}")
            raw = self._get(endpoint)
            source_bytes += len(raw)
            if source_bytes > selection.max_bytes:
                raise AcquisitionError("row-selection source transfer ceiling exceeded")
            payload = json.loads(raw.decode())
            returned = payload.get("rows", [])
            if not returned:
                break
            for entry in returned:
                row_number = int(entry.get("row_idx", offset))
                if requested_rows and row_number not in requested_rows:
                    continue
                row = entry.get("row", {})
                selected_value = row.get(selection.selector_field)
                if selected_value is None or str(selected_value) not in selected_values:
                    continue
                if any(field not in row for field in selection.fields):
                    raise AcquisitionError("row-selection schema drift: a selected field is missing")
                projected = {field: row[field] for field in selection.fields}
                encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                row_hash = hashlib.sha256(encoded.encode()).hexdigest()
                collected.append(projected)
                hashes.append(row_hash)
                acquired_rows.append(row_number)
            if not requested_rows and len(returned) < length:
                break
        if not collected:
            raise AcquisitionError("no rows matched the selected values")
        seen_values = {str(record.get(selection.selector_field)) for record in collected}
        missing = selected_values - seen_values
        if missing:
            raise AcquisitionError(f"selected values not found in dataset: {sorted(missing)}")
        return RowAcquisitionResult(
            repository=selection.repository,
            revision=selection.revision,
            config=selection.config,
            split=selection.split,
            records=tuple(collected),
            row_sha256=tuple(hashes),
            selector_field=selection.selector_field,
            row_numbers=tuple(acquired_rows),
            matched_value_count=len(seen_values),
            row_count=len(collected),
            source_bytes=source_bytes,
            license=selection.license,
        )

    @staticmethod
    def _parse(file: str, content: bytes) -> list[Mapping[str, Any]]:
        text = content.decode("utf-8")
        if file.endswith(".jsonl"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if file.endswith(".json"):
            value = json.loads(text)
            return value if isinstance(value, list) else value.get("data", [])
        if file.endswith(".csv"):
            return list(csv.DictReader(io.StringIO(text)))
        raise AcquisitionError("unsupported source file format; use JSON, JSONL, or CSV")
