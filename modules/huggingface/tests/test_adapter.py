import hashlib
import io
import json
import unittest
import zipfile
from urllib.error import HTTPError
from unittest.mock import patch

from igor_huggingface import AcquisitionError, DatasetSelection, HuggingFaceAdapter, ImageSelection, RowSelection
from igor_huggingface.adapter import _row_windows


class AdapterTest(unittest.TestCase):
    def test_mutable_revision_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            DatasetSelection("owner/name", "main", "default", "train", "data.jsonl", (0,), ("text",), "mit")

    def test_acquisition_projects_fields_and_verifies_hashes(self):
        payload = b'{"text":"hello","label":"hidden"}\n{"text":"world","label":"hidden"}\n'
        projected = json.dumps({"text": "world"}, sort_keys=True, separators=(",", ":"))
        row_hash = hashlib.sha256(projected.encode()).hexdigest()
        selection = DatasetSelection("owner/name", "a" * 40, "default", "train", "data.jsonl", (1,), ("text",), "mit", hashlib.sha256(payload).hexdigest(), (row_hash,))

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({"private": False, "gated": False, "disabled": False, "sha": selection.revision, "cardData": {"license": "mit"}}).encode()
                    return payload
            return Response()

        result = HuggingFaceAdapter(opener=fake).acquire(selection)
        self.assertEqual(result.records, ({"text": "world"},))
        self.assertEqual(result.row_sha256, (row_hash,))

    def test_image_acquisition_extracts_bytes_and_seals_labels(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as value:
            value.writestr("validation/healthy/healthy_val.0.jpg", b"healthy-bytes")
        payload = archive.getvalue()
        member = "validation/healthy/healthy_val.0.jpg"
        selection = ImageSelection("owner/name", "a" * 40, "default", "validation", "data/validation.zip", (member,), "mit", hashlib.sha256(payload).hexdigest())

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({"private": False, "gated": False, "disabled": False, "sha": selection.revision, "cardData": {"license": ["mit"]}}).encode()
                    return payload
            return Response()

        result = HuggingFaceAdapter(opener=fake).acquire_images(selection, self._tmpdir())
        self.assertEqual(result.records[0]["image_sha256"], "sha256:" + hashlib.sha256(b"healthy-bytes").hexdigest())
        self.assertNotIn("label", result.records[0])
        self.assertEqual(result.judgments[0]["label"], "healthy")

    def test_row_selection_mutable_revision_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            RowSelection("owner/name", "main", "metadata-en", "metadata", "event_id",
                         ("event-1",), ("event_id", "text"), "cc-by-4.0")

    def test_amendment_empty_event_ids_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            RowSelection("owner/name", "a" * 40, "metadata-en", "metadata", "event_id",
                         (), ("event_id",), "cc-by-4.0")

    def test_amendment_duplicate_event_ids_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            RowSelection("owner/name", "a" * 40, "metadata-en", "metadata", "event_id",
                         ("event-1", "event-1"), ("event_id",), "cc-by-4.0")

    def test_amendment_empty_fields_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            RowSelection("owner/name", "a" * 40, "metadata-en", "metadata", "event_id",
                         ("event-1",), (), "cc-by-4.0")

    def test_amendment_zero_max_rows_fails_closed(self):
        with self.assertRaises(AcquisitionError):
            RowSelection("owner/name", "a" * 40, "metadata-en", "metadata", "event_id",
                         ("event-1",), ("event_id",), "cc-by-4.0",
                               max_rows=0)

    def test_amendment_valid_selection_constructs(self):
        selection = RowSelection(
            "kasys/ReCaRe", "a" * 40, "metadata-en", "metadata",
            "event_id", ("event-1", "event-2"), ("event_id", "text", "change_type"),
            "cc-by-4.0", max_bytes=10 * 1024 * 1024, max_rows=100,
        )
        self.assertEqual(selection.selector_values, ("event-1", "event-2"))
        self.assertEqual(selection.max_rows, 100)

    def test_row_windows_coalesce_sparse_offsets_without_exceeding_page_limit(self):
        self.assertEqual(_row_windows((24, 25, 99, 417, 418, 516, 700)),
                         ((24, 76), (417, 100), (700, 1)))

    def test_amendment_acquisition_filters_and_hashes(self):
        rows_page = {
            "rows": [
                {"row_idx": 0, "row": {"amendment_law_id": None, "law_id": "31964L0432",
                                        "type_of_change": None, "text_before": "scope"}},
                {"row_idx": 1, "row": {"amendment_law_id": "32013L0020", "law_id": "31964L0432",
                                        "type_of_change": "potential_mod_-ct", "text_before": "defs"}},
                {"row_idx": 2, "row": {"amendment_law_id": None, "law_id": "31964L0432",
                                        "type_of_change": None, "text_before": "reqs"}},
            ]
        }
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "amendment_law_id", ("32013L0020",),
            ("amendment_law_id", "law_id", "type_of_change"),
            "cc-by-4.0", max_rows=100,
        )

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({
                            "private": False, "gated": False, "disabled": False,
                            "sha": selection.revision,
                            "cardData": {"license": "cc-by-4.0"},
                        }).encode()
                    return json.dumps(rows_page).encode()
            return Response()

        result = HuggingFaceAdapter(opener=fake).acquire_rows(selection)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["amendment_law_id"], "32013L0020")
        self.assertEqual(result.records[0]["law_id"], "31964L0432")
        self.assertEqual(result.matched_value_count, 1)
        self.assertEqual(result.row_count, 1)
        projected = json.dumps(result.records[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        expected_hash = hashlib.sha256(projected.encode()).hexdigest()
        self.assertEqual(result.row_sha256, (expected_hash,))

    def test_amendment_acquisition_coalesces_sparse_row_windows(self):
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "amendment_law_id", ("32013L0020",),
            ("amendment_law_id", "law_id", "type_of_change"),
            "cc-by-4.0", row_numbers=(10, 12, 90, 130), max_rows=100,
        )
        calls = []

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({
                            "private": False, "gated": False, "disabled": False,
                            "sha": selection.revision,
                            "cardData": {"license": "cc-by-4.0"},
                        }).encode()
                    calls.append(url.full_url)
                    if "offset=10&length=81" in url.full_url:
                        return json.dumps({"rows": [
                            {"row_idx": 10, "row": {"amendment_law_id": "32013L0020", "law_id": "a",
                                                    "type_of_change": "modify"}},
                            {"row_idx": 11, "row": {"amendment_law_id": "ignore", "law_id": "skip",
                                                    "type_of_change": "skip"}},
                            {"row_idx": 12, "row": {"amendment_law_id": "32013L0020", "law_id": "b",
                                                    "type_of_change": "modify"}},
                            {"row_idx": 90, "row": {"amendment_law_id": "32013L0020", "law_id": "c",
                                                    "type_of_change": "modify"}},
                        ]}).encode()
                    return json.dumps({"rows": [
                        {"row_idx": 130, "row": {"amendment_law_id": "32013L0020", "law_id": "d",
                                                 "type_of_change": "modify"}},
                    ]}).encode()
            return Response()

        result = HuggingFaceAdapter(opener=fake).acquire_rows(selection)

        self.assertEqual([record["law_id"] for record in result.records], ["a", "b", "c", "d"])
        self.assertEqual(result.row_numbers, (10, 12, 90, 130))
        self.assertEqual(len(calls), 2)

    def test_huggingface_get_retries_transient_http_errors(self):
        calls = 0

        def fake(url, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError(url.full_url, 429, "too many requests", {}, None)

            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner): return b'{"ok": true}'
            return Response()

        self.assertEqual(HuggingFaceAdapter(opener=fake, sleep=lambda delay: None)._get("https://example.test/data"), b'{"ok": true}')
        self.assertEqual(calls, 2)

    def test_amendment_acquisition_rejects_private_repo(self):
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "event_id", ("event-1",), ("event_id",), "cc-by-4.0",
        )

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    return json.dumps({"private": True, "gated": False, "disabled": False,
                                       "sha": selection.revision,
                                       "cardData": {"license": "cc-by-4.0"}}).encode()
            return Response()

        with self.assertRaises(AcquisitionError):
            HuggingFaceAdapter(opener=fake).acquire_rows(selection)

    def test_amendment_acquisition_rejects_license_mismatch(self):
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "event_id", ("event-1",), ("event_id",), "cc-by-4.0",
        )

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    return json.dumps({"private": False, "gated": False, "disabled": False,
                                       "sha": selection.revision,
                                       "cardData": {"license": "mit"}}).encode()
            return Response()

        with self.assertRaises(AcquisitionError):
            HuggingFaceAdapter(opener=fake).acquire_rows(selection)

    def test_amendment_acquisition_rejects_missing_events(self):
        rows_page = {
            "rows": [
                {"row_idx": 0, "row": {"amendment_law_id": "32013L0020", "law_id": "31964L0432"}},
            ]
        }
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "amendment_law_id", ("32013L0020", "32099L9999"),
            ("amendment_law_id", "law_id"),
            "cc-by-4.0", max_rows=100,
        )

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({"private": False, "gated": False, "disabled": False,
                                           "sha": selection.revision,
                                           "cardData": {"license": "cc-by-4.0"}}).encode()
                    return json.dumps(rows_page).encode()
            return Response()

        with self.assertRaises(AcquisitionError):
            HuggingFaceAdapter(opener=fake).acquire_rows(selection)

    def test_amendment_acquisition_multi_event(self):
        rows_page = {
            "rows": [
                {"row_idx": 0, "row": {"amendment_law_id": "32013L0020", "law_id": "31964L0432",
                                        "type_of_change": "potential_mod_-ct"}},
                {"row_idx": 1, "row": {"amendment_law_id": "32019R2175", "law_id": "32010R1095",
                                        "type_of_change": "mod_--t"}},
                {"row_idx": 2, "row": {"amendment_law_id": "32019R2175", "law_id": "32010R1095",
                                        "type_of_change": "mod_--t"}},
            ]
        }
        selection = RowSelection(
            "owner/name", "a" * 40, "metadata-en", "metadata",
            "amendment_law_id", ("32013L0020", "32019R2175"),
            ("amendment_law_id", "law_id", "type_of_change"),
            "cc-by-4.0", max_rows=100,
        )

        def fake(url, **kwargs):
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self_inner):
                    if "/api/datasets/" in url.full_url:
                        return json.dumps({"private": False, "gated": False, "disabled": False,
                                           "sha": selection.revision,
                                           "cardData": {"license": "cc-by-4.0"}}).encode()
                    return json.dumps(rows_page).encode()
            return Response()

        result = HuggingFaceAdapter(opener=fake).acquire_rows(selection)
        self.assertEqual(result.matched_value_count, 2)
        self.assertEqual(result.row_count, 3)
        self.assertEqual(len(result.row_sha256), 3)

    def test_amendment_result_as_dict(self):
        from igor_huggingface.adapter import RowAcquisitionResult
        result = RowAcquisitionResult(
            repository="owner/name", revision="a" * 40,
            config="metadata-en", split="metadata",
            records=({"event_id": "event-1", "text": "example"},),
            row_sha256=("abc123",), selector_field="event_id", row_numbers=(7,),
            matched_value_count=1, row_count=1,
            source_bytes=1024, license="cc-by-4.0",
        )
        d = result.as_dict()
        self.assertEqual(d["repository"], "owner/name")
        self.assertEqual(d["matched_value_count"], 1)
        self.assertIsInstance(d["records"], list)
        self.assertIsInstance(d["row_sha256"], list)

    @staticmethod
    def _tmpdir():
        import tempfile
        return tempfile.mkdtemp()


if __name__ == "__main__":
    unittest.main()
