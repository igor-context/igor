from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from igor_dlt import IngestConfig, IngestionLimits, McpSource, RestSource, SourceLimitError, SqlSource, ingest_source
from igor_core import stable_identity


class _ApiState:
    pages = [
        {"records": [{"id": "a", "text": "alpha"}], "next": None},
    ]


class _Handler(BaseHTTPRequestHandler):
    paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        self.__class__.paths.append(self.path)
        page = int(self.path.rsplit("/", 1)[-1]) if self.path.rsplit("/", 1)[-1].isdigit() else 1
        body = json.dumps(_ApiState.pages[min(page - 1, len(_ApiState.pages) - 1)]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def _api() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/1"


def _config(tmp_path, name: str) -> IngestConfig:
    return IngestConfig("test", "transport", str(tmp_path / "lance"), f"test/{name}", dataset_name=name, state_dir=str(tmp_path / "state"), pipeline_dir=str(tmp_path / "pipelines"))


def test_rest_dlt_lance_snapshot_reuse_change_and_tombstone(tmp_path) -> None:
    server, uri = _api()
    try:
        source = RestSource(uri, "ticket", field_mapping={"text": "text"}, mode="snapshot")
        first = ingest_source(source, _config(tmp_path, "rest"))
        assert first.records[0]["record_id"] == "a"
        assert first.load_metadata["source"]["capabilities"]["pagination"] is True
        assert {node.node_type for node in first.lineage(stable_identity("run-rest")).nodes} >= {"source_record", "source_observation", "ingestion_load", "context_unit"}
        identical = ingest_source(source, _config(tmp_path, "rest"))
        assert identical.records == ()
        assert identical.load_metadata["dlt"]["skipped"] is True
        assert identical.load_metadata["counters"]["reused_records"] == 1

        _ApiState.pages = [{"records": [{"id": "b", "text": "bravo"}], "next": None}]
        second = ingest_source(source, _config(tmp_path, "rest"))
        assert {row["record_id"] for row in second.records} == {"a", "b"}
        assert second.load_metadata["counters"]["deleted_records"] == 1
        assert any(row["deleted"] for row in second.records)
    finally:
        server.shutdown()


def test_sql_dlt_lance_cursor_merge_and_declared_tombstone(tmp_path) -> None:
    database = tmp_path / "fixture.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (id TEXT PRIMARY KEY, text TEXT, updated INTEGER, _deleted INTEGER DEFAULT 0)")
        connection.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", [("a", "alpha", 1, 0), ("b", "bravo", 2, 0)])
        connection.commit()
    source = SqlSource(str(database), "item", "items", "id", cursor_column="updated", field_mapping={"text": "text", "updated": "updated"}, mode="incremental")
    first = ingest_source(source, _config(tmp_path, "sql"))
    assert len(first.records) == 2
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE items SET text = ?, updated = ? WHERE id = ?", ("alpha changed", 3, "a"))
        connection.execute("INSERT INTO items VALUES (?, ?, ?, ?)", ("c", "charlie", 4, 0))
        connection.execute("UPDATE items SET _deleted = 1, updated = ? WHERE id = ?", (5, "b"))
        connection.commit()
    second = ingest_source(source, _config(tmp_path, "sql"))
    assert {row["record_id"] for row in second.records} == {"a", "c", "b"}
    assert second.load_metadata["counters"]["deleted_records"] == 1
    assert second.load_metadata["source"]["capabilities"]["predicate_pushdown"] is True


def test_rest_cursor_state_is_sent_on_incremental_rerun(tmp_path) -> None:
    server, uri = _api()
    try:
        _Handler.paths.clear()
        _ApiState.pages = [{"records": [{"id": "a", "text": "alpha", "updated": 1}], "next": None}]
        source = RestSource(uri, "ticket", cursor_field="updated", field_mapping={"text": "text"}, mode="incremental")
        ingest_source(source, _config(tmp_path, "rest-cursor"))
        _ApiState.pages = [{"records": [{"id": "b", "text": "bravo", "updated": 2}], "next": None}]
        ingest_source(source, _config(tmp_path, "rest-cursor"))
        assert any("cursor=1" in path for path in _Handler.paths)
    finally:
        server.shutdown()


class _Mcp:
    def __init__(self) -> None:
        self.resources = [{"uri": "mcp://fixture/a", "name": "a", "mimeType": "text/plain", "revision": "1"}]

    def list_resources(self, cursor=None):
        return self.resources, None, {"snapshot": False, "updates": False, "deletions": False}

    def read_resource(self, uri):
        return {"uri": uri, "text": "hello", "mimeType": "text/plain"}


def test_mcp_resource_adapter_emits_common_observation_and_capabilities(tmp_path) -> None:
    result = ingest_source(McpSource("resource", _Mcp()), _config(tmp_path, "mcp"), IngestionLimits(max_records=2))
    assert result.records[0]["source_system"] == "mcp"
    assert result.records[0]["content_ref"] == "mcp://fixture/a"
    assert result.load_metadata["source"]["capabilities"]["deletions"] is False


def test_limits_fail_closed_before_load(tmp_path) -> None:
    server, uri = _api()
    try:
        with pytest.raises(SourceLimitError):
            ingest_source(RestSource(uri, "ticket"), _config(tmp_path, "limited"), IngestionLimits(max_records=0))
    finally:
        server.shutdown()
