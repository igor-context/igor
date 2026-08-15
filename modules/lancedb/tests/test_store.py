from pathlib import Path

import pytest

from igor_lancedb import LanceStore, LanceStoreConfig


ROWS = [{"id": 1, "group_name": "a", "amount": 10}, {"id": 2, "group_name": "b", "amount": 20}]


def test_create_add_read_and_metadata(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    store.create("events", ROWS[:1])
    store.add("events", ROWS[1:])
    assert store.read("events").to_pylist() == ROWS
    assert store.metadata("events") == {"name": "events", "columns": ["id", "group_name", "amount"], "row_count": 2}


def test_replace_and_reopen_persisted_table(tmp_path: Path) -> None:
    path = tmp_path / "db"
    LanceStore(path).create("events", ROWS)
    LanceStore(path).replace("events", [{"id": 3, "group_name": "c", "amount": 30}])
    assert LanceStore(path).read("events").to_pylist() == [{"id": 3, "group_name": "c", "amount": 30}]


def test_missing_table_and_empty_writes_are_deterministic(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    with pytest.raises(ValueError, match="unknown Lance table: missing"):
        store.read("missing")
    with pytest.raises(ValueError, match="cannot create a table from empty rows"):
        store.create("empty", [])


def test_local_namespace_is_routed_and_reopened(tmp_path: Path) -> None:
    config = LanceStoreConfig(str(tmp_path / "root"), "test", "support")
    store = LanceStore(config)
    store.create("events", ROWS)

    reopened = LanceStore(config)
    assert reopened.uri.endswith("/test/support")
    assert reopened.names() == ["events"]
    assert reopened.read("events").to_pylist() == ROWS


def test_namespaces_do_not_discover_each_other(tmp_path: Path) -> None:
    support = LanceStoreConfig(str(tmp_path / "root"), "test", "support")
    finance = LanceStoreConfig(str(tmp_path / "root"), "test", "finance")
    LanceStore(support).create("events", ROWS[:1])

    assert LanceStore(support).names() == ["events"]
    assert LanceStore(finance).names() == []
    with pytest.raises(ValueError, match="unknown Lance table: events"):
        LanceStore(finance).read("events")


def test_namespace_and_provider_options_are_validated() -> None:
    with pytest.raises(ValueError, match="environment and domain are required together"):
        LanceStore("s3://bucket/root", environment="test")
    with pytest.raises(ValueError, match="path-safe identifier"):
        LanceStoreConfig("s3://bucket/root", "test/prod", "support")


def test_remote_configuration_passes_provider_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []

    def connect(uri: str, *, storage_options: dict[str, str] | None = None):
        calls.append((uri, storage_options))
        return object()

    monkeypatch.setattr("igor_lancedb.store.lancedb.connect", connect)
    config = LanceStoreConfig(
        "s3://bucket/lance",
        "qualification",
        "support",
        {"aws_endpoint_url": "http://minio:9000"},
    )
    store = LanceStore(config)

    assert store.uri == "s3://bucket/lance/qualification/support"
    assert calls == [(store.uri, {"aws_endpoint_url": "http://minio:9000"})]
