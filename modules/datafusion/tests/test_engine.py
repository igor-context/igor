from pathlib import Path

import pytest

from igor_datafusion import AnalyticalEngine
from igor_lancedb import LanceStore


def make_engine(tmp_path: Path) -> AnalyticalEngine:
    store = LanceStore(tmp_path / "db")
    store.create("events", [{"id": 1, "group_name": "a", "amount": 10}, {"id": 2, "group_name": "a", "amount": 20}, {"id": 3, "group_name": "b", "amount": 5}])
    return AnalyticalEngine(store)


def test_scan_and_aggregation(tmp_path: Path) -> None:
    rows = make_engine(tmp_path).query("SELECT group_name, SUM(amount) AS total FROM events GROUP BY group_name ORDER BY group_name").to_pylist()
    assert rows == [{"group_name": "a", "total": 30}, {"group_name": "b", "total": 5}]


def test_join(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.store.create("groups", [{"group_name": "a", "label": "Alpha"}, {"group_name": "b", "label": "Beta"}])
    rows = engine.query("SELECT e.id, g.label FROM events e JOIN groups g ON e.group_name = g.group_name ORDER BY e.id").to_pylist()
    assert rows == [{"id": 1, "label": "Alpha"}, {"id": 2, "label": "Alpha"}, {"id": 3, "label": "Beta"}]


def test_invalid_sql_and_missing_table_are_deterministic(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(ValueError, match="analytical query failed"):
        engine.query("SELECT missing FROM events")
    with pytest.raises(ValueError, match="analytical query failed: unknown Lance table: absent"):
        engine.query("SELECT * FROM absent", "absent")


def test_semantic_search_is_a_table_function_and_delegates_to_port(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    store.create("context_catalog", [{
        "canonical_identity": "sha256:a", "display_name": "Duplicate charge",
        "context_model_id": "support", "object_kind": "message", "source_system": "crm",
        "source_key": "ticket-1", "preview": "charged twice", "status": "active",
    }])

    class DeterministicRetrieval:
        def __init__(self):
            self.requests = []

        def retrieve(self, request):
            self.requests.append(request)
            return ({"context_identity": "sha256:a", "score": 0.91, "rank": 0},)

    retrieval = DeterministicRetrieval()
    rows = AnalyticalEngine(store, retrieval).query(
        "SELECT canonical_identity, display_name, semantic_score "
        "FROM semantic_search('context_catalog', 'charged twice', 'full_text', 10) "
        "WHERE context_model_id = 'support' AND status = 'active'"
    ).to_pylist()
    assert rows == [{"canonical_identity": "sha256:a", "display_name": "Duplicate charge", "semantic_score": 0.91}]
    assert retrieval.requests[0].table_name == "context_catalog"
    assert retrieval.requests[0].search_mode == "full_text"


def test_semantic_search_requires_positive_limit(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    store.create("context_catalog", [{"canonical_identity": "sha256:a"}])

    class EmptyRetrieval:
        def retrieve(self, request):
            return ()

    with pytest.raises(ValueError, match="semantic_search limit must be positive"):
        AnalyticalEngine(store, EmptyRetrieval()).query(
            "SELECT * FROM semantic_search('context_catalog', 'x', 'full_text', 0)"
        )


def test_empty_semantic_search_is_an_empty_relation(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    store.create("context_catalog", [{"canonical_identity": "sha256:a"}])

    class EmptyRetrieval:
        def retrieve(self, request):
            return ()

    rows = AnalyticalEngine(store, EmptyRetrieval()).query(
        "SELECT COUNT(*) AS result_count FROM semantic_search('context_catalog', 'x', 'full_text', 3)"
    ).to_pylist()
    assert rows == [{"result_count": 0}]
