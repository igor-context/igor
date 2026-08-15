"""Runner-owned analytical query command helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from igor_datafusion import AnalyticalEngine
from igor_lancedb import LanceRetrievalAdapter, LanceStore


def query_standard_relations(store: str | Path, sql: str) -> dict[str, Any]:
    if not sql.strip():
        raise ValueError("query SQL cannot be empty")
    store_path = Path(store)
    if not store_path.is_dir():
        raise ValueError(f"query relation store does not exist: {store_path}")
    lance_store = LanceStore(store_path)
    relation_names = lance_store.names()
    if not relation_names:
        raise ValueError(f"query relation store has no Lance tables: {store_path}")
    default_relation = "context_catalog" if "context_catalog" in relation_names else relation_names[0]
    table = AnalyticalEngine(lance_store, LanceRetrievalAdapter(lance_store)).query(
        sql,
        table_name=default_relation,
    )
    rows = [_jsonable(row) for row in table.to_pylist()]
    return {
        "engine": "datafusion",
        "store": str(store_path),
        "relations": relation_names,
        "columns": table.column_names,
        "row_count": len(rows),
        "rows": rows,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "as_py"):
        return _jsonable(value.as_py())
    return value
