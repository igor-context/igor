from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import pyarrow as pa
import pyarrow.dataset as ds
from datafusion import SessionContext
from igor_lancedb import LanceStore


class RetrievalPort(Protocol):
    def retrieve(self, query: Any) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class SemanticSearchRequest:
    table_name: str
    query: str
    search_mode: str
    limit: int
    query_vector: tuple[float, ...] = ()
    query_vector_ref: str | None = None
    prefilter: str | None = None
    retrieval_table_name: str | None = None


class AnalyticalEngine:
    """Execute SQL and plan semantic retrieval through a neutral port."""

    def __init__(self, store: LanceStore, retrieval: RetrievalPort | None = None,
                 query_vectors: Mapping[str, tuple[float, ...]] | None = None):
        self.store = store
        self.retrieval = retrieval
        self.query_vectors = dict(query_vectors or {})

    def query(self, sql: str, table_name: str = "events") -> pa.Table:
        if not sql.strip():
            raise ValueError("analytical SQL cannot be empty")
        try:
            context = SessionContext()
            snapshots = {name: self.store.read(name) for name in self.store.names()}
            for name in self.store.names():
                context.register_table(name, context.from_arrow(snapshots[name]))
            if self.retrieval is not None:
                context.register_udtf(self._semantic_search_function(context, snapshots))
            if table_name not in self.store.names() and "semantic_search" not in sql:
                self.store.read(table_name)
            batches = context.sql(sql).collect()
            return pa.Table.from_batches(batches)
        except Exception as error:
            raise ValueError(f"analytical query failed: {error}") from error

    def _semantic_search_function(self, context: SessionContext, snapshots: dict[str, pa.Table]):
        from datafusion import TableFunction

        def semantic_search(table: Any, query: Any, mode: Any, limit: Any,
                            query_vector_ref: Any | None = None, prefilter: Any | None = None,
                            *, session: SessionContext):
            def value_of(expression: Any) -> Any:
                value = expression.python_value()
                return value.as_py() if hasattr(value, "as_py") else value

            vector_ref = None if query_vector_ref is None else str(value_of(query_vector_ref))
            if vector_ref is None:
                vector = ()
            else:
                try:
                    vector = tuple(self.query_vectors[vector_ref])
                except KeyError as error:
                    raise ValueError(f"unknown query vector reference: {vector_ref}") from error
            request = SemanticSearchRequest(
                table_name=value_of(table), query=value_of(query),
                search_mode=value_of(mode), limit=int(value_of(limit)),
                query_vector=vector,
                query_vector_ref=vector_ref,
                prefilter=None if prefilter is None else value_of(prefilter),
            )
            if request.limit <= 0:
                raise ValueError("semantic_search limit must be positive")
            facts = self.retrieval.retrieve(request)
            catalog = {row["canonical_identity"]: row for row in snapshots[request.table_name].to_pylist()}
            rows = []
            for fact in facts:
                value = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else dict(fact)
                identity = value.get("context_identity")
                source = catalog.get(identity, {})
                rows.append({
                    "canonical_identity": identity,
                    "display_name": source.get("display_name", identity),
                    "context_model_id": source.get("context_model_id"),
                    "object_kind": source.get("object_kind"),
                    "status": source.get("status", "active"),
                    "source_system": source.get("source_system"),
                    "source_key": source.get("source_key"),
                    "media_type": source.get("media_type", "application/octet-stream"),
                    "evidence_preview": source.get("evidence_preview", source.get("preview", "")),
                    "content_ref": source.get("content_ref"),
                    "semantic_score": value.get("score", 0.0),
                    "authority_status": source.get("authority_status", "unknown"),
                    "temporal_status": source.get("temporal_status", "unknown"),
                    "policy_id": source.get("policy_id"),
                    "as_of": source.get("as_of"),
                    "retrieval_rank": value.get("rank", 0),
                    "retrieval_outcome": "candidate",
                })
            schema = pa.schema([
                ("canonical_identity", pa.string()), ("display_name", pa.string()),
                ("context_model_id", pa.string()), ("object_kind", pa.string()),
                ("status", pa.string()), ("source_system", pa.string()),
                ("source_key", pa.string()), ("media_type", pa.string()),
                ("evidence_preview", pa.string()), ("content_ref", pa.string()),
                ("semantic_score", pa.float64()), ("authority_status", pa.string()),
                ("temporal_status", pa.string()), ("policy_id", pa.string()),
                ("as_of", pa.string()), ("retrieval_rank", pa.int64()),
                ("retrieval_outcome", pa.string()),
            ])
            if not rows:
                empty = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
                return ds.dataset(empty)
            return ds.dataset(pa.Table.from_pylist(rows, schema=schema))

        return TableFunction("semantic_search", semantic_search, context, with_session=True)
