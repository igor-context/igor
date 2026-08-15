# DataFusion module

Own analytical SQL planning and execution over Lance relations. The `AnalyticalEngine` interface accepts a `LanceStore`, SQL request, and optional provider-neutral retrieval port. It registers Arrow snapshots and the `semantic_search(table, query, mode, limit[, query_vector])` table function, returning complete bounded evidence rows without exposing DataFusion lifecycle to callers.

```text
AnalyticalEngine(store, retrieval).query(sql) -> pyarrow.Table
```

DataFusion is the only SQL engine in this path; storage lifecycle remains in `modules/lancedb/`. Retrieval is delegated through the port; DataFusion does not import LanceDB retrieval code. Unsupported tables, invalid SQL, and invalid semantic-search limits raise deterministic `ValueError` errors.

Verification: `docker compose run --rm --build datafusion-test`.
