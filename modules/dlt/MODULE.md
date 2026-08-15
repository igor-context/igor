# dlt module

Own source ingestion, normalization, merge keys, and load state through dlt's
Lance destination. The module consumes source records and an explicit storage
configuration, then emits canonical records and serializable load metadata.

## Interface

```python
IngestConfig(environment, domain, bucket_url, namespace_name, dataset_name, storage_options, state_dir, pipeline_dir)
ingest_records(records, config) -> IngestResult
ingest_bound_records(records, config, source_contract, connector_binding) -> IngestResult
ingest_source(source_definition, config, limits) -> IngestResult
```

The canonical record preserves `record_id`, `channel`, and `text`, and adds the
environment, domain, and deterministic `context_unit_id`. The configured
namespace is part of the result identity. `load_metadata` is the dlt load
information converted to JSON-compatible data. `IngestResult.lineage(run_identity)` emits portable source-record → ingestion-load → context-unit facts with a non-secret configuration revision.

`ingest_bound_records` validates the deployment binding against the domain source
contract, maps concrete fields into required concepts, checks media/hash/observation
metadata, and emits stable source and context identities.

`ingest_source` is the transport-neutral acquisition seam. `RestSource` performs
bounded declarative endpoint reads with pagination, retry, authentication headers,
mapping, and snapshot tombstones. `SqlSource` performs bounded table reads with
validated identifiers, cursor predicate pushdown, batching, merge/upsert keys, and
declared tombstone rows. `McpSource` accepts an IGOR-owned resource adapter that uses
`resources/list` and `resources/read`; tool invocation is rejected by this seam.
All three produce the same `Observation` envelope and load through dlt's Lance
destination with merge state persisted under `state_dir`.

Source contract and connector-binding identities are accepted on each source
definition and participate in source identity. They remain caller-supplied domain and
deployment declarations; dlt does not invent scope or meaning. MCP synchronization
capabilities are reported as returned by the server and are never inferred from the
transport.

The module does not answer queries, use dlt's DuckDB-backed dataset interface,
generate embeddings, or perform semantic derivation. Callers use LanceDB for
retrieval and DataFusion for analytical SQL.

## Artifacts and invariants

- State and dlt pipeline packages are written only under ignored `.igor/` paths in
  Compose qualifications or caller-provided runtime directories.
- `load_metadata` includes source/snapshot identities, capabilities, redacted bounded
  counters, and dlt load metadata; credentials are never copied into identities.
- `lineage(run_identity)` emits source-record → source-observation → ingestion-load →
  context-unit edges when observation metadata is present.
- `replace` remains available only through the legacy explicit interface; transport
  sources use merge with `record_id` as primary key and represent supported deletion
  behavior as tombstones.

## Commands

- `docker compose run --rm dlt-test`
- `docker compose run --rm dlt-rest-qualification`
- `docker compose run --rm dlt-sql-qualification`
- `docker compose run --rm dlt-mcp-qualification`
- `docker compose --profile qualification run --rm --build dlt-remote-smoke`
