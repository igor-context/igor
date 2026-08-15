# LanceDB module

Own operational Lance relations, vector indexes, retrieval semantics, and tool-neutral storage adapters. The `LanceStore` interface hides connection, table, and persistence lifecycle while exposing Arrow tables to the permitted analytical seam.

Implemented interface:

```text
LanceStore(path).create(name, rows)
LanceStore(path).add(name, rows)
LanceStore(path).replace(name, rows)
LanceStore(path).read(name) -> pyarrow.Table
LanceStore(path).metadata(name) -> dict
LanceStore(LanceStoreConfig(root_uri, environment, domain, storage_options)).names()
```

`LanceStoreConfig` resolves `<root_uri>/<environment>/<domain>` and keeps provider-owned
storage options opaque to callers. A plain local path remains supported for existing
callers; a configured local root is also routed beneath its namespace. The URI is the
persistence boundary: callers that need state across processes must keep the root
mounted or point it at durable object storage. Writers are serialized by the caller.
The module does not execute analytical SQL.

`LanceContextOutputStore(store)` implements the context compiler's immutable semantic
output port with identity compare-and-set publication and JSON values stored in a
namespaced Lance relation. The compiler and runner depend only on this port.

`LanceRetrievalAdapter(store)` implements the compiler and semantic-SQL retrieval ports
using LanceDB's native vector, full-text, or hybrid search request and returns
provider-neutral ranked identities, scores, and retrieval facts. Vector and hybrid
searches name the vector column explicitly; `ensure_vector_index()` builds and reports
the bounded native IVF index used by qualification. Full-text search preserves IGOR's
neutral `full_text` request mode while mapping it to LanceDB's native `fts` query type
and creating a bounded native text index before execution.

`LanceReadableContextCatalog(store)` owns the vendor-neutral readable projections
`context_catalog`, `lineage_nodes`, and `lineage_edges`. Canonical hashes are complete
and remain the only join keys; display names, context-model labels, schema labels,
operation labels, statuses, and bounded previews are query/report presentation data.
The relation names are stable and independent of hash-derived physical table names.

`LanceContextModelMaterializer(store)` owns generic Context Model materialization for
standard Lance relations. It accepts a resolved manifest object, portable manifest
artifact, or normalized manifest mapping plus runtime observations, outputs,
assertions, retrieval facts, resolution facts, lineage facts, packages, and package
items, then replaces the standard relation families:
`context_models`, `context_objects`, `context_snapshots`, `context_assertions`,
`context_catalog`, `context_outputs`, `context_retrieval`, `authority_policies`,
`resolution_decisions`, `resolution_candidates`, `lineage_nodes`, `lineage_edges`,
`context_packages`, `package_items`, `context_contracts`,
`context_package_contracts`, and `contract_evaluations`. The interface remains
tool-neutral and does not import the Context Compiler; the runner supplies manifest
and runtime facts while LanceDB owns relation persistence. Every standard relation
family is present after a materialization pass; relation families with no rows are
created as zero-row Lance tables with explicit Arrow schemas. Runtime outputs
automatically produce generic `context_assertions` that preserve the output role,
value, subject identity, and evidence identities; callers may still supply explicit
assertion rows when a domain publishes more specific claims. Resolution decisions,
context contracts, contract evaluations, and context packages also receive standard
readable `lineage_nodes`; `lineage_edges` remain identity-only and get readable
source/target names by joining those nodes.

`validate_standard_context_relations(store)` and
`LanceContextModelMaterializer(store).validate()` run domain-neutral conformance
checks over the standard relation set. `execute_context_model_tests(store, manifest)`
and `LanceContextModelMaterializer(store).execute_tests(manifest)` execute resolved
Context Model test declarations over those relations. The generic declarations
covered now include `identity_unique`, `snapshot_integrity`, `evidence_required`,
`lineage_complete`, `retrieval_indexed`, `authority_resolved`, and
`no_orphan_outputs`; unsupported domain-specific test declarations return structured
failed diagnostics instead of being silently ignored. The checks verify relation
presence, primary-key uniqueness, snapshot/object references, no orphan outputs,
retrieval projections, lineage closure, resolution candidate references, package item
references, package context references, assertion subject references, exactly one
contract binding for every published package, package-contract references, and
allow-only package publication. Results are structured rows so callers can surface all
failures without coupling reports to LanceDB internals.

`LanceStore` is structured-row-only. Namespaces organize and route table identity but
are not an authorization boundary. Binary/blob payloads follow the binary and blob
storage contract in the [IGOR specification](https://github.com/igor-context/spec) and
require a separately assigned and qualified adapter; this module does not expose blob
operations.

The runner's support smoke uses the configured `IGOR_LANCE_ROOT` and a namespaced
`LanceStoreConfig`; the live Compose service mounts `.igor/support-live-lance` at that
root so tables survive container exit. LanceDB's documented `merge_insert` pattern is
appropriate for future keyed bulk refreshes; immutable compiler outputs continue to
use compare-and-set publication through the adapter.

Verification: `docker compose run --rm --build lancedb-test`; remote backend qualification:
`docker compose --profile qualification run --rm --build lancedb-remote-smoke`.
