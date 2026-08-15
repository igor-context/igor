# relational module

Own the IGOR-native relational model runner. A versioned manifest describes SQL models, dependencies, output relations, and tests. `RelationalRunner` executes each model through `AnalyticalEngine`, preserves the Arrow result, and persists it through `LanceStore`.

Implemented interface:

```text
RelationalRunner(engine, store).run(manifest) -> RelationalRun
```

`RelationalRun.lineage_graph(run_identity, manifest)` emits portable stored-relation dependency facts. The runner does not own a graph store.

The runner rejects cycles, unknown dependencies, unsupported materializations, failed assertions, and empty model results. It does not ingest sources, perform retrieval, initialize models, or execute SQL through another engine.

Verification:

```text
docker compose run --rm --build relational-test
docker compose run --rm --build relational-smoke
docker compose run --rm architecture-check
```
