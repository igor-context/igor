# Evaluator module

The evaluator owns artifact-only conformance, deterministic scenario scoring, and operational warnings. Its command is:

```text
igor-evaluator evaluate <scenario-pack> <run-directory> [--json] [--output <path>]
```

It reads the scenario manifest, declared scenario artifacts, the submitted run manifest/index/stages, and the run's declared artifacts. It imports only `igor_core` and never imports or inspects a reference implementation.

The v0.1 support seam expects the required `metrics` artifact to be JSON with `predictions`, each containing `context_unit_id` and either `intent_label` or `abstain: true`. It emits a conformance matrix, deterministic metric vector, operational warnings, and a human-readable summary. Invalid submissions return a non-zero exit code while preserving any safely computable diagnostic quality metrics.

The `report` command renders an artifact-only visual report with a result-first
summary, pipeline flow, ticket-to-enrichment table, retrieval and mutation charts,
resolution outcomes, an SVG lineage path, five SQL result tables, a scorecard chart,
and an explicit operational warning. It reads only the submitted run directory and
evaluator JSON; it does not inspect live Lance state or call providers.

The `semantic-sql` command independently validates bounded semantic-SQL artifacts
for complete evidence rows, native predicates, indexed retrieval, score ordering,
composition path, and the no-full-table-scan proof.

Verification:

```text
docker compose run --rm --build evaluator-test
docker compose run --rm evaluator-fixture
docker compose run --rm architecture-check
```
