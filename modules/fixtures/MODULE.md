# Fixtures module

Own deterministic scenario-fixture generation and portable interchange. It may depend on core contracts and must not treat portable files as an operational database.

## Interface

`igor-fixtures scenario validate <scenario-directory> [--json]` validates a versioned scenario pack and its artifact hashes. `igor-fixtures scenario generate <output-directory>` writes the canonical support v0.1 pack. The Python interface exposes `generate_support_pack()` and `validate_scenario_pack()` for callers that need typed results.

Scenario-specific selections may reference one canonical context-model semantic
artifact with `context_model_ref`; callers resolve the file and use its content hash,
not the relative path, in execution identity.

## Invariants

- Source fixtures contain no gold labels, expected outcomes, or evaluator-only judgments.
- Context-unit identities are stable and unique within a pack.
- Mutation impact sets are deterministic, disjoint, and derived from declared source identities.
- Every declared artifact is present, stays within the pack, and matches its recorded hash and size.
- The scorecard defines a reproducible metric, population, abstention/no-answer treatment, and threshold.
- The pack uses namespaced support extensions and does not redefine core run-artifact identity.

## Configuration and artifacts

The command is offline and takes only a scenario directory or output directory. The pack produces JSON source, mapping, mutation, judgment, and scorecard artifacts under `scenarios/support/v0.1/`.

## Verification

```sh
docker compose run --rm --build scenario-test
docker compose run --rm --build scenario-validate
docker compose run --rm --build core-test
docker compose run --rm architecture-check
```
