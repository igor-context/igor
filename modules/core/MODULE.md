# Core module

Core now also owns the vendor-neutral `ContextSourceContract`, `ConnectorBinding`,
`ContentPart`, and `EnrichmentRecipe` values. Registered enrichment output descriptors
carry Draft 2020-12 JSON Schema; untrusted provider payloads must pass complete schema
validation before publication. Core defines these identities and constraints but does
not select source records, resolve credentials, dereference media, or interpret domain
taxonomies.

## Interface

Core presents tool-neutral typed contracts and deterministic functions. Callers may validate an independent embedding or completion profile through `igor-core profile validate-model <path> --capability <embedding|completion>`, resolve and validate a compiler composition through `igor-core profile validate <path>`, validate a complete run directory through `igor-core contract validate <path>`, validate a canonical ContextPackage artifact through `igor-core context validate <path>`, or export JSON Schemas through the corresponding schema commands. The public `igor_core` exports provide typed profile, manifest, artifact, stage, index, lineage, and Context IR models. Inputs are plain typed values or versioned files; outputs are validated values, canonical JSON, stable identities, or structured validation errors.

## Invariants

- Core imports no reference tool module or provider SDK.
- Canonical serialization is deterministic for equivalent values.
- Model-profile identities include every non-secret capability selection and parameter.
- Compiler compositions reject inline combined selections and capability-mismatched references.
- Resolved reference-profile identity depends on expanded component values, not filesystem paths.
- Unknown configuration fields fail validation.
- Core commands perform no network access.
- Every artifact has a schema version, relative path, producer identity, media type, byte size, integrity hash, and derived stable identity.
- A run identity links the benchmark contract, scenario pack, scorecard, implementation, source fixture, transformation configuration, and evaluator.
- A complete run rejects missing, mismatched, duplicate, undeclared, or unsafe artifact references.
- A declared `artifacts/lineage.json` is validated as one closed, run-scoped graph with stable node/edge identities and no undeclared endpoints.
- Context IR records are immutable, schema-qualified, identity-addressable, and tool-neutral; core validates structure and dependency declarations but does not derive, embed, enrich, retrieve, or compile.

## Configuration

Core validates embedding and completion profiles independently, resolves their references from a compiler composition, and returns one complete run-level `ReferenceProfile`. Capability owners later receive only their resolved configuration slice. Credentials never enter any tracked profile.

## Verification

```sh
docker compose run --rm --build core-test
docker compose run --rm --build core-contract
docker compose run --rm --build core-contract igor-core contract schema
docker compose run --rm --build core-context-contract
```
