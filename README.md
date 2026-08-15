# IGOR — AI Context Infrastructure

IGOR continuously turns changing organizational data into reliable, traceable,
task-ready context for AI agents. It is not a retrieval wrapper: it preserves what
was observed, tracks what changed, records how meaning was derived, resolves which
sources are authoritative and valid for the task's time, and compiles a bounded
context package with evidence — before an agent ever sees it.

## The problem

Agents don't usually fail because the model is weak. They fail because the context
they receive is wrong: a stale fact, a superseded document version, a source that
was never authoritative for the question being asked. Retrieval alone answers "what
might be relevant." It does not answer whether that candidate is *allowed*, whether
it was *valid at the time the task cares about*, or *why* it was selected.

## What IGOR does

- **Traceable** — every context unit, derivation, and package carries an 8-layer
  lineage: source → snapshot → enrichment → retrieval → resolution → contract →
  package → item. You can always answer "where did this come from and how was it
  derived."
- **Temporally correct** — the same data queried at a different time can produce a
  different, correct answer. Authority and temporal resolution are explicit policy
  decisions, not incidental retrieval ranking.
- **Contract-governed** — publication is allow, deny, or abstain, and fails closed.
  Nothing reaches an agent that the governing Context Contract didn't explicitly
  permit.
- **Selectively invalidated** — identity is content-addressed (SHA-256) at every
  layer, so a change invalidates only what actually depends on it. Nothing is
  re-derived on a blind TTL.

## Quick start

```sh
docker compose run --rm --build igor-project-cli igor init recare-cli --template /opt/scenarios/recare-legal/v0.1
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor deps'
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor sync --live'
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor resolve'
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor qualify --live'
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor evaluate --live'   # 18/18 checks
```

`--live` steps call real embedding and completion providers and need
`IGOR_EMBEDDING_API_KEY` and `IGOR_COMPLETION_API_KEY` in
`modules/context-compiler/.env` (gitignored, never commit it). Everything else in the
repo runs deterministically, with no network access and no credentials:

```sh
docker compose run --rm runner-test
docker compose run --rm architecture-check
```

## The ReCaRe qualification

`scenarios/recare-legal/v0.1` is a real end-to-end scenario over the [ReCaRe
dataset](https://huggingface.co/datasets/kasys/ReCaRe) — EU legal amendment records
pinned to a fixed Hugging Face revision. A live run through the full CLI chain above
produced:

| Metric | Value |
| --- | --- |
| Rows acquired (pinned revision) | 8 |
| Bytes acquired | 183,336 |
| Article snapshots resolved | 16 |
| Qualification wall time | 16.8s |
| Lineage nodes / edges | 29 / 33 |
| Independent evaluation checks | **18 / 18 passed** |
| DataFusion semantic query (`amendment rationale`) | 3 ranked rows |

Every check — artifact integrity, authority correctness, temporal correctness,
snapshot preservation, native semantic retrieval, fail-closed publication, mutation
invalidation, deterministic ordering, and more — passed against the independent
evaluator, not a self-report from the runner that produced the qualification.

Run the same DataFusion semantic SQL yourself:

```sh
docker compose run --rm igor-project-cli sh -c 'cd recare-cli && igor query --target semantic-filter --sql "SELECT display_name, semantic_score FROM semantic_search(\"context_catalog\", \"amendment rationale\", \"full_text\", 5) ORDER BY retrieval_rank"'
```

## Architecture

Each module owns one capability and only imports the modules it declares in its
`module.toml`:

| Module | Owns |
| --- | --- |
| `core` | Identity, hashing, canonical serialization, benchmark contracts |
| `context-compiler` | Semantic derivation, selective invalidation, retrieval resolution, budgeting, `ContextPackage` compilation |
| `lancedb` | Lance relations, vector indexes, retrieval and storage adapters |
| `datafusion` | Analytical SQL execution over Lance relations |
| `dlt` | Source ingestion, normalization, merge keys via dlt's Lance destination |
| `relational` | Declarative relational models and lineage over DataFusion and Lance |
| `huggingface` | Pinned dataset discovery, bounded acquisition, content identity verification |
| `evaluator` | Artifact-only conformance and scenario-quality scoring |
| `fixtures` | Deterministic fixture creation and portable interchange |
| `runner` | Reference-stage composition, ordering, and run-artifact assembly — the project CLI's implementation |

`core` stays tool-neutral; `runner` is the composition root that wires everything
together for a given scenario. No module imports outside its declared dependencies —
`harnesses/architecture` enforces this on every change.

## Specification

The normative contracts IGOR implements — identity, lineage, temporal resolution,
Context Contracts, evaluation — live in the implementation-neutral spec:
[github.com/igor-context/spec](https://github.com/igor-context/spec).

## Project structure

```
modules/            capability-owned reference implementation modules
scenarios/           versioned scenario packs (recare-legal, document-huggingface)
harnesses/           repository-wide architecture and integration checks
config/              non-secret model and compiler profiles
scripts/             utility scripts
compose.yaml         the Docker Compose runtime — the only project execution topology
```

A scenario package declares everything domain-specific in one place:

```
scenarios/recare-legal/v0.1/
├── igor_project.yml       project identity and context model selection
├── packages.yml           dependency lock manifest
├── sources/                source contract and connector binding
├── schemas/                domain representation schema
├── enrichments/            enrichment recipe and taxonomy
├── policies/                authority and temporal resolution policy
├── retrievals/              retrieval definition
├── contracts/                Context Contract (allow/deny/abstain rules)
├── models/                   embedding and completion provider profiles
├── profiles/                 resolved compiler profile
└── evals/                    evaluation fixtures
```

No IGOR core module contains ReCaRe-specific code — the scenario package is the only
place domain meaning lives.

## License

Apache 2.0. See [LICENSE](LICENSE).
