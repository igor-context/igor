# Runner module

The deterministic document qualification is composed through the existing dlt,
parallel enrichment, Lance, and DataFusion seams. It sends verified original page
content and task text through a capability-qualified direct multimodal adapter. It
writes a portable JSON artifact and report under `.igor/document-qualification/`.

The document composition loads the scenario's approved domain schema and semantic
definition. The pinned Hugging Face source columns are described explicitly;
reference answers and supplied embeddings are forbidden from prompts. Relevant
`question_types` guidance and general answerability rules are projected into each
request. Context-model identity participates in output/work identity and live-cache
reuse, so a taxonomy revision cannot silently reuse stale enrichment.

Run the deterministic qualification with:

```text
docker compose run --rm --build document-qualification
```

Own the reference implementation's composition root: stage ordering, resolved configuration slices, adapter wiring, and run-artifact assembly. Compose invokes it; tool modules do not orchestrate one another.

Implemented interface:

```text
CompositionRoot.run(config, stages, output) -> validated RunPackage
materialize_context_model_lifecycle(...) -> ContextModelLifecycleResult
compile_context_model_lifecycle(...) -> ContextModelLifecycleResult
igor init|deps|validate|compile|plan|package
```

The `igor` CLI is the runner-owned user-facing composition layer for Context Projects.
It discovers `igor_project.yml`, loads project definitions, resolves deterministic
local dependencies, delegates manifest compilation to `igor_context_compiler`, renders
diagnostics, and writes generated artifacts under `.igor/target/`. `igor validate`,
`igor compile`, `igor plan`, and `igor package` are deterministic and must not
acquire source data, call providers, write Lance relations directly, or discover
credentials beyond presence-only checks in `plan --live`. `igor package` binds one
manifest-declared output role to its resolved Context Contract, delegates allow/deny/
abstain evaluation plus standard relation persistence to the lifecycle owner, and
fails closed when publication is not allowed.

The project CLI also exposes a deterministic chain of project-state commands:
`igor sync`, `igor build`, `igor diff`, `igor resolve`, `igor query`, `igor explain`,
`igor test`, `igor evaluate`, `igor qualify`, `igor observe`, `igor tune`,
`igor serve`, and `igor status`. They remain inside the project boundary, write
their JSON summaries under `.igor/target/`, and reuse the same manifest/validation
seams rather than introducing a second runtime path.

For live project execution, `igor sync --live`, `igor qualify --live`, and
`igor evaluate --live` may execute declarative command bindings from project
`runtime.yml`. The runner treats those bindings as neutral `igor-*` command
interfaces: it expands only project-bounded placeholders, executes without a shell,
captures stdout/stderr/exit code, records produced artifact hashes, and writes the
summary under `.igor/target/`. Source adapters, scenario qualification commands, and
evaluators remain separately owned executables; the project CLI does not import their
tool modules or encode their vendor/domain semantics.

`igor query --sql <SQL>` executes analytical SQL over a project-bounded Lance
relation store through the DataFusion owner. By default it opens
`.igor/runtime/qualification/allowed-relations`; `--store` may select another
project-relative relation store. The query artifact records the DataFusion engine,
available relation names, columns, row count, and JSON-safe rows under
`.igor/target/query.json`. The project CLI validates that the store remains inside
the project boundary and does not special-case a source vendor, dataset, or scenario
domain. SQL may also call the DataFusion-owned `semantic_search(...)` table function;
the runner command seam wires that function to the LanceDB retrieval adapter rather
than making the project CLI import storage or retrieval modules.

`RunConfig` resolves a non-secret compiler composition and its independently referenced embedding and completion profiles into one `ReferenceProfile` and the required run identity links. Reference paths are authoring details; the fully expanded component values participate in transformation identity. The runner also carries `DomainConfig` plus `StructuredStorageConfig` values. The domain/environment pair deterministically forms the structured-storage namespace; the provider-neutral root URI, lifecycle class, isolation tier, and optional credential reference remain a runtime configuration slice, not model-profile data. `StageSpec` is the typed port for a stage; a handler receives the artifacts produced by earlier stages and returns JSON artifact payloads. The root executes stages in the declared order, records input/output artifact identities, writes the manifest and index, and validates the complete run through the core contract. The support smoke wires fixture loading, domain-aware dlt ingestion, LanceStore/DataFusion relational materialization, and required artifact assembly through this seam. Stages may attach core lineage facts to `ArtifactPayload.lineage`; the root merges them into the single `artifacts/lineage.json` run artifact and validates graph closure through core.

The v0.1 runner composes only active owner interfaces. Its support prepare stage invokes the active dlt ingestion port, persists canonical rows through LanceStore, and passes them to DataFusion/relational stages. The support compiler stage creates qualified Context IR, invokes the compiler with LanceDB storage/retrieval adapters, and emits compiler artifacts. Evaluator execution remains a separate terminal process; evaluator code is never imported by runner code. dbt-lance is not a runtime dependency. Stage failures are surfaced as `RunnerError` and do not produce a successful run package.

The Context Model lifecycle seam is the runner-owned composition shell over the
manifest compiler direction and active owner modules. `materialize_context_model_lifecycle`
accepts a resolved manifest object, portable manifest artifact, or normalized mapping
with runtime observations, outputs, assertions, retrieval facts, typed resolution
requests or manifest-derived resolution inputs plus a domain resolution port, package
specs or package inputs for overrides, lineage facts, and already materialized facts
when needed. It executes the reusable post-retrieval resolution/package slice, then
delegates standard relation persistence to
`LanceContextModelMaterializer` and exposes the LanceDB-owned declared-test execution
result on `ContextModelLifecycleResult.test_execution`. Runtime outputs produce
generic output assertions through the LanceDB materializer, and the lifecycle also
derives standard `lineage_edges` for post-retrieval decisions and package inclusion
from the resolved request/result/package rows, so callers do not hand-author generic
assertion rows or the readable resolution-to-package chain. `compile_context_model_lifecycle`
first materializes observations and retrieval projections, executes declared compiler
derivations through `plan`/`execute`, preserves the compiler output-store JSON values,
then refreshes standard relations with resolution decisions, candidates, lineage,
packages, SQL-ready relation names, and manifest-declared test results. Context
Compiler remains Lance-free; LanceDB remains the storage/materialization owner.
When callers do not supply a full `CompilationRequest`, the lifecycle can build one
from the resolved manifest's declared derivation roles plus runtime role identity
bindings or per-instance derivation facts, runtime model profiles, embedding space,
and code revision. That path lets a runner execute declared derivations without
hand-authoring `DerivationSpec` rows. The support compiler path now supplies
per-record derivation facts and runtime content to this lifecycle builder instead of
constructing its own compiler request. The `image-context` composition uses the same
path for image representations, task projections, embeddings, enrichments, and package
compilation. For selective-invalidation checks, the lifecycle can return the compiler
plan without executing providers.
Runtime retrieval inputs for manifest-declared retrieval roles are compiled into
`RetrievalQuery` values through the Context Compiler helper; compiler retrieval facts
are then projected into the standard `resolution_candidates` relation even before a
domain policy is asked to select or reject them.
The materialization seam also accepts repeated runtime retrieval input rows. Each row
names a manifest retrieval role plus task metadata; the lifecycle compiles the typed
retrieval query, invokes the supplied retrieval port, returns the executed retrieval
rows, and uses those rows to derive resolution requests and default packages. The live
document pipeline now uses this seam instead of constructing `RetrievalQuery` objects
in scenario code. Replay retrieval rows used only to derive resolution requests are
not written to `context_retrieval` unless they carry a vector or explicitly request
retrieval materialization, so standard retrieval indexes contain searchable
projections rather than policy-case metadata.
Runtime resolution inputs for manifest-declared retrieval roles are compiled into
typed `ResolutionRequest` values through the Context Compiler helper; the lifecycle can
then invoke the supplied domain `ResolutionPort` or validate supplied resolution
results and materialize the resulting standard resolution rows without callers
hand-building compiler request objects. If a caller supplies a `CompilationRequest`
without an attached resolution request, the lifecycle attaches the manifest-derived
resolution request before executing the compiler. When explicit resolution inputs are
absent, runtime retrieval rows that carry a manifest `retrieval_role`, task identity,
`as_of`, and candidate context identity are lifted into typed resolution requests
before the domain port runs. The support compiler path uses this lifecycle helper to
attach its compiler `ResolutionRequest` from the resolved support manifest,
preserving the support domain's policy ID through manifest authority settings. Its
showcase lifecycle replay turns the compiler's retrieval-enriched portable resolution
artifact into retrieval rows and lets the lifecycle derive typed request objects for
standard relation materialization. The support resolution showcase now does the same
for its conflict/authority/temporal examples instead of constructing typed resolution
requests in scenario code.
When no explicit package specs, package rows, or package inputs are supplied, the
lifecycle derives default `ContextPackage` items from selected resolution candidates,
preserving candidate evidence IDs and task/as-of metadata without callers
hand-building package specs. Package inputs remain supported for task-specific
overrides. The support showcase, document qualification, and image qualification now
rely on this default package derivation for standard `context_packages` and
`package_items`.
The lifecycle already contains reusable Context Contract evaluation helpers and
scenario-level tests for allow/deny/abstain package decisions. The project CLI now
reuses that lifecycle through `igor package`: a request file names one manifest output
role plus runtime package inputs, the runner resolves the declared contract from the
compiled manifest, and the lifecycle evaluates allow/deny/abstain semantics before
publishing any package rows. Standard relation writes still occur only inside the
lifecycle/materializer owner.
For compiler execution, the lifecycle also derives compiler package items from
`required_context_identities` when callers do not provide explicit `ContextItem`
objects. Support and image compiler paths now pass required context identities instead
of constructing package items; cached required identities remain packageable even when
they are reused from inventory and are not re-emitted as fresh derivation outputs.
`ContextModelLifecycleResult.query_standard_sql(sql, ...)` exposes the materialized
standard relations through the DataFusion owner while the runner supplies the
LanceDB-backed retrieval adapter for `semantic_search(...)`.
The support, document, image, and bounded semantic-SQL lifecycle paths now build their
model with the Context Compiler's `compile_context_model` manifest compiler before
handing observations and runtime outputs to this seam.
The support, document, and image qualifications now pass retrieval rows instead of
manifest-keyed resolution inputs for their standard lifecycle replay. No path
constructs the standard resolution and package rows itself.
Their standard `context_catalog`, retrieval, and readable-lineage SQL reads go through
the lifecycle result instead of opening a separate scenario-specific standard-relation
SQL path.
The live document pipeline's delivery SQL also joins the lifecycle-maintained
`resolution_decisions`, `resolution_candidates`, `context_packages`, and
`package_items` relations directly, instead of writing custom document resolution or
package mirror tables after replay.
The architecture harness now rejects direct `create`, `replace`, or `add` calls for
standard Context Model relation names outside the LanceDB materializer and runner
lifecycle owner file, so scenario runners cannot reintroduce manual standard relation
maintenance.

`semantic_sql_query(store, sql)` is the semantic SQL composition seam. It wires the
LanceDB-owned retrieval adapter into DataFusion's vendor-neutral `semantic_search`
table function; callers provide SQL and do not learn LanceDB configuration.

The bounded live qualification is:

```text
docker compose --profile qualification run --rm --build semantic-sql-live
```

It resolves its small support Context Model through the manifest compiler before
materializing `context_catalog` and `context_retrieval`.

Verification:

```text
docker compose run --rm --build runner-test pytest tests/test_project_cli.py -q
docker compose run --rm --build runner-test
docker compose run --rm --build context-compiler-test
docker compose run --rm --build runner-smoke
docker compose run --rm --build igor-project-smoke
docker compose run --rm --build architecture-test
docker compose run --rm architecture-check
docker compose run --rm docs-site validate
```

The composition root resolves the tracked compiler profile and selects deterministic
providers for the reproducible profile or the Context Compiler's Voyage embedding and
DeepSeek enrichment adapters for `config/compiler/live-v0.yaml`. The opt-in live
support run is:

```text
docker compose --profile qualification run --rm --build support-live
```

It requires only `IGOR_EMBEDDING_API_KEY` and `IGOR_COMPLETION_API_KEY` in the ignored
Context Compiler `.env` file; credentials never enter profiles or artifacts.

The pinned Hugging Face support qualification consumes a portable acquisition
artifact through the runner:

```text
docker compose --profile qualification run --rm --build huggingface-qualification
docker compose run --rm --build support-hf-smoke
```

The qualification runner's showcase stage materializes the standard Context Model
relations through the generic lifecycle seam, then emits `artifacts/semantic-sql.json`
from the same `context_catalog` and `context_retrieval` relations. The qualification
evaluator consumes that portable artifact:

```text
docker compose --profile qualification run --rm --build support-hf-live-e2e
docker compose --profile qualification run --rm --build semantic-sql-evaluate
```

The `image-context` composition demonstrates how a second domain enters without a
compiler branch. The media package supplies its source contract, fixture connector
binding, image and projection representations, output schema, and recipe. The mock
command proves metadata-only selective invalidation through the lifecycle-built plan;
the live command sends actual PNG bytes to Voyage and the declared task content to
DeepSeek, persists both semantic outputs in LanceDB, and compiles one valid package:

```text
docker compose run --rm image-context-e2e
docker compose run --rm image-context-live
```

The ordinary-image Hugging Face qualification consumes the image acquisition artifact
through the same dlt-bound source and direct multimodal ports. It is capped at 12
deterministic images and a smaller live slice under the embedding request ceiling:

```text
docker compose run --rm image-huggingface-deterministic
docker compose --profile qualification run --rm image-huggingface-live
```

The image qualification loads its context-model declaration from the scenario semantic
definition, resolves it through the Context Compiler manifest compiler, and supplies
that manifest object with observations, runtime outputs, and retrieval rows to the
generic lifecycle materialization seam. The lifecycle derives typed resolution
requests from the manifest policy, executes the domain resolution port, compiles
default package rows from selected decisions, and derives readable lineage from output
dependencies; the portable SQL and HTML report retain canonical SHA identities while
adding configuration-driven readable lineage names.

The bounded ReCaRe qualification is an ordinary IGOR project under
`scenarios/recare-legal/v0.1`. Its source mapping, schemas, taxonomy, recipes, model
profiles, authority policy, retrieval, Context Contract, outputs, and evaluations are
declarative. The runner compiles that project, derives snapshot times from source
revision identifiers, indexes vector-bearing standard outputs once, retrieves through
LanceDB ANN, resolves authority and time, and publishes only contract-allowed packages.
The generic Hugging Face row selector acquires the live slice; the independent
authoritative-context evaluator consumes only the serialized qualification artifact.

```text
docker compose --profile qualification run --rm recare-deterministic-qualification
docker compose --profile qualification run --rm recare-authoritative-evaluate
```
