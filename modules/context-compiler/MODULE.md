# Context Compiler module

Reserve this capability for the IGOR-native Context Compiler defined by the
Context Compiler module interface in the [IGOR specification](https://github.com/igor-context/spec).
It is the IGOR-native owner of deterministic planning, selective invalidation,
provider-neutral semantic execution, and context-package compilation.

Resolution uses the typed `ResolutionRequest`/`ResolutionResult` seam. The compiler
validates policy identity and revision, UTC `as_of`, candidate permutation stability,
assertion/evidence references, complete ordered decisions, explicit outcome/basis
codes, and resolution artifacts. Domain packages implement `ResolutionPort`; they
own authority and temporal semantics and are registered by the runner.

Implemented interface:

```text
plan(request, inventory) -> CompilationPlan
execute(plan, ports) -> CompilationResult
compile_context_model(request) -> ResolvedContextModelManifest
```

`plan` is pure and deterministic. `execute` only invokes work declared by the plan,
validates core values before publication, performs typed retrieval/resolution and
deterministic budgeting, and reuses immutable identities through supplied ports. Ports
are provider-neutral protocols; LanceDB adapters and the runner remain outside this
module.

Semantic provider calls use `EmbeddingRequest` and `CompletionRequest`. Each request
contains qualified `Representation` identities, ordered typed `ContentPart` values, the
capability-typed `ModelProfile`, and the relevant embedding space or output schema.
Providers return `ProviderOutcome` values with a typed status, bounded attempt count,
and non-secret `ProviderMetadata`; they never publish Context IR records. The compiler
constructs and validates core `Embedding` and `EnrichmentOutput` values before calling
`OutputStore.publish`. Runtime content is excluded from plan/artifact payloads and is
represented only by content-part identities in plan identity material. Completion
requests also carry a versioned `EnrichmentRecipe`; the compiler validates accepted
representation/media types, evidence, prompt/taxonomy revision, output-schema identity,
and the complete registered JSON Schema before publishing an enrichment.

Context Model manifest compilation is the first executable slice of the declarative
Context Model contract. `compile_context_model` accepts a mapping-shaped authoring
declaration plus an identity-bearing reference registry and returns a
`ResolvedContextModelManifest`. It resolves source contracts, connector bindings,
schemas, semantic definitions, recipes, model profiles, authority policies,
Context Contracts, retrievals, outputs, evaluation declarations, presentation
declarations, and generic tests into stable identities; rejects duplicate roles,
missing references, unknown object kinds, dependency cycles, and malformed sections;
and emits a portable `resolved-context-model-manifest` artifact dictionary.
Display-only fields such as title and `display_name` remain in the manifest for
reports but do not affect semantic identity. This slice does not materialize Lance
relations, execute retrieval/resolution, or publish packages from the manifest.

`compile_context_model_derivations(manifest, role_identities, code_revision=...)`
compiles declared semantic, retrieval-projection, and authority-assertion object roles
into typed `DerivationSpec`s from runtime role identity bindings. It is deterministic
and fails closed when a declared dependency role has no runtime identity. The helper
does not acquire source data, construct provider request content, import storage, or
execute the derivations; the runner supplies runtime profiles, ports, and content.

`compile_context_model_derivation_instances(manifest, instances, code_revision=...)`
compiles multiple runtime instances of declared derivation roles into typed
`DerivationSpec`s. Each instance names the manifest role, runtime input identities,
and optional output/configuration/model identities. This lets the runner execute
per-record or per-object compiler work from a resolved Context Model manifest without
hand-authoring compiler derivation rows.

`compile_context_model_retrieval_queries(manifest, retrieval_inputs)` compiles
manifest-declared retrieval roles into typed `RetrievalQuery` values from runtime query
text/vector inputs. It preserves the manifest's search mode and candidate limit,
derives stable query identities when none are supplied, and rejects unknown retrieval
roles. It does not execute retrieval or resolve authority; the runner wires the
retrieval and resolution ports.

`compile_context_model_resolution_request(manifest, ...)` compiles a
manifest-declared retrieval's resolution policy into a typed `ResolutionRequest` from
runtime task metadata, UTC `as_of`, candidates, evidence, and assertion envelopes. It
uses the retrieval's `resolution.policy` role to select the resolved authority policy,
preserves the policy reference and revision, honors an explicit manifest `policy_id`
when a domain policy ID differs from the registry reference, produces stable request
identities, sorts candidate/evidence/assertion identities deterministically, and fails
closed when the retrieval role is unknown or has no resolution policy. It does not
execute domain policy logic; qualified domain packages still implement
`ResolutionPort`.

The live qualification adapters are `VoyageMultimodalEmbeddingAdapter`,
`DeepSeekCompletionAdapter`, `GeminiCompletionAdapter`, and
`MistralCompletionAdapter`. They validate the resolved capability profile before
making a request, discover only the capability-scoped runtime key, classify
HTTP/transport/shape failures into typed outcomes, and preserve profile revisions
plus provider response metadata. The opt-in qualification command writes only
redacted results under `.igor/`.

Voyage translates interleaved text and image parts and verifies referenced bytes
against their declared SHA-256 before Base64 dispatch. DeepSeek completion accepts
ordered text, image, and document parts and sends verified binary content alongside
task text; unsupported content kinds fail before provider execution. Gemini uses the
same ordered content contract with the REST `generateContent` API and the
`GEMINI_API_KEY` environment variable. Completion is a multimodal capability boundary,
not a text-only projection boundary.

Mistral completion accepts verified text, image, and PDF document parts through the same
neutral request type, but the tracked profiles are workload-specific. Each economy
(`ministral-8b-2512`) and general (`mistral-small-2603`) tier has separate text, direct
vision, and document-QnA profiles. The adapter rejects any content kind outside the
selected profile before provider execution. All use JSON mode and retain IGOR's
independent output-schema validation. PDF Document QnA is a provider-managed composition
that internally uses Mistral Document AI/OCR; IGOR does not insert, store, or require a
separate OCR projection. A standalone OCR output remains a separate optional capability.

Provider and model selections are independent, tracked, non-secret model profiles.
`config/reference.yaml` composes the deterministic embedding and completion profiles for
regression. `config/compiler/live-v0.yaml` composes the independently reusable
`config/providers/embeddings/voyage-multimodal-3.yaml` and
`config/providers/completions/deepseek-v4-flash.yaml` selections for live qualification.
The Gemini alternative is composed by `config/compiler/live-gemini-v0.yaml` and uses
the `gemini-3.1-flash-lite` model.
The `live-{text,image,document}-{economy,general}-v0.yaml` compiler compositions select
one workload-qualified Mistral completion profile alongside the independent embedding
selection. Their names describe workload and service tier rather than creating permanent
vendor pairs.
The completion API is infrastructure used by IGOR's enrichment operation; it is not the
enrichment schema, prompt, or taxonomy itself.

`embedding_adapter_for(profile)` and `completion_adapter_for(profile)` keep provider
dispatch inside the Context Compiler owner. Runners select validated capability
profiles and never branch directly on vendor names.
Provider adapters apply profile-declared retry and batching controls at this same
boundary: transient and timeout outcomes can be retried up to `max_attempts`, and
embedding requests can be split by `max_batch_size` without domain- or vendor-specific
caller logic.

Copy `.env.example` to `.env` and configure only runtime credentials:

```text
IGOR_EMBEDDING_API_KEY
IGOR_COMPLETION_API_KEY
GEMINI_API_KEY
```

Public provider endpoints are tracked alongside model behavior in each model profile. A
private, credential-bearing gateway URL must use a future opaque runtime credential
or deployment reference instead of entering tracked configuration.

The real `.env` is ignored. Environment values must not override provider, model,
revision, or inference parameters because those values participate in profile and
transformation identity. Non-secret request batching and throttling settings are also
tracked in its model profile so a run's complete provider behavior can be reproduced.
API keys and secret-bearing endpoint details must not enter model/compiler profiles, plans,
artifacts, lineage, logs, or tracked documentation.

The compiler-owned `runtime` seam adds provider-neutral `EnrichmentWorkItem` values,
stable compatible `BatchPlan` values, `RuntimeLimits`, an atomic
`InProcessQuotaCoordinator`, and `run_enrichment`. Work is sorted by identity for
publication, while independent calls run concurrently up to the declared ceiling.
Batch retries isolate transient members, and direct ports plus bounded agent runners
adapt to the same `ProviderOutcome` shape. Runtime artifacts contain only work and
batch identities, redacted accounting, attempt statuses, queue/provider timing, CPU,
and peak-memory measurements; content and credentials never enter them.

Verification:

```text
docker compose run --rm --build context-compiler-test
docker compose run --rm --build context-compiler-contract
docker compose run --rm --build context-compiler-runtime-qualification
docker compose run --rm --build runner-test
docker compose run --rm --build architecture-test
docker compose run --rm architecture-check
docker compose --profile qualification run --rm --build context-compiler-live-qualification
docker compose run --rm image-context-e2e
docker compose run --rm image-context-live
```
