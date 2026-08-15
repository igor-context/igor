from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, Any, Literal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from time import sleep

from pydantic import BaseModel, ConfigDict, Field, model_validator

from igor_core import (
    ContentPart, ContextItem, ContextPackage, Derivation, Embedding, EmbeddingSpace,
    EnrichmentOutput, EnrichmentRecipe, Evidence, ModelProfile, Representation,
    SchemaDescriptor, stable_identity, validate_schema_payload,
)

Identity = str


def representation_identity(value: Representation) -> Identity:
    return value.identity

FailureCategory = Literal[
    "contract_invalid", "plan_conflict", "unsupported_operation", "port_unavailable",
    "missing_content", "provider_rejected", "transient_failure", "timeout",
    "malformed_output", "retry_exhausted", "persistence_conflict", "budget_unsatisfied",
]

OutcomeStatus = Literal[
    "succeeded", "abstained", "transient_failure", "permanent_rejection",
    "timeout", "malformed_output", "retry_exhausted",
]


class CompilerFailure(ValueError):
    def __init__(self, category: FailureCategory, message: str):
        super().__init__(f"{category}: {message}")
        self.category = category


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: Identity
    query_type: Literal["vector", "full_text", "hybrid"]
    text: str = ""
    vector: tuple[float, ...] = ()
    limit: int = Field(default=10, gt=0)
    space_identity: str | None = None


class RetrievalFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    result_identity: Identity
    context_identity: Identity
    score: float
    rank: int = Field(ge=0)
    facts: dict[str, str] = {}


class AuthorityAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    assertion_identity: Identity
    subject_identity: Identity
    issuer_ref: str
    scope_ref: str
    evidence_identities: tuple[Identity, ...] = Field(min_length=1)


class TemporalAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    assertion_identity: Identity
    subject_identity: Identity
    observed_at: datetime | None = None
    effective_from: datetime | None = None
    effective_until: datetime | None = None
    evidence_identities: tuple[Identity, ...] = Field(min_length=1)


class ResolutionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_identity: Identity
    context_identity: Identity
    retrieval: RetrievalFact | None = None
    evidence_identities: tuple[Identity, ...] = ()
    authority_assertion_ids: tuple[Identity, ...] = ()
    temporal_assertion_ids: tuple[Identity, ...] = ()


class ResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_identity: Identity
    ir_version: str = "0.1"
    task_id: str
    task_schema_revision: str
    as_of: datetime
    candidates: tuple[ResolutionCandidate, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    authority_assertions: tuple[AuthorityAssertion, ...] = ()
    temporal_assertions: tuple[TemporalAssertion, ...] = ()
    policy_id: str
    policy_revision: str

    @model_validator(mode="after")
    def validate_temporal_clock(self) -> "ResolutionRequest":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be a timezone-aware UTC instant")
        if self.as_of.utcoffset().total_seconds() != 0:
            raise ValueError("as_of must be UTC")
        return self

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "request_identity": self.request_identity,
            "ir_version": self.ir_version,
            "task_id": self.task_id,
            "task_schema_revision": self.task_schema_revision,
            "as_of": self.as_of.isoformat(),
            "candidates": [item.model_dump(mode="json") for item in sorted(self.candidates, key=lambda item: item.candidate_identity)],
            "evidence": [item.model_dump(mode="json") for item in sorted(self.evidence, key=lambda item: item.identity)],
            "authority_assertions": [item.model_dump(mode="json") for item in sorted(self.authority_assertions, key=lambda item: item.assertion_identity)],
            "temporal_assertions": [item.model_dump(mode="json") for item in sorted(self.temporal_assertions, key=lambda item: item.assertion_identity)],
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
        })


ResolutionOutcome = Literal["selected", "rejected", "abstained", "conflict"]
AuthorityBasis = Literal["satisfied", "not_satisfied", "unknown", "conflicting"]
TemporalBasis = Literal["valid", "expired", "future", "unknown", "conflicting"]


class ResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_identity: Identity
    outcome: ResolutionOutcome
    reason_code: str
    authority_basis_ids: tuple[Identity, ...] = ()
    temporal_basis_ids: tuple[Identity, ...] = ()
    authority_basis: AuthorityBasis
    temporal_basis: TemporalBasis
    as_of: datetime
    policy_id: str
    policy_revision: str
    conflict_set_id: Identity | None = None

    @property
    def selected(self) -> bool:
        return self.outcome == "selected"


class ResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_identity: Identity
    resolution_identity: Identity
    decisions: tuple[ResolutionDecision, ...]
    selected_identities: tuple[Identity, ...] = ()
    rejected_identities: tuple[Identity, ...] = ()
    abstained_identities: tuple[Identity, ...] = ()
    conflict_identities: tuple[Identity, ...] = ()
    error: str | None = None


class RetrievalPort(Protocol):
    def retrieve(self, query: RetrievalQuery) -> tuple[RetrievalFact, ...]: ...


class ResolutionPort(Protocol):
    def resolve(self, request: ResolutionRequest) -> ResolutionResult: ...


class DerivationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: str
    input_identities: tuple[Identity, ...] = Field(min_length=1)
    output_identity: Identity
    code_revision: str
    configuration_identity: str
    model_revision: str | None = None

    def as_core(self) -> Derivation:
        return Derivation(
            ir_version="0.1", operation=self.operation,
            input_identities=self.input_identities, output_identity=self.output_identity,
            code_revision=self.code_revision, configuration_identity=self.configuration_identity,
            model_revision=self.model_revision,
        )


class DependencyInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    derivations: tuple[DerivationSpec, ...] = ()
    known_output_identities: tuple[Identity, ...] = ()
    valid_output_identities: tuple[Identity, ...] = ()
    changed_identities: tuple[Identity, ...] = ()

    @property
    def identity(self) -> Identity:
        return stable_identity(self)


class CompilationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_identity: Identity
    task_id: str
    required_output_identities: tuple[Identity, ...] = Field(min_length=1)
    derivations: tuple[DerivationSpec, ...] = ()
    embedding_profile: ModelProfile
    completion_profile: ModelProfile
    embedding_space: EmbeddingSpace
    embedding_inputs: tuple["QualifiedRepresentation", ...] = ()
    completion_inputs: tuple["QualifiedRepresentation", ...] = ()
    completion_output_schema: SchemaDescriptor | None = None
    enrichment_recipe: EnrichmentRecipe | None = None
    prompt_version: str = "0.1"
    taxonomy_version: str = "0.1"
    code_revision: str
    budget_tokens: int | None = Field(default=None, gt=0)
    package_schema_revision: str = "0.1"
    package_items: tuple[ContextItem, ...] = ()
    retrieval_queries: tuple[RetrievalQuery, ...] = ()
    required_context_identities: tuple[Identity, ...] = ()
    resolution: ResolutionRequest | None = None

    @property
    def profile_identity(self) -> str:
        return stable_identity({"embedding": self.embedding_profile, "completion": self.completion_profile})

    def identity_material(self) -> dict[str, object]:
        """Return plan identity inputs without runtime source content."""
        value = self.model_dump(mode="json", exclude={"embedding_inputs", "completion_inputs"})
        value["embedding_inputs"] = [item.identity_material() for item in self.embedding_inputs]
        value["completion_inputs"] = [item.identity_material() for item in self.completion_inputs]
        return value


class QualifiedRepresentation(BaseModel):
    """A qualified IR identity paired with resolved, ephemeral runtime content."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    representation: Representation
    parts: tuple[ContentPart, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()

    @property
    def identity(self) -> Identity:
        return representation_identity(self.representation)

    def identity_material(self) -> dict[str, object]:
        return {
            "representation": self.representation.identity,
            "content_parts": [part.identity for part in self.parts],
        }

    @property
    def text(self) -> str:
        return "\n\n".join(part.text for part in self.parts if part.kind == "text" and part.text)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_identity: Identity
    input: QualifiedRepresentation
    space: EmbeddingSpace
    profile: ModelProfile


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_identity: Identity
    inputs: tuple[QualifiedRepresentation, ...] = Field(min_length=1)
    output_schema: SchemaDescriptor
    recipe: EnrichmentRecipe
    prompt_version: str
    taxonomy_version: str
    profile: ModelProfile


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_attempts: int = Field(default=1, ge=1, le=8)
    backoff_seconds: float = Field(default=0.25, ge=0, le=60)


class ProviderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str
    model: str
    response_id: str | None = None
    usage: dict[str, int] = {}
    provider_revision: str | None = None
    model_fingerprint: str | None = None


class ProviderOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    status: OutcomeStatus
    attempts: int = Field(ge=1)
    value: object | None = None
    metadata: ProviderMetadata | None = None
    error: str | None = None


class PlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_identity: Identity
    operation: str
    input_identities: tuple[Identity, ...]
    dependencies: tuple[Identity, ...]
    reusable: bool


class CompilationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_identity: Identity
    inventory_identity: Identity
    invalidation_seeds: tuple[Identity, ...]
    invalidation_closure: tuple[Identity, ...]
    cache_hits: tuple[Identity, ...]
    nodes: tuple[PlanNode, ...]
    expected_output_identities: tuple[Identity, ...]

    @property
    def identity(self) -> Identity:
        return stable_identity(self)


class CompilationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    plan_identity: Identity
    outcomes: Mapping[Identity, str]
    published_output_identities: tuple[Identity, ...] = ()
    package_identity: Identity | None = None
    lineage: tuple[tuple[str, Identity, Identity], ...] = ()
    attempts: Mapping[Identity, int] = {}
    retrieval_facts: tuple[RetrievalFact, ...] = ()
    resolution: tuple[ResolutionDecision, ...] = ()
    resolution_result: ResolutionResult | None = None
    failures: tuple[dict[str, str], ...] = ()
    artifact_records: Mapping[str, object] = {}
    semantic_outputs: Mapping[Identity, object] = {}


class OutputStore(Protocol):
    def has_valid(self, identity: Identity) -> bool: ...
    def publish(self, identity: Identity, value: object) -> None: ...


class EmbeddingPort(Protocol):
    def embed(self, request: EmbeddingRequest) -> ProviderOutcome: ...


class EnrichmentPort(Protocol):
    def enrich(self, request: CompletionRequest) -> ProviderOutcome: ...


class BudgetPort(Protocol):
    def apply(self, items: Sequence[ContextItem], budget_tokens: int) -> tuple[ContextItem, ...]: ...


@dataclass(frozen=True)
class CompilerPorts:
    store: OutputStore
    embedding: EmbeddingPort | None = None
    enrichment: EnrichmentPort | None = None
    budget: BudgetPort | None = None
    retrieval: RetrievalPort | None = None
    resolution: ResolutionPort | None = None
    retry: RetryPolicy = RetryPolicy()


class DeterministicProviders:
    """Credential-free substitutes used by the official reproducibility profile."""

    def __init__(self, enrichment_by_recipe: Mapping[str, Mapping[str, object]] | None = None):
        self._enrichment_by_recipe = {
            key: dict(value) for key, value in (enrichment_by_recipe or {}).items()
        }

    def embed(self, request: EmbeddingRequest) -> ProviderOutcome:
        material = "|".join(part.identity for part in request.input.parts)
        digest = sha256((material + request.profile.identity).encode()).digest()
        vector = [round(byte / 255, 6) for byte in digest[:request.space.dimension]]
        if len(vector) != request.space.dimension:
            return ProviderOutcome(status="malformed_output", attempts=1, error="dimension mismatch",
                                   metadata=ProviderMetadata(provider=request.profile.provider, model=request.profile.model))
        return ProviderOutcome(status="succeeded", attempts=1, value=vector,
                               metadata=ProviderMetadata(provider=request.profile.provider, model=request.profile.model))

    def embed_batch(self, requests: Sequence[EmbeddingRequest]) -> tuple[ProviderOutcome, ...]:
        return tuple(self.embed(request) for request in requests)

    def enrich(self, request: CompletionRequest) -> ProviderOutcome:
        configured = self._enrichment_by_recipe.get(request.recipe.recipe_id)
        if configured is not None:
            return ProviderOutcome(
                status="succeeded", attempts=1, value=dict(configured),
                metadata=ProviderMetadata(provider=request.profile.provider, model=request.profile.model),
            )
        text = " ".join(item.text.lower() for item in request.inputs)
        if any(term in text for term in ("beneficiary", "beneficiaries", "transfer failed", "transfer is not possible", "unable to transfer", "couldn't transfer", "could not transfer", "transfer did not go through")):
            label, action = "beneficiary_not_allowed", "explain_beneficiary_restriction"
        elif any(term in text for term in ("apple pay", "apple watch", "google pay", "google play")):
            label, action = "apple_pay_or_google_pay", "explain_apple_pay_or_google_pay"
        elif any(term in text for term in ("lost", "left my phone", "no longer have my phone", "stolen phone", "phone was stolen", "phone is missing", "can't find my phone", "cannot find my phone", "mugged")):
            label, action = "lost_or_stolen_phone", "lost_phone_security_steps"
        elif any(term in text for term in ("expire", "expires", "expired", "expiration", "new card", "replacement card", "replace my card", "card by phone")):
            label, action = "card_about_to_expire", "explain_card_expiry"
        elif any(term in text for term in ("charged for using", "charged more", "charged extra", "fee", "pay money for paying", "extra charge")):
            label, action = "card_payment_fee_charged", "explain_card_payment_fee"
        elif "top up" in text or "top-up" in text:
            label, action = "apple_pay_or_google_pay", "explain_apple_pay_or_google_pay"
        elif any(term in text for term in ("what types of cards", "what cards", "which cards", "currencies", "credit card from", "usa credit card", "american express", "use any card")):
            label, action = "supported_cards_and_currencies", "explain_supported_cards_and_currencies"
        elif any(term in text for term in ("visa", "mastercard")):
            label, action = "visa_or_mastercard", "explain_visa_or_mastercard"
        elif any(term in text for term in ("cancel", "transfer", "transaction", "revert", "reverse", "wrong account", "wrong payment", "change it to the right account")):
            label, action = "cancel_transfer", "cancel_transfer"
        elif any(term in text for term in ("change my pin", "change pin", "new pin", "pin")):
            label, action = "change_pin", "send_pin_change_steps"
        elif any(term in text for term in ("spare card", "more cards", "extra card", "additional card", "duplicate card", "physical cards", "active cards", "another card")):
            label, action = "getting_spare_card", "explain_spare_card_request"
        else:
            label, action = "unclassified", "request_more_information"
        value = {"intent_label": label, "urgency": "normal", "recommended_action": action}
        return ProviderOutcome(status="succeeded", attempts=1, value=value,
                               metadata=ProviderMetadata(provider=request.profile.provider, model=request.profile.model))


class _DeterministicBudget:
    def apply(self, items: Sequence[ContextItem], budget_tokens: int) -> tuple[ContextItem, ...]:
        total = 0
        accepted: list[ContextItem] = []
        for item in sorted(items, key=lambda value: (value.rank, value.representation_identity)):
            if total + item.token_estimate <= budget_tokens:
                accepted.append(item)
                total += item.token_estimate
        return tuple(accepted)


def _closure(seeds: set[Identity], derivations: Sequence[DerivationSpec]) -> set[Identity]:
    return {str(value) for value in __import__("igor_core").affected_outputs(seeds, [item.as_core() for item in derivations])}


def _resolution_request(request: CompilationRequest, facts: Sequence[RetrievalFact]) -> ResolutionRequest:
    if request.resolution is None:
        raise CompilerFailure("contract_invalid", "resolution request is required when a resolution port is supplied")
    base = request.resolution
    evidence_ids = {item.identity for item in base.evidence}
    authority = {item.assertion_identity: item for item in base.authority_assertions}
    temporal = {item.assertion_identity: item for item in base.temporal_assertions}
    for assertion in (*base.authority_assertions, *base.temporal_assertions):
        if set(assertion.evidence_identities) - evidence_ids:
            raise CompilerFailure("contract_invalid", f"assertion references missing evidence: {assertion.assertion_identity}")
    candidates = []
    for fact in facts:
        candidate = next((item for item in base.candidates if item.context_identity == fact.context_identity), None)
        if candidate is None:
            candidate = ResolutionCandidate(
                candidate_identity=stable_identity({"candidate": fact.context_identity}),
                context_identity=fact.context_identity, retrieval=fact,
            )
        elif candidate.retrieval != fact:
            candidate = candidate.model_copy(update={"retrieval": fact})
        if set(candidate.evidence_identities) - evidence_ids:
            raise CompilerFailure("contract_invalid", f"candidate references missing evidence: {candidate.context_identity}")
        if set(candidate.authority_assertion_ids) - set(authority):
            raise CompilerFailure("contract_invalid", f"candidate references missing authority assertion: {candidate.context_identity}")
        if set(candidate.temporal_assertion_ids) - set(temporal):
            raise CompilerFailure("contract_invalid", f"candidate references missing temporal assertion: {candidate.context_identity}")
        candidates.append(candidate)
    if len({item.candidate_identity for item in candidates}) != len(candidates):
        raise CompilerFailure("contract_invalid", "duplicate resolution candidate identity")
    return base.model_copy(update={"candidates": tuple(sorted(candidates, key=lambda item: item.candidate_identity))})


def _validate_resolution(request: ResolutionRequest, result: ResolutionResult) -> ResolutionResult:
    if result.request_identity != request.request_identity or result.resolution_identity != request.identity:
        raise CompilerFailure("contract_invalid", "resolution result identity does not match request")
    expected = {item.candidate_identity for item in request.candidates}
    actual = {item.candidate_identity for item in result.decisions}
    if actual != expected or len(actual) != len(result.decisions):
        raise CompilerFailure("contract_invalid", "resolution must contain exactly one decision per candidate")
    for decision in result.decisions:
        if (decision.as_of != request.as_of or decision.policy_id != request.policy_id or
                decision.policy_revision != request.policy_revision):
            raise CompilerFailure("contract_invalid", "decision basis does not match resolution request")
        if decision.outcome == "conflict" and not decision.conflict_set_id:
            raise CompilerFailure("contract_invalid", "conflict decision requires conflict_set_id")
    ordered = tuple(sorted(result.decisions, key=lambda item: item.candidate_identity))
    if ordered != result.decisions:
        raise CompilerFailure("contract_invalid", "resolution decisions are not stably ordered")
    groups = {"selected": [], "rejected": [], "abstained": [], "conflict": []}
    for decision in ordered:
        groups[decision.outcome].append(decision.candidate_identity)
    return result.model_copy(update={
        "selected_identities": tuple(groups["selected"]), "rejected_identities": tuple(groups["rejected"]),
        "abstained_identities": tuple(groups["abstained"]), "conflict_identities": tuple(groups["conflict"]),
    })


def plan(request: CompilationRequest, inventory: DependencyInventory) -> CompilationPlan:
    """Build an immutable, deterministic work plan without performing I/O."""
    by_output: dict[Identity, DerivationSpec] = {}
    for item in (*inventory.derivations, *request.derivations):
        previous = by_output.get(item.output_identity)
        if previous is not None and previous != item:
            raise ValueError("duplicate derivation producers")
        by_output[item.output_identity] = item
    if set(inventory.changed_identities) - set(inventory.known_output_identities) and inventory.known_output_identities:
        raise CompilerFailure("plan_conflict", "changed identity is absent from inventory")
    seeds = set(inventory.changed_identities)
    closure = _closure(seeds, tuple(by_output.values()))
    valid = set(inventory.valid_output_identities)
    nodes: list[PlanNode] = []
    cache_hits: list[Identity] = []
    visiting: set[Identity] = set()
    visited: set[Identity] = set()

    def visit(output: Identity) -> None:
        if output in visited:
            return
        if output in visiting:
            raise ValueError("derivation cycle")
        derivation = by_output.get(output)
        if derivation is None:
            if output not in valid and output not in inventory.known_output_identities:
                raise ValueError(f"missing dependency: {output}")
            visited.add(output)
            return
        reusable = output in valid and output not in closure
        if reusable:
            cache_hits.append(output)
            visited.add(output)
            return
        visiting.add(output)
        for dependency in sorted(derivation.input_identities):
            visit(dependency)
        visiting.remove(output)
        nodes.append(PlanNode(output_identity=output, operation=derivation.operation,
                              input_identities=derivation.input_identities,
                              dependencies=tuple(sorted(set(derivation.input_identities) & set(by_output))),
                              reusable=False))
        visited.add(output)

    for output in request.required_output_identities:
        visit(output)
    return CompilationPlan(
        request_identity=stable_identity(request.identity_material()), inventory_identity=inventory.identity,
        invalidation_seeds=tuple(sorted(seeds)), invalidation_closure=tuple(sorted(closure)),
        cache_hits=tuple(sorted(cache_hits)), nodes=tuple(nodes),
        expected_output_identities=tuple(sorted(request.required_output_identities)),
    )


def execute(plan_value: CompilationPlan, ports: CompilerPorts, request: CompilationRequest | None = None) -> CompilationResult:
    """Execute only planned nodes and publish each identity at most once."""
    outcomes: dict[Identity, str] = {identity: "reused" for identity in plan_value.cache_hits}
    published: list[Identity] = []
    lineage: list[tuple[str, Identity, Identity]] = []
    failures: list[dict[str, str]] = []
    semantic_outputs: dict[Identity, object] = {}
    attempt_counts: dict[Identity, int] = {identity: 1 for identity in outcomes}

    def invoke(call: Any) -> ProviderOutcome:
        outcome = call()
        attempts = 1
        while outcome.status in ("transient_failure", "timeout") and attempts < ports.retry.max_attempts:
            if ports.retry.backoff_seconds:
                sleep(ports.retry.backoff_seconds * (2 ** (attempts - 1)))
            attempts += 1
            outcome = call()
        if outcome.status in ("transient_failure", "timeout") and attempts >= ports.retry.max_attempts:
            outcome = outcome.model_copy(update={"status": "retry_exhausted", "attempts": attempts})
        return outcome.model_copy(update={"attempts": attempts})

    def invoke_batch(call: Any) -> tuple[ProviderOutcome, ...]:
        outcomes = tuple(call())
        attempts = 1
        while any(item.status in ("transient_failure", "timeout") for item in outcomes) and attempts < ports.retry.max_attempts:
            if ports.retry.backoff_seconds:
                sleep(ports.retry.backoff_seconds * (2 ** (attempts - 1)))
            attempts += 1
            outcomes = tuple(call())
        if attempts > 1:
            outcomes = tuple(item.model_copy(update={"attempts": attempts}) for item in outcomes)
        return outcomes

    provider_results: dict[Identity, ProviderOutcome] = {}
    semantic_nodes = [node for node in plan_value.nodes if node.operation.startswith(("embedding", "enrichment"))]
    if semantic_nodes and request is not None:
        embedding_inputs = {item.identity: item for item in request.embedding_inputs}
        completion_inputs = {item.identity: item for item in request.completion_inputs}

        embedding_nodes = [node for node in semantic_nodes if node.operation.startswith("embedding")]
        embedding_batch_size = max(1, int(request.embedding_profile.parameters.get("request_chunk_size", 1)))
        if embedding_nodes and hasattr(ports.embedding, "embed_batch") and embedding_batch_size > 1:
            for start in range(0, len(embedding_nodes), embedding_batch_size):
                chunk = embedding_nodes[start:start + embedding_batch_size]
                requests = tuple(EmbeddingRequest(
                    output_identity=node.output_identity, input=embedding_inputs[node.input_identities[0]],
                    space=request.embedding_space, profile=request.embedding_profile
                ) for node in chunk)
                batch = invoke_batch(lambda requests=requests: ports.embedding.embed_batch(requests))  # type: ignore[union-attr]
                provider_results.update({node.output_identity: outcome for node, outcome in zip(chunk, batch)})

        def dispatch(node: PlanNode) -> ProviderOutcome:
            if node.operation.startswith("embedding"):
                item = embedding_inputs[node.input_identities[0]]
                return invoke(lambda: ports.embedding.embed(EmbeddingRequest(
                    output_identity=node.output_identity, input=item, space=request.embedding_space,
                    profile=request.embedding_profile)))  # type: ignore[union-attr]
            first = completion_inputs[node.input_identities[0]]
            return invoke(lambda: ports.enrichment.enrich(CompletionRequest(
                output_identity=node.output_identity, inputs=(first,),
                output_schema=request.completion_output_schema, recipe=request.enrichment_recipe,
                prompt_version=request.prompt_version, taxonomy_version=request.taxonomy_version,
                profile=request.completion_profile)))  # type: ignore[union-attr]

        configured_workers = request.embedding_profile.parameters.get(
            "max_concurrency", request.completion_profile.parameters.get("max_concurrency", 8)
        )
        max_workers = max(1, min(32, int(configured_workers)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {node.output_identity: executor.submit(dispatch, node) for node in semantic_nodes if node.output_identity not in provider_results}
            provider_results = {identity: future.result() for identity, future in futures.items()}

    for node in plan_value.nodes:
        if node.operation.startswith("embedding") and ports.embedding is None:
            raise CompilerFailure("port_unavailable", "embedding")
        if node.operation.startswith("enrichment") and ports.enrichment is None:
            raise CompilerFailure("port_unavailable", "enrichment")
        value: object = {"operation": node.operation, "inputs": list(node.input_identities)}
        provider_outcome: ProviderOutcome | None = None
        if node.operation.startswith("embedding"):
            if request is None:
                raise CompilerFailure("contract_invalid", "semantic execution requires a request")
            inputs = {item.identity: item for item in request.embedding_inputs}
            item = inputs.get(node.input_identities[0]) if node.input_identities else None
            if item is None or not item.parts:
                raise CompilerFailure("missing_content", node.output_identity)
            if request.embedding_profile.capability != "embedding":
                raise CompilerFailure("contract_invalid", "embedding profile capability mismatch")
            provider_outcome = provider_results.get(node.output_identity) or invoke(lambda: ports.embedding.embed(EmbeddingRequest(
                output_identity=node.output_identity, input=item, space=request.embedding_space,
                profile=request.embedding_profile)))  # type: ignore[union-attr]
            if provider_outcome.status == "succeeded":
                try:
                    value = Embedding(ir_version="0.1", representation_identity=item.identity,
                                      space_identity=request.embedding_space.identity,
                                      vector=list(provider_outcome.value or []))
                except Exception as error:
                    provider_outcome = provider_outcome.model_copy(update={"status": "malformed_output", "error": str(error)})
        elif node.operation.startswith("enrichment"):
            if request is None or request.completion_output_schema is None or request.enrichment_recipe is None:
                raise CompilerFailure("contract_invalid", "completion output schema and enrichment recipe are required")
            if request.completion_profile.capability != "completion":
                raise CompilerFailure("contract_invalid", "completion profile capability mismatch")
            if not request.completion_inputs or any(not item.parts for item in request.completion_inputs):
                raise CompilerFailure("missing_content", node.output_identity)
            completion_inputs = {item.identity: item for item in request.completion_inputs}
            first = completion_inputs.get(node.input_identities[0]) if node.input_identities else None
            if first is None:
                raise CompilerFailure("missing_content", node.output_identity)
            recipe = request.enrichment_recipe
            if recipe.output_schema_identity != request.completion_output_schema.identity:
                raise CompilerFailure("contract_invalid", "enrichment recipe output-schema mismatch")
            if recipe.prompt_version != request.prompt_version or recipe.taxonomy_version != request.taxonomy_version:
                raise CompilerFailure("contract_invalid", "enrichment recipe prompt/taxonomy mismatch")
            if first.representation.representation_type not in recipe.accepted_representation_types:
                raise CompilerFailure("contract_invalid", "representation type not accepted by enrichment recipe")
            if any(part.media_type not in recipe.accepted_media_types for part in first.parts):
                raise CompilerFailure("contract_invalid", "content media type not accepted by enrichment recipe")
            if recipe.evidence_required and not first.evidence:
                raise CompilerFailure("contract_invalid", "enrichment recipe requires evidence")
            provider_outcome = provider_results.get(node.output_identity) or invoke(lambda: ports.enrichment.enrich(CompletionRequest(
                output_identity=node.output_identity, inputs=(first,),
                output_schema=request.completion_output_schema, recipe=recipe, prompt_version=request.prompt_version,
                taxonomy_version=request.taxonomy_version, profile=request.completion_profile)))  # type: ignore[union-attr]
            if provider_outcome.status == "succeeded":
                try:
                    derivation = next(item for item in request.derivations if item.output_identity == node.output_identity)
                    payload = dict(provider_outcome.value or {})
                    validate_schema_payload(request.completion_output_schema, payload)
                    value = EnrichmentOutput(
                        ir_version="0.1", representation_identity=first.identity,
                        schema_ref=request.completion_output_schema, payload=payload,
                        evidence_identities=tuple(item.identity for item in first.evidence),
                        derivation_identity=derivation.as_core().identity, status="accepted", confidence=1.0,
                    )
                except Exception as error:
                    provider_outcome = provider_outcome.model_copy(update={"status": "malformed_output", "error": str(error)})
        if provider_outcome is not None and provider_outcome.status != "succeeded":
            category = {
                "abstained": "provider_rejected", "transient_failure": "transient_failure",
                "permanent_rejection": "provider_rejected", "timeout": "timeout",
                "malformed_output": "malformed_output", "retry_exhausted": "retry_exhausted",
            }.get(provider_outcome.status, "provider_rejected")
            outcomes[node.output_identity] = provider_outcome.status
            attempt_counts[node.output_identity] = provider_outcome.attempts
            failures.append({"identity": node.output_identity, "category": category,
                             "message": provider_outcome.error or provider_outcome.status})
            continue
        if provider_outcome is not None:
            semantic_outputs[node.output_identity] = value
        if not ports.store.has_valid(node.output_identity):
            ports.store.publish(node.output_identity, value)
            published.append(node.output_identity)
        outcomes[node.output_identity] = "published"
        attempt_counts[node.output_identity] = provider_outcome.attempts if provider_outcome else 1
        lineage.extend(("derived_from", node.output_identity, source) for source in node.input_identities)
    retrieval_facts: list[RetrievalFact] = []
    resolution: list[ResolutionDecision] = []
    resolution_result: ResolutionResult | None = None
    if request is not None and request.retrieval_queries:
        if ports.retrieval is None:
            raise CompilerFailure("port_unavailable", "retrieval")
        for query in request.retrieval_queries:
            retrieval_facts.extend(RetrievalFact.model_validate(item) for item in ports.retrieval.retrieve(query))
        if ports.resolution is not None:
            resolution_request = _resolution_request(request, tuple(retrieval_facts))
            resolution_result = _validate_resolution(resolution_request, ports.resolution.resolve(resolution_request))
            resolution.extend(resolution_result.decisions)
    package_identity = None
    if request is not None and request.package_items and request.budget_tokens is not None:
        if request.resolution is not None and resolution_result is not None:
            selected_contexts = {
                candidate.context_identity for candidate in _resolution_request(request, retrieval_facts).candidates
                if candidate.candidate_identity in set(resolution_result.selected_identities)
            }
            if not selected_contexts.issuperset({item.representation_identity for item in request.package_items if item.representation_identity in set(request.required_context_identities)}):
                raise CompilerFailure("budget_unsatisfied", f"required context item was not selected by resolution policy (selected={len(selected_contexts)}, required={len({item.representation_identity for item in request.package_items if item.representation_identity in set(request.required_context_identities)})})")
        budget = ports.budget or _DeterministicBudget()
        items = budget.apply(request.package_items, request.budget_tokens)
        required = set(request.required_context_identities)
        accepted = {item.representation_identity for item in items}
        if not required.issubset(accepted):
            raise CompilerFailure("budget_unsatisfied", "required context item was not accepted")
        package = ContextPackage(ir_version="0.1", task_id=request.task_id,
                                 schema_revision=request.package_schema_revision, items=items,
                                 budget_tokens=request.budget_tokens, created_at=datetime(1970, 1, 1),
                                 metadata=(
                                     {"resolution_identity": resolution_result.resolution_identity,
                                      "policy_id": resolution_result.decisions[0].policy_id,
                                      "policy_revision": resolution_result.decisions[0].policy_revision}
                                     if resolution_result and resolution_result.decisions else {}
                                 ))
        package_identity = package.identity
        ports.store.publish(package_identity, package)
    records = {
        "plan": plan_value.model_dump(mode="json"),
        "derivations": [{"output_identity": key, "outcome": value} for key, value in sorted(outcomes.items())],
        "failures": failures,
        "retrieval": [item.model_dump(mode="json") for item in retrieval_facts],
        "resolution": [item.model_dump(mode="json") for item in resolution],
        "resolution_request": resolution_result and _resolution_request(request, retrieval_facts).model_dump(mode="json"),
        "resolution_result": resolution_result and resolution_result.model_dump(mode="json"),
        "lineage": [{"type": edge[0], "output_identity": edge[1], "input_identity": edge[2]} for edge in lineage],
        "package_identity": package_identity,
    }
    return CompilationResult(plan_identity=plan_value.identity, outcomes=outcomes,
                             published_output_identities=tuple(published), package_identity=package_identity,
                             lineage=tuple(lineage), attempts=attempt_counts,
                             retrieval_facts=tuple(retrieval_facts), resolution=tuple(resolution),
                             resolution_result=resolution_result,
                             failures=tuple(failures), artifact_records=records,
                             semantic_outputs=semantic_outputs)
