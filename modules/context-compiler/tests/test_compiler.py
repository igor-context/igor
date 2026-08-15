from __future__ import annotations

from datetime import datetime, timezone

from igor_context_compiler import (
    CompilationRequest, CompilerPorts, DependencyInventory, DerivationSpec,
    DeterministicProviders, execute, plan,
    QualifiedRepresentation,
)
from igor_context_compiler import (
    AuthorityAssertion, ResolutionCandidate, ResolutionDecision, ResolutionRequest, ResolutionResult,
    TemporalAssertion, RetrievalFact,
)
from igor_core import ContentPart, EmbeddingSpace, ModelProfile, Representation, SchemaDescriptor, stable_identity


def ident(value):
    return stable_identity({"test": value})


class MemoryStore:
    def __init__(self, valid=()):
        self.values = {identity: object() for identity in valid}

    def has_valid(self, identity):
        return identity in self.values

    def publish(self, identity, value):
        if identity in self.values and self.values[identity] != value:
            raise ValueError("persistence_conflict")
        self.values[identity] = value


def derivation(output, *inputs, operation="representation"):
    return DerivationSpec(operation=operation, input_identities=tuple(inputs), output_identity=output,
                          code_revision="code:1", configuration_identity="config:1")


def request(outputs, derivations):
    return CompilationRequest(run_identity="sha256:" + "1" * 64, task_id="task-1",
                              required_output_identities=tuple(outputs), derivations=tuple(derivations),
                              embedding_profile=ModelProfile(schema_version="0.1", profile_id="deterministic-embedding",
                                  capability="embedding", provider="deterministic", model="hash", revision="1"),
                              completion_profile=ModelProfile(schema_version="0.1", profile_id="deterministic-completion",
                                  capability="completion", provider="deterministic", model="fixture", revision="1"),
                              embedding_space=EmbeddingSpace(ir_version="0.1", provider="deterministic", model="hash",
                                  model_revision="1", dimension=8, dtype="float64", metric="cosine", normalized=False,
                                  input_schema_identity="schema:deterministic"), code_revision="code:1")


def test_plan_is_stable_and_orders_dependencies():
    source, middle, out = (ident(value) for value in ("source", "middle", "out"))
    req = request((out,), (derivation(out, middle), derivation(middle, source)))
    inventory = DependencyInventory(known_output_identities=(source,))
    value = plan(req, inventory)
    assert [node.output_identity for node in value.nodes] == [middle, out]
    assert value.identity == plan(req, inventory).identity


def test_changed_identity_invalidates_only_transitive_outputs():
    source, middle, out = (ident(value) for value in ("source", "middle", "out"))
    req = request((out,), (derivation(out, middle), derivation(middle, source)))
    inventory = DependencyInventory(known_output_identities=(source,), changed_identities=(source,))
    value = plan(req, inventory)
    assert value.invalidation_closure == tuple(sorted((middle, out, source)))


def test_execution_is_idempotent_and_uses_deterministic_provider():
    source, output = ident("source"), ident("embedding-output")
    representation = Representation(ir_version="0.1", representation_type="text",
        schema_ref=SchemaDescriptor(schema_version="0.1", schema_id="test", revision="1"),
        source_snapshot_ids=(source,), payload="source content")
    req = request((output,), (derivation(output, representation.identity, operation="embedding.v1"),))
    req = req.model_copy(update={"embedding_inputs": (QualifiedRepresentation(
        representation=representation,
        parts=(ContentPart(kind="text", media_type="text/plain", text="source content",
                           content_sha256=stable_identity("source content")),),
    ),)})
    inventory = DependencyInventory(known_output_identities=(source, representation.identity))
    store = MemoryStore()
    ports = CompilerPorts(store=store, embedding=DeterministicProviders())
    compiled = execute(plan(req, inventory), ports, req)
    repeated = execute(plan(req, inventory), ports, req)
    assert compiled.published_output_identities == (output,)
    assert repeated.published_output_identities == ()
    assert compiled.plan_identity == repeated.plan_identity


def test_cycles_fail_closed():
    a, b = ident("a"), ident("b")
    req = request((a,), (derivation(a, b), derivation(b, a)))
    try:
        plan(req, DependencyInventory())
    except ValueError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("cycle should be rejected")


def test_resolution_contract_is_order_independent_and_requires_complete_decisions():
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence_id = ident("evidence")
    candidate_ids = [ident("a"), ident("b")]
    facts = tuple(RetrievalFact(result_identity=ident(f"result-{i}"), context_identity=cid, score=1 - i / 10, rank=i) for i, cid in enumerate(candidate_ids))
    authority = tuple(AuthorityAssertion(assertion_identity=ident(f"authority-{i}"), subject_identity=cid,
        issuer_ref="support", scope_ref="ticket", evidence_identities=(evidence_id,)) for i, cid in enumerate(candidate_ids))
    temporal = tuple(TemporalAssertion(assertion_identity=ident(f"temporal-{i}"), subject_identity=cid,
        effective_from=datetime(2025, 1, 1, tzinfo=timezone.utc), effective_until=None,
        evidence_identities=(evidence_id,)) for i, cid in enumerate(candidate_ids))
    candidates = tuple(ResolutionCandidate(candidate_identity=ident(f"candidate-{i}"), context_identity=cid,
        retrieval=facts[i], evidence_identities=(evidence_id,), authority_assertion_ids=(authority[i].assertion_identity,),
        temporal_assertion_ids=(temporal[i].assertion_identity,)) for i, cid in enumerate(candidate_ids))
    request = ResolutionRequest(request_identity=ident("request"), task_id="task", task_schema_revision="1", as_of=as_of,
        candidates=candidates, policy_id="support.authority-temporal", policy_revision="1")
    selected = tuple(ResolutionDecision(candidate_identity=item.candidate_identity, outcome="selected", reason_code="valid",
        authority_basis_ids=item.authority_assertion_ids, temporal_basis_ids=item.temporal_assertion_ids,
        authority_basis="satisfied", temporal_basis="valid", as_of=as_of, policy_id=request.policy_id,
        policy_revision=request.policy_revision) for item in sorted(candidates, key=lambda value: value.candidate_identity))
    result = ResolutionResult(request_identity=request.request_identity, resolution_identity=request.identity, decisions=selected)
    assert tuple(item.candidate_identity for item in result.decisions) == tuple(sorted(item.candidate_identity for item in candidates))
    assert request.model_copy(update={"candidates": tuple(reversed(candidates))}).identity == request.identity
