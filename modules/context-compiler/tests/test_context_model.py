from __future__ import annotations

from datetime import datetime, timezone

from igor_context_compiler import (
    ContextModelCompileRequest,
    ContextModelReference,
    ResolutionCandidate,
    compile_context_model,
    compile_context_model_derivation_instances,
    compile_context_model_derivations,
    compile_context_model_resolution_request,
    compile_context_model_retrieval_queries,
)
from igor_context_compiler.compiler import CompilerFailure
from igor_core import stable_identity


def ref(kind: str, name: str) -> ContextModelReference:
    return ContextModelReference(kind=kind, ref=name, identity=stable_identity({"kind": kind, "ref": name}))


REFERENCES = (
    ref("source_contract", "contracts.v1"),
    ref("connector_binding", "acme-contracts-prod.v1"),
    ref("schema", "commercial.payment-terms.v1"),
    ref("semantic_definition", "commercial.payment-terms-rubric.v1"),
    ref("recipe", "commercial.payment-terms-recipe.v1"),
    ref("model_profile", "commercial-text-general.v1"),
    ref("authority_policy", "commercial.payment-terms-precedence.v1"),
)


def declaration(*, title: str = "Customer Commercial Context", display: str = "Payment terms for {{ customer_name }}"):
    return {
        "context_model": {"id": "customer.commercial", "revision": "1", "title": title},
        "sources": {
            "customer_contract": {
                "source_contract": "contracts.v1",
                "connector_binding": "acme-contracts-prod.v1",
                "identity_fields": ["contract_id"],
            },
        },
        "objects": {
            "contract": {
                "kind": "business_object",
                "source": "customer_contract",
                "display_name": "Contract {{ contract_number }}",
            },
            "contract_snapshot": {
                "kind": "source_snapshot",
                "source": "customer_contract",
                "display_name": "{{ contract_number }} observed at {{ observed_at }}",
            },
            "payment_terms": {
                "kind": "semantic_derivation",
                "derived_from": ["contract_snapshot"],
                "operation": "enrichment.structured.v1",
                "schema": "commercial.payment-terms.v1",
                "semantic_definition": "commercial.payment-terms-rubric.v1",
                "recipe": "commercial.payment-terms-recipe.v1",
                "model_profile": "commercial-text-general.v1",
                "display_name": display,
            },
        },
        "authority": {
            "payment_terms": {
                "policy": "commercial.payment-terms-precedence.v1",
                "reject_expired": True,
                "conflict_behavior": "abstain",
            },
        },
        "retrievals": {
            "authoritative_payment_terms": {
                "search": "hybrid",
                "candidate_limit": 20,
                "target_roles": ["payment_terms"],
                "eligibility": {"active_only": True, "valid_at_task_time": True},
                "resolution": {"policy": "payment_terms", "accepted_outcomes": ["selected"]},
            },
        },
        "context_contracts": {
            "commercial.agent.v1": {
                "contract_id": "commercial.agent",
                "version": "1",
                "permitted_consumers": ["agent"],
                "permitted_tasks": ["answer"],
                "permitted_purposes": ["commercial-review"],
                "allowed_modalities": ["text"],
                "required_authority_level": "verified",
                "historical_evidence_allowed": False,
                "allowed_as_of_max_age_days": 30,
                "budgets": {"max_package_items": 8, "max_tokens": 1200, "max_bytes": 64000},
                "freshness_hours": 24,
                "citations_required": True,
                "abstain_conditions": ["missing_evidence"],
                "prohibited_uses": ["training"],
            }
        },
        "outputs": {
            "payment_terms_package": {
                "output_type": "context_package",
                "retrieval": "authoritative_payment_terms",
                "contract": "commercial.agent.v1",
                "target_roles": ["payment_terms"],
                "format": "markdown",
                "destination": "agent",
            }
        },
        "evaluations": {
            "payment_terms_retrieval": {
                "category": "retrieval",
                "description": "retrieval relevance",
                "target_roles": ["payment_terms"],
                "target_outputs": ["payment_terms_package"],
                "target_contracts": ["commercial.agent.v1"],
                "target_retrievals": ["authoritative_payment_terms"],
                "fixtures": {"source_cases": ["valid-case"]},
                "checks": ["semantic_retrieval_relevance"],
            }
        },
        "presentation": {
            "payment_terms": {"preview_fields": ["customer_name", "payment_terms", "effective_from"]},
        },
        "tests": [
            {"identity_unique": {"object": "contract"}},
            {"snapshot_integrity": {"object": "contract_snapshot"}},
            {"evidence_required": {"object": "payment_terms"}},
            {"lineage_complete": {"object": "payment_terms"}},
            {"no_orphan_outputs": {}},
        ],
    }


def compile_fixture(value=None, references=REFERENCES):
    return compile_context_model(ContextModelCompileRequest(declaration=value or declaration(), references=references))


def test_manifest_resolves_references_and_emits_portable_artifact():
    manifest = compile_fixture()
    source = manifest.sources[0]
    payment_terms = next(item for item in manifest.objects if item.role == "payment_terms")
    authority = manifest.authority[0]
    retrieval = manifest.retrievals[0]
    contract = manifest.context_contracts[0]
    output = manifest.outputs[0]
    evaluation = manifest.evaluations[0]
    presentation = manifest.presentation[0]
    artifact = manifest.artifact()
    assert source.source_contract_identity == REFERENCES[0].identity
    assert payment_terms.schema_identity == REFERENCES[2].identity
    assert payment_terms.semantic_definition_identity == REFERENCES[3].identity
    assert payment_terms.recipe_identity == REFERENCES[4].identity
    assert payment_terms.model_profile_identity == REFERENCES[5].identity
    assert authority.policy_identity == REFERENCES[6].identity
    assert retrieval.resolution_policy_role == "payment_terms"
    assert retrieval.target_roles == ("payment_terms",)
    assert contract.contract_id == "commercial.agent"
    assert output.contract_identity == contract.identity
    assert output.retrieval_identity == retrieval.identity
    assert evaluation.target_outputs == ("payment_terms_package",)
    assert presentation.preview_fields == ("customer_name", "payment_terms", "effective_from")
    assert {item.test_type for item in manifest.tests} >= {"identity_unique", "lineage_complete", "no_orphan_outputs"}
    assert artifact["artifact_type"] == "resolved-context-model-manifest"
    assert artifact["manifest_identity"] == manifest.identity


def test_manifest_identity_is_stable_for_equivalent_ordering():
    value = declaration()
    value["objects"] = [
        {"payment_terms": value["objects"]["payment_terms"]},
        {"contract_snapshot": value["objects"]["contract_snapshot"]},
        {"contract": value["objects"]["contract"]},
    ]
    assert compile_fixture(value).identity == compile_fixture().identity


def test_display_only_names_do_not_change_semantic_identity():
    original = compile_fixture()
    renamed = compile_fixture(declaration(title="Renamed", display="Terms preview"))
    original_payment_terms = next(item for item in original.objects if item.role == "payment_terms")
    renamed_payment_terms = next(item for item in renamed.objects if item.role == "payment_terms")
    assert renamed.identity == original.identity
    assert renamed_payment_terms.identity == original_payment_terms.identity
    assert renamed_payment_terms.display_name == "Terms preview"


def test_authoring_reference_aliases_do_not_change_semantic_identity():
    value = declaration()
    value["sources"]["customer_contract"]["source_contract"] = "contracts.alias"
    value["authority"]["payment_terms"]["policy"] = "policy.alias"
    references = (
        ContextModelReference(kind="source_contract", ref="contracts.alias", identity=REFERENCES[0].identity),
        ContextModelReference(kind="authority_policy", ref="policy.alias", identity=REFERENCES[6].identity),
        *REFERENCES[1:6],
    )
    assert compile_fixture(value, references=references).identity == compile_fixture().identity


def test_missing_reference_fails_closed():
    try:
        compile_fixture(references=REFERENCES[:-1])
    except CompilerFailure as error:
        assert error.category == "contract_invalid"
        assert "missing authority_policy reference" in str(error)
    else:
        raise AssertionError("missing references must fail closed")


def test_duplicate_roles_fail_closed():
    value = declaration()
    value["objects"] = [
        {"role": "contract", "kind": "business_object", "source": "customer_contract"},
        {"role": "contract", "kind": "source_snapshot", "source": "customer_contract"},
    ]
    try:
        compile_fixture(value)
    except CompilerFailure as error:
        assert "duplicate Context Model role" in str(error)
    else:
        raise AssertionError("duplicate roles must fail")


def test_unknown_object_kind_fails_closed():
    value = declaration()
    value["objects"]["payment_terms"]["kind"] = "provider_specific_table"
    try:
        compile_fixture(value)
    except CompilerFailure as error:
        assert "unknown Context Model object kind" in str(error)
    else:
        raise AssertionError("unknown object kinds must fail")


def test_missing_object_dependency_fails_closed():
    value = declaration()
    value["objects"]["payment_terms"]["derived_from"] = ["missing_snapshot"]
    try:
        compile_fixture(value)
    except CompilerFailure as error:
        assert "missing dependency" in str(error)
    else:
        raise AssertionError("missing dependencies must fail")


def test_object_dependency_cycle_fails_closed():
    value = declaration()
    value["objects"]["contract_snapshot"]["derived_from"] = ["payment_terms"]
    try:
        compile_fixture(value)
    except CompilerFailure as error:
        assert "dependency cycle" in str(error)
    else:
        raise AssertionError("cycles must fail")


def test_manifest_derivations_compile_from_runtime_role_identities():
    manifest = compile_fixture()
    snapshot_identity = stable_identity({"snapshot": "contract-14"})

    derivations = compile_context_model_derivations(
        manifest,
        {"contract_snapshot": snapshot_identity},
        code_revision="test",
    )

    assert len(derivations) == 1
    derivation = derivations[0]
    assert derivation.operation == "enrichment.structured.v1"
    assert derivation.input_identities == (snapshot_identity,)
    assert derivation.configuration_identity == REFERENCES[2].identity
    assert derivation.model_revision == REFERENCES[5].identity
    assert derivation.output_identity.startswith("sha256:")
    assert derivation.output_identity == compile_context_model_derivations(
        manifest,
        {"contract_snapshot": snapshot_identity},
        code_revision="test",
    )[0].output_identity


def test_manifest_derivation_instances_compile_per_runtime_input():
    manifest = compile_fixture()
    first = stable_identity({"snapshot": "contract-14", "page": 1})
    second = stable_identity({"snapshot": "contract-14", "page": 2})
    explicit_output = stable_identity({"payment_terms": second})

    derivations = compile_context_model_derivation_instances(
        manifest,
        (
            {"role": "payment_terms", "input_identities": (first,)},
            {"role": "payment_terms", "input_identities": (second,), "output_identity": explicit_output},
        ),
        code_revision="test",
    )

    assert len(derivations) == 2
    assert derivations[0].operation == "enrichment.structured.v1"
    assert derivations[0].input_identities == (first,)
    assert derivations[0].configuration_identity == REFERENCES[2].identity
    assert derivations[0].output_identity.startswith("sha256:")
    assert derivations[1].output_identity == explicit_output
    assert derivations == compile_context_model_derivation_instances(
        manifest,
        (
            {"role": "payment_terms", "input_identities": (first,)},
            {"role": "payment_terms", "input_identities": (second,), "output_identity": explicit_output},
        ),
        code_revision="test",
    )


def test_manifest_derivation_instances_fail_for_non_derivable_role():
    try:
        compile_context_model_derivation_instances(
            compile_fixture(),
            ({"role": "contract", "input_identities": (stable_identity("contract"),)},),
            code_revision="test",
        )
    except CompilerFailure as error:
        assert error.category == "contract_invalid"
        assert "not derivable" in str(error)
    else:
        raise AssertionError("non-derivable roles must fail closed")


def test_manifest_derivations_fail_without_runtime_dependency_identity():
    try:
        compile_context_model_derivations(compile_fixture(), {}, code_revision="test")
    except CompilerFailure as error:
        assert error.category == "contract_invalid"
        assert "missing runtime identity" in str(error)
    else:
        raise AssertionError("missing runtime role identities must fail closed")


def test_manifest_retrieval_queries_compile_from_runtime_inputs():
    manifest = compile_fixture()

    queries = compile_context_model_retrieval_queries(
        manifest,
        {
            "authoritative_payment_terms": {
                "text": "payment terms for Acme",
                "vector": (0.1, 0.2, 0.3),
                "space_identity": "commercial-contract-v1",
            },
        },
    )

    assert len(queries) == 1
    query = queries[0]
    assert query.query_type == "hybrid"
    assert query.text == "payment terms for Acme"
    assert query.vector == (0.1, 0.2, 0.3)
    assert query.limit == 20
    assert query.space_identity == "commercial-contract-v1"
    assert query.query_id == compile_context_model_retrieval_queries(
        manifest,
        {
            "authoritative_payment_terms": {
                "text": "payment terms for Acme",
                "vector": (0.1, 0.2, 0.3),
                "space_identity": "commercial-contract-v1",
            },
        },
    )[0].query_id


def test_manifest_retrieval_queries_fail_for_unknown_role():
    try:
        compile_context_model_retrieval_queries(compile_fixture(), {"missing": {"text": "x"}})
    except CompilerFailure as error:
        assert error.category == "contract_invalid"
        assert "unknown Context Model retrieval role" in str(error)
    else:
        raise AssertionError("unknown retrieval roles must fail closed")


def test_manifest_resolution_request_compiles_from_retrieval_policy():
    value = declaration()
    value["authority"]["payment_terms"]["policy_id"] = "commercial.payment-terms-precedence"
    value["authority"]["payment_terms"]["policy_revision"] = "2026-08-14"
    manifest = compile_fixture(value)
    first = ResolutionCandidate(
        candidate_identity=stable_identity({"candidate": "first"}),
        context_identity=stable_identity({"context": "first"}),
    )
    second = ResolutionCandidate(
        candidate_identity=stable_identity({"candidate": "second"}),
        context_identity=stable_identity({"context": "second"}),
    )

    request = compile_context_model_resolution_request(
        manifest,
        retrieval_role="authoritative_payment_terms",
        task_id="commercial-task",
        task_schema_revision="1",
        as_of=datetime(2026, 8, 14, tzinfo=timezone.utc),
        candidates=(second, first),
    )

    assert request.policy_id == "commercial.payment-terms-precedence"
    assert request.policy_revision == "2026-08-14"
    assert request.candidates == tuple(sorted((first, second), key=lambda item: item.candidate_identity))
    assert request.request_identity == compile_context_model_resolution_request(
        manifest,
        retrieval_role="authoritative_payment_terms",
        task_id="commercial-task",
        task_schema_revision="1",
        as_of=datetime(2026, 8, 14, tzinfo=timezone.utc),
        candidates=(first, second),
    ).request_identity


def test_manifest_resolution_request_fails_without_resolution_policy():
    value = declaration()
    value["retrievals"]["authoritative_payment_terms"].pop("resolution")
    manifest = compile_fixture(value)

    try:
        compile_context_model_resolution_request(
            manifest,
            retrieval_role="authoritative_payment_terms",
            task_id="commercial-task",
            task_schema_revision="1",
            as_of=datetime(2026, 8, 14, tzinfo=timezone.utc),
        )
    except CompilerFailure as error:
        assert error.category == "contract_invalid"
        assert "no resolution policy" in str(error)
    else:
        raise AssertionError("authority-aware resolution requires a manifest policy")
