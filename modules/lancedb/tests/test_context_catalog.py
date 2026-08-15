from pathlib import Path

from igor_core import stable_identity
from igor_lancedb import (
    STANDARD_CONTEXT_RELATIONS,
    LanceContextModelMaterializer,
    LanceReadableContextCatalog,
    LanceRetrievalAdapter,
    LanceStore,
    execute_context_model_tests,
    validate_standard_context_relations,
)


def test_readable_catalog_and_lineage_relations_are_stable_and_queryable(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    catalog = LanceReadableContextCatalog(store)
    catalog.replace(
        context_catalog=[{
            "canonical_identity": "sha256:" + "1" * 64,
            "display_name": "Invoice page 2",
            "context_model_id": "document.qa.v1",
            "object_kind": "source_snapshot",
            "schema_id": "document.page",
            "schema_revision": "1",
            "source_system": "huggingface",
            "source_key": "page-form-02",
            "operation_name": "ingest",
            "status": "observed",
            "preview": "Invoice date…",
            "physical_relation": "document_pages",
        }],
        lineage_nodes=[{
            "canonical_node_id": "sha256:" + "1" * 64,
            "display_name": "Invoice page 2",
            "context_model_id": "document.qa.v1",
            "node_kind": "source_snapshot",
            "source_system": "huggingface",
            "source_key": "page-form-02",
            "schema_id": "document.page",
            "schema_revision": "1",
            "operation_name": "ingest",
            "status": "observed",
            "preview": "Invoice date…",
        }],
        lineage_edges=[],
    )
    assert catalog.relation_names == ("context_catalog", "lineage_edges", "lineage_nodes")
    assert store.read("context_catalog").to_pylist()[0]["display_name"] == "Invoice page 2"
    assert store.read("lineage_nodes").to_pylist()[0]["canonical_node_id"].startswith("sha256:")


def test_context_model_materializer_writes_all_standard_relations(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    materializer = LanceContextModelMaterializer(store)
    manifest_identity = stable_identity({"context_model": "customer.commercial", "revision": "1"})
    snapshot_identity = stable_identity({"snapshot": "contract-14", "content": "net-30"})
    output_identity = stable_identity({"output": snapshot_identity, "role": "payment_terms"})
    package_identity = stable_identity({"package": output_identity})
    retrieval_identity = stable_identity({"retrieval": output_identity})
    contract_identity = stable_identity({"contract": "customer.commercial"})
    evaluation_identity = stable_identity({"evaluation": package_identity, "contract": contract_identity})
    manifest = {
        "context_model_id": "customer.commercial",
        "context_model_revision": "1",
        "context_model_identity": manifest_identity,
        "title": "Customer Commercial Context",
        "authority": [{
            "role": "payment_terms",
            "target_role": "payment_terms",
            "policy_identity": stable_identity({"policy": "payment_terms"}),
            "settings": {"precedence": ["signed_contract", "crm"]},
        }],
        "tests": [
            {"identity_unique": {"object": "contract"}},
            {"snapshot_integrity": {"object": "contract_snapshot"}},
            {"evidence_required": {"object": "payment_terms"}},
            {"lineage_complete": {"object": "payment_terms"}},
            {"retrieval_indexed": {"object": "payment_terms"}},
            {"no_orphan_outputs": {}},
        ],
    }

    result = materializer.replace(
        manifest=manifest,
        observations=[{
            "object_role": "contract",
            "snapshot_role": "contract_snapshot",
            "source_system": "contract_repo",
            "source_key": "ACME-2026-14",
            "snapshot_identity": snapshot_identity,
            "content_ref": "contracts/acme-2026-14.pdf",
            "content_sha256": stable_identity({"content": "net-30"}),
            "media_type": "application/pdf",
            "observed_at": "2026-08-14T00:00:00Z",
            "display_name": "Contract ACME-2026-14",
            "preview": "Net 30 payment terms",
        }],
        outputs=[{
            "identity": output_identity,
            "role": "payment_terms",
            "kind": "semantic_derivation",
            "derived_from": None,
            "input_identities": [snapshot_identity],
            "evidence_identities": None,
            "schema_id": "commercial.payment-terms.v1",
            "display_name": "Payment terms for Acme",
            "payload": {"payment_terms": "Net 30"},
            "vector": [0.1, 0.2, 0.3],
            "text": "Payment terms are Net 30.",
            "authority_status": "selected",
            "temporal_status": "valid",
            "policy_id": "payment_terms",
            "as_of": "2026-08-14",
        }],
        retrieval=[{
            "identity": output_identity,
            "context_identity": output_identity,
            "retrieval_identity": retrieval_identity,
            "vector": [0.1, 0.2, 0.3],
            "text": "Payment terms are Net 30.",
            "status": "active",
        }],
        resolution_candidates=[{
            "candidate_identity": stable_identity({"candidate": output_identity}),
            "decision_identity": stable_identity({"decision": output_identity}),
            "context_identity": output_identity,
            "rank": 1,
            "score": 0.99,
        }],
        resolution_decisions=[{
            "decision_identity": stable_identity({"decision": output_identity}),
            "query_identity": stable_identity({"query": "payment_terms"}),
            "selected_identity": output_identity,
            "decision_status": "selected",
        }],
        packages=[{
            "package_identity": package_identity,
            "package_kind": "task_context",
            "items": [{"context_identity": output_identity, "item_role": "answer_evidence"}],
        }],
        context_contracts=[{
            "contract_identity": contract_identity,
            "contract_id": "customer.commercial.package",
            "contract_version": "1",
            "rules_json": "{}",
            "status": "active",
        }],
        context_package_contracts=[{
            "package_identity": package_identity,
            "contract_identity": contract_identity,
            "contract_version": "1",
            "evaluation_identity": evaluation_identity,
            "decision": "allow",
        }],
        contract_evaluations=[{
            "evaluation_identity": evaluation_identity,
            "package_identity": package_identity,
            "contract_identity": contract_identity,
            "contract_version": "1",
            "decision": "allow",
            "violated_rules": [],
            "explanation": "allowed",
            "evaluated_at": "2026-08-14T00:00:00Z",
            "evaluator_identity": "test",
        }],
    )

    assert tuple(STANDARD_CONTEXT_RELATIONS) == LanceContextModelMaterializer.relation_names
    assert set(STANDARD_CONTEXT_RELATIONS).issubset(store.names())
    assert result.relation_counts["context_models"] == 1
    assert result.relation_counts["context_snapshots"] == 1
    assert result.relation_counts["context_outputs"] == 1
    assert result.relation_counts["context_assertions"] == 1
    assert result.relation_counts["context_retrieval"] == 1
    assert result.relation_counts["context_contracts"] == 1
    assert result.relation_counts["context_package_contracts"] == 1
    assert result.relation_counts["contract_evaluations"] == 1
    assert result.relation_counts["package_items"] == 1
    assert store.read("context_catalog").to_pylist()[0]["context_model_id"] == "customer.commercial"
    assertion = store.read("context_assertions").to_pylist()[0]
    assert assertion["assertion_subject"] == output_identity
    assert assertion["assertion_type"] == "payment_terms"
    assert assertion["evidence_identities"] == [snapshot_identity]
    assert store.read("lineage_edges").to_pylist()[0]["source_node_id"] == snapshot_identity
    conformance = materializer.validate()
    assert conformance.passed
    assert not conformance.failures
    declared_tests = materializer.execute_tests(manifest)
    assert declared_tests.passed
    assert {item.test_type for item in declared_tests.tests} >= {"evidence_required", "lineage_complete"}


def test_context_model_materializer_creates_empty_standard_relations(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    result = LanceContextModelMaterializer(store).replace(
        manifest={
            "context_model_id": "empty.lifecycle",
            "context_model_revision": "1",
            "context_model_identity": stable_identity({"context_model": "empty.lifecycle"}),
        },
    )

    assert set(STANDARD_CONTEXT_RELATIONS) == set(store.names())
    assert result.relation_counts["context_models"] == 1
    assert result.relation_counts["context_outputs"] == 0
    assert store.read("context_outputs").num_rows == 0
    assert validate_standard_context_relations(store).passed


def test_lance_retrieval_adapter_maps_full_text_to_native_fts(tmp_path: Path) -> None:
    class Query:
        search_mode = "full_text"
        query = "payment terms"
        limit = 1

    store = LanceStore(tmp_path / "db")
    store.create(
        "context_retrieval",
        [
            {
                "identity": "retrieval:1",
                "context_identity": "ctx:1",
                "retrieval_identity": "retrieval:1",
                "vector": [0.1, 0.2, 0.3],
                "text": "Payment terms are Net 30.",
                "status": "active",
            }
        ],
    )

    results = LanceRetrievalAdapter(store).retrieve(Query())

    assert len(results) == 1
    assert results[0]["context_identity"] == "ctx:1"
    assert results[0]["facts"]["query_type"] == "full_text"


def test_context_model_conformance_reports_broken_standard_relations(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    output_identity = stable_identity({"output": "orphan"})
    LanceContextModelMaterializer(store).replace(
        manifest={
            "context_model_id": "broken.lifecycle",
            "context_model_revision": "1",
            "context_model_identity": stable_identity({"context_model": "broken.lifecycle"}),
        },
        outputs=[{
            "identity": output_identity,
            "role": "broken_output",
            "kind": "context_representation",
            "payload": {"ok": False},
        }],
    )
    store.replace("context_catalog", [{
        "canonical_identity": stable_identity({"catalog": "wrong"}),
        "display_name": "Wrong catalog row",
        "context_model_id": "broken.lifecycle",
        "context_model_revision": "1",
        "object_kind": "context_representation",
        "schema_id": "",
        "schema_revision": "",
        "source_system": "",
        "source_key": "",
        "media_type": "",
        "evidence_preview": "",
        "preview": "",
        "content_ref": "",
        "operation_name": "",
        "status": "active",
        "authority_status": "",
        "temporal_status": "",
        "policy_id": "",
        "as_of": "",
        "physical_relation": "context_outputs",
    }])
    store.replace("package_items", [{
        "package_item_identity": stable_identity({"package_item": "orphan"}),
        "package_identity": stable_identity({"package": "missing"}),
        "representation_identity": stable_identity({"context": "missing"}),
        "context_model_id": "broken.lifecycle",
        "context_model_revision": "1",
    }])

    result = validate_standard_context_relations(store)

    assert not result.passed
    failures = {item["check_id"]: item for item in result.failures}
    assert output_identity in failures["no_orphan_outputs"]["missing_catalog_identities"]
    assert failures["package_items_reference_packages"]["missing_package_identities"]
    assert failures["package_items_reference_context"]["missing_context_identities"]


def test_context_model_declared_tests_report_missing_evidence(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "db")
    manifest = {
        "context_model_id": "declared.tests",
        "context_model_revision": "1",
        "context_model_identity": stable_identity({"context_model": "declared.tests"}),
        "tests": [
            {"evidence_required": {"object": "summary"}},
            {"unknown_domain_test": {"object": "summary"}},
        ],
    }
    output_identity = stable_identity({"output": "summary-without-evidence"})
    LanceContextModelMaterializer(store).replace(
        manifest=manifest,
        outputs=[{
            "identity": output_identity,
            "role": "summary",
            "kind": "semantic_derivation",
            "payload": {"summary": "No source edge"},
        }],
    )

    result = execute_context_model_tests(store, manifest)

    assert not result.passed
    failures = {item["check_id"]: item for item in result.failures}
    assert output_identity in failures["evidence_required"]["missing_evidence_identities"]
    assert failures["unknown_domain_test"]["reason_code"] == "unsupported_context_model_test"
