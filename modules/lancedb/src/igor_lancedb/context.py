from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from igor_core import canonical_json, stable_identity

from igor_lancedb.store import LanceStore


STANDARD_CONTEXT_RELATIONS = (
    "context_models",
    "context_objects",
    "context_snapshots",
    "context_assertions",
    "context_catalog",
    "context_outputs",
    "context_retrieval",
    "authority_policies",
    "context_contracts",
    "context_package_contracts",
    "contract_evaluations",
    "resolution_decisions",
    "resolution_candidates",
    "lineage_nodes",
    "lineage_edges",
    "context_packages",
    "package_items",
)


STANDARD_CONTEXT_SCHEMAS = {
    "context_models": pa.schema([
        ("context_model_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("semantic_identity", pa.string()),
        ("title", pa.string()), ("status", pa.string()), ("manifest_json", pa.string()),
    ]),
    "context_objects": pa.schema([
        ("object_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("object_role", pa.string()),
        ("object_kind", pa.string()), ("source_system", pa.string()),
        ("source_key", pa.string()), ("current_snapshot_identity", pa.string()),
        ("status", pa.string()), ("display_name", pa.string()),
    ]),
    "context_snapshots": pa.schema([
        ("snapshot_identity", pa.string()), ("object_identity", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("snapshot_role", pa.string()), ("source_system", pa.string()),
        ("source_key", pa.string()), ("observed_at", pa.string()),
        ("content_ref", pa.string()), ("content_sha256", pa.string()),
        ("media_type", pa.string()), ("byte_length", pa.int64()),
        ("change_type", pa.string()), ("status", pa.string()), ("payload_json", pa.string()),
    ]),
    "context_assertions": pa.schema([
        ("assertion_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("assertion_subject", pa.string()),
        ("assertion_type", pa.string()), ("evidence_identities", pa.list_(pa.string())),
        ("status", pa.string()), ("value_json", pa.string()),
    ]),
    "context_catalog": pa.schema([
        ("canonical_identity", pa.string()), ("display_name", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("object_kind", pa.string()), ("schema_id", pa.string()), ("schema_revision", pa.string()),
        ("source_system", pa.string()), ("source_key", pa.string()), ("media_type", pa.string()),
        ("evidence_preview", pa.string()), ("preview", pa.string()), ("content_ref", pa.string()),
        ("operation_name", pa.string()), ("status", pa.string()), ("authority_status", pa.string()),
        ("temporal_status", pa.string()), ("policy_id", pa.string()), ("as_of", pa.string()),
        ("physical_relation", pa.string()),
    ]),
    "context_outputs": pa.schema([
        ("identity", pa.string()), ("output_identity", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("output_role", pa.string()), ("output_kind", pa.string()), ("status", pa.string()),
        ("value", pa.string()), ("value_json", pa.string()), ("payload_json", pa.string()),
    ]),
    "context_retrieval": pa.schema([
        ("identity", pa.string()), ("context_identity", pa.string()), ("retrieval_identity", pa.string()),
        ("vector", pa.list_(pa.float64())), ("text", pa.string()), ("context_model_id", pa.string()),
        ("status", pa.string()),
    ]),
    "authority_policies": pa.schema([
        ("policy_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("policy_role", pa.string()),
        ("target_role", pa.string()), ("policy_ref", pa.string()), ("settings_json", pa.string()),
        ("status", pa.string()),
    ]),
    "context_contracts": pa.schema([
        ("contract_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("contract_id", pa.string()),
        ("contract_version", pa.string()), ("semantic_identity", pa.string()),
        ("rules_json", pa.string()), ("status", pa.string()),
    ]),
    "context_package_contracts": pa.schema([
        ("binding_identity", pa.string()), ("package_identity", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("contract_identity", pa.string()), ("contract_version", pa.string()),
        ("evaluation_identity", pa.string()), ("decision", pa.string()),
    ]),
    "contract_evaluations": pa.schema([
        ("evaluation_identity", pa.string()), ("package_identity", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("contract_identity", pa.string()), ("contract_version", pa.string()),
        ("decision", pa.string()), ("violated_rules", pa.list_(pa.string())),
        ("explanation", pa.string()), ("evaluated_at", pa.string()),
        ("evaluator_identity", pa.string()),
    ]),
    "resolution_decisions": pa.schema([
        ("decision_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("candidate_identity", pa.string()),
        ("outcome", pa.string()), ("reason_code", pa.string()),
    ]),
    "resolution_candidates": pa.schema([
        ("candidate_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("context_identity", pa.string()),
    ]),
    "lineage_nodes": pa.schema([
        ("canonical_node_id", pa.string()), ("display_name", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
        ("node_kind", pa.string()), ("node_type", pa.string()), ("source_system", pa.string()),
        ("source_key", pa.string()), ("schema_id", pa.string()), ("schema_revision", pa.string()),
        ("operation_name", pa.string()), ("status", pa.string()), ("preview", pa.string()),
        ("content_ref", pa.string()), ("media_type", pa.string()),
    ]),
    "lineage_edges": pa.schema([
        ("edge_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("source_node_id", pa.string()),
        ("target_node_id", pa.string()), ("edge_type", pa.string()),
        ("operation_identity", pa.string()), ("run_identity", pa.string()),
    ]),
    "context_packages": pa.schema([
        ("package_identity", pa.string()), ("context_model_id", pa.string()),
        ("context_model_revision", pa.string()), ("task_id", pa.string()),
        ("package_kind", pa.string()),
    ]),
    "package_items": pa.schema([
        ("package_item_identity", pa.string()), ("package_identity", pa.string()),
        ("context_model_id", pa.string()), ("context_model_revision", pa.string()),
    ]),
}


@dataclass(frozen=True)
class ContextModelMaterializationResult:
    relation_counts: dict[str, int]
    relations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextModelConformanceResult:
    checks: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return all(bool(item["passed"]) for item in self.checks)

    @property
    def failures(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.checks if not item["passed"])


@dataclass(frozen=True)
class ContextModelDeclaredTestResult:
    test_identity: str
    test_type: str
    target_role: str | None
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextModelTestExecutionResult:
    relation_conformance: ContextModelConformanceResult
    tests: tuple[ContextModelDeclaredTestResult, ...] = ()

    @property
    def passed(self) -> bool:
        return self.relation_conformance.passed and all(item.passed for item in self.tests)

    @property
    def failures(self) -> tuple[dict[str, Any], ...]:
        relation_failures = tuple({"check_id": item["check_id"], **item} for item in self.relation_conformance.failures)
        declared_failures = tuple({
            "check_id": item.test_type,
            "test_identity": item.test_identity,
            "target_role": item.target_role,
            **item.evidence,
        } for item in self.tests if not item.passed)
        return (*relation_failures, *declared_failures)


class LanceContextModelMaterializer:
    """Materialize standard Context Model relations from manifest and runtime facts."""

    relation_names = STANDARD_CONTEXT_RELATIONS

    def __init__(self, store: LanceStore):
        self.store = store

    def replace(self, *, manifest: Any,
                observations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                outputs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                assertions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                retrieval: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                resolution_decisions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                resolution_candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                context_contracts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                context_package_contracts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                contract_evaluations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                lineage_edges: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                packages: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
                package_items: list[dict[str, Any]] | tuple[dict[str, Any], ...] = ()) -> ContextModelMaterializationResult:
        normalized_manifest = _manifest(manifest)
        relations = _standard_relations(
            normalized_manifest,
            list(observations),
            list(outputs),
            list(assertions),
            list(retrieval),
            list(resolution_decisions),
            list(resolution_candidates),
            list(context_contracts),
            list(context_package_contracts),
            list(contract_evaluations),
            list(lineage_edges),
            list(packages),
            list(package_items),
        )
        for name in self.relation_names:
            rows: list[dict[str, Any]] | pa.Table = relations[name] or _empty_relation(name)
            if name in self.store.names():
                self.store.replace(name, rows)
            else:
                self.store.create(name, rows)
        return ContextModelMaterializationResult(
            relation_counts={name: len(relations[name]) for name in self.relation_names},
            relations=relations,
        )

    def read(self, relation: str):
        if relation not in self.relation_names:
            raise ValueError(f"unknown Context Model relation: {relation}")
        return self.store.read(relation)

    def validate(self) -> ContextModelConformanceResult:
        return validate_standard_context_relations(self.store)

    def execute_tests(self, manifest: Any) -> ContextModelTestExecutionResult:
        return execute_context_model_tests(self.store, manifest)


def validate_standard_context_relations(store: LanceStore) -> ContextModelConformanceResult:
    """Validate domain-neutral consistency of the standard Context Model relations."""
    names = set(store.names())
    checks: list[dict[str, Any]] = []
    missing = sorted(set(STANDARD_CONTEXT_RELATIONS) - names)
    _check(checks, "standard_relations_present", not missing, missing=missing)
    if missing:
        return ContextModelConformanceResult(tuple(checks))

    rows = {name: store.read(name).to_pylist() for name in STANDARD_CONTEXT_RELATIONS}
    for name, values in rows.items():
        key = _primary_key(name)
        keys = [str(row.get(key, "")) for row in values]
        duplicates = sorted({item for item in keys if item and keys.count(item) > 1})
        blanks = sum(1 for item in keys if not item)
        _check(checks, f"{name}_identity_unique", not duplicates and blanks == 0,
               duplicates=duplicates, blank_identities=blanks)

    objects = {row["object_identity"] for row in rows["context_objects"]}
    snapshots = {row["snapshot_identity"] for row in rows["context_snapshots"]}
    output_ids = {row["identity"] for row in rows["context_outputs"]}
    catalog_ids = {row["canonical_identity"] for row in rows["context_catalog"]}
    node_ids = {row["canonical_node_id"] for row in rows["lineage_nodes"]}
    retrieval_ids = {row["retrieval_identity"] for row in rows["context_retrieval"]}
    candidates = {row["candidate_identity"] for row in rows["resolution_candidates"]}
    contracts = {row["contract_identity"] for row in rows["context_contracts"]}
    packages = {row["package_identity"] for row in rows["context_packages"]}
    bindings_by_package: dict[str, list[dict[str, Any]]] = {}
    evaluations = {row["evaluation_identity"]: row for row in rows["contract_evaluations"]}

    snapshot_orphans = sorted({row["object_identity"] for row in rows["context_snapshots"] if row.get("object_identity") not in objects})
    _check(checks, "snapshot_objects_exist", not snapshot_orphans, missing_object_identities=snapshot_orphans)
    current_snapshot_orphans = sorted({
        row["current_snapshot_identity"] for row in rows["context_objects"]
        if row.get("current_snapshot_identity") and row.get("current_snapshot_identity") not in snapshots
    })
    _check(checks, "object_current_snapshots_exist", not current_snapshot_orphans,
           missing_snapshot_identities=current_snapshot_orphans)
    catalog_output_orphans = sorted(output_ids - catalog_ids)
    _check(checks, "no_orphan_outputs", not catalog_output_orphans, missing_catalog_identities=catalog_output_orphans)
    retrieval_context_orphans = sorted({
        row["context_identity"] for row in rows["context_retrieval"]
        if row.get("context_identity") not in output_ids and row.get("context_identity") not in catalog_ids
    })
    _check(checks, "retrieval_contexts_exist", not retrieval_context_orphans,
           missing_context_identities=retrieval_context_orphans)
    retrieval_identity_blanks = sorted({row.get("context_identity", "") for row in rows["context_retrieval"] if not row.get("retrieval_identity")})
    _check(checks, "retrieval_indexed", not retrieval_identity_blanks and len(retrieval_ids) == len(rows["context_retrieval"]),
           missing_retrieval_identities=retrieval_identity_blanks)
    lineage_orphans = sorted({
        endpoint for row in rows["lineage_edges"]
        for endpoint in (row.get("source_node_id"), row.get("target_node_id"))
        if endpoint and endpoint not in node_ids
    })
    _check(checks, "lineage_closure", not lineage_orphans, missing_node_identities=lineage_orphans)
    candidate_orphans = sorted({
        row["candidate_identity"] for row in rows["resolution_decisions"]
        if row.get("candidate_identity") and row.get("candidate_identity") not in candidates
    })
    _check(checks, "resolution_candidates_exist", not candidate_orphans,
           missing_candidate_identities=candidate_orphans)
    package_orphans = sorted({
        row["package_identity"] for row in rows["package_items"]
        if row.get("package_identity") not in packages
    })
    _check(checks, "package_items_reference_packages", not package_orphans,
           missing_package_identities=package_orphans)
    package_context_orphans = sorted({
        row["representation_identity"] for row in rows["package_items"]
        if row.get("representation_identity") and row.get("representation_identity") not in output_ids and row.get("representation_identity") not in catalog_ids
    })
    _check(checks, "package_items_reference_context", not package_context_orphans,
           missing_context_identities=package_context_orphans)
    assertion_orphans = sorted({
        row["assertion_subject"] for row in rows["context_assertions"]
        if row.get("assertion_subject") and row.get("assertion_subject") not in output_ids and row.get("assertion_subject") not in catalog_ids and row.get("assertion_subject") not in snapshots
    })
    _check(checks, "assertions_reference_context", not assertion_orphans,
           missing_context_identities=assertion_orphans)
    for row in rows["context_package_contracts"]:
        bindings_by_package.setdefault(row.get("package_identity", ""), []).append(row)
    unbound_packages = sorted(package for package in packages if len(bindings_by_package.get(package, ())) != 1)
    _check(checks, "packages_reference_exactly_one_contract", not unbound_packages,
           package_identities=unbound_packages)
    binding_contract_orphans = sorted({
        row["contract_identity"] for row in rows["context_package_contracts"]
        if row.get("contract_identity") not in contracts
    })
    _check(checks, "package_contracts_reference_contracts", not binding_contract_orphans,
           missing_contract_identities=binding_contract_orphans)
    binding_eval_orphans = sorted({
        row["evaluation_identity"] for row in rows["context_package_contracts"]
        if row.get("evaluation_identity") not in evaluations
    })
    _check(checks, "package_contracts_reference_evaluations", not binding_eval_orphans,
           missing_evaluation_identities=binding_eval_orphans)
    non_allow_bindings = sorted({
        row["package_identity"] for row in rows["context_package_contracts"]
        if row.get("decision") != "allow"
    })
    _check(checks, "published_packages_have_allow_contract_decision", not non_allow_bindings,
           package_identities=non_allow_bindings)
    evaluation_contract_orphans = sorted({
        row["contract_identity"] for row in rows["contract_evaluations"]
        if row.get("contract_identity") not in contracts
    })
    _check(checks, "contract_evaluations_reference_contracts", not evaluation_contract_orphans,
           missing_contract_identities=evaluation_contract_orphans)
    return ContextModelConformanceResult(tuple(checks))


def execute_context_model_tests(store: LanceStore, manifest: Any) -> ContextModelTestExecutionResult:
    """Execute resolved manifest test declarations against standard Context Model relations."""
    relation_conformance = validate_standard_context_relations(store)
    normalized_manifest = _manifest(manifest)
    rows = _standard_relation_rows(store) if relation_conformance.passed else {}
    results = tuple(
        _execute_declared_test(test, rows, relation_conformance)
        for test in normalized_manifest.get("tests", ())
    )
    return ContextModelTestExecutionResult(relation_conformance=relation_conformance, tests=results)


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, **evidence: Any) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), **evidence})


def _standard_relation_rows(store: LanceStore) -> dict[str, list[dict[str, Any]]]:
    return {name: store.read(name).to_pylist() for name in STANDARD_CONTEXT_RELATIONS}


def _execute_declared_test(test: Any, rows: dict[str, list[dict[str, Any]]],
                           relation_conformance: ContextModelConformanceResult) -> ContextModelDeclaredTestResult:
    declaration = _test_declaration(test)
    test_type = declaration["test_type"]
    target_role = declaration.get("target_role")
    identity = declaration["identity"]
    if not relation_conformance.passed:
        return ContextModelDeclaredTestResult(
            test_identity=identity,
            test_type=test_type,
            target_role=target_role,
            passed=False,
            evidence={"reason_code": "standard_relations_invalid"},
        )
    if test_type == "identity_unique":
        return _identity_unique_test(identity, test_type, target_role, rows)
    if test_type == "snapshot_integrity":
        return _snapshot_integrity_test(identity, test_type, target_role, rows)
    if test_type == "evidence_required":
        return _evidence_required_test(identity, test_type, target_role, rows)
    if test_type == "lineage_complete":
        return _lineage_complete_test(identity, test_type, target_role, rows)
    if test_type == "retrieval_indexed":
        return _retrieval_indexed_test(identity, test_type, target_role, rows)
    if test_type == "authority_resolved":
        return _authority_resolved_test(identity, test_type, target_role, rows)
    if test_type == "no_orphan_outputs":
        check = _conformance_check(relation_conformance, "no_orphan_outputs")
        return ContextModelDeclaredTestResult(
            test_identity=identity,
            test_type=test_type,
            target_role=target_role,
            passed=bool(check and check["passed"]),
            evidence={key: value for key, value in (check or {}).items() if key not in {"check_id", "passed"}},
        )
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=False,
        evidence={"reason_code": "unsupported_context_model_test"},
    )


def _test_declaration(test: Any) -> dict[str, Any]:
    value = dict(test) if isinstance(test, Mapping) else _manifest_mapping(test)
    if "test_type" in value:
        test_type = str(value["test_type"])
        target_role = value.get("target_role")
        parameters = dict(value.get("parameters", {}))
    elif len(value) == 1:
        test_type, raw = next(iter(value.items()))
        parameters = dict(raw) if isinstance(raw, Mapping) else {}
        target_role = raw if isinstance(raw, str) else parameters.get("object") or parameters.get("target")
    else:
        test_type = "invalid"
        target_role = None
        parameters = {}
    target = str(target_role) if target_role is not None else None
    identity = str(value.get("identity") or stable_identity({
        "test_type": test_type,
        "target_role": target,
        "parameters": parameters,
    }))
    return {"identity": identity, "test_type": str(test_type), "target_role": target, "parameters": parameters}


def _identity_unique_test(identity: str, test_type: str, target_role: str | None,
                          rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    target_rows = _rows_for_role(rows, target_role)
    identities = [item[1] for item in target_rows]
    duplicates = sorted({item for item in identities if item and identities.count(item) > 1})
    blanks = sum(1 for item in identities if not item)
    passed = bool(target_rows) and not duplicates and blanks == 0
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={"target_rows": len(target_rows), "duplicates": duplicates, "blank_identities": blanks},
    )


def _snapshot_integrity_test(identity: str, test_type: str, target_role: str | None,
                             rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    snapshots = [
        row for row in rows["context_snapshots"]
        if target_role is None or row.get("snapshot_role") == target_role
    ]
    objects = {row["object_identity"] for row in rows["context_objects"]}
    missing_objects = sorted({row.get("object_identity", "") for row in snapshots if row.get("object_identity") not in objects})
    missing_content = sorted({row["snapshot_identity"] for row in snapshots if not row.get("content_sha256")})
    missing_observed_at = sorted({row["snapshot_identity"] for row in snapshots if not row.get("observed_at")})
    passed = bool(snapshots) and not missing_objects and not missing_content and not missing_observed_at
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={
            "target_rows": len(snapshots),
            "missing_object_identities": missing_objects,
            "missing_content_sha256": missing_content,
            "missing_observed_at": missing_observed_at,
        },
    )


def _evidence_required_test(identity: str, test_type: str, target_role: str | None,
                            rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    outputs = _output_rows_for_role(rows, target_role)
    edge_targets = {row.get("target_node_id") for row in rows["lineage_edges"] if row.get("source_node_id")}
    missing = sorted({row["identity"] for row in outputs if row["identity"] not in edge_targets})
    passed = bool(outputs) and not missing
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={"target_rows": len(outputs), "missing_evidence_identities": missing},
    )


def _lineage_complete_test(identity: str, test_type: str, target_role: str | None,
                           rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    target_rows = _rows_for_role(rows, target_role)
    node_ids = {row["canonical_node_id"] for row in rows["lineage_nodes"]}
    missing = sorted({item[1] for item in target_rows if item[1] not in node_ids})
    passed = bool(target_rows) and not missing
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={"target_rows": len(target_rows), "missing_lineage_nodes": missing},
    )


def _retrieval_indexed_test(identity: str, test_type: str, target_role: str | None,
                            rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    outputs = _output_rows_for_role(rows, target_role)
    indexed = {row.get("context_identity") for row in rows["context_retrieval"] if row.get("retrieval_identity")}
    missing = sorted({row["identity"] for row in outputs if row["identity"] not in indexed})
    passed = bool(outputs) and not missing
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={"target_rows": len(outputs), "missing_retrieval_identities": missing},
    )


def _authority_resolved_test(identity: str, test_type: str, target_role: str | None,
                             rows: dict[str, list[dict[str, Any]]]) -> ContextModelDeclaredTestResult:
    outputs = _output_rows_for_role(rows, target_role)
    candidate_by_context = {
        row.get("context_identity"): row.get("candidate_identity") for row in rows["resolution_candidates"]
    }
    decided = {row.get("candidate_identity") for row in rows["resolution_decisions"]}
    missing = sorted({
        row["identity"] for row in outputs
        if not candidate_by_context.get(row["identity"]) or candidate_by_context[row["identity"]] not in decided
    })
    passed = bool(outputs) and not missing
    return ContextModelDeclaredTestResult(
        test_identity=identity,
        test_type=test_type,
        target_role=target_role,
        passed=passed,
        evidence={"target_rows": len(outputs), "missing_resolution_identities": missing},
    )


def _rows_for_role(rows: dict[str, list[dict[str, Any]]], target_role: str | None) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    selected.extend(
        ("context_objects", row.get("object_identity", ""))
        for row in rows["context_objects"]
        if target_role is None or row.get("object_role") == target_role
    )
    selected.extend(
        ("context_snapshots", row.get("snapshot_identity", ""))
        for row in rows["context_snapshots"]
        if target_role is None or row.get("snapshot_role") == target_role
    )
    selected.extend(
        ("context_outputs", row.get("identity", ""))
        for row in rows["context_outputs"]
        if target_role is None or row.get("output_role") == target_role
    )
    return selected


def _output_rows_for_role(rows: dict[str, list[dict[str, Any]]], target_role: str | None) -> list[dict[str, Any]]:
    return [
        row for row in rows["context_outputs"]
        if target_role is None or row.get("output_role") == target_role
    ]


def _conformance_check(result: ContextModelConformanceResult, check_id: str) -> dict[str, Any] | None:
    return next((item for item in result.checks if item["check_id"] == check_id), None)



class LanceReadableContextCatalog:
    """Stable, query-oriented projections for readable context and lineage."""

    relation_names = ("context_catalog", "lineage_edges", "lineage_nodes")

    def __init__(self, store: LanceStore):
        self.store = store

    def replace(self, *, context_catalog: list[dict[str, Any]],
                lineage_nodes: list[dict[str, Any]],
                lineage_edges: list[dict[str, Any]]) -> None:
        rows = {
            "context_catalog": context_catalog,
            "lineage_nodes": lineage_nodes,
            "lineage_edges": lineage_edges,
        }
        for name, values in rows.items():
            if not values:
                continue
            if name in self.store.names():
                self.store.replace(name, values)
            else:
                self.store.create(name, values)

    def read(self, relation: str):
        if relation not in self.relation_names:
            raise ValueError(f"unknown readable context relation: {relation}")
        return self.store.read(relation)


class LanceContextOutputStore:
    """Lance-backed implementation of the compiler's immutable output port."""

    table_name = "context_outputs"

    def __init__(self, store: LanceStore):
        self.store = store

    def _rows(self) -> list[dict[str, str]]:
        if self.table_name not in self.store.names():
            return []
        return self.store.read(self.table_name).to_pylist()

    def has_valid(self, identity: str) -> bool:
        return any(row["identity"] == identity for row in self._rows())

    def publish(self, identity: str, value: Any) -> None:
        encoded = canonical_json(value)
        rows = self._rows()
        existing = next((row for row in rows if row["identity"] == identity), None)
        if existing is not None:
            if existing["value"] != encoded:
                raise ValueError("persistence_conflict")
            return
        row = {"identity": identity, "value": encoded}
        if self.table_name in self.store.names() and not rows:
            self.store.replace(self.table_name, [row])
        elif rows:
            self.store.add(self.table_name, [row])
        else:
            self.store.create(self.table_name, [row])

    def read(self, identity: str) -> Any:
        for row in self._rows():
            if row["identity"] == identity:
                return json.loads(row["value"])
        raise KeyError(identity)


class LanceRetrievalAdapter:
    """Provider-neutral retrieval port backed by LanceDB's native search API."""

    def __init__(self, store: LanceStore, table_name: str = "context_retrieval"):
        self.store = store
        self.table_name = table_name
        self.last_execution: dict[str, Any] = {}

    def ensure_vector_index(self, *, metric: str = "cosine") -> tuple[dict[str, Any], ...]:
        """Build the bounded native index and return redacted index facts."""
        table = self.store._table(self.table_name)
        table.create_index(metric, num_partitions=1, vector_column_name="vector", index_type="IVF_FLAT")
        return tuple({
            "name": item.name,
            "index_type": item.index_type,
            "num_indexed_rows": item.num_indexed_rows,
            "num_unindexed_rows": item.num_unindexed_rows,
            "num_segments": item.num_segments,
        } for item in table.list_indices())

    def retrieve(self, query: Any) -> tuple[Any, ...]:
        table_name = getattr(query, "retrieval_table_name", None) or self.table_name
        if table_name not in self.store.names():
            return ()
        search_mode = getattr(query, "search_mode", getattr(query, "query_type", "vector"))
        query_text = getattr(query, "query", getattr(query, "text", ""))
        if search_mode == "full_text":
            self.ensure_full_text_index(table_name=table_name)
            table = self.store._table(table_name)
            request = table.search(query_text, query_type="fts")
        elif search_mode == "hybrid":
            table = self.store._table(table_name)
            vector = list(getattr(query, "query_vector", getattr(query, "vector", ())))
            request = table.search(vector, vector_column_name="vector").text(query_text)
        else:
            table = self.store._table(table_name)
            vector = list(getattr(query, "query_vector", getattr(query, "vector", ())))
            request = table.search(vector, vector_column_name="vector")
        if getattr(query, "prefilter", None):
            request = request.where(query.prefilter)
        request = request.limit(query.limit)
        plan = request.explain_plan()
        rows = request.to_list()
        self.last_execution = {
            "adapter": "LanceRetrievalAdapter",
            "table": table_name,
            "search_mode": search_mode,
            "prefilter": getattr(query, "prefilter", None),
            "limit": int(query.limit),
            "plan": plan,
        }
        return tuple(
            {
                "result_identity": row.get("retrieval_identity") or row.get("identity"),
                "context_identity": row.get("context_identity", row.get("identity")),
                # Lance exposes distance (lower is better); IGOR exposes score (higher is better).
                "score": float(row["_score"]) if row.get("_score") is not None else -float(row.get("_distance", 0.0)),
                "rank": rank,
                "facts": {"table": table_name, "query_type": search_mode, "query_id": getattr(query, "query_id", "semantic-sql")},
            }
            for rank, row in enumerate(rows)
        )

    def ensure_full_text_index(self, *, table_name: str | None = None, field_name: str = "text") -> tuple[dict[str, Any], ...]:
        """Build the bounded native full-text index and return redacted index facts."""
        table = self.store._table(table_name or self.table_name)
        try:
            table.create_fts_index(field_name, replace=True)
        except Exception as error:
            message = str(error).lower()
            if "already exists" not in message and "exists already" not in message:
                raise
        return tuple({
            "name": item.name,
            "index_type": item.index_type,
            "num_indexed_rows": item.num_indexed_rows,
            "num_unindexed_rows": item.num_unindexed_rows,
            "num_segments": item.num_segments,
        } for item in table.list_indices())


def _manifest(value: Any) -> dict[str, Any]:
    outer = _manifest_mapping(value)
    manifest = _manifest_mapping(outer.get("manifest", outer))
    identity = (
        outer.get("manifest_identity")
        or manifest.get("context_model_identity")
        or manifest.get("identity")
        or stable_identity(manifest)
    )
    return {
        "context_model_id": str(manifest["context_model_id"]),
        "revision": str(manifest.get("revision", manifest.get("context_model_revision", ""))),
        "title": manifest.get("title"),
        "identity": identity,
        "semantic_identity": outer.get("semantic_identity") or manifest.get("semantic_identity") or identity,
        "objects": list(manifest.get("objects", ())),
        "authority": list(manifest.get("authority", ())),
        "contracts": list(manifest.get("contracts", manifest.get("context_contracts", ()))),
        "retrievals": list(manifest.get("retrievals", ())),
        "sources": list(manifest.get("sources", ())),
        "tests": list(manifest.get("tests", ())),
    }


def _manifest_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    artifact = getattr(value, "artifact", None)
    if callable(artifact):
        return dict(artifact())
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dict(dump(mode="json"))
        identity = getattr(value, "identity", None)
        if identity is not None:
            result.setdefault("context_model_identity", str(identity))
        semantic_identity = getattr(value, "semantic_identity", None)
        if semantic_identity is not None:
            result.setdefault("semantic_identity", str(semantic_identity))
        return result
    raise TypeError("resolved Context Model manifest must be a mapping or expose artifact()/model_dump()")


def _empty_relation(name: str) -> pa.Table:
    return pa.Table.from_pylist([], schema=STANDARD_CONTEXT_SCHEMAS[name])


def _standard_relations(manifest: dict[str, Any], observations: list[dict[str, Any]],
                        outputs: list[dict[str, Any]], assertions: list[dict[str, Any]],
                        retrieval: list[dict[str, Any]], resolution_decisions: list[dict[str, Any]],
                        resolution_candidates: list[dict[str, Any]], context_contracts: list[dict[str, Any]],
                        context_package_contracts: list[dict[str, Any]], contract_evaluations: list[dict[str, Any]],
                        lineage_edges: list[dict[str, Any]],
                        packages: list[dict[str, Any]], package_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    context_model_id = manifest["context_model_id"]
    revision = manifest["revision"]
    model_identity = manifest["identity"]
    relations = {name: [] for name in STANDARD_CONTEXT_RELATIONS}
    catalog: list[dict[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}
    relations["context_models"].append({
        "context_model_identity": model_identity,
        "context_model_id": context_model_id,
        "context_model_revision": revision,
        "semantic_identity": manifest["semantic_identity"],
        "title": manifest.get("title") or context_model_id,
        "status": "active",
        "manifest_json": canonical_json(manifest),
    })
    for item in manifest.get("authority", ()):
        role = str(item.get("role", item.get("target_role", "authority_policy")))
        policy_identity = str(item.get("policy_identity", item.get("identity", stable_identity(item))))
        relations["authority_policies"].append({
            "policy_identity": policy_identity,
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "policy_role": role,
            "target_role": str(item.get("target_role", role)),
            "policy_ref": str(item.get("policy_ref", "")),
            "settings_json": canonical_json(item.get("settings", {})),
            "status": "active",
        })
    for item in (context_contracts or manifest.get("contracts", ())):
        value = dict(item)
        contract_identity = str(value.get("contract_identity", value.get("identity", stable_identity(value))))
        value.setdefault("contract_identity", contract_identity)
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        value.setdefault("contract_id", value.get("contract_id", "context-package-contract"))
        value.setdefault("contract_version", value.get("version", value.get("revision", contract_identity)))
        value.setdefault("semantic_identity", value["contract_identity"])
        if "rules_json" not in value:
            value["rules_json"] = canonical_json(value.get("rules", value))
        value.setdefault("status", "active")
        relations["context_contracts"].append({
            "contract_identity": contract_identity,
            "context_model_id": str(value["context_model_id"]),
            "context_model_revision": str(value["context_model_revision"]),
            "contract_id": str(value["contract_id"]),
            "contract_version": str(value["contract_version"]),
            "semantic_identity": str(value["semantic_identity"]),
            "rules_json": str(value["rules_json"]),
            "status": str(value["status"]),
        })
        nodes[contract_identity] = {
            "canonical_node_id": contract_identity,
            "display_name": str(value.get("display_name", f"Context contract: {value.get('contract_id', contract_identity)}")),
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "node_kind": "context_contract",
            "node_type": "context_contract",
            "source_system": "",
            "source_key": str(value.get("contract_version", "")),
            "schema_id": "",
            "schema_revision": "",
            "operation_name": "resolve_context_contract",
            "status": str(value.get("status", "active")),
            "preview": str(value.get("contract_id", "")),
            "content_ref": "",
            "media_type": "",
        }
    for observation in observations:
        source_system = str(observation.get("source_system", "unknown"))
        source_key = str(observation.get("source_key", observation.get("record_id", observation.get("id", ""))))
        object_role = str(observation.get("object_role", observation.get("role", "source_object")))
        snapshot_role = str(observation.get("snapshot_role", observation.get("kind", "source_snapshot")))
        object_identity = str(observation.get("object_identity") or stable_identity({
            "context_model": model_identity, "role": object_role, "source_system": source_system,
            "source_key": source_key,
        }))
        snapshot_identity = str(observation.get("snapshot_identity") or observation.get("canonical_identity") or stable_identity({
            "object_identity": object_identity,
            "content_sha256": observation.get("content_sha256", ""),
            "observed_at": observation.get("observed_at", ""),
            "change_type": observation.get("change_type", "observed"),
        }))
        relations["context_objects"].append({
            "object_identity": object_identity,
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "object_role": object_role,
            "object_kind": str(observation.get("object_kind", object_role)),
            "source_system": source_system,
            "source_key": source_key,
            "current_snapshot_identity": snapshot_identity,
            "status": str(observation.get("object_status", observation.get("status", "active"))),
            "display_name": str(observation.get("object_display_name", observation.get("display_name", source_key))),
        })
        relations["context_snapshots"].append({
            "snapshot_identity": snapshot_identity,
            "object_identity": object_identity,
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "snapshot_role": snapshot_role,
            "source_system": source_system,
            "source_key": source_key,
            "observed_at": str(observation.get("observed_at", "")),
            "content_ref": str(observation.get("content_ref", "")),
            "content_sha256": str(observation.get("content_sha256", snapshot_identity)),
            "media_type": str(observation.get("media_type", "")),
            "byte_length": int(observation.get("byte_length", observation.get("byte_count", 0)) or 0),
            "change_type": str(observation.get("change_type", "observed")),
            "status": str(observation.get("snapshot_status", observation.get("status", "observed"))),
            "payload_json": canonical_json(observation.get("payload", {})),
        })
        row = _catalog_row(context_model_id, revision, snapshot_identity, observation, "context_snapshots", snapshot_role)
        row["display_name"] = str(observation.get("display_name", source_key))
        row["object_kind"] = snapshot_role
        row["operation_name"] = str(observation.get("operation_name", "observe"))
        row["status"] = str(observation.get("status", "observed"))
        catalog.append(row)

    for assertion in assertions:
        row = dict(assertion)
        row.setdefault("assertion_identity", assertion.get("identity") or stable_identity(assertion))
        row.setdefault("context_model_id", context_model_id)
        row.setdefault("context_model_revision", revision)
        row.setdefault("assertion_subject", str(assertion.get("subject_identity", assertion.get("assertion_subject", ""))))
        row.setdefault("assertion_type", str(assertion.get("predicate", assertion.get("assertion_type", ""))))
        row.setdefault("evidence_identities", list(assertion.get("evidence_identities", ())))
        row.setdefault("status", "active")
        if "value_json" not in row:
            row["value_json"] = canonical_json(assertion.get("value", assertion.get("payload", {})))
        relations["context_assertions"].append(row)

    for output in outputs:
        identity = str(output.get("identity", output.get("output_identity", stable_identity(output))))
        payload = output.get("payload", {})
        value = output.get("value", payload)
        row = dict(output)
        row.pop("vector", None)
        if "payload" in row:
            row["payload_json"] = canonical_json(row.pop("payload"))
        else:
            row.setdefault("payload_json", canonical_json(payload))
        row["identity"] = identity
        row.setdefault("output_identity", identity)
        row.setdefault("context_model_id", context_model_id)
        row.setdefault("context_model_revision", revision)
        row.setdefault("output_role", output.get("role", output.get("object_role", output.get("output_kind", "context_output"))))
        row.setdefault("output_kind", output.get("kind", row["output_role"]))
        row.setdefault("status", "active")
        row.setdefault("value", canonical_json(value))
        row.setdefault("value_json", row["value"])
        relations["context_outputs"].append(row)
        derived_values = output.get("derived_from") or output.get("input_identities") or ()
        if bool(output.get("publish_assertion", True)):
            evidence_values = output.get("evidence_identities") or derived_values
            evidence_identities = tuple(str(item) for item in evidence_values)
            assertion_type = str(output.get("assertion_type", row["output_role"]))
            assertion = {
                "assertion_identity": stable_identity({
                    "context_model_identity": model_identity,
                    "assertion_subject": identity,
                    "assertion_type": assertion_type,
                    "value_json": row["value_json"],
                    "evidence_identities": evidence_identities,
                }),
                "context_model_id": context_model_id,
                "context_model_revision": revision,
                "assertion_subject": identity,
                "assertion_type": assertion_type,
                "evidence_identities": list(evidence_identities),
                "status": str(row["status"]),
                "value_json": row["value_json"],
            }
            relations["context_assertions"].append(assertion)
        catalog_row = _catalog_row(
            context_model_id,
            revision,
            identity,
            output,
            "context_outputs",
            str(output.get("object_kind", row["output_kind"])),
        )
        catalog_row["display_name"] = str(output.get("display_name", identity))
        catalog_row["operation_name"] = str(output.get("operation_name", output.get("operation", row["output_kind"])))
        catalog_row["status"] = str(row["status"])
        catalog.append(catalog_row)
        for source in derived_values:
            lineage_edges.append({
                "source_node_id": str(source),
                "target_node_id": identity,
                "edge_type": str(output.get("edge_type", "derived_from")),
                "operation_identity": str(output.get("operation_identity", output.get("operation", ""))),
            })
        if output.get("vector") is not None:
            retrieval.append({
                "identity": identity,
                "context_identity": str(output.get("context_identity", identity)),
                "retrieval_identity": str(output.get("retrieval_identity", stable_identity({"retrieval": identity}))),
                "vector": output["vector"],
                "text": str(output.get("text", output.get("preview", ""))),
                "status": str(output.get("status", "active")),
                "context_model_id": context_model_id,
            })

    relations["context_catalog"].extend(_dedupe(catalog, "canonical_identity"))

    for row in relations["context_catalog"]:
        node_id = str(row["canonical_identity"])
        nodes[node_id] = {
            "canonical_node_id": node_id,
            "display_name": row.get("display_name", node_id),
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "node_kind": row.get("object_kind", row.get("node_kind", "")),
            "node_type": row.get("object_kind", row.get("node_type", "")),
            "source_system": row.get("source_system", ""),
            "source_key": row.get("source_key", ""),
            "schema_id": row.get("schema_id", ""),
            "schema_revision": row.get("schema_revision", ""),
            "operation_name": row.get("operation_name", ""),
            "status": row.get("status", "active"),
            "preview": row.get("preview", row.get("evidence_preview", "")),
            "content_ref": row.get("content_ref", ""),
            "media_type": row.get("media_type", ""),
        }

    for row in retrieval:
        value = dict(row)
        identity = str(value.get("identity", value.get("context_identity", stable_identity(value))))
        value.setdefault("identity", identity)
        value.setdefault("context_identity", identity)
        value.setdefault("retrieval_identity", stable_identity({"retrieval": value["context_identity"]}))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("status", "active")
        relations["context_retrieval"].append(value)
    for row in resolution_decisions:
        value = dict(row)
        value.setdefault("decision_identity", stable_identity(value))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        relations["resolution_decisions"].append(value)
        decision_identity = str(value["decision_identity"])
        nodes[decision_identity] = {
            "canonical_node_id": decision_identity,
            "display_name": str(value.get(
                "display_name",
                f"Resolution decision: {value.get('outcome', value.get('decision_status', 'evaluated'))}",
            )),
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "node_kind": "resolution_decision",
            "node_type": "resolution_decision",
            "source_system": "",
            "source_key": str(value.get("candidate_identity", "")),
            "schema_id": "",
            "schema_revision": "",
            "operation_name": "resolve_authority",
            "status": str(value.get("outcome", value.get("decision_status", "evaluated"))),
            "preview": str(value.get("reason_code", "")),
            "content_ref": "",
            "media_type": "",
        }
    for row in resolution_candidates:
        value = dict(row)
        value.setdefault("candidate_identity", value.get("identity", stable_identity(value)))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        relations["resolution_candidates"].append(value)
    for row in contract_evaluations:
        value = dict(row)
        value.setdefault("evaluation_identity", stable_identity(value))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        value.setdefault("violated_rules", [])
        relations["contract_evaluations"].append(value)
        evaluation_identity = str(value["evaluation_identity"])
        nodes[evaluation_identity] = {
            "canonical_node_id": evaluation_identity,
            "display_name": str(value.get("display_name", f"Contract evaluation: {value.get('decision', 'unknown')}")),
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "node_kind": "contract_evaluation",
            "node_type": "contract_evaluation",
            "source_system": "",
            "source_key": str(value.get("package_identity", "")),
            "schema_id": "",
            "schema_revision": "",
            "operation_name": "evaluate_context_contract",
            "status": str(value.get("decision", "unknown")),
            "preview": str(value.get("explanation", "")),
            "content_ref": "",
            "media_type": "",
        }
    for package in packages:
        value = dict(package)
        package_identity = str(value.get("package_identity", value.get("identity", stable_identity(package))))
        value.setdefault("package_identity", package_identity)
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        relations["context_packages"].append(value)
        nodes[package_identity] = {
            "canonical_node_id": package_identity,
            "display_name": str(value.get("display_name", f"Context package: {value.get('task_id', package_identity)}")),
            "context_model_id": context_model_id,
            "context_model_revision": revision,
            "node_kind": "context_package",
            "node_type": "context_package",
            "source_system": "",
            "source_key": str(value.get("task_id", "")),
            "schema_id": "",
            "schema_revision": "",
            "operation_name": "compile_package",
            "status": str(value.get("status", "compiled")),
            "preview": str(value.get("package_kind", "task_context")),
            "content_ref": "",
            "media_type": "",
        }
        for index, item in enumerate(value.get("items", ())):
            item_row = dict(item)
            item_row.setdefault("package_identity", package_identity)
            item_row.setdefault("package_item_identity", stable_identity({"package": package_identity, "index": index, "item": item}))
            package_items.append(item_row)
    for row in context_package_contracts:
        value = dict(row)
        value.setdefault("binding_identity", stable_identity(value))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        relations["context_package_contracts"].append(value)
    for item in package_items:
        value = dict(item)
        value.setdefault("package_item_identity", stable_identity(value))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        relations["package_items"].append(value)
    for edge in lineage_edges:
        value = dict(edge)
        value.setdefault("edge_identity", stable_identity(value))
        value.setdefault("context_model_id", context_model_id)
        value.setdefault("context_model_revision", revision)
        value.setdefault("operation_identity", "")
        value.setdefault("run_identity", "")
        relations["lineage_edges"].append(value)
        for endpoint in (value.get("source_node_id"), value.get("target_node_id")):
            if endpoint and str(endpoint) not in nodes:
                nodes[str(endpoint)] = {
                    "canonical_node_id": str(endpoint),
                    "display_name": str(endpoint),
                    "context_model_id": context_model_id,
                    "context_model_revision": revision,
                    "node_kind": "referenced",
                    "node_type": "referenced",
                    "source_system": "",
                    "source_key": "",
                    "schema_id": "",
                    "schema_revision": "",
                    "operation_name": "",
                    "status": "referenced",
                    "preview": "",
                    "content_ref": "",
                    "media_type": "",
                }
    relations["lineage_nodes"] = _dedupe(list(nodes.values()), "canonical_node_id")
    for name in STANDARD_CONTEXT_RELATIONS:
        key = _primary_key(name)
        relations[name] = _dedupe(relations[name], key)
    return relations


def _catalog_row(context_model_id: str, revision: str, identity: str, value: dict[str, Any],
                 physical_relation: str, object_kind: str) -> dict[str, Any]:
    return {
        "canonical_identity": identity,
        "display_name": str(value.get("display_name", identity)),
        "context_model_id": context_model_id,
        "context_model_revision": revision,
        "object_kind": object_kind,
        "schema_id": str(value.get("schema_id", "")),
        "schema_revision": str(value.get("schema_revision", "")),
        "source_system": str(value.get("source_system", "")),
        "source_key": str(value.get("source_key", "")),
        "media_type": str(value.get("media_type", "")),
        "evidence_preview": str(value.get("evidence_preview", value.get("preview", ""))),
        "preview": str(value.get("preview", value.get("evidence_preview", ""))),
        "content_ref": str(value.get("content_ref", "")),
        "operation_name": str(value.get("operation_name", "")),
        "status": str(value.get("status", "active")),
        "authority_status": str(value.get("authority_status", "")),
        "temporal_status": str(value.get("temporal_status", "")),
        "policy_id": str(value.get("policy_id", "")),
        "as_of": str(value.get("as_of", "")),
        "physical_relation": physical_relation,
    }


def _primary_key(name: str) -> str:
    return {
        "context_models": "context_model_identity",
        "context_objects": "object_identity",
        "context_snapshots": "snapshot_identity",
        "context_assertions": "assertion_identity",
        "context_catalog": "canonical_identity",
        "context_outputs": "identity",
        "context_retrieval": "retrieval_identity",
        "authority_policies": "policy_identity",
        "context_contracts": "contract_identity",
        "context_package_contracts": "binding_identity",
        "contract_evaluations": "evaluation_identity",
        "resolution_decisions": "decision_identity",
        "resolution_candidates": "candidate_identity",
        "lineage_nodes": "canonical_node_id",
        "lineage_edges": "edge_identity",
        "context_packages": "package_identity",
        "package_items": "package_item_identity",
    }[name]


def _dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_key = str(row.get(key, stable_identity(row)))
        row[key] = row_key
        values[row_key] = row
    return [values[item] for item in sorted(values)]
