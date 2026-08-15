"""Functional core and filesystem shell for the first support scenario pack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from igor_core.canonical import canonical_json, stable_identity

PACK_VERSION = "0.1.0"
PACK_ROOT = Path("scenarios/support/v0.1")
ARTIFACT_PATHS = (
    "fixtures/source.json",
    "fixtures/context-units.json",
    "mutations/source-change.json",
    "judgments/classification.json",
    "scorecard.json",
)

SOURCE_RECORDS = [
    {"record_id": "ticket-001", "channel": "email", "text": "I cannot reset my password."},
    {"record_id": "ticket-002", "channel": "chat", "text": "I was charged twice for the same order."},
    {"record_id": "ticket-003", "channel": "web", "text": "The service is unavailable for our whole team."},
    {"record_id": "ticket-004", "channel": "email", "text": "Where can I download my monthly invoice?"},
]
CONTEXT_UNITS = [
    {"context_unit_id": "ctx:support:001", "record_id": "ticket-001", "igor_support": {"kind": "conversation"}},
    {"context_unit_id": "ctx:support:002", "record_id": "ticket-002", "igor_support": {"kind": "conversation"}},
    {"context_unit_id": "ctx:support:003", "record_id": "ticket-003", "igor_support": {"kind": "conversation"}},
    {"context_unit_id": "ctx:support:004", "record_id": "ticket-004", "igor_support": {"kind": "conversation"}},
]
JUDGMENTS = {
    "schema_version": "0.1",
    "task_id": "support-intent-v0.1",
    "judgment_policy": {"source": "independent-human-review", "sealed_from_implementation": True},
    "labels": [
        {"context_unit_id": "ctx:support:001", "acceptable_labels": ["account_access"]},
        {"context_unit_id": "ctx:support:002", "acceptable_labels": ["billing_duplicate_charge"]},
        {"context_unit_id": "ctx:support:003", "acceptable_labels": ["service_outage"]},
        {"context_unit_id": "ctx:support:999", "acceptable_labels": [], "no_answer": True},
    ],
    "abstention_policy": "abstention is incorrect for answerable units and required for the no-answer unit",
}
SCORECARD = {
    "scorecard_id": "support-intent-accuracy-v0.1",
    "version": "0.1.0",
    "task_id": "support-intent-v0.1",
    "metrics": [{
        "metric_id": "answerable_exact_accuracy",
        "formula": "correct_answerable_predictions / answerable_judgments",
        "population": "judgments where no_answer is false",
        "acceptable_alternatives": "any label in acceptable_labels counts as correct",
        "abstention_treatment": "abstention is incorrect on answerable judgments",
        "no_answer_treatment": "excluded from accuracy; false-positive behavior is reported diagnostically",
        "threshold": {"operator": ">=", "value": 1.0},
    }],
    "failure_policy": "below-threshold quality is reported; conformance remains a separate pass/fail layer",
}


def resolve_context_model_reference(scenario: str | Path) -> dict[str, Any] | None:
    """Resolve a scenario's canonical semantic artifact by content identity."""
    root = Path(scenario)
    manifest = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    reference = manifest.get("context_model_ref")
    if reference is None:
        return None
    target = (root / reference).resolve()
    if not target.is_file():
        raise ValueError("context model reference does not resolve to a file")
    payload = target.read_bytes()
    return {"path": str(target), "content_sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}"}


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _artifact_descriptor(root: Path, relative: str) -> dict[str, Any]:
    payload = (root / relative).read_bytes()
    return {"path": relative, "media_type": "application/json", "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}", "size_bytes": len(payload)}


def _manifest(artifact_descriptors: list[dict[str, Any]], source_fixture: str) -> dict[str, Any]:
    identity_input = {
        "scenario_id": "support",
        "version": PACK_VERSION,
        "benchmark_contract": "0.1",
        "artifacts": artifact_descriptors,
        "scorecard_id": SCORECARD["scorecard_id"],
    }
    return {
        "schema_version": "0.1",
        "scenario_id": "support",
        "version": PACK_VERSION,
        "license": "Apache-2.0",
        "compatibility": {"benchmark_contract": "0.1", "evaluator": ">=0.1,<0.2"},
        "scenario_identity": stable_identity(identity_input),
        "source_fixture_identity": source_fixture,
        "artifacts": artifact_descriptors,
        "tasks": [{"task_id": "support-intent-v0.1", "inputs": ["fixtures/source.json", "fixtures/context-units.json"], "outputs": ["intent_label"], "judgments": "judgments/classification.json"}],
        "extensions": {"namespace": "igor_support", "fields": ["kind"]},
        "leakage_controls": {"judgments_are_separate": True, "implementation_inputs_exclude": ["judgments/", "mutations/source-change.json", "scorecard.json"]},
        "required_run_artifacts": ["context", "semantic", "retrieval", "provenance", "metrics"],
        "scorecard": "scorecard.json",
    }


def generate_support_pack(output: str | Path) -> Path:
    root = Path(output)
    for relative in ARTIFACT_PATHS:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / ARTIFACT_PATHS[0]).write_bytes(_json_bytes({"schema_version": "0.1", "records": SOURCE_RECORDS}))
    (root / ARTIFACT_PATHS[1]).write_bytes(_json_bytes({"schema_version": "0.1", "context_units": CONTEXT_UNITS}))
    (root / ARTIFACT_PATHS[2]).write_bytes(_json_bytes({"schema_version": "0.1", "mutation_id": "support-source-change-001", "base_fixture_identity": stable_identity(SOURCE_RECORDS), "changed_record_id": "ticket-002", "changed_fields": ["text"], "expected_affected": ["ctx:support:002"], "expected_unaffected": ["ctx:support:001", "ctx:support:003", "ctx:support:004"]}))
    (root / ARTIFACT_PATHS[3]).write_bytes(_json_bytes(JUDGMENTS))
    (root / ARTIFACT_PATHS[4]).write_bytes(_json_bytes(SCORECARD))
    descriptors = [_artifact_descriptor(root, relative) for relative in ARTIFACT_PATHS]
    manifest = _manifest(descriptors, stable_identity(SOURCE_RECORDS))
    (root / "scenario.json").write_bytes(_json_bytes(manifest))
    return root


def validate_scenario_pack(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    if not str(manifest["scenario_id"]).startswith("support") or manifest["version"] != PACK_VERSION:
        raise ValueError("unsupported support scenario identity or version")
    declared = manifest["artifacts"]
    if [item["path"] for item in declared] != list(ARTIFACT_PATHS):
        raise ValueError("scenario artifacts must be declared in canonical order")
    for item in declared:
        actual = _artifact_descriptor(root, item["path"])
        if actual != item:
            raise ValueError(f"artifact descriptor mismatch: {item['path']}")
    source = json.loads((root / "fixtures/source.json").read_text(encoding="utf-8"))
    units = json.loads((root / "fixtures/context-units.json").read_text(encoding="utf-8"))["context_units"]
    if any("label" in record or "intent" in record for record in source["records"]):
        raise ValueError("source fixture leaks gold labels")
    if len({unit["context_unit_id"] for unit in units}) != len(units):
        raise ValueError("context-unit identities must be unique")
    mutation = json.loads((root / "mutations/source-change.json").read_text(encoding="utf-8"))
    affected = set(mutation["expected_affected"])
    unaffected = set(mutation["expected_unaffected"])
    known = {unit["context_unit_id"] for unit in units}
    if not affected or affected & unaffected or not (affected | unaffected) <= known:
        raise ValueError("mutation impact sets must be disjoint and cover known context units")
    judgments = json.loads((root / "judgments/classification.json").read_text(encoding="utf-8"))
    if not judgments["judgment_policy"]["sealed_from_implementation"]:
        raise ValueError("judgments must be sealed from the implementation path")
    scorecard = json.loads((root / "scorecard.json").read_text(encoding="utf-8"))
    metric = scorecard["metrics"][0]
    for field in ("formula", "population", "abstention_treatment", "no_answer_treatment", "threshold"):
        if not metric.get(field):
            raise ValueError(f"scorecard metric missing {field}")
    return {"valid": True, "scenario_identity": manifest["scenario_identity"], "artifact_count": len(declared), "task_count": len(manifest["tasks"])}
