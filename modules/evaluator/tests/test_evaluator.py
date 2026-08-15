from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from igor_evaluator.evaluate import evaluate_document_live_artifact, evaluate_image_artifact, evaluate_submission, write_static_report


SCENARIO = Path("/opt/scenarios/support/v0.1")


def _make_run(tmp_path: Path, *, predictions: list[dict], broken_hash: bool = False, incomplete: bool = False) -> Path:
    root = tmp_path / "run"
    (root / "artifacts").mkdir(parents=True)
    metrics = json.dumps({"predictions": predictions}, sort_keys=True).encode() + b"\n"
    (root / "artifacts/metrics.json").write_bytes(metrics)
    if not incomplete:
        for name in ("context", "semantic", "retrieval", "provenance"):
            (root / f"artifacts/{name}.json").write_text("{}\n")
    links = {"benchmark_contract": "igor-benchmark-contract:0.1", "scenario_pack": "support:0.1", "scorecard": "support-scorecard:0.1", "implementation": "test:0.1", "source_fixture": "support-fixture:0.1", "transformation_config": "deterministic:0.1", "evaluator": "igor-evaluator:0.1"}
    manifest = {"schema_version": "0.1", "identity_links": links, "artifact_index_path": "artifact-index.json", "stage_result_paths": []}
    from igor_core.canonical import stable_identity
    descriptors = []
    for path in sorted((root / "artifacts").glob("*.json")):
        payload = path.read_bytes()
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        descriptors.append({"schema_version": "0.1", "path": str(path.relative_to(root)), "producer": {"name": "test", "revision": "1"}, "media_type": "application/json", "sha256": "sha256:" + "0" * 64 if broken_hash and path.name == "metrics.json" else digest, "size_bytes": len(payload)})
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "artifact-index.json").write_text(json.dumps({"schema_version": "0.1", "run_identity": stable_identity(links), "artifacts": descriptors}))
    return root


@pytest.fixture
def scenario(tmp_path: Path) -> Path:
    target = tmp_path / "scenario"
    shutil.copytree(SCENARIO, target)
    return target


def good_predictions() -> list[dict]:
    return [{"context_unit_id": "ctx:support:001", "intent_label": "account_access"}, {"context_unit_id": "ctx:support:002", "intent_label": "billing_duplicate_charge"}, {"context_unit_id": "ctx:support:003", "intent_label": "service_outage"}, {"context_unit_id": "ctx:support:999", "abstain": True}]


def test_valid_abstaining_submission_scores(scenario: Path, tmp_path: Path) -> None:
    result = evaluate_submission(scenario, _make_run(tmp_path, predictions=good_predictions()))
    assert result["conforming"] is True
    assert result["quality_threshold_met"] is True
    assert result["metric_vector"]["answerable_exact_accuracy"]["value"] == 1.0


@pytest.mark.parametrize("kwargs", [{"incomplete": True}, {"broken_hash": True}])
def test_malformed_and_incomplete_runs_are_nonconforming(scenario: Path, tmp_path: Path, kwargs: dict) -> None:
    result = evaluate_submission(scenario, _make_run(tmp_path, predictions=good_predictions(), **kwargs))
    assert result["conforming"] is False
    assert result["official_result"] is False


def test_wrong_and_no_answer_predictions_remain_diagnostic(scenario: Path, tmp_path: Path) -> None:
    predictions = [{"context_unit_id": "ctx:support:001", "intent_label": "wrong"}, {"context_unit_id": "ctx:support:999", "intent_label": "false_positive"}]
    result = evaluate_submission(scenario, _make_run(tmp_path, predictions=predictions))
    assert result["conforming"] is True
    assert result["quality_threshold_met"] is False
    assert result["metric_vector"]["no_answer_false_positive"] is True


def test_static_report_is_visual_and_artifact_only(scenario: Path, tmp_path: Path) -> None:
    run = _make_run(tmp_path, predictions=good_predictions())
    evaluation = evaluate_submission(scenario, run)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    report = write_static_report(run, evaluation_path, tmp_path / "report.html")
    html = report.read_text(encoding="utf-8")
    assert "Pipeline flow" in html
    assert "Evaluation scorecard" in html
    assert "Operational warning" in html
    assert "<svg class='lineage'" in html
    assert "<pre>" not in html


def test_document_live_evaluator_rejects_provider_mismatch(tmp_path: Path) -> None:
    artifact = {
        "profile": {"provider": "mistral", "model": "mistral-small-2603", "revision": "r1"},
        "outcomes": [{"status": "succeeded", "value": {"answer": "x", "abstain": False, "evidence_page_id": "p"},
                      "metadata": {"provider": "gemini", "model": "gemini-3.1-flash-lite", "provider_revision": "g1"}} for _ in range(8)],
        "stages": ["dlt", "lancedb", "retrieval", "resolution", "package", "datafusion-sql"],
        "enrichments": [], "sql": {"rows": [], "relations": []}, "mutation_reuse": {},
    }
    judgments = {"judgments": []}
    artifact_path, judgments_path = tmp_path / "artifact.json", tmp_path / "judgments.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    judgments_path.write_text(json.dumps(judgments), encoding="utf-8")
    result = evaluate_document_live_artifact(artifact_path, judgments_path)
    assert result["conforming"] is False
    assert next(item for item in result["conformance"] if item["check_id"] == "outcome_profile_matches_declared")["passed"] is False


def test_image_evaluator_is_artifact_only_and_requires_sealed_labels(tmp_path: Path) -> None:
    artifact = {
        "baseline": [{"source_record": {"asset_id": "a"},
                      "provider_outcomes": {"embedding": [{"status": "succeeded"}], "completion": [{"status": "succeeded"}]}}],
        "selective_invalidation": {"expected_reused": ["sha256:a"], "reused": ["sha256:a"]},
    }
    path = tmp_path / "image.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    result = evaluate_image_artifact(path)
    assert result["conforming"] is False
    assert next(item for item in result["conformance"] if item["check_id"] == "sealed-labels")["passed"] is True
    artifact["baseline"][0]["source_record"]["label"] = "hidden"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert next(item for item in evaluate_image_artifact(path)["conformance"] if item["check_id"] == "sealed-labels")["passed"] is False


def test_image_evaluator_rejects_declared_but_unexecuted_matrix(tmp_path: Path) -> None:
    artifact = {
        "mode": "deterministic",
        "baseline": [],
        "mutation_matrix": [{"mutation": str(index), "status": "passed"} for index in range(7)],
        "resolution": [],
    }
    path = tmp_path / "image-incomplete.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    result = evaluate_image_artifact(path)
    assert next(item for item in result["conformance"] if item["check_id"] == "mutation-matrix")["passed"] is False
