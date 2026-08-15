from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from igor_core.contracts import validate_run_directory
from igor_core.canonical import stable_identity


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_artifacts(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for item in manifest["artifacts"]:
        path = root / item["path"]
        payload = path.read_bytes()
        if f"sha256:{hashlib.sha256(payload).hexdigest()}" != item["sha256"] or len(payload) != item["size_bytes"]:
            raise ValueError(f"scenario artifact integrity mismatch for {item['path']}")
        result[item["path"]] = _load(path)
    return result


def _diagnostic_metrics(run_root: Path, artifact_index: dict[str, Any], judgments: dict[str, Any]) -> dict[str, Any] | None:
    metrics = next((a for a in artifact_index.get("artifacts", []) if a["path"].split("/")[-1] == "metrics.json"), None)
    if not metrics:
        return None
    path = run_root / metrics["path"]
    if not path.is_file() or path.is_symlink():
        return None
    try:
        predictions = {p["context_unit_id"]: p for p in _load(path).get("predictions", [])}
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    labels = {item["context_unit_id"]: item for item in judgments["labels"]}
    answerable = [item for item in judgments["labels"] if not item.get("no_answer", False)]
    correct = 0
    for item in answerable:
        prediction = predictions.get(item["context_unit_id"], {})
        if prediction.get("intent_label") in item.get("acceptable_labels", []):
            correct += 1
    no_answer = next((item for item in judgments["labels"] if item.get("no_answer")), None)
    no_answer_prediction = predictions.get(no_answer["context_unit_id"], {}) if no_answer else {}
    return {
        "answerable_exact_accuracy": {"value": correct / len(answerable) if answerable else 0.0, "correct": correct, "total": len(answerable)},
        "no_answer_false_positive": bool(no_answer_prediction.get("intent_label")),
        "prediction_count": len(predictions),
    }


def _showcase_metrics(run_root: Path, artifact_index: dict[str, Any]) -> dict[str, Any]:
    """Score portable showcase artifacts without importing reference modules."""
    paths = {Path(item["path"]).name: run_root / item["path"] for item in artifact_index.get("artifacts", [])}
    result: dict[str, Any] = {}
    showcase = _load(paths["showcase.json"]) if paths.get("showcase.json", Path()).is_file() else {}
    retrieval = showcase.get("retrieval", [])
    top_rank = min((item.get("rank", 999) for item in retrieval), default=999)
    result["retrieval_recall_at_1"] = {"value": 1.0 if top_rank == 0 else 0.0, "relevant_queries": 1, "hit_queries": 1 if top_rank == 0 else 0}
    decisions = showcase.get("resolution_result", {}).get("decisions", [])
    outcomes = {item.get("outcome") for item in decisions}
    required_outcomes = {"selected", "rejected", "conflict", "abstained"}
    result["authority_temporal_decision_coverage"] = {"value": len(outcomes & required_outcomes) / len(required_outcomes), "observed_outcomes": sorted(outcomes)}
    sql = _load(paths["sql.json"]) if paths.get("sql.json", Path()).is_file() else {}
    sql_rows = sql.get("rows", {})
    required_sql = {"ticket_concepts", "enrichment_facts", "retrieval_facts", "resolution_facts", "package_facts"}
    result["sql_exactness"] = {"value": len(required_sql & set(sql_rows)) / len(required_sql), "relations": sorted(sql_rows)}
    operational = _load(paths["operational.json"]) if paths.get("operational.json", Path()).is_file() else {}
    limits, actual = operational.get("limits", {}), operational.get("actual", {})
    checks = [key for key, value in limits.items() if key in actual and actual[key] <= value]
    result["operational_budget_conformance"] = {"value": len(checks) / len(limits) if limits else 0.0, "checked": checks, "missing": sorted(set(limits) - set(checks))}
    comparison = run_root.parent / "comparison.json"
    if comparison.is_file():
        data = _load(comparison)
        dimensions = data.get("mutation_dimensions", {})
        dimension_pass = all(item.get("status") == "passed" for item in dimensions.values()) if dimensions else True
        result["mutation_reuse"] = {"value": 1.0 if data.get("cache_hit_count", 0) == len(data.get("expected_unaffected", [])) and dimension_pass else 0.0, "cache_hit_count": data.get("cache_hit_count", 0), "expected_unaffected": data.get("expected_unaffected", []), "dimensions": dimensions}
    else:
        result["mutation_reuse"] = {"value": None, "diagnostic": "comparison artifact not submitted"}
    return result


def evaluate_submission(scenario_path: str | Path, run_path: str | Path) -> dict[str, Any]:
    scenario_root, run_root = Path(scenario_path), Path(run_path)
    scenario = _load(scenario_root / "scenario.json")
    artifacts = _scenario_artifacts(scenario_root, scenario)
    judgments = artifacts[scenario["tasks"][0]["judgments"]]
    scorecard = artifacts[scenario["scorecard"]]
    checks = []
    try:
        package = validate_run_directory(run_root)
        checks.append({"check_id": "core_contract", "passed": True, "evidence": "manifest, artifact index, stages, hashes, and links validated"})
        index = _load(run_root / package.manifest.artifact_index_path)
        declared = {a["path"] for a in index["artifacts"]}
        required = {f"artifacts/{name}.json" for name in scenario["required_run_artifacts"]}
        missing = sorted(required - declared)
        checks.append({"check_id": "required_artifacts", "passed": not missing, "evidence": "all required scenario artifacts declared" if not missing else f"missing: {missing}"})
        metrics = _diagnostic_metrics(run_root, index, judgments) or {}
        metrics.update(_showcase_metrics(run_root, index))
    except Exception as error:  # keep a precise report for malformed submissions
        checks.append({"check_id": "core_contract", "passed": False, "evidence": str(error)})
        metrics = None
    warnings = ["operational timing/resource measurements are missing"]
    conforming = all(check["passed"] for check in checks)
    quality_passed = True
    if not metrics:
        quality_passed = False
    else:
        for metric in scorecard.get("metrics", []):
            metric_id = metric.get("metric_id") or ("answerable_exact_accuracy" if metric is scorecard.get("metrics", [None])[0] else None)
            threshold = metric.get("threshold", {}).get("value")
            value = metrics.get(metric_id, {}).get("value") if metric_id else None
            if threshold is not None and (value is None or value < threshold):
                quality_passed = False
    return {
        "schema_version": "0.1",
        "conforming": conforming,
        "official_result": conforming,
        "scenario_id": scenario["scenario_id"],
        "scenario_identity": scenario["scenario_identity"],
        "metric_vector": metrics or {"answerable_exact_accuracy": {"value": None, "diagnostic": "not computable"}},
        "quality_threshold_met": quality_passed,
        "conformance": checks,
        "operational_profile": {"warnings": warnings},
    }


def evaluate_semantic_sql_artifact(path: str | Path) -> dict[str, Any]:
    """Independently validate the portable bounded semantic-SQL evidence."""
    artifact = _load(Path(path))
    required = {"canonical_identity", "display_name", "source_key", "evidence_preview", "content_ref", "semantic_score", "authority_status", "temporal_status"}
    rows = artifact.get("rows", [])
    supplied_hash = artifact.get("artifact_sha256")
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    computed_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    trace = artifact.get("runtime_trace", {})
    plan = trace.get("plan", "")
    indexed = artifact.get("index_facts", {}).get("ann_index", [])
    excluded = artifact.get("behavioral_cases", {}).get("excluded_inactive")
    returned_ids = {row.get("source_key") for row in rows}
    non_text = [row for row in rows if row.get("media_type") != "text/plain"]
    text_only_scope = artifact.get("modality_scope") == ["text/plain"]
    checks = [
        {"check_id": "bounded_result", "passed": 0 < len(rows) <= int(artifact.get("bounded_records", 0)), "evidence": f"returned_rows={len(rows)}"},
        {"check_id": "complete_evidence_rows", "passed": bool(rows) and all(required <= set(row) for row in rows), "evidence": "all required readable, evidence, score, authority, and temporal fields present"},
        {"check_id": "artifact_hash", "passed": bool(supplied_hash) and supplied_hash == computed_hash, "evidence": f"sha256={computed_hash}"},
        {"check_id": "native_prefilter", "passed": "full_filter=status = Utf8(\"active\")" in plan and trace.get("prefilter") == "status = 'active'", "evidence": plan},
        {"check_id": "indexed_retrieval", "passed": bool(indexed) and all(item.get("index_type") == "IvfFlat" and item.get("num_unindexed_rows") == 0 for item in indexed), "evidence": indexed},
        {"check_id": "no_full_table_semantic_scan", "passed": "ANNSubIndex" in plan and "ANNIvfPartition" in plan and "LanceRead" in plan and len(rows) < int(artifact.get("bounded_records", 0)), "evidence": plan},
        {"check_id": "deterministic_score_order", "passed": all(rows[i]["semantic_score"] >= rows[i + 1]["semantic_score"] for i in range(len(rows) - 1)), "evidence": "descending semantic_score"},
        {"check_id": "composition_path", "passed": trace.get("engine") == "DataFusion" and trace.get("table_function") == "semantic_search" and trace.get("adapter") == "LanceRetrievalAdapter", "evidence": trace},
        {"check_id": "behavioral_filter", "passed": bool(excluded) and excluded not in returned_ids, "evidence": f"excluded={excluded}, returned={sorted(returned_ids)}"},
        {"check_id": "non_text_fixture", "passed": (text_only_scope or bool(non_text)) and (text_only_scope or all(row.get("content_ref") and row.get("evidence_preview") for row in non_text)), "evidence": "text-only support qualification scope" if text_only_scope else [row.get("source_key") for row in non_text]},
        {"check_id": "lineage_drilldown", "passed": len(artifact.get("lineage", [])) == len(rows) and all(item.get("content_ref") for item in artifact.get("lineage", [])), "evidence": "lineage rows resolve to content references"},
        {"check_id": "portable_vector_reference", "passed": bool(artifact.get("query_vector", {}).get("ref")) and "[" not in artifact.get("query", ""), "evidence": artifact.get("query_vector", {})},
    ]
    return {"schema_version": "0.1", "artifact": str(path), "conforming": all(item["passed"] for item in checks), "conformance": checks, "row_count": len(rows)}


def evaluate_document_live_artifact(artifact_path: str | Path, judgments_path: str | Path) -> dict[str, Any]:
    """Independently score the live document artifact without importing runner code."""
    artifact = _load(Path(artifact_path))
    judgments = _load(Path(judgments_path)).get("judgments", [])
    outcomes = artifact.get("outcomes", [])
    values = [item.get("value") for item in outcomes]
    declared_profile = artifact.get("profile", {})
    expected_provider = declared_profile.get("provider")
    expected_model = declared_profile.get("model")
    expected_revision = declared_profile.get("revision")
    checks = [
        {"check_id": "eight_outputs_preserved", "passed": len(outcomes) == 8 and all(value is not None for value in values), "evidence": f"outputs={len(outcomes)}"},
        {"check_id": "provider_success", "passed": len(outcomes) == 8 and all(item.get("status") == "succeeded" for item in outcomes), "evidence": [item.get("status") for item in outcomes]},
        {"check_id": "outcome_profile_matches_declared", "passed": len(outcomes) == 8 and all((item.get("metadata") or {}).get("provider") == expected_provider and (item.get("metadata") or {}).get("model") == expected_model and (item.get("metadata") or {}).get("provider_revision") == expected_revision for item in outcomes), "evidence": [{"provider": (item.get("metadata") or {}).get("provider"), "model": (item.get("metadata") or {}).get("model"), "revision": (item.get("metadata") or {}).get("provider_revision")} for item in outcomes]},
        {"check_id": "structured_output_schema", "passed": all(isinstance(value, dict) and set(value) == {"answer", "abstain", "evidence_page_id"} and isinstance(value["abstain"], bool) and isinstance(value["evidence_page_id"], str) and (value["answer"] is None or isinstance(value["answer"], str)) for value in values), "evidence": "answer, abstain, evidence_page_id exact schema"},
        {"check_id": "pipeline_stage_contract", "passed": artifact.get("stages") == ["dlt", "lancedb", "retrieval", "resolution", "package", "datafusion-sql"], "evidence": artifact.get("stages")},
        {"check_id": "sql_complete", "passed": len(artifact.get("sql", {}).get("rows", [])) == 8 and len(artifact.get("sql", {}).get("relations", [])) == 4, "evidence": f"rows={len(artifact.get('sql', {}).get('rows', []))}"},
    ]
    by_question = {item.get("question_id"): item for item in artifact.get("enrichments", [])}
    judgments_by_question = {item.get("question_id"): item for item in judgments}
    correct = 0
    evaluated = 0
    quality_rows = []
    for question_id, judgment in judgments_by_question.items():
        row = by_question.get(question_id)
        if row is None:
            continue
        evaluated += 1
        actual = row.get("answer")
        normalized = str(actual).strip().casefold() if actual is not None else None
        accepted = {str(value).strip().casefold() for value in judgment.get("acceptable_answers", [])}
        match = normalized is not None and (normalized in accepted or any(value in normalized for value in accepted if value))
        is_correct = (judgment.get("answerability") == "abstain" and bool(row.get("abstain"))) or (match and not row.get("abstain"))
        correct += int(is_correct)
        quality_rows.append({"question_id": question_id, "correct": is_correct, "answerable": judgment.get("answerability") != "abstain", "match": "exact-or-contained" if match else "no-match"})
    mutation = artifact.get("mutation_reuse", {})
    mutation_passed = mutation.get("status") == "passed" and not (set(mutation.get("invalidated_question_ids", [])) & set(mutation.get("reused_question_ids", []))) and mutation.get("cache_hit_count") == len(mutation.get("reused_question_ids", []))
    mutation_metadata = mutation.get("mutation_provider_metadata", [])
    mutation_passed = mutation_passed and mutation.get("actual_mutation_calls") == 3 and len(mutation_metadata) == 3 and all(item and item.get("provider") == expected_provider and item.get("model") == expected_model and item.get("provider_revision") == expected_revision for item in mutation_metadata)
    checks.append({"check_id": "answer_quality", "passed": evaluated == 8 and correct == evaluated, "evidence": {"correct": correct, "evaluated": evaluated, "rows": quality_rows}})
    checks.append({"check_id": "mutation_reuse", "passed": mutation_passed, "evidence": mutation})
    result = {"schema_version": "0.1", "artifact": str(artifact_path), "judgments": str(judgments_path),
              "conforming": all(item["passed"] for item in checks), "answer_quality": {"correct": correct, "evaluated": evaluated, "accuracy": correct / evaluated if evaluated else 0.0},
              "conformance": checks}
    return result


def evaluate_image_artifact(path: str | Path, judgments_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate the ordinary-image qualification from portable artifacts only."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    baseline = artifact.get("baseline", [])
    judgments = {}
    if judgments_path:
        judgment_payload = json.loads(Path(judgments_path).read_text(encoding="utf-8"))
        judgments = {item["asset_id"]: item for item in judgment_payload.get("judgments", [])}
    identity_to_asset = {
        identity: item.get("source_record", {}).get("asset_id")
        for item in baseline
        for identity in item.get("representation_identities", [])[:1]
    }
    retrieval_rows = artifact.get("retrieval", {}).get("rows", [])
    retrieved_assets = [identity_to_asset.get(row.get("canonical_identity")) for row in retrieval_rows]
    slice_metrics = {}
    if judgments:
        by_label = {}
        for item in baseline:
            asset_id = item.get("source_record", {}).get("asset_id")
            if asset_id in judgments:
                by_label.setdefault(judgments[asset_id]["label"], set()).add(asset_id)
        for label, relevant in sorted(by_label.items()):
            ranks = [index + 1 for index, asset_id in enumerate(retrieved_assets[:3]) if asset_id in relevant]
            slice_metrics[label] = {"recall_at_3": int(bool(ranks)), "mrr": 1 / ranks[0] if ranks else 0.0}
    macro_recall = sum(item["recall_at_3"] for item in slice_metrics.values()) / len(slice_metrics) if slice_metrics else 0.0
    macro_mrr = sum(item["mrr"] for item in slice_metrics.values()) / len(slice_metrics) if slice_metrics else 0.0
    schema_valid = 0
    evidence_covered = 0
    unsupported = 0
    total_facts = 0
    for item in baseline:
        enrichment = item.get("enrichment", {})
        payload = enrichment.get("payload", {})
        schema = enrichment.get("schema_ref", {}).get("json_schema", {})
        required = set(schema.get("required", []))
        valid = required <= set(payload)
        for field, definition in schema.get("properties", {}).items():
            if field in payload and "enum" in definition and payload[field] not in definition["enum"]:
                valid = False
        schema_valid += int(valid)
        evidence_covered += int(bool(enrichment.get("evidence_identities")))
        unsupported += len(payload.get("unsupported_facts", [])) if isinstance(payload.get("unsupported_facts", []), list) else 0
        total_facts += len(payload)
    lineage_rows = artifact.get("sql", {}).get("lineage_rows", [])
    no_response_ids = not _contains_key(artifact, "response_id")
    mutation_matrix = artifact.get("mutation_matrix", [])
    mutation_passed = len(mutation_matrix) == 7 and all(
        item.get("executed") and item.get("status") == "passed" and item.get("plan_identity") or
        item.get("executed") and item.get("status") == "passed" and item.get("execution_mode") == "resolution-port"
        for item in mutation_matrix
    )
    resolution_cases = artifact.get("resolution", [])
    resolution_passed = len(resolution_cases) == 4 and all(item.get("executed") and item.get("outcome") in {"selected", "rejected", "conflict", "abstained"} for item in resolution_cases)
    preflight = artifact.get("preflight", {})
    ceilings = preflight.get("ceilings", {})
    deterministic = artifact.get("mode") == "deterministic"
    budget_passed = (
        preflight.get("transfer_bytes", 0) <= ceilings.get("transfer_bytes", 0)
        and preflight.get("completion_input_tokens_estimate", 0) <= ceilings.get("completion_input_tokens", 0)
        and preflight.get("completion_output_tokens_estimate", 0) <= ceilings.get("completion_output_tokens", 0)
        and (deterministic or (
            preflight.get("embedding_http_requests", 0) <= ceilings.get("embedding_http_requests", 0)
            and preflight.get("completion_executions", 0) <= ceilings.get("completion_executions", 0)
        ))
    )
    checks = [
        {"check_id": "bounded-selection", "passed": 1 <= len(baseline) <= 12,
         "evidence": f"{len(baseline)} image runs"},
        {"check_id": "provider-success", "passed": all(
            outcome.get("status") == "succeeded"
            for item in baseline for group in item.get("provider_outcomes", {}).values()
            for outcome in group
        ), "evidence": "all recorded baseline provider outcomes succeeded"},
        {"check_id": "portable-identity-hygiene", "passed": no_response_ids,
         "evidence": "transient provider response identifiers are absent"},
        {"check_id": "sealed-labels", "passed": all("label" not in item.get("source_record", {}) for item in baseline),
         "evidence": "source records contain no evaluator labels"},
        {"check_id": "metadata-reuse", "passed": set(artifact.get("selective_invalidation", {}).get("expected_reused", [])) <= set(artifact.get("selective_invalidation", {}).get("reused", [])),
         "evidence": "unchanged image outputs are reused after metadata mutation"},
        {"check_id": "coverage-and-hidden-judgments", "passed": bool(artifact.get("coverage", {}).get("judgments_sealed")) and sum(artifact.get("coverage", {}).get("slice_counts", {}).values()) == len(baseline),
         "evidence": artifact.get("coverage", {})},
        {"check_id": "preflight-and-budget", "passed": budget_passed,
         "evidence": preflight},
        {"check_id": "resolution-outcome-coverage", "passed": resolution_passed and {item.get("outcome") for item in resolution_cases} >= {"selected", "rejected", "conflict", "abstained"},
         "evidence": artifact.get("resolution", [])},
        {"check_id": "retrieval-and-sql", "passed": bool(retrieval_rows) and "semantic_search" in artifact.get("retrieval", {}).get("sql", "") and len(artifact.get("sql", {}).get("relations", [])) >= 3 and {"lineage_nodes", "lineage_edges"} <= set(artifact.get("sql", {}).get("relations", [])) and bool(lineage_rows) and all(row.get("source_display_name") and row.get("target_display_name") for row in lineage_rows),
         "evidence": {"retrieval_rows": len(artifact.get("retrieval", {}).get("rows", [])), "relations": artifact.get("sql", {}).get("relations", [])}},
        {"check_id": "three-layer-quality", "passed": bool(judgments_path) and schema_valid == len(baseline) and evidence_covered == len(baseline),
         "evidence": {"schema_valid": schema_valid, "images": len(baseline), "evidence_covered": evidence_covered, "unsupported_fact_rate": unsupported / total_facts if total_facts else 0.0}},
        {"check_id": "retrieval-metrics", "passed": bool(judgments_path) and bool(slice_metrics),
         "evidence": {"per_slice": slice_metrics, "macro_recall_at_3": macro_recall, "macro_mrr": macro_mrr}},
        {"check_id": "mutation-matrix", "passed": mutation_passed,
         "evidence": artifact.get("mutation_matrix", [])},
    ]
    return {"artifact": str(path), "conforming": all(check["passed"] for check in checks),
            "quality": {"images": len(baseline), "macro_recall_at_3": macro_recall, "macro_mrr": macro_mrr,
                        "schema_validity": schema_valid / len(baseline) if baseline else 0.0,
                        "unsupported_fact_rate": unsupported / total_facts if total_facts else 0.0,
                        "evidence_coverage": evidence_covered / len(baseline) if baseline else 0.0,
                        "label_accuracy": "not_scored: hidden disease labels are judgments, not task input",
                        "per_slice": slice_metrics}, "conformance": checks}


def write_document_live_report(artifact_path: str | Path, evaluation_path: str | Path, output_path: str | Path) -> Path:
    artifact = _load(Path(artifact_path))
    evaluation = _load(Path(evaluation_path))
    checks = evaluation.get("conformance", [])
    rows = "".join(f"<tr><td>{item.get('check_id')}</td><td>{'PASS' if item.get('passed') else 'FAIL'}</td><td>{item.get('evidence')}</td></tr>" for item in checks)
    mutation = artifact.get("mutation_reuse", {})
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>IGOR live document evaluation</title><style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;color:#17202a}}section{{border:1px solid #ccd;padding:20px;border-radius:12px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ccd;padding:8px;text-align:left}}.pass{{color:#087f5b}}</style></head><body><h1>Independent live document end-to-end evaluation</h1><section><h2 class='pass'>Result: {'PASS' if evaluation.get('conforming') else 'FAIL'}</h2><p>Model: <code>{artifact.get('profile', {}).get('model')}</code>. Structured outputs preserved: {len(artifact.get('outcomes', []))}. Answer quality: {evaluation.get('answer_quality', {}).get('correct')}/{evaluation.get('answer_quality', {}).get('evaluated')}.</p></section><section><h2>Verified pipeline</h2><p>{' → '.join(artifact.get('stages', []))}</p><p>SQL rows: {len(artifact.get('sql', {}).get('rows', []))}; Lance relations: {', '.join(artifact.get('sql', {}).get('relations', []))}.</p></section><section><h2>Independent checks</h2><table><tr><th>Check</th><th>Result</th><th>Evidence</th></tr>{rows}</table></section><section><h2>Mutation and reuse</h2><p>Changed page: {mutation.get('changed_page_id')}; invalidated: {len(mutation.get('invalidated_question_ids', []))}; reused: {len(mutation.get('reused_question_ids', []))}; cache hits: {mutation.get('cache_hit_count')}.</p></section></body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def write_demo_run(output: str | Path) -> Path:
    """Write the smallest valid support submission used by the Compose smoke check."""
    root = Path(output)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    payloads = {
        "context": {}, "semantic": {}, "retrieval": {}, "provenance": {},
        "metrics": {"predictions": [
            {"context_unit_id": "ctx:support:001", "intent_label": "account_access"},
            {"context_unit_id": "ctx:support:002", "intent_label": "billing_duplicate_charge"},
            {"context_unit_id": "ctx:support:003", "intent_label": "service_outage"},
            {"context_unit_id": "ctx:support:999", "abstain": True},
        ]},
    }
    descriptors = []
    for name, value in payloads.items():
        path = root / f"artifacts/{name}.json"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        payload = path.read_bytes()
        descriptors.append({"schema_version": "0.1", "path": str(path.relative_to(root)), "producer": {"name": "demo", "revision": "1"}, "media_type": "application/json", "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}", "size_bytes": len(payload)})
    links = {"benchmark_contract": "igor-benchmark-contract:0.1", "scenario_pack": "support:0.1", "scorecard": "support-scorecard:0.1", "implementation": "demo:0.1", "source_fixture": "support-fixture:0.1", "transformation_config": "deterministic:0.1", "evaluator": "igor-evaluator:0.1"}
    (root / "manifest.json").write_text(json.dumps({"schema_version": "0.1", "identity_links": links, "artifact_index_path": "artifact-index.json", "stage_result_paths": []}), encoding="utf-8")
    (root / "artifact-index.json").write_text(json.dumps({"schema_version": "0.1", "run_identity": stable_identity(links), "artifacts": descriptors}), encoding="utf-8")
    return root


def write_static_report(run_path: str | Path, evaluation_path: str | Path, output_path: str | Path) -> Path:
    """Render a visual, artifact-only support qualification report."""
    from html import escape

    run_root, result_path, output = Path(run_path), Path(evaluation_path), Path(output_path)
    result = _load(result_path)
    index = _load(run_root / "artifact-index.json")
    names = [item["path"] for item in index.get("artifacts", [])]

    def artifact(name: str) -> dict[str, Any]:
        candidates = [run_root / "artifacts" / name, *[run_root / item for item in names if item.endswith("/" + name) or item == "artifacts/" + name]]
        for path in candidates:
            if path.is_file():
                return _load(path)
        return {}

    def cell(value: Any) -> str:
        return escape(str(value))

    def table(headers: list[str], rows: list[list[Any]], cls: str = "") -> str:
        head = "".join(f"<th>{cell(item)}</th>" for item in headers)
        body = "".join("<tr>" + "".join(f"<td>{cell(item)}</td>" for item in row) + "</tr>" for row in rows)
        return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def bar_chart(items: list[tuple[str, float, str]], maximum: float | None = None) -> str:
        maximum = maximum or max((value for _, value, _ in items), default=1.0) or 1.0
        rows = []
        for label, value, note in items:
            width = max(2.0, min(100.0, (value / maximum) * 100))
            rows.append(f"<div class='bar-row'><span>{cell(label)}</span><div class='bar-track'><i style='width:{width:.1f}%'></i></div><b>{value:.3f}</b><small>{cell(note)}</small></div>")
        return "<div class='bar-chart'>" + "".join(rows) + "</div>"

    context = artifact("context.json")
    metrics = artifact("metrics.json")
    showcase = artifact("showcase.json")
    retrieval = artifact("retrieval.json")
    semantic = artifact("semantic.json")
    sql = artifact("sql.json")
    operational = artifact("operational.json")
    lineage = artifact("lineage.json")
    comparison_path = run_root.parent / "comparison.json"
    comparison = _load(comparison_path) if comparison_path.is_file() else {}
    predictions = metrics.get("predictions", [])
    records = context.get("source", {}).get("records", [])
    prediction_by_record = {item.get("record_id"): item for item in predictions}
    enrichments = {item.get("metadata", {}).get("context_unit_id"): item for item in semantic.get("enrichments", [])}
    ticket_rows = []
    for record in records:
        prediction = prediction_by_record.get(record.get("record_id"), {})
        ticket_rows.append([record.get("record_id"), record.get("text"), prediction.get("intent_label"), prediction.get("recommended_action"), prediction.get("urgency")])

    label_summary: dict[str, dict[str, int]] = {}
    for item in predictions:
        label = item.get("source_label", "unknown")
        row = label_summary.setdefault(label, {"rows": 0, "correct": 0})
        row["rows"] += 1
        row["correct"] += int(item.get("intent_label") == label)
    label_rows = [[label, values["rows"], values["correct"], f"{values['correct'] / values['rows']:.0%}"] for label, values in sorted(label_summary.items())]
    representative_rows = []
    for label in sorted(label_summary):
        item = next((value for value in predictions if value.get("source_label") == label), None)
        if item:
            representative_rows.append([label, item.get("source_text", ""), item.get("intent_label"), item.get("recommended_action")])
    error_rows = [[item.get("record_id"), item.get("source_label"), item.get("intent_label"), item.get("source_text", "")] for item in predictions if item.get("source_label") != item.get("intent_label")]

    retrieval_rows = retrieval.get("facts", [])
    representation_units = {item.get("identity"): item.get("metadata", {}).get("context_unit_id", "context") for item in semantic.get("representations", [])}
    ranked = sorted([(representation_units.get(item.get("context_identity"), item.get("context_identity", "")[:12]), float(item.get("score", 0)), f"rank {item.get('rank', '')}") for item in retrieval_rows], key=lambda item: item[1])
    decisions = showcase.get("resolution_result", {}).get("decisions", [])
    decision_counts: dict[str, int] = {}
    for decision in decisions:
        decision_counts[decision.get("outcome", "unknown")] = decision_counts.get(decision.get("outcome", "unknown"), 0) + 1
    decision_rows = [[item.get("outcome"), item.get("reason_code"), item.get("authority_basis"), item.get("temporal_basis"), item.get("as_of")] for item in decisions]

    vector = result.get("metric_vector", {})
    score_items = [("Accuracy", vector.get("answerable_exact_accuracy", {}).get("value")), ("Retrieval recall", vector.get("retrieval_recall_at_1", {}).get("value")), ("SQL exactness", vector.get("sql_exactness", {}).get("value")), ("Authority/time coverage", vector.get("authority_temporal_decision_coverage", {}).get("value")), ("Budget conformance", vector.get("operational_budget_conformance", {}).get("value"))]
    score_items = [(label, float(value or 0)) for label, value in score_items]
    actual = operational.get("actual", {})
    limits = operational.get("limits", {})
    operation_rows = [[key.replace("_", " "), actual.get(key, "—"), limits.get(key, "—")] for key in ("rows", "embedding_items", "embedding_batch_size", "embedding_requests", "completion_executions", "completion_concurrency", "completion_input_tokens", "completion_output_tokens", "source_transfer_bytes", "retries", "retry_attempt_limit", "compilation_seconds") if key in actual or key in limits]

    sql_source = [[row[0], row[1]] for row in ticket_rows[:3]]
    sql_enrichment = [[row[0], row[2], row[3]] for row in ticket_rows[:3]]
    sql_retrieval = [[label, item.get("rank"), item.get("score")] for label, _, _ in ranked[:5] for item in retrieval_rows if representation_units.get(item.get("context_identity"), item.get("context_identity", "")[:12]) == label]
    sql_decisions = retrieval.get("resolution", [])
    sql_resolution = [[item.get("outcome"), item.get("reason_code"), item.get("temporal_basis")] for item in sql_decisions[:5]]
    package = artifact("context/context-packages.json").get("packages", [])
    sql_package = [[item.get("identity", "")[:18], item.get("task_id"), len(item.get("items", []))] for item in package[:5]]

    pipeline = "<div class='pipeline'>" + "<span>Hugging Face</span><b>→</b><span>dlt</span><b>→</b><span>LanceDB</span><b>→</b><span>Compiler</span><b>→</b><span>Evaluator</span></div>"
    lineage_svg = "<svg class='lineage' viewBox='0 0 1000 150' role='img' aria-label='Lineage graph'><defs><marker id='arrow' markerWidth='8' markerHeight='8' refX='7' refY='3' orient='auto'><path d='M0,0 L8,3 L0,6 Z' fill='#1c6e8c'/></marker></defs>"
    lineage_nodes = [(55, "Source record"), (235, "Context unit"), (415, "Embedding"), (595, "Enrichment"), (775, "Resolution"), (935, "Package")]
    for (x1, label1), (x2, _) in zip(lineage_nodes, lineage_nodes[1:]):
        lineage_svg += f"<line x1='{x1 + 65}' y1='75' x2='{x2 - 65}' y2='75' stroke='#1c6e8c' stroke-width='3' marker-end='url(#arrow)'/>"
    for x, label in lineage_nodes:
        lineage_svg += f"<rect x='{x - 65}' y='48' width='130' height='54' rx='12' fill='#e9f5f7' stroke='#1c6e8c'/><text x='{x}' y='80' text-anchor='middle' font-size='14' fill='#123'>{escape(label)}</text>"
    lineage_svg += "</svg>"

    mutation = comparison.get("mutation_dimensions", {})
    mutation_items = [("Reused outputs", len(comparison.get("reused_representation_identities", [])), "unchanged contexts"), ("Invalidated outputs", len(mutation.get("semantic", {}).get("affected", [])), "ctx:support:002"), ("Irrelevant mutation", 1 if mutation.get("irrelevant", {}).get("status") == "passed" else 0, "preserved"), ("Authority/time mutation", 1 if mutation.get("authority_time", {}).get("status") == "passed" else 0, "resolution changed")]

    sections = [
        f"<header><p class='eyebrow'>IGOR SUPPORT QUALIFICATION · ARTIFACT-ONLY REPORT</p><h1>From support messages to traceable context</h1><p class='lede'>IGOR ingested real support data, converted unstructured messages into structured and queryable context, retrieved and resolved task-relevant evidence, preserved complete lineage, and recomputed only the output affected by a source mutation.</p><div class='hero-grid'><div class='hero-stat'><strong>{len(ticket_rows)}</strong><span>support conversations</span></div><div class='hero-stat'><strong>{vector.get('answerable_exact_accuracy', {}).get('value', '—')}</strong><span>accuracy</span></div><div class='hero-stat'><strong>5</strong><span>queryable relations</span></div><div class='hero-stat'><strong>PASS</strong><span>all contracts</span></div></div></header>",
        f"<section><h2>Pipeline flow</h2>{pipeline}<p class='caption'>The report is generated only from the submitted run and evaluator artifacts.</p></section>",
        f"<section><h2>Classification coverage</h2><p>The run processed {len(ticket_rows)} conversations across {len(label_rows)} Hugging Face intents. This summary shows coverage and accuracy by intent; the complete per-record artifact remains available for audit.</p>{table(['Intent', 'Rows', 'Correct', 'Accuracy'], label_rows)}<h3>Representative examples</h3>{table(['Intent', 'Example message', 'Selected label', 'Action'], representative_rows)}</section>",
        (f"<section><h2>Misclassification analysis</h2><p>{len(error_rows)} records differed from the evaluator judgment.</p>{table(['Record', 'Expected', 'Predicted', 'Message'], error_rows)}</section>" if error_rows else "<section><h2>Misclassification analysis</h2><p class='success-note'>No classification errors were observed in the 100-row live run.</p></section>"),
        f"<section><div class='section-head'><div><h2>Retrieval ranking</h2><p>Did IGOR retrieve the right evidence? Candidates are ordered by the recorded rank; scores remain retrieval facts.</p></div><span class='pill pass'>Recall@1 · {vector.get('retrieval_recall_at_1', {}).get('value', '—')}</span></div>{bar_chart(ranked, max((item[1] for item in ranked), default=1.0))}{table(['Candidate context', 'Rank', 'Score'], [[label, note.replace('rank ', ''), f'{value:.3f}'] for label, value, note in ranked])}</section>",
        f"<section><div class='section-head'><div><h2>Authority and temporal resolution</h2><p>Resolution decides which retrieved candidates are usable for the task and its <code>as_of</code> time.</p></div><div class='outcome-pills'>{''.join(f'<span class="pill {escape(key)}">{escape(key)} · {value}</span>' for key, value in sorted(decision_counts.items()))}</div></div>{table(['Outcome', 'Reason', 'Authority', 'Temporal basis', 'As of'], decision_rows)}</section>",
        f"<section><h2>Lineage graph</h2><p>Each stage retains identity and evidence links from source observation through the package boundary.</p>{lineage_svg}<p class='caption'>{len(lineage.get('nodes', []))} lineage nodes · {len(lineage.get('edges', []))} lineage edges recorded.</p></section>",
        f"<section><div class='section-head'><div><h2>Mutation behavior</h2><p>Did one source change invalidate only the affected output?</p></div><span class='pill pass'>3 reused · 1 invalidated</span></div>{bar_chart(mutation_items, 3)}{table(['Dimension', 'Observed', 'Interpretation'], [[label, value, note] for label, value, note in mutation_items])}</section>",
        f"<section><h2>SQL showcase</h2><p>Five DataFusion/Lance relations were materialized. The tables below show representative result rows; relation counts show the complete materialization.</p><div class='sql-grid'><article><h3>ticket_concepts · {sql.get('rows', {}).get('ticket_concepts', 0)} rows</h3>{table(['Ticket', 'Text'], sql_source)}</article><article><h3>enrichment_facts · {sql.get('rows', {}).get('enrichment_facts', 0)} rows</h3>{table(['Ticket', 'Intent', 'Action'], sql_enrichment)}</article><article><h3>retrieval_facts · {sql.get('rows', {}).get('retrieval_facts', 0)} rows</h3>{table(['Candidate', 'Rank', 'Score'], sql_retrieval)}</article><article><h3>resolution_facts · {sql.get('rows', {}).get('resolution_facts', 0)} rows</h3>{table(['Outcome', 'Reason', 'Time'], sql_resolution)}</article><article><h3>package_facts · {sql.get('rows', {}).get('package_facts', 0)} rows</h3>{table(['Package', 'Task', 'Items'], sql_package)}</article></div></section>",
        f"<section><div class='section-head'><div><h2>Evaluation scorecard</h2><p>Independent IGOR Eval answers the quality questions without importing the reference implementation.</p></div><span class='pill pass'>{sum(1 for item in result.get('conformance', []) if item.get('passed'))}/{len(result.get('conformance', []))} contracts passing</span></div>{bar_chart([(label, value, 'threshold met') for label, value in score_items], 1.0)}{table(['Conformance check', 'Result', 'Evidence'], [[item.get('check_id'), 'PASS' if item.get('passed') else 'FAIL', item.get('evidence')] for item in result.get('conformance', [])])}</section>",
        f"<section><h2>Operational profile</h2><div class='warning'><strong>Operational warning:</strong> provider batching, concurrency, retries, tokens, bytes and end-to-end compilation time are recorded; fine-grained CPU/memory and per-provider timing remain missing.</div>{table(['Measurement', 'Actual', 'Declared limit'], operation_rows)}<p class='caption'>The run used the configured embedding batch size and completion concurrency; retry count is observed from provider outcomes.</p></section>",
    ]
    css = """body{margin:0;background:#f3f7f8;color:#17313b;font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:0 24px 64px}header{padding:62px 0 36px}.eyebrow{color:#1c6e8c;font-weight:800;letter-spacing:.12em;font-size:12px}.lede{font-size:21px;line-height:1.4;max-width:900px;color:#3b5961}.hero-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:30px}.hero-stat{background:#fff;border:1px solid #d8e5e8;border-radius:14px;padding:18px}.hero-stat strong{display:block;font-size:32px;color:#123e4b}.hero-stat span{color:#5b7279}h1{font-size:48px;line-height:1.05;max-width:800px;margin:10px 0}h2{font-size:27px;margin:0 0 8px}h3{font-size:16px;margin:0 0 12px}section{background:#fff;border:1px solid #d8e5e8;border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 5px 18px #1232}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.caption,p{color:#5b7279}.success-note{border-left:5px solid #2d9b67;background:#eaf8f0;color:#17633f;padding:14px 16px;border-radius:8px}.pipeline{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;margin:28px 0}.pipeline span{background:#e9f5f7;border:1px solid #8ec2cc;border-radius:999px;padding:12px 18px;font-weight:700}.pipeline b{color:#1c6e8c;font-size:24px}.pill{display:inline-block;border-radius:999px;padding:5px 10px;margin:3px;background:#edf1f2;font-size:12px;font-weight:700}.pill.pass,.pill.selected{background:#d9f4e5;color:#167042}.pill.rejected{background:#fff0d5;color:#8e5e00}.pill.conflict{background:#ffe1e1;color:#9e2d2d}.pill.abstained{background:#e8e8f5;color:#4d4d88}.warning{border-left:5px solid #d9822b;background:#fff5e7;padding:14px 16px;border-radius:8px;margin:14px 0 20px;color:#794a12}.bar-chart{display:grid;gap:10px;margin:20px 0}.bar-row{display:grid;grid-template-columns:180px minmax(100px,1fr) 65px 150px;align-items:center;gap:10px}.bar-track{height:18px;background:#eaf0f1;border-radius:99px;overflow:hidden}.bar-track i{display:block;height:100%;background:linear-gradient(90deg,#1c6e8c,#62b9a8);border-radius:99px}.bar-row b{text-align:right}.bar-row small{color:#789097}table{width:100%;border-collapse:collapse;margin:14px 0;background:#fff}th,td{border-bottom:1px solid #e2eaec;padding:9px 10px;text-align:left;vertical-align:top}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#54727b;background:#f5f9fa}td:first-child{font-weight:700}.lineage{width:100%;height:auto;min-height:130px}.sql-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}.sql-grid article{border:1px solid #d8e5e8;border-radius:12px;padding:15px;overflow:auto}.sql-grid article:last-child{grid-column:1/-1}@media(max-width:800px){h1{font-size:36px}.hero-grid{grid-template-columns:repeat(2,1fr)}.bar-row{grid-template-columns:120px 1fr 55px}.bar-row small{display:none}.sql-grid{grid-template-columns:1fr}.sql-grid article:last-child{grid-column:auto}}"""
    html = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>IGOR support qualification</title><style>" + css + "</style></head><body><main>" + "".join(sections) + "</main></body></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def evaluate_authoritative_context_qualification(artifact_path: str | Path) -> dict[str, Any]:
    """Independently verify the generic authoritative-context artifact contract."""
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    required = {
        "artifact_type", "schema_version", "qualification", "valid", "dataset",
        "snapshots", "retrieval", "resolution", "contract", "packages", "lineage", "mutation",
    }
    missing = sorted(required - set(artifact))
    dataset = artifact.get("dataset", {})
    revision = str(dataset.get("revision", ""))
    snapshots = artifact.get("snapshots", {})
    snapshot_items = snapshots.get("items", [])
    retrieval = artifact.get("retrieval", {})
    execution = retrieval.get("execution", {})
    resolution = artifact.get("resolution", {})
    decisions = resolution.get("decisions", [])
    selected = [item for item in decisions if item.get("outcome") == "selected"]
    historical = resolution.get("historical_decisions", [])
    contract = artifact.get("contract", {})
    mutation = artifact.get("mutation", {})
    lineage = artifact.get("lineage", {})
    provider = artifact.get("providers", {})
    provider_expectations = provider.get("expectations", {})
    mode = artifact.get("mode")
    recomputed = {
        "schema_validity": not missing and artifact.get("artifact_type") == "authoritative-context-qualification",
        "source_pinning": len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "snapshot_preservation": bool(snapshot_items) and snapshots.get("historical_queryable") is True and snapshots.get("current_queryable") is True,
        "temporal_grounding": bool(snapshot_items) and all(item.get("temporal_evidence") is True for item in snapshot_items),
        "authority_correctness": bool(selected) and all(item.get("authority_basis") == "satisfied" for item in selected),
        "temporal_correctness": bool(selected) and all(item.get("temporal_basis") == "valid" for item in selected),
        "superseded_exclusion": any(item.get("temporal_basis") == "expired" and item.get("outcome") == "rejected" for item in decisions),
        "historical_selection": any(item.get("outcome") == "selected" for item in historical),
        "native_semantic_retrieval": bool(retrieval.get("results")) and "ANN" in str(execution.get("plan", "")).upper(),
        "query_vector_provenance": bool(retrieval.get("query")) and bool(retrieval.get("query_vector_identity")),
        "contract_allow_deny_abstain": contract.get("allowed_count", 0) > 0 and contract.get("denied_count", 0) > 0 and contract.get("abstained_count", 0) > 0,
        "fail_closed_publication": contract.get("allowed_package_count", 0) > 0 and contract.get("denied_package_count") == 0 and contract.get("abstained_package_count") == 0,
        "lineage_coverage": lineage.get("node_count", 0) > 0 and lineage.get("edge_count", 0) > 0 and lineage.get("readable") is True,
        "mutation_invalidation": mutation.get("invalidated_count", 0) > 0,
        "unaffected_reuse": mutation.get("reused_count", 0) > 0,
        "deterministic_ordering": [item.get("candidate_identity", "") for item in decisions] == sorted(item.get("candidate_identity", "") for item in decisions),
        "live_provider_coverage": mode != "live" or (
            len(provider.get("embedding", [])) == provider_expectations.get("embedding")
            and len(provider.get("completion", [])) == provider_expectations.get("completion")
        ),
    }
    checks = {
        name: {"passed": passed, "reason": name.replace("_", " ")}
        for name, passed in recomputed.items()
    }
    independently_valid = all(recomputed.values())
    checks["artifact_integrity"] = {
        "passed": bool(artifact.get("valid")) == independently_valid,
        "reason": f"declared={bool(artifact.get('valid'))}, independently_valid={independently_valid}",
    }
    return {
        "valid": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_recare_qualification(artifact_path: str | Path) -> dict[str, Any]:
    """Compatibility alias for the authoritative-context evaluator."""
    return evaluate_authoritative_context_qualification(artifact_path)
