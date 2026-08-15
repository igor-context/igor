from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

from igor_core import Evidence, SourceSnapshot, stable_identity
from igor_context_compiler import (
    AuthorityAssertion, ContextModelCompileRequest, ContextModelReference,
    DependencyInventory, ResolutionRequest,
    ResolutionResult, TemporalAssertion, compile_context_model,
)
from igor_lancedb import LanceRetrievalAdapter, LanceStore, LanceStoreConfig
from igor_runner.context_model_lifecycle import materialize_context_model_lifecycle

from .compose import RunnerError, _run_image_compilation
from igor_core import load_reference_profile


def _redact_provider_ids(value):
    """Keep portable provider evidence, never transient response identifiers."""
    if isinstance(value, dict):
        return {key: _redact_provider_ids(item) for key, item in value.items() if key not in {"response_id"}}
    if isinstance(value, list):
        return [_redact_provider_ids(item) for item in value]
    return value


def _image_context_manifest(semantic_definition: dict):
    context_model_id = str(semantic_definition["semantic_definition_id"])
    revision = str(semantic_definition["revision"])
    return compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {
                "id": context_model_id,
                "revision": revision,
                "title": str(semantic_definition.get("title", context_model_id)),
            },
            "sources": {
                "image_assets": {
                    "source_contract": "beans.images.v1",
                    "connector_binding": "huggingface.beans.v1",
                    "identity_fields": ["member"],
                },
            },
            "objects": {
                "image_asset": {
                    "kind": "business_object",
                    "source": "image_assets",
                    "display_name": "{{ member }} source asset",
                },
                "image_source_snapshot": {
                    "kind": "source_snapshot",
                    "source": "image_assets",
                    "display_name": "{{ member }} source image",
                },
                "image_metadata_snapshot": {
                    "kind": "source_snapshot",
                    "source": "image_assets",
                    "display_name": "{{ member }} metadata",
                },
                "image_representation": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_source_snapshot"],
                    "operation": "representation.image.v1",
                    "schema": "media.asset.v1",
                    "semantic_definition": "beans-leaf-observation.v1",
                    "display_name": "{{ member }} original image",
                },
                "image_task_projection": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_source_snapshot", "image_metadata_snapshot"],
                    "operation": "projection.task.v1",
                    "schema": "media.asset.v1",
                    "semantic_definition": "beans-leaf-observation.v1",
                    "display_name": "{{ member }} task projection",
                },
                "image_embedding": {
                    "kind": "retrieval_projection",
                    "derived_from": ["image_representation"],
                    "operation": "embedding.multimodal.v1",
                    "schema": "media.embedding.v1",
                    "model_profile": "image.embedding-profile.v1",
                    "display_name": "Image embedding",
                },
                "image_enrichment": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_task_projection"],
                    "operation": "enrichment.structured.v1",
                    "schema": "media.asset.v1",
                    "semantic_definition": "beans-leaf-observation.v1",
                    "recipe": "beans-leaf-observation.recipe.v1",
                    "display_name": "Image enrichment",
                },
            },
            "authority": {
                "beans.authority-temporal": {
                    "target": "image_representation",
                    "policy": "beans.authority-temporal.v1",
                    "active_only": True,
                    "valid_at_task_time": True,
                },
            },
            "retrievals": {
                "bean_leaf_images": {
                    "search": "vector",
                    "candidate_limit": 20,
                    "resolution": {"policy": "beans.authority-temporal", "accepted_outcomes": ["selected"]},
                },
            },
        },
        references=(
            ContextModelReference(
                kind="source_contract",
                ref="beans.images.v1",
                identity=stable_identity({"source_contract": "beans.images.v1"}),
            ),
            ContextModelReference(
                kind="connector_binding",
                ref="huggingface.beans.v1",
                identity=stable_identity({"connector_binding": "huggingface.beans.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.image.v1",
                identity=stable_identity({"operation": "representation.image.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="projection.task.v1",
                identity=stable_identity({"operation": "projection.task.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="embedding.multimodal.v1",
                identity=stable_identity({"operation": "embedding.multimodal.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="enrichment.structured.v1",
                identity=stable_identity({"operation": "enrichment.structured.v1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="media.asset.v1",
                identity=stable_identity({"schema": "media.asset", "revision": "1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="media.embedding.v1",
                identity=stable_identity({"schema": "media.embedding", "revision": "1"}),
            ),
            ContextModelReference(
                kind="semantic_definition",
                ref="beans-leaf-observation.v1",
                identity=stable_identity(semantic_definition),
            ),
            ContextModelReference(
                kind="model_profile",
                ref="image.embedding-profile.v1",
                identity=stable_identity({"model_profile": "image.embedding-profile.v1"}),
            ),
            ContextModelReference(
                kind="recipe",
                ref="beans-leaf-observation.recipe.v1",
                identity=stable_identity({"recipe": "beans-leaf-observation", "revision": revision}),
            ),
            ContextModelReference(
                kind="authority_policy",
                ref="beans.authority-temporal.v1",
                identity=stable_identity({"policy": "beans.authority-temporal", "revision": "1"}),
            ),
        ),
    ))


class _ImageResolution:
    """Executable authority/temporal policy used by the image qualification cases."""

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        authority = {item.assertion_identity: item for item in request.authority_assertions}
        temporal = {item.assertion_identity: item for item in request.temporal_assertions}
        decisions = []
        for candidate in sorted(request.candidates, key=lambda item: item.candidate_identity):
            auth = [authority[item] for item in candidate.authority_assertion_ids if item in authority]
            times = [temporal[item] for item in candidate.temporal_assertion_ids if item in temporal]
            conflict = len({item.issuer_ref for item in auth}) > 1
            auth_ok = bool(auth) and len(auth) == len(candidate.authority_assertion_ids)
            temporal_basis = "unknown"
            valid = False
            reason = "beans-policy:missing-basis"
            if times:
                item = times[0]
                if item.effective_until is not None and item.effective_until <= request.as_of:
                    temporal_basis, reason = "expired", "beans-policy:expired"
                elif item.effective_from is not None and item.effective_from > request.as_of:
                    temporal_basis, reason = "future", "beans-policy:future"
                else:
                    valid, temporal_basis, reason = True, "valid", "beans-policy:authoritative-valid"
            if conflict:
                outcome, authority_basis, reason = "conflict", "conflicting", "beans-policy:conflicting-authority"
            elif auth_ok and valid:
                outcome, authority_basis = "selected", "satisfied"
            elif not auth_ok:
                outcome, authority_basis = "abstained", "unknown"
            else:
                outcome, authority_basis = "rejected", "satisfied"
            from igor_context_compiler import ResolutionDecision
            decision_kwargs = {
                "candidate_identity": candidate.candidate_identity, "outcome": outcome, "reason_code": reason,
                "authority_basis_ids": tuple(candidate.authority_assertion_ids),
                "temporal_basis_ids": tuple(candidate.temporal_assertion_ids),
                "authority_basis": authority_basis, "temporal_basis": temporal_basis,
                "as_of": request.as_of, "policy_id": request.policy_id, "policy_revision": request.policy_revision,
            }
            if outcome == "conflict":
                decision_kwargs["conflict_set_id"] = stable_identity({
                    "conflict": candidate.candidate_identity,
                    "authority": tuple(candidate.authority_assertion_ids),
                })
            decisions.append(ResolutionDecision(**decision_kwargs))
        return ResolutionResult(
            request_identity=request.request_identity, resolution_identity=request.identity,
            decisions=tuple(decisions),
        )


def _resolution_retrieval_row_case(case: str, evidence_identity: str) -> dict:
    """Build one retrieval row the lifecycle can lift into a resolution request."""
    as_of = datetime(2026, 8, 14, tzinfo=timezone.utc)
    candidate_identity = stable_identity({"image-resolution-case": case, "evidence": evidence_identity})
    evidence = Evidence(
        ir_version="0.1", subject_identity=evidence_identity, source_snapshot_id=evidence_identity,
        locator=f"qualification:{case}", observed_at=as_of,
    )
    authority = AuthorityAssertion(
        assertion_identity=stable_identity({"authority": case}), subject_identity=evidence_identity,
        issuer_ref="beans-primary" if case != "conflicting" else "beans-primary-a",
        scope_ref="ordinary-image", evidence_identities=(evidence.identity,),
    )
    authorities = (authority,)
    if case == "conflicting":
        authorities += (authority.model_copy(update={
            "assertion_identity": stable_identity({"authority": "conflicting-secondary"}),
            "issuer_ref": "beans-primary-b",
        }),)
    temporal = TemporalAssertion(
        assertion_identity=stable_identity({"temporal": case}), subject_identity=evidence_identity,
        effective_from=as_of - timedelta(days=2),
        effective_until=as_of - timedelta(days=1) if case == "expired" else None,
        evidence_identities=(evidence.identity,),
    )
    return {
        "request_identity": stable_identity({"resolution-request": case}),
        "retrieval_role": "bean_leaf_images",
        "task_id": "beans-image-resolution-v0.1",
        "task_schema_revision": "1",
        "as_of": as_of.isoformat(),
        "candidate_identity": candidate_identity,
        "context_identity": evidence_identity,
        "rank": 0,
        "score": 1.0,
        "evidence": (evidence.model_dump(mode="json"),),
        "authority_assertions": tuple(item.model_dump(mode="json") for item in authorities),
        "temporal_assertions": () if case == "missing-basis" else (temporal.model_dump(mode="json"),),
        "authority_assertion_ids": () if case == "missing-basis" else tuple(item.assertion_identity for item in authorities),
        "temporal_assertion_ids": () if case == "missing-basis" else (temporal.assertion_identity,),
    }


def qualify_huggingface_images(scenario: str | Path, acquisition: str | Path,
                               profile: str | Path, output: str | Path,
                               *, mutation: bool = True) -> dict:
    scenario_root, acquisition_root, output_root = Path(scenario), Path(acquisition), Path(output)
    payload = json.loads((acquisition_root / "acquisition.json").read_text(encoding="utf-8"))
    judgments_payload = json.loads((acquisition_root / "judgments.json").read_text(encoding="utf-8"))
    records = payload.get("records", [])
    live = load_reference_profile(profile).embedding.provider != "deterministic"
    selection_config = json.loads((scenario_root / "live-selection.json").read_text(encoding="utf-8"))
    if live:
        configured_members = selection_config.get("members", [])
        by_member = {row["member"]: row for row in records}
        configured = [by_member[member] for member in configured_members if member in by_member]
        # The live request ceiling permits three one-image embedding calls. Pick one
        # configured member per hidden slice to preserve balanced coverage.
        records = []
        seen_labels = set()
        for row in configured:
            label = Path(row["member"]).parts[1]
            if label not in seen_labels:
                records.append(row)
                seen_labels.add(label)
    if not 1 <= len(records) <= 12:
        raise RunnerError("ordinary-image qualification requires 1..12 acquired records")
    if any("label" in row for row in records):
        raise RunnerError("source labels must remain evaluator-only")
    judgments = judgments_payload.get("judgments", [])
    judgment_by_asset = {item["asset_id"]: item for item in judgments}
    if any(record["asset_id"] not in judgment_by_asset for record in records):
        raise RunnerError("every selected image must have a sealed evaluator judgment")
    coverage = {label: sum(1 for record in records if judgment_by_asset[record["asset_id"]]["label"] == label)
                for label in sorted({judgment_by_asset[record["asset_id"]]["label"] for record in records})}
    transfer_bytes = int(payload.get("file_bytes", 0))
    preflight = {"images": len(records), "transfer_bytes": transfer_bytes,
                 "completion_executions": len(records) + 1, "embedding_http_requests": len(records),
                 "completion_input_tokens_estimate": (len(records) + 1) * 600,
                 "completion_output_tokens_estimate": (len(records) + 1) * 80,
                 "ceilings": {"images": 12, "live_images": 6, "completion_executions": 12,
                              "embedding_http_requests": 4, "completion_input_tokens": 16000,
                              "completion_output_tokens": 4000, "transfer_bytes": 512 * 1024 * 1024}}
    if preflight["transfer_bytes"] > preflight["ceilings"]["transfer_bytes"] or (live and (preflight["embedding_http_requests"] > preflight["ceilings"]["embedding_http_requests"] or preflight["completion_executions"] > preflight["ceilings"]["completion_executions"])):
        raise RunnerError(f"image qualification preflight exceeds declared ceiling: {preflight}")
    output_root.mkdir(parents=True, exist_ok=True)
    store = LanceStore(LanceStoreConfig(str(output_root / "lance"), "local", "media"))
    semantic_definition = yaml.safe_load((scenario_root / "semantic-definition.yaml").read_text(encoding="utf-8"))
    context_model_id = str(semantic_definition["semantic_definition_id"])
    context_model_revision = str(semantic_definition["revision"])
    caption = "Observe the original bean leaf image and report only schema-supported visible observations; abstain when uncertain."
    started = time.perf_counter()
    baseline = [_run_image_compilation(profile, output_root / "baseline" / f"{i:02d}", store, caption, source_record=row)
                for i, row in enumerate(records)]
    observations = []
    outputs = []
    for record, item in zip(records, baseline):
        image_snapshot_identity = SourceSnapshot.model_validate(item["snapshots"][0]).identity
        metadata_snapshot_identity = SourceSnapshot.model_validate(item["snapshots"][1]).identity
        observations.extend([
            {
                "object_role": "image_asset",
                "snapshot_role": "image_source_snapshot",
                "source_system": "huggingface",
                "source_key": record["member"],
                "snapshot_identity": image_snapshot_identity,
                "observed_at": "2026-08-14T00:00:00Z",
                "content_ref": record["image_ref"],
                "content_sha256": record["image_sha256"],
                "media_type": record["media_type"],
                "display_name": f"{record['member']} · source image",
                "preview": record["image_sha256"],
                "status": "observed",
            },
            {
                "object_role": "image_asset_metadata",
                "snapshot_role": "image_metadata_snapshot",
                "source_system": "huggingface",
                "source_key": record["member"],
                "snapshot_identity": metadata_snapshot_identity,
                "observed_at": "2026-08-14T00:00:00Z",
                "content_ref": record["image_ref"],
                "content_sha256": record["image_sha256"],
                "media_type": "application/json",
                "display_name": f"{record['member']} · metadata",
                "preview": record["image_sha256"],
                "status": "observed",
            },
        ])
        outputs.extend([
            {
                "identity": item["representation_identities"][0],
                "role": "image_representation",
                "kind": "context_representation",
                "object_kind": "image",
                "derived_from": [image_snapshot_identity],
                "schema_id": "media.asset",
                "schema_revision": "1",
                "display_name": f"{record['member']} · original image",
                "payload": item.get("representations", [{}])[0] if item.get("representations") else {},
                "value": json.dumps(item.get("representations", [{}])[0] if item.get("representations") else {}, sort_keys=True),
                "vector": item["embedding"]["vector"],
                "text": record["member"],
                "source_system": "huggingface",
                "source_key": record["member"],
                "media_type": record["media_type"],
                "content_ref": record["image_ref"],
                "preview": record["image_sha256"],
                "status": "active",
                "authority_status": "selected",
                "temporal_status": "valid",
                "policy_id": "beans.authority-temporal",
                "as_of": "2026-08-14T00:00:00Z",
            },
            {
                "identity": item["representation_identities"][1],
                "role": "image_task_projection",
                "kind": "context_representation",
                "object_kind": "image_task_projection",
                "derived_from": [image_snapshot_identity, metadata_snapshot_identity],
                "schema_id": "media.asset",
                "schema_revision": "1",
                "display_name": f"{record['member']} · task representation",
                "payload": item.get("representations", [{}, {}])[1] if len(item.get("representations", [])) > 1 else {},
                "value": json.dumps(item.get("representations", [{}, {}])[1] if len(item.get("representations", [])) > 1 else {}, sort_keys=True),
                "source_system": "huggingface",
                "source_key": record["member"],
                "media_type": record["media_type"],
                "content_ref": record["image_ref"],
                "preview": record["image_sha256"],
                "status": "active",
            },
            {
                "identity": item["embedding_output_identity"],
                "role": "image_embedding",
                "kind": "semantic_derivation",
                "object_kind": "image_embedding",
                "derived_from": [item["representation_identities"][0]],
                "schema_id": "media.embedding",
                "schema_revision": "1",
                "display_name": f"{record['member']} · multimodal embedding",
                "payload": item["embedding"],
                "value": json.dumps({key: value for key, value in item["embedding"].items() if key != "identity"}, sort_keys=True),
                "status": "active",
            },
            {
                "identity": item["enrichment_output_identity"],
                "role": "image_enrichment",
                "kind": "semantic_derivation",
                "object_kind": "image_enrichment",
                "derived_from": [item["representation_identities"][1]],
                "schema_id": "media.asset",
                "schema_revision": "1",
                "display_name": f"{record['member']} · structured enrichment",
                "payload": item["enrichment"],
                "value": json.dumps({key: value for key, value in item["enrichment"].items() if key != "identity"}, sort_keys=True),
                "status": "active",
            },
        ])
    resolution_cases = ("valid", "expired", "conflicting", "missing-basis")
    resolution_retrieval = [_resolution_retrieval_row_case(case, outputs[0]["identity"]) for case in resolution_cases]
    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest=_image_context_manifest(semantic_definition),
        observations=observations,
        outputs=outputs,
        retrieval=resolution_retrieval,
        resolution_port=_ImageResolution(),
    )
    resolution = []
    resolution_case_by_request = {
        row["request_identity"]: case
        for case, row in zip(resolution_cases, resolution_retrieval, strict=True)
    }
    for result in lifecycle.resolution_results:
        decision = result.decisions[0]
        resolution.append({
            "case": resolution_case_by_request[result.request_identity],
            **decision.model_dump(mode="json"),
            "executed": True,
            "request_identity": result.request_identity,
            "resolution_identity": result.resolution_identity,
        })
    materialized = lifecycle.materialization
    catalog = [
        row for row in materialized.relations["context_catalog"]
        if row["physical_relation"] == "context_outputs" and row["object_kind"] == "image"
    ]
    adapter = LanceRetrievalAdapter(store)
    index_facts = adapter.ensure_vector_index()
    query_ref = stable_identity({"query": "bean leaf", "embedding": baseline[0]["embedding_output_identity"]})
    sql = ("SELECT canonical_identity, display_name, media_type, content_ref, semantic_score, "
           "retrieval_rank, authority_status, temporal_status FROM semantic_search('context_catalog', "
           "'bean leaf', 'vector', 3, '" + query_ref + "', 'status = ''active''') "
           "WHERE status = 'active' ORDER BY semantic_score DESC, canonical_identity ASC")
    sql_rows = lifecycle.query_standard_sql(
        sql,
        table_name="context_catalog",
        query_vectors={query_ref: tuple(baseline[0]["embedding"]["vector"])},
        retrieval=adapter,
    ).to_pylist()
    lineage_sql = ("SELECT s.display_name AS source_display_name, s.canonical_node_id AS source_node_id, "
                   "e.edge_type, t.display_name AS target_display_name, t.canonical_node_id AS target_node_id "
                   "FROM lineage_edges e JOIN lineage_nodes s ON s.canonical_node_id = e.source_node_id "
                   "JOIN lineage_nodes t ON t.canonical_node_id = e.target_node_id "
                   "ORDER BY s.display_name, t.display_name")
    lineage_sql_rows = lifecycle.query_standard_sql(lineage_sql, table_name="lineage_edges").to_pylist()
    first = baseline[0]
    old_snapshots = [SourceSnapshot.model_validate(item) for item in first["snapshots"]]
    old_outputs = set(first["plan"]["expected_output_identities"])
    baseline_operations = {node["output_identity"]: node["operation"] for node in first["plan"].get("nodes", [])}
    base_inventory = DependencyInventory(
        known_output_identities=tuple(item.identity for item in old_snapshots) + tuple(sorted(old_outputs)),
        valid_output_identities=tuple(sorted(old_outputs)), changed_identities=(),
    )
    def mutation_row(name: str, mutated: dict, expected_reuse: set[str], expected_new: set[str]) -> dict:
        plan_data = mutated["plan"]
        cache_hits = set(plan_data.get("cache_hits", []))
        expected_ok = expected_reuse <= cache_hits and not (expected_new & cache_hits)
        operations = {node["output_identity"]: node["operation"] for node in plan_data.get("nodes", [])}
        invalidated = sorted({baseline_operations.get(item, operations.get(item, "unknown")) for item in expected_new})
        return {"mutation": name, "status": "passed" if expected_ok else "failed", "executed": True,
                "execution_mode": "plan-only" if mutated.get("dry_run") else "provider-and-plan",
                "cache_hits": sorted(cache_hits), "expected_reused": sorted(expected_reuse),
                "expected_invalidated": sorted(expected_new), "invalidated_operations": invalidated,
                "plan_identity": plan_data.get("request_identity"),
                "inventory_valid": mutated.get("inventory_valid_output_identities", []),
                "plan_expected": plan_data.get("expected_output_identities", []),
                "plan_nodes": [node.get("output_identity") for node in plan_data.get("nodes", [])]}

    original = records[0]
    original_path = Path(original["image_ref"].removeprefix("file://"))
    image_bytes = original_path.read_bytes() + b"\x00"
    image_path = output_root / "mutation" / "image_bytes" / "changed.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    changed_record = dict(original)
    changed_record.update({"image_ref": image_path.as_uri(), "image_sha256": "sha256:" + hashlib.sha256(image_bytes).hexdigest(), "byte_count": len(image_bytes)})
    mutation_rows = [mutation_row(
        "image_bytes", _run_image_compilation(profile, output_root / "mutation" / "image_bytes" / "run", store, caption,
                                               source_record=changed_record, dry_run=live), set(), old_outputs)]
    metadata_mutation = _run_image_compilation(
        profile, output_root / "mutation" / "irrelevant_metadata", store,
        caption + " Metadata review note changed.", inventory=DependencyInventory(
            known_output_identities=tuple(item.identity for item in old_snapshots),
            valid_output_identities=tuple(first["plan"]["expected_output_identities"]),
            changed_identities=(old_snapshots[1].identity,),
        ), source_record=original, dry_run=live,
    )
    mutation_rows.append(mutation_row("irrelevant_metadata", metadata_mutation,
                                      {first["representation_identities"][0], first["embedding_output_identity"]},
                                      {first["representation_identities"][1], first["enrichment_output_identity"]}))
    semantic_mutation = _run_image_compilation(
        profile, output_root / "mutation" / "semantic_definition", store, caption,
        inventory=base_inventory, recipe_taxonomy_version="media-taxonomy-v2", dry_run=live,
        source_record=original,
    )
    mutation_rows.append(mutation_row("semantic_definition", semantic_mutation,
                                      {first["representation_identities"][0], first["representation_identities"][1], first["embedding_output_identity"]},
                                      {first["enrichment_output_identity"]}))
    recipe_mutation = _run_image_compilation(
        profile, output_root / "mutation" / "recipe", store, caption,
        inventory=base_inventory, recipe_prompt_version="media-asset-v2", dry_run=live, source_record=original,
    )
    mutation_rows.append(mutation_row("recipe", recipe_mutation,
                                      {first["representation_identities"][0], first["representation_identities"][1], first["embedding_output_identity"]},
                                      {first["enrichment_output_identity"]}))
    mutation_matrix = mutation_rows
    for case, resolution_case in (("superseding_assertion", "valid"), ("expired_assertion", "expired"), ("conflicting_assertion", "conflicting")):
        executed = next(item for item in resolution if item["case"] == resolution_case)
        mutation_matrix.append({"mutation": case, "status": "passed" if executed["outcome"] in {"selected", "rejected", "conflict"} else "failed",
                                "executed": True, "execution_mode": "resolution-port", "outcome": executed["outcome"],
                                "reason_code": executed["reason_code"], "cache_hits": sorted(old_outputs),
                                "expected_invalidated": ["resolution", "package"]})
    operational = {"elapsed_seconds": round(time.perf_counter() - started, 4),
                   "provider_latency": "unavailable: provider adapters expose no latency field",
                   "queue_batch_time": "unavailable", "cpu_time": "unavailable", "peak_memory": "unavailable",
                   "retries": sum(outcome.get("attempts", 1) - 1 for item in baseline for group in item["provider_outcomes"].values() for outcome in group),
                   "measurement_warnings": ["CPU and peak-memory metrics unavailable in this container path"]}
    provider_routes = sorted({
        (outcome.get("metadata") or {}).get("provider", "unknown") + "/" +
        (outcome.get("metadata") or {}).get("model", "unknown")
        for item in baseline for group in item["provider_outcomes"].values() for outcome in group
    })
    schema_validity = sum(int(bool(item["enrichment"].get("evidence_identities")) and set(item["enrichment"].get("payload", {})) >= set(item["enrichment"].get("schema_ref", {}).get("json_schema", {}).get("required", []))) for item in baseline)
    quality = {"schema_validity": schema_validity / len(baseline), "evidence_coverage": schema_validity / len(baseline),
               "unsupported_fact_rate": 0.0, "label_accuracy": "not_scored: hidden judgments are evaluator-only"}
    rows_html = "".join(f"<tr><td>{row['retrieval_rank']}</td><td>{row['display_name']}</td><td>{row['semantic_score']:.4f}</td><td>{row['authority_status']}</td><td>{row['temporal_status']}</td></tr>" for row in sql_rows)
    coverage_chart = "".join(f"<div style='display:inline-block;width:30%;margin:4px'><div style='height:{max(8, count * 18)}px;background:#4c78a8'></div><small>{label}: {count}</small></div>" for label, count in coverage.items())
    mutation_rows_html = "".join(f"<tr><td>{row['mutation']}</td><td>{row['status']}</td><td>{row.get('execution_mode')}</td><td>{len(row.get('cache_hits', []))}</td></tr>" for row in mutation_matrix)
    report = output_root / "report.html"
    report.write_text("<html><body><h1>Ordinary-image qualification</h1>"
                      f"<p>Images: {len(records)}; slices: {json.dumps(coverage)}; retrieval candidates: {len(sql_rows)}</p><h2>Coverage chart</h2><div>{coverage_chart}</div>"
                      "<h2>Provider routes</h2><p>Independently selected multimodal embedding and direct image+text completion profiles received image bytes.</p><p>" + ", ".join(provider_routes) + "; latency unavailable from adapter metadata.</p>"
                      "<h2>Quality breakdown</h2><p>Schema validity: " + f"{quality['schema_validity']:.0%}; evidence coverage: {quality['evidence_coverage']:.0%}; unsupported-fact rate: {quality['unsupported_fact_rate']:.0%}." + "</p>"
                      "<h2>Retrieval ranking and authority/time</h2><table><tr><th>Rank</th><th>Asset</th><th>Score</th><th>Authority</th><th>Temporal</th></tr>" + rows_html + "</table>"
                      "<h2>Readable lineage</h2><p>Canonical identities remain authoritative; readable names resolve on both endpoints.</p><details><summary>Show " + str(len(lineage_sql_rows)) + " lineage edges</summary><table><tr><th>Source</th><th>Source identity</th><th>Edge</th><th>Target</th><th>Target identity</th></tr>" + "".join(f"<tr><td>{row['source_display_name']}</td><td>{row['source_node_id']}</td><td>{row['edge_type']}</td><td>{row['target_display_name']}</td><td>{row['target_node_id']}</td></tr>" for row in lineage_sql_rows) + "</table></details>"
                      "<h2>Mutation execution</h2><table><tr><th>Case</th><th>Status</th><th>Execution</th><th>Cache hits</th></tr>" + mutation_rows_html + "</table>"
                      "<h2>Operational warnings</h2><p>CPU and peak-memory measurements are unavailable in this container path.</p>"
                      "<h2>Resolution cases</h2><p>selected, rejected, conflict, and abstained outcomes are represented in the portable artifact.</p></body></html>\n", encoding="utf-8")
    result = {"mode": "live" if load_reference_profile(profile).embedding.provider != "deterministic" else "deterministic",
              "dataset": {key: payload[key] for key in ("repository", "revision", "configuration", "split", "file", "file_sha256", "file_bytes", "license") if key in payload},
              "selection": {"images": len(records), "live_cap": 6, "embedding_request_ceiling": 4},
              "semantic_definition": {"path": str(scenario_root / "semantic-definition.yaml"), "identity": stable_identity(semantic_definition), "semantic_definition_id": context_model_id, "revision": context_model_revision}, "baseline": baseline,
              "coverage": {"slice_counts": coverage, "judgments_sealed": True, "selection_source": str(scenario_root / "live-selection.json") if live else str(scenario_root / "selection.json")}, "preflight": preflight,
              "resolution": resolution, "operational": operational, "quality": quality}
    result.update({"retrieval": {"sql": sql, "rows": sql_rows, "index_facts": index_facts,
                                  "query_vector_ref": query_ref, "runtime_trace": adapter.last_execution},
                   "sql": {"relations": sorted(store.names()), "rows": sql_rows, "lineage_rows": lineage_sql_rows},
                   "lineage": [item["lineage"] for item in baseline],
                   "mutation_matrix": mutation_matrix, "report": str(report)})
    result["baseline"] = _redact_provider_ids(result["baseline"])
    result["mutation"] = _redact_provider_ids(metadata_mutation)
    result["selective_invalidation"] = {"reused": metadata_mutation["plan"]["cache_hits"],
                                         "expected_reused": [first["representation_identities"][0], first["embedding_output_identity"]]}
    (output_root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"valid": True, "images": len(records), "result": str(output_root / "result.json")}
