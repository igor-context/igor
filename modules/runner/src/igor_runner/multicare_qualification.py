"""Bounded authoritative-context qualification for MultiCaRe multimodal clinical-case context.

Mirrors the structure of ``recare_qualification.py`` but exercises two joined sources
(case narrative text and linked diagnostic images) through the multimodal image
pipeline: :class:`igor_core.ContentPart` with ``kind="image"``, the Voyage
multimodal embedding adapter, and the Mistral completion adapter (both resolved
generically through ``embedding_adapter_for``/``completion_adapter_for`` from the
Context Model's declared model profiles). Only CC-BY licensed images are ever
turned into context; every other license is filtered out before any derivation,
embedding, or enrichment happens.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import yaml

from igor_context_compiler import (
    AuthorityAssertion,
    CompletionRequest,
    EmbeddingRequest,
    QualifiedRepresentation,
    ResolutionDecision,
    ResolutionRequest,
    ResolutionResult,
    TemporalAssertion,
    compile_context_model_retrieval_queries,
    completion_adapter_for,
    embedding_adapter_for,
)
from igor_core import (
    ConnectorBinding,
    ConnectorFieldBinding,
    ConnectorResourceBinding,
    ContentPart,
    ContextSourceContract,
    EmbeddingSpace,
    EnrichmentRecipe,
    Evidence,
    Representation,
    SchemaDescriptor,
    load_reference_profile,
    stable_identity,
    validate_schema_payload,
)
from igor_dlt import IngestConfig, ingest_bound_records
from igor_lancedb import LanceRetrievalAdapter, LanceStore

from .compose import RunnerError
from .context_model_lifecycle import materialize_context_model_lifecycle
from .project import compile_project, deps_project, validate_project

CASES_DATASET_REVISION = "e5b0a5418dbb6edce6b731c47664cb4f269833bb"
IMAGES_DATASET_REVISION = "c8517124928d2fe3651ee6cb6c560fce66e02344"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected a mapping in {path}")
    return value


def _license_accepted(value: str, allowed_prefixes: tuple[str, ...] = ("cc-by",)) -> bool:
    """Token-aware CC-BY acceptance check (mirrors igor_huggingface's adapter logic).

    The runner does not import the huggingface module (only ``core``, ``fixtures``,
    ``dlt``, ``lancedb``, ``datafusion``, ``relational``, and ``context-compiler`` are
    permitted project imports for this module), so this small, self-contained check is
    duplicated here rather than shared. It rejects any designation carrying a
    restriction token (``nc``, ``nd``, ``sa``) even when it textually starts with an
    allowed prefix, e.g. ``"CC BY-NC"`` is rejected even though ``"CC BY"`` is allowed.
    """
    normalized = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", value.strip().lower())).strip()
    tokens = set(normalized.split())
    if tokens & {"nc", "nd", "sa"}:
        return False
    for prefix in allowed_prefixes:
        prefix_normalized = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", prefix.strip().lower())).strip()
        if normalized == prefix_normalized or normalized.startswith(prefix_normalized + " "):
            return True
    return False


def _compiled_project(scenario: Path, output: Path) -> tuple[dict[str, Any], Path]:
    """Compile a writable copy so qualifications never mutate the source project."""
    project = output / "compiled-project"
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(scenario, project, ignore=shutil.ignore_patterns(".igor"))
    deps_project(project)
    errors, warnings = validate_project(project)
    if errors:
        raise RunnerError("IGOR project validation failed: " + "; ".join(item.message for item in errors))
    compiled = compile_project(project)
    (output / "project-warnings.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in warnings], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return compiled.manifest, project


def _source_seam(project: Path, contract_ref: str, binding_ref: str) -> tuple[ContextSourceContract, ConnectorBinding]:
    document = _load_yaml(project / "sources" / "clinical_cases.yml")
    contract = ContextSourceContract.model_validate(document["source_contracts"][contract_ref])
    raw_binding = document["connector_bindings"][binding_ref]
    resources = tuple(
        ConnectorResourceBinding(
            resource_id=item["resource_id"],
            source_resource=item["source_resource"],
            cursor_field=item.get("cursor_field"),
            fields=tuple(ConnectorFieldBinding.model_validate(field) for field in item["fields"]),
            filters=item.get("filters", {}),
        )
        for item in raw_binding["resources"]
    )
    binding = ConnectorBinding(
        binding_id=raw_binding["binding_id"],
        revision=str(raw_binding["revision"]),
        source_contract_identity=contract.identity,
        connector=raw_binding["connector"],
        deployment_ref=raw_binding["deployment_ref"],
        resources=resources,
    )
    binding.validate_against(contract)
    return contract, binding


def _partition_source_records(
    records: list[dict[str, Any]],
    *,
    contract: ContextSourceContract,
    binding: ConnectorBinding,
    resource_id: str,
    identity_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = {item.resource_id: item for item in contract.resources}
    bindings = {item.resource_id: item for item in binding.resources}
    requirement = requirements[resource_id]
    resource_binding = bindings[resource_id]
    field_by_concept = {field.concept_id: field.source_field for field in resource_binding.fields}
    required = {field.concept_id for field in requirement.fields if field.required} | set(requirement.identity_concepts)
    complete: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = sorted(
            concept for concept in required
            if not str(record.get(field_by_concept.get(concept, concept), "") or "").strip()
        )
        if not missing:
            complete.append(record)
            continue
        issues.append({
            "record_index": index,
            "identity": str(record.get(identity_field, "")),
            "reason_code": "missing_required_source_concepts",
            "missing_concepts": missing,
            "action": "excluded_from_context_ingestion",
        })
    return complete, issues


def _normalize_records(payload: Mapping[str, Any], *, key: str) -> list[dict[str, Any]]:
    records = payload.get(key)
    if records is None:
        records = payload.get("records")
    if not isinstance(records, list):
        raise RunnerError(f"multicare qualification requires a '{key}' or 'records' list payload")
    return [dict(item) for item in records]


def _case_snapshots(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one snapshot per case, staggered deterministically in list order.

    MultiCaRe case reports are standalone publications (no amendment chain like the
    ReCaRe legal scenario), so each case gets exactly one snapshot. Snapshots are
    staggered in time by list order so temporal resolution has something meaningful
    to select between a "historical" as_of (only earlier cases visible) and a
    "current" as_of (every case visible).
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    snapshots: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(records, key=lambda item: str(item.get("case_id", "")))):
        case_id = str(record.get("case_id", ""))
        article_id = str(record.get("article_id", ""))
        case_text = str(record.get("case_text", ""))
        identity = stable_identity({"case_id": case_id, "article_id": article_id, "case_text": case_text})
        effective_from = base + timedelta(days=60 * index)
        snapshots.append({
            "snapshot_identity": identity,
            "case_identity": stable_identity({"case_id": case_id, "article_id": article_id}),
            "case_id": case_id,
            "article_id": article_id,
            "gender": str(record.get("gender", "") or ""),
            "age": str(record.get("age", "") or ""),
            "case_text": case_text,
            "keywords": str(record.get("keywords", "") or ""),
            "source_identity": identity,
            "effective_from": effective_from,
            "effective_until": None,
            "temporal_evidence": True,
        })
    return snapshots


def _image_snapshots(records: list[dict[str, Any]], case_snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one snapshot per CC-BY-licensed image, inheriting its case's effective_from."""
    by_case = {item["case_id"]: item for item in case_snapshots}
    snapshots: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (str(item.get("case_id", "")), str(item.get("image_id", "")))):
        case_id = str(record.get("case_id", ""))
        image_id = str(record.get("image_id", ""))
        case_snapshot = by_case.get(case_id)
        if case_snapshot is None:
            continue
        content_sha256 = str(record.get("content_sha256", ""))
        identity = stable_identity({"case_id": case_id, "image_id": image_id, "content_sha256": content_sha256})
        snapshots.append({
            "snapshot_identity": identity,
            "case_id": case_id,
            "image_id": image_id,
            "figure_id": str(record.get("figure_id", "") or ""),
            "caption": str(record.get("caption", "") or ""),
            "image_path": str(record.get("image_path", "")),
            "media_type": str(record.get("media_type", "")),
            "content_sha256": content_sha256,
            "width": record.get("width"),
            "height": record.get("height"),
            "license": str(record.get("license", "")),
            "source_identity": identity,
            "effective_from": case_snapshot["effective_from"],
            "effective_until": case_snapshot["effective_until"],
            "temporal_evidence": True,
        })
    return snapshots


def _resolved_image_path(snapshot: Mapping[str, Any], project_root: Path) -> Path:
    image_path = Path(snapshot["image_path"])
    if not image_path.is_absolute():
        image_path = (project_root / image_path).resolve()
    return image_path


def _image_content_part(snapshot: Mapping[str, Any], project_root: Path) -> ContentPart:
    image_path = _resolved_image_path(snapshot, project_root)
    return ContentPart(
        kind="image",
        media_type=snapshot["media_type"],
        content_ref=image_path.as_uri(),
        content_sha256=snapshot["content_sha256"],
    )


def _multimodal_representation(
    case_snapshot: Mapping[str, Any],
    image_snapshots: list[dict[str, Any]],
    schema: SchemaDescriptor,
    project_root: Path,
) -> QualifiedRepresentation:
    text_part = ContentPart(
        kind="text", media_type="text/plain", text=case_snapshot["case_text"],
        content_sha256=stable_identity(case_snapshot["case_text"]),
    )
    parts: list[ContentPart] = [text_part]
    for image_snapshot in image_snapshots:
        parts.append(_image_content_part(image_snapshot, project_root))
    representation = Representation(
        ir_version="0.1",
        representation_type="text",
        schema_ref=schema,
        source_snapshot_ids=(case_snapshot["snapshot_identity"], *(item["snapshot_identity"] for item in image_snapshots)),
        payload=case_snapshot["case_text"],
    )
    return QualifiedRepresentation(representation=representation, parts=tuple(parts))


def _deterministic_vector(text: str, dimension: int) -> list[float]:
    values = [0.0] * dimension
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        values[int.from_bytes(digest[:2], "big") % dimension] += 1.0
    norm = sum(item * item for item in values) ** 0.5 or 1.0
    return [item / norm for item in values]


def _enrichment_input_text(case_snapshot: Mapping[str, Any], image_snapshots: list[dict[str, Any]]) -> str:
    header = (
        "Return only a JSON object matching the schema.\n"
        "Keep clinical_summary to one concise sentence under 200 characters.\n"
        "Do not copy the source text verbatim.\n\n"
        f"Case: {case_snapshot['case_id']} (article {case_snapshot['article_id']})\n"
    )
    if image_snapshots:
        captions = "; ".join(item["caption"] for item in image_snapshots if item["caption"])
        header += f"Linked figure caption(s): {captions}\n"
    else:
        header += "No linked figure is available for this case.\n"
    return header + f"\nCase narrative:\n{case_snapshot['case_text']}"


def _runtime_rows(
    case_snapshots: list[dict[str, Any]],
    images_by_case: dict[str, list[dict[str, Any]]],
    vectors_by_case: dict[str, list[float]],
    *,
    observed_at: datetime,
    source_system: str,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for snapshot in case_snapshots:
        case_id = snapshot["case_id"]
        vector = vectors_by_case[case_id]
        display_name = f"Case {case_id} — {snapshot['keywords'] or 'clinical case'}"
        observations.append({
            "object_role": "clinical_case",
            "snapshot_role": "case_snapshot",
            "source_system": source_system,
            "source_key": snapshot["source_identity"],
            "snapshot_identity": snapshot["snapshot_identity"],
            "observed_at": observed_at.isoformat(),
            "content_ref": f"{source_system}://{snapshot['source_identity']}",
            "content_sha256": stable_identity(snapshot["case_text"]),
            "media_type": "text/plain",
            "display_name": display_name,
            "preview": snapshot["case_text"][:240],
            "schema_id": "clinical.case",
            "schema_revision": "1",
            "operation_name": "ingest",
            "status": "observed",
        })
        outputs.append({
            "identity": snapshot["snapshot_identity"],
            "role": "case_version",
            "kind": "context_representation",
            "object_kind": "case_version",
            "derived_from": [snapshot["snapshot_identity"]],
            "schema_id": "clinical.case",
            "schema_revision": "1",
            "display_name": display_name,
            "payload": {
                "case_id": case_id,
                "article_id": snapshot["article_id"],
                "gender": snapshot["gender"],
                "age": snapshot["age"],
                "case_text": snapshot["case_text"],
                "keywords": snapshot["keywords"],
                "effective_from": snapshot["effective_from"].isoformat() if snapshot["effective_from"] else None,
            },
            "vector": vector,
            "text": snapshot["case_text"],
            "source_system": source_system,
            "source_key": snapshot["source_identity"],
            "media_type": "text/plain",
            "content_ref": f"{source_system}://{snapshot['source_identity']}",
            "status": "active",
            "authority_status": "authoritative",
            "authority_level": "source-authoritative" if snapshot["temporal_evidence"] else "",
            "temporal_status": "known" if snapshot["temporal_evidence"] else "unknown",
            "policy_id": "clinical.official-source-policy:1",
            "as_of": observed_at.isoformat(),
            "evidence_preview": snapshot["case_text"][:240],
        })
        for image_snapshot in images_by_case.get(case_id, []):
            image_display_name = f"Image {image_snapshot['image_id']} for Case {case_id}"
            observations.append({
                "object_role": "clinical_image",
                "snapshot_role": "image_snapshot",
                "source_system": source_system,
                "source_key": image_snapshot["source_identity"],
                "snapshot_identity": image_snapshot["snapshot_identity"],
                "observed_at": observed_at.isoformat(),
                "content_ref": _resolved_image_path(image_snapshot, project_root).as_uri(),
                "content_sha256": image_snapshot["content_sha256"],
                "media_type": image_snapshot["media_type"],
                "display_name": image_display_name,
                "preview": image_snapshot["caption"][:240],
                "schema_id": "clinical.image",
                "schema_revision": "1",
                "operation_name": "ingest",
                "status": "observed",
            })
            outputs.append({
                "identity": image_snapshot["snapshot_identity"],
                "role": "image_version",
                "kind": "context_representation",
                "object_kind": "image_version",
                "derived_from": [image_snapshot["snapshot_identity"]],
                "schema_id": "clinical.image",
                "schema_revision": "1",
                "display_name": image_display_name,
                "payload": {
                    "case_id": case_id,
                    "image_id": image_snapshot["image_id"],
                    "figure_id": image_snapshot["figure_id"],
                    "caption": image_snapshot["caption"],
                    "media_type": image_snapshot["media_type"],
                    "content_sha256": image_snapshot["content_sha256"],
                    "license": image_snapshot["license"],
                },
                # The image shares the case's jointly-embedded multimodal vector: it
                # was produced by one Voyage request over text+image parts together.
                "vector": vector,
                "text": image_snapshot["caption"],
                "source_system": source_system,
                "source_key": image_snapshot["source_identity"],
                "media_type": image_snapshot["media_type"],
                "content_ref": _resolved_image_path(image_snapshot, project_root).as_uri(),
                "status": "active",
                "authority_status": "authoritative",
                "authority_level": "source-authoritative" if image_snapshot["temporal_evidence"] else "",
                "temporal_status": "known" if image_snapshot["temporal_evidence"] else "unknown",
                "policy_id": "clinical.official-source-policy:1",
                "as_of": observed_at.isoformat(),
                "evidence_preview": image_snapshot["caption"][:240],
            })
    return observations, outputs


def _retrieval_rows(
    facts: tuple[Any, ...],
    by_identity: dict[str, dict[str, Any]],
    *,
    as_of: datetime,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence, authority, temporal = [], [], []
    rows = []
    seen_contexts: set[str] = set()
    request_identity = stable_identity({"query": runtime["requests"]["query"], "as_of": as_of.isoformat()})
    for fact in facts:
        value = fact if isinstance(fact, Mapping) else fact.model_dump(mode="json")
        context_identity = str(value["context_identity"])
        if context_identity in seen_contexts or context_identity not in by_identity:
            continue
        seen_contexts.add(context_identity)
        snapshot = by_identity[context_identity]
        preview = snapshot.get("case_text", snapshot.get("caption", ""))[:240]
        evidence_item = Evidence(
            ir_version="0.1",
            subject_identity=snapshot["snapshot_identity"],
            source_snapshot_id=snapshot["snapshot_identity"],
            locator=f"source:{snapshot['source_identity']}",
            excerpt=preview,
            observed_at=as_of,
        )
        authority_item = AuthorityAssertion(
            assertion_identity=stable_identity({"authority": snapshot["snapshot_identity"], "issuer": runtime["authority"]["issuer_ref"]}),
            subject_identity=snapshot["snapshot_identity"],
            issuer_ref=str(runtime["authority"]["issuer_ref"]),
            scope_ref=str(runtime["authority"]["scope_ref"]),
            evidence_identities=(evidence_item.identity,),
        )
        temporal_item = TemporalAssertion(
            assertion_identity=stable_identity({"temporal": snapshot["snapshot_identity"]}),
            subject_identity=snapshot["snapshot_identity"],
            observed_at=as_of,
            effective_from=snapshot["effective_from"],
            effective_until=snapshot["effective_until"],
            evidence_identities=(evidence_item.identity,),
        )
        evidence.append(evidence_item)
        authority.append(authority_item)
        temporal.append(temporal_item)
        rows.append({
            "retrieval_role": "case_search",
            "request_identity": request_identity,
            "task_id": str(runtime["requests"]["task"]),
            "task_schema_revision": "1",
            "as_of": as_of.isoformat(),
            "candidate_identity": stable_identity({"request": request_identity, "context": snapshot["snapshot_identity"]}),
            "context_identity": snapshot["snapshot_identity"],
            "retrieval_identity": str(value["result_identity"]),
            "rank": int(value["rank"]),
            "score": float(value["score"]),
            "facts": dict(value.get("facts", {})),
            "evidence_identities": (evidence_item.identity,),
            "authority_assertion_ids": (authority_item.assertion_identity,),
            "temporal_assertion_ids": (temporal_item.assertion_identity,),
            "evidence_locator": evidence_item.locator,
            "source_snapshot_id": snapshot["snapshot_identity"],
            "evidence_excerpt": evidence_item.excerpt,
        })
    if rows:
        rows[0]["evidence"] = tuple(item.model_dump(mode="json") for item in evidence)
        rows[0]["authority_assertions"] = tuple(item.model_dump(mode="json") for item in authority)
        rows[0]["temporal_assertions"] = tuple(item.model_dump(mode="json") for item in temporal)
    return rows


class _AuthorityTemporalResolution:
    """Evaluate supplied authority and temporal assertions without domain field knowledge."""

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        authority = {item.assertion_identity: item for item in request.authority_assertions}
        temporal = {item.assertion_identity: item for item in request.temporal_assertions}
        decisions: list[ResolutionDecision] = []
        for candidate in sorted(request.candidates, key=lambda item: item.candidate_identity):
            authority_items = [authority.get(identity) for identity in candidate.authority_assertion_ids]
            authority_ok = bool(authority_items) and all(item is not None for item in authority_items)
            temporal_items = [temporal.get(identity) for identity in candidate.temporal_assertion_ids]
            temporal_item = next((item for item in temporal_items if item is not None), None)
            if not authority_ok:
                outcome, authority_basis, temporal_basis, reason = "abstained", "unknown", "unknown", "missing_authority"
            elif temporal_item is None or temporal_item.effective_from is None:
                outcome, authority_basis, temporal_basis, reason = "abstained", "satisfied", "unknown", "missing_temporal_evidence"
            elif temporal_item.effective_from > request.as_of:
                outcome, authority_basis, temporal_basis, reason = "rejected", "satisfied", "future", "not_yet_effective"
            elif temporal_item.effective_until is not None and request.as_of >= temporal_item.effective_until:
                outcome, authority_basis, temporal_basis, reason = "rejected", "satisfied", "expired", "superseded"
            else:
                outcome, authority_basis, temporal_basis, reason = "selected", "satisfied", "valid", "authoritative_at_as_of"
            decisions.append(ResolutionDecision(
                candidate_identity=candidate.candidate_identity,
                outcome=outcome,
                reason_code=reason,
                authority_basis=authority_basis,
                temporal_basis=temporal_basis,
                as_of=request.as_of,
                policy_id=request.policy_id,
                policy_revision=request.policy_revision,
            ))
        selected = tuple(item.candidate_identity for item in decisions if item.outcome == "selected")
        rejected = tuple(item.candidate_identity for item in decisions if item.outcome == "rejected")
        abstained = tuple(item.candidate_identity for item in decisions if item.outcome == "abstained")
        return ResolutionResult(
            request_identity=request.request_identity,
            resolution_identity=request.identity,
            decisions=tuple(decisions),
            selected_identities=selected,
            rejected_identities=rejected,
            abstained_identities=abstained,
        )


def _resolution_inputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    first = rows[0]
    return [{
        "retrieval_role": first["retrieval_role"],
        "request_identity": first["request_identity"],
        "task_id": first["task_id"],
        "task_schema_revision": first["task_schema_revision"],
        "as_of": first["as_of"],
        "candidates": [{
            "candidate_identity": row["candidate_identity"],
            "context_identity": row["context_identity"],
            "retrieval": {
                "result_identity": row["retrieval_identity"],
                "context_identity": row["context_identity"],
                "score": row["score"],
                "rank": row["rank"],
                "facts": row["facts"],
            },
            "evidence_identities": row["evidence_identities"],
            "authority_assertion_ids": row["authority_assertion_ids"],
            "temporal_assertion_ids": row["temporal_assertion_ids"],
        } for row in rows],
        "evidence": first.get("evidence", ()),
        "authority_assertions": first.get("authority_assertions", ()),
        "temporal_assertions": first.get("temporal_assertions", ()),
    }]


def _contract_context(runtime: Mapping[str, Any], *, outcome: str, as_of: datetime) -> dict[str, Any]:
    request = runtime["requests"]
    if outcome == "deny":
        return {
            "consumer": request["denied_consumer"],
            "task": request["task"],
            "purpose": request["prohibited_purpose"],
            "authority_level": runtime["authority"]["level"],
            "evaluated_at": as_of.isoformat(),
        }
    if outcome == "prohibited_task":
        return {
            "consumer": request["consumer"],
            "task": request["prohibited_task"],
            "purpose": request["purpose"],
            "authority_level": runtime["authority"]["level"],
            "evaluated_at": as_of.isoformat(),
        }
    return {
        "consumer": request["consumer"],
        "task": request["task"],
        "purpose": request["purpose"],
        "authority_level": "" if outcome == "abstain" else runtime["authority"]["level"],
        "evaluated_at": as_of.isoformat(),
    }


def _evaluate_lifecycle(
    *,
    output: Path,
    name: str,
    manifest: Mapping[str, Any],
    observations: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    contract_context: Mapping[str, Any],
):
    store = LanceStore(output / name)
    return materialize_context_model_lifecycle(
        store=store,
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval=[],
        resolution_inputs=_resolution_inputs(retrieval_rows),
        resolution_port=_AuthorityTemporalResolution(),
        contract_context=contract_context,
    )


def _run(
    scenario: Path,
    output: Path,
    *,
    cases_payload: Mapping[str, Any],
    images_payload: Mapping[str, Any],
    embedding_profile_path: Path | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    manifest, project = _compiled_project(scenario, output)
    runtime = _load_yaml(project / "runtime.yml")
    limits = runtime["limits"]

    case_records = _normalize_records(cases_payload, key="cases")
    if len(case_records) > int(limits["max_cases"]):
        raise RunnerError("multicare qualification case ceiling exceeded")
    case_contract, case_binding = _source_seam(project, runtime["source"]["case_source_contract"], runtime["source"]["case_connector_binding"])
    case_records, case_issues = _partition_source_records(
        case_records, contract=case_contract, binding=case_binding,
        resource_id=str(runtime["source"]["case_resource_id"]), identity_field="case_id",
    )
    if not case_records:
        raise RunnerError("multicare qualification has no complete case records")

    image_records_all = _normalize_records(images_payload, key="images")
    if len(image_records_all) > int(limits["max_images"]):
        raise RunnerError("multicare qualification image ceiling exceeded")
    image_contract, image_binding = _source_seam(project, runtime["source"]["image_source_contract"], runtime["source"]["image_connector_binding"])
    image_records_all, image_issues = _partition_source_records(
        image_records_all, contract=image_contract, binding=image_binding,
        resource_id=str(runtime["source"]["image_resource_id"]), identity_field="image_id",
    )
    # License gate: only CC-BY (or CC-BY-<variant>) images ever become context. This
    # is the license-aware governance boundary the scenario demonstrates — a
    # CC-BY-NC/-ND/-SA image is filtered out here, before any snapshot, embedding, or
    # enrichment work happens, and can never reach a published package. In live mode
    # the acquisition adapter (igor_huggingface.acquire_case_images) already applies
    # this identical token-aware CC-BY check before ever writing image bytes to
    # disk, so a rejected image is typically already absent from `image_records_all`
    # by the time this second, defense-in-depth check runs here; the adapter's own
    # `skipped_license_count` (carried through `images_payload`) is added so the
    # qualification accounts for every rejection regardless of which layer caught it.
    acquisition_skipped_license_count = int(images_payload.get("skipped_license_count", 0) or 0)
    image_records = [item for item in image_records_all if _license_accepted(str(item.get("license", "")))]
    skipped_license_count = acquisition_skipped_license_count + (len(image_records_all) - len(image_records))

    case_snapshots = _case_snapshots(case_records)
    image_snapshots = _image_snapshots(image_records, case_snapshots)
    images_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in image_snapshots:
        images_by_case[item["case_id"]].append(item)

    observed_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
    raw_cases = [{
        "_resource_id": runtime["source"]["case_resource_id"],
        "_media_type": "text/plain",
        "_content_ref": f"source://{stable_identity(item)}",
        "_content_sha256": stable_identity(item),
        "_observed_at": observed_at.isoformat(),
        **item,
    } for item in case_records]
    ingestion = ingest_bound_records(
        raw_cases,
        IngestConfig("qualification", "clinical", str(output / "ingestion-lance"), "qualification/clinical"),
        case_contract,
        case_binding,
    )

    schema_document = _load_yaml(project / "schemas" / "clinical.yml")
    case_schema = SchemaDescriptor.model_validate(schema_document["schemas"]["clinical.case.v1"])
    enrichment_schema = SchemaDescriptor.model_validate(schema_document["schemas"]["clinical.enrichment.v1"])

    query_text = str(runtime["requests"]["query"])
    query_part = ContentPart(kind="text", media_type="text/plain", text=query_text, content_sha256=stable_identity(query_text))
    query_representation = QualifiedRepresentation(
        representation=Representation(
            ir_version="0.1", representation_type="text", schema_ref=case_schema,
            source_snapshot_ids=(stable_identity({"query": query_text}),), payload=query_text,
        ),
        parts=(query_part,),
    )

    # Build the joint text+image multimodal representation for every case
    # unconditionally, in both deterministic and live mode. This exercises the real
    # ContentPart(kind="image") construction path — content_ref resolution against
    # the compiled project's copy of the fixture (or acquired) image files and
    # content_sha256 identity validation — even when no live provider is called, so
    # the deterministic path proves the full ContentPart -> embedding -> enrichment ->
    # package pipeline shape without any network or API access.
    multimodal_reps: dict[str, QualifiedRepresentation] = {}
    for snapshot in case_snapshots:
        case_id = snapshot["case_id"]
        multimodal_reps[case_id] = _multimodal_representation(
            snapshot, images_by_case.get(case_id, []), case_schema, project,
        )

    provider_outcomes: list[dict[str, Any]] = []
    enrichment_outcomes: list[dict[str, Any]] = []
    vectors_by_case: dict[str, list[float]] = {}
    if embedding_profile_path is None:
        dimension = int(limits["deterministic_dimension"])
        for snapshot in case_snapshots:
            case_id = snapshot["case_id"]
            combined = snapshot["case_text"] + "|" + "|".join(
                item["content_sha256"] for item in images_by_case.get(case_id, [])
            )
            vectors_by_case[case_id] = _deterministic_vector(combined, dimension)
        query_vector = _deterministic_vector(query_text, dimension)
        embedding_space_identity = stable_identity({"provider": "deterministic", "dimension": dimension})
        mode = "deterministic"
    else:
        profile = load_reference_profile(embedding_profile_path)
        embedding_profile = profile.embedding
        dimension = int(embedding_profile.parameters["dimensions"])
        space = EmbeddingSpace(
            ir_version="0.1", provider=embedding_profile.provider, model=embedding_profile.model,
            model_revision=embedding_profile.revision, dimension=dimension, dtype="float32",
            metric="cosine", normalized=False, input_schema_identity=case_schema.identity,
        )
        case_order = [item["case_id"] for item in case_snapshots]
        all_representations = tuple(multimodal_reps[case_id] for case_id in case_order) + (query_representation,)
        requests = tuple(
            EmbeddingRequest(
                output_identity=stable_identity({"embedding": item.identity, "profile": embedding_profile.identity}),
                input=item, space=space, profile=embedding_profile,
            )
            for item in all_representations
        )
        outcomes = embedding_adapter_for(embedding_profile).embed_batch(requests)
        if not all(item.status == "succeeded" for item in outcomes):
            raise RunnerError("one or more configured multimodal embedding requests failed")
        provider_outcomes = [item.model_dump(mode="json", exclude_none=True) for item in outcomes]
        values = [list(item.value or []) for item in outcomes]
        for case_id, vector in zip(case_order, values[:-1], strict=True):
            vectors_by_case[case_id] = vector
        query_vector = values[-1]
        embedding_space_identity = space.identity
        mode = "live"

        recipe_document = _load_yaml(project / "enrichments" / "clinical.yml")["recipes"]["clinical.case-enrichment-recipe.v1"]
        recipe = EnrichmentRecipe(
            recipe_id=recipe_document["recipe_id"], revision=str(recipe_document["revision"]),
            accepted_representation_types=tuple(recipe_document["accepted_representation_types"]),
            accepted_media_types=tuple(recipe_document["accepted_media_types"]),
            output_schema_identity=enrichment_schema.identity,
            prompt_version=str(recipe_document["prompt_version"]), taxonomy_version=str(recipe_document["taxonomy_version"]),
            evidence_required=bool(recipe_document["evidence_required"]), abstention_allowed=bool(recipe_document["abstention_allowed"]),
        )
        completion_adapter = completion_adapter_for(profile.completion)
        for snapshot in case_snapshots:
            case_id = snapshot["case_id"]
            case_images = images_by_case.get(case_id, [])
            event_text = _enrichment_input_text(snapshot, case_images)
            text_part = ContentPart(kind="text", media_type="text/plain", text=event_text, content_sha256=stable_identity(event_text))
            image_parts = tuple(_image_content_part(item, project) for item in case_images)
            event_input = QualifiedRepresentation(
                representation=Representation(
                    ir_version="0.1", representation_type="text", schema_ref=case_schema,
                    source_snapshot_ids=(snapshot["snapshot_identity"], *(item["snapshot_identity"] for item in case_images)),
                    payload=event_text,
                ),
                parts=(text_part, *image_parts),
            )
            request = CompletionRequest(
                output_identity=stable_identity({"enrichment": snapshot["snapshot_identity"], "profile": profile.completion.identity}),
                inputs=(event_input,), output_schema=enrichment_schema, recipe=recipe,
                prompt_version=recipe.prompt_version, taxonomy_version=recipe.taxonomy_version, profile=profile.completion,
            )
            outcome = completion_adapter.enrich(request)
            if outcome.status != "succeeded":
                enrichment_outcomes.append({"case_identity": snapshot["case_identity"], **outcome.model_dump(mode="json", exclude_none=True)})
                continue
            validate_schema_payload(enrichment_schema, outcome.value)
            enrichment_outcomes.append({"case_identity": snapshot["case_identity"], **outcome.model_dump(mode="json", exclude_none=True)})

    observations, outputs = _runtime_rows(
        case_snapshots, images_by_case, vectors_by_case,
        observed_at=observed_at, source_system="huggingface-multicare" if mode == "live" else "multicare-fixture",
        project_root=project,
    )
    index_store = LanceStore(output / "retrieval-lance")
    materialize_context_model_lifecycle(store=index_store, manifest=manifest, observations=observations, outputs=outputs, retrieval=[])
    retrieval = LanceRetrievalAdapter(index_store)
    index_facts = retrieval.ensure_vector_index()
    query = compile_context_model_retrieval_queries(manifest, {
        "case_search": {
            "text": query_text, "vector": query_vector,
            "limit": int(limits["retrieval_limit"]), "space_identity": embedding_space_identity,
        }
    })[0]
    facts = retrieval.retrieve(query)
    if not facts:
        raise RunnerError("multimodal semantic retrieval returned no candidates")
    retrieval_plan = dict(retrieval.last_execution)

    by_identity = {item["snapshot_identity"]: item for item in case_snapshots}
    by_identity.update({item["snapshot_identity"]: item for item in image_snapshots})

    effective_times = sorted(item["effective_from"] for item in case_snapshots if item["effective_from"] is not None)
    current_as_of = effective_times[-1] + timedelta(days=1)
    historical_as_of = effective_times[0] + timedelta(days=1)

    # Synthetic supersession fixture. Real MultiCaRe case reports are standalone
    # publications with no amendment/republication chain — unlike ReCaRe's legal
    # domain, nothing in the acquired sample is ever naturally superseded. The
    # authoritative-context contract's supersession-rejection path is still a real,
    # load-bearing behavior of the temporal resolution engine (it is what keeps a
    # retracted or replaced version from ever being selected as current), so this
    # constructs one explicit, clearly synthetic "retracted and republished"
    # administrative-correction fixture from the first acquired case: an earlier
    # snapshot whose effective_until equals the real snapshot's effective_from. It
    # is a resolution-input-only fixture — never embedded, enriched, indexed, or
    # published — that exists solely to exercise the same supersession-rejection
    # path ReCaRe's real amendment history exercises naturally.
    anchor = case_snapshots[0]
    synthetic_prior_identity = stable_identity({"synthetic_prior_of": anchor["snapshot_identity"]})
    synthetic_prior = {
        **anchor,
        "snapshot_identity": synthetic_prior_identity,
        "source_identity": synthetic_prior_identity,
        "case_id": f"{anchor['case_id']}-prior-fixture",
        "case_text": "[synthetic superseded-fixture] " + anchor["case_text"],
        "effective_from": anchor["effective_from"] - timedelta(days=30),
        "effective_until": anchor["effective_from"],
    }
    by_identity[synthetic_prior_identity] = synthetic_prior
    synthetic_fact = {
        "context_identity": synthetic_prior_identity,
        "result_identity": stable_identity({"synthetic_result_for": synthetic_prior_identity}),
        "rank": len(facts) + 1,
        "score": 0.0,
        "facts": {"synthetic_supersession_fixture": "true"},
    }

    current_rows = _retrieval_rows((*facts, synthetic_fact), by_identity, as_of=current_as_of, runtime=runtime)
    historical_rows = _retrieval_rows(facts, by_identity, as_of=historical_as_of, runtime=runtime)

    allowed = _evaluate_lifecycle(
        output=output, name="allowed-relations", manifest=manifest,
        observations=observations, outputs=outputs, retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="allow", as_of=current_as_of),
    )
    denied = _evaluate_lifecycle(
        output=output, name="denied-relations", manifest=manifest,
        observations=observations, outputs=outputs, retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="deny", as_of=current_as_of),
    )
    abstain_outputs = [{**item, "authority_level": "", "authority_status": "unknown"} for item in outputs]
    abstained = _evaluate_lifecycle(
        output=output, name="abstained-relations", manifest=manifest,
        observations=observations, outputs=abstain_outputs, retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="abstain", as_of=current_as_of),
    )
    prohibited_task = _evaluate_lifecycle(
        output=output, name="prohibited-task-relations", manifest=manifest,
        observations=observations, outputs=outputs, retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="prohibited_task", as_of=current_as_of),
    )
    historical = _evaluate_lifecycle(
        output=output, name="historical-relations", manifest=manifest,
        observations=observations, outputs=outputs, retrieval_rows=historical_rows,
        contract_context=_contract_context(runtime, outcome="allow", as_of=historical_as_of),
    )

    def decisions(lifecycle: Any) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for result in lifecycle.resolution_results for item in result.decisions]

    current_decisions = decisions(allowed)
    historical_decisions = decisions(historical)

    mutation_records = [dict(item) for item in case_records]
    mutation_records[0]["case_text"] = str(mutation_records[0].get("case_text", "")) + " [bounded mutation]"
    mutated_case_snapshots = _case_snapshots(mutation_records)
    mutated_image_snapshots = _image_snapshots(image_records, mutated_case_snapshots)
    original_ids = {item["snapshot_identity"] for item in case_snapshots} | {item["snapshot_identity"] for item in image_snapshots}
    mutated_ids = {item["snapshot_identity"] for item in mutated_case_snapshots} | {item["snapshot_identity"] for item in mutated_image_snapshots}
    # The mutated case's own snapshot and its jointly-embedded multimodal image
    # invalidate together (both derive from the same joint text+image embedding
    # request); every other case's image snapshots are byte-for-byte unaffected and
    # are expected to be reused (cache hit).
    mutated_case_id = str(mutation_records[0].get("case_id", ""))
    reused_other_case_images = {
        item["snapshot_identity"] for item in image_snapshots if item["case_id"] != mutated_case_id
    }

    allowed_count = sum(item["decision"] == "allow" for item in allowed.contract_evaluations)
    denied_count = sum(item["decision"] == "deny" for item in denied.contract_evaluations)
    abstained_count = sum(item["decision"] == "abstain" for item in abstained.contract_evaluations)
    prohibited_task_denied_or_abstained = sum(
        item["decision"] in {"deny", "abstain"} for item in prohibited_task.contract_evaluations
    )
    ann_active = any("IVF" in str(item.get("index_type", "")).upper() for item in index_facts)
    lineage_rows = allowed.materialization.relations.get("lineage_edges", [])
    lineage_nodes = allowed.materialization.relations.get("lineage_nodes", [])

    retrieved_context_identities = {row["context_identity"] for row in current_rows}
    retrieved_case_ids = retrieved_context_identities & {item["snapshot_identity"] for item in case_snapshots}
    retrieved_image_ids = retrieved_context_identities & {item["snapshot_identity"] for item in image_snapshots}

    checks = {
        "project_compiled": bool(manifest.get("manifest_identity")),
        "cases_source_pinned": cases_payload.get("cases_dataset_revision", cases_payload.get("revision")) == CASES_DATASET_REVISION or mode != "live",
        "images_source_pinned": images_payload.get("images_dataset_revision", images_payload.get("revision")) == IMAGES_DATASET_REVISION or mode != "live",
        "temporal_grounded": all(item["temporal_evidence"] for item in case_snapshots),
        "native_ann": ann_active and "ANN" in str(retrieval_plan.get("plan", "")).upper(),
        "current_selected": any(item["outcome"] == "selected" for item in current_decisions),
        "historical_selected": any(item["outcome"] == "selected" for item in historical_decisions),
        "future_case_rejected": any(item["temporal_basis"] == "future" for item in historical_decisions),
        "retrieval_includes_text_and_image": bool(retrieved_case_ids) and bool(retrieved_image_ids),
        "license_gate_enforced": skipped_license_count > 0 and all(
            _license_accepted(item["license"]) for item in image_snapshots
        ),
        "allowed_published": allowed_count > 0 and bool(allowed.packages),
        "denied_failed_closed": denied_count > 0 and not denied.packages,
        "abstained_failed_closed": abstained_count > 0 and not abstained.packages,
        "prohibited_task_failed_closed": prohibited_task_denied_or_abstained > 0 and not prohibited_task.packages,
        "readable_lineage": bool(lineage_rows) and all(item.get("display_name") for item in lineage_nodes),
        "mutation_invalidation": bool(original_ids - mutated_ids),
        "unaffected_image_reuse": bool(reused_other_case_images & mutated_ids),
        "live_enrichment": mode != "live" or (
            len(enrichment_outcomes) == len(case_snapshots)
            and all(item.get("status") == "succeeded" for item in enrichment_outcomes)
        ),
        "image_content_hash_verified": bool(image_snapshots) and all(
            item["content_sha256"] == "sha256:" + hashlib.sha256(_resolved_image_path(item, project).read_bytes()).hexdigest()
            for item in image_snapshots
        ),
    }
    artifact = {
        "artifact_type": "authoritative-context-qualification",
        "schema_version": "0.1",
        "qualification": "bounded-multimodal-clinical-context-v1",
        "mode": mode,
        "valid": all(checks.values()),
        "checks": checks,
        "dataset": {
            # `revision` is the generic authoritative-context artifact contract's
            # single pinned-revision field (the same field ReCaRe's single-source
            # artifact populates); this scenario joins two sources, so it is set to
            # the primary (case) source's revision, with both revisions still
            # available individually below.
            "revision": cases_payload.get("cases_dataset_revision", cases_payload.get("revision")),
            "cases_repository": "OpenMed/multicare-cases",
            "cases_revision": cases_payload.get("cases_dataset_revision", cases_payload.get("revision")),
            "images_repository": "OpenMed/multicare-case-images",
            "images_revision": images_payload.get("images_dataset_revision", images_payload.get("revision")),
            "acquired_cases": len(_normalize_records(cases_payload, key="cases")),
            "processed_cases": len(case_records),
            "acquired_images": len(_normalize_records(images_payload, key="images")),
            "license_kept_images": len(image_records),
            "license_skipped_images": skipped_license_count,
        },
        "source_issues": case_issues + image_issues,
        "cases": len(case_snapshots),
        "case_list": [{"case_id": item["case_id"], "display_name": f"Case {item['case_id']}"} for item in case_snapshots],
        "ingestion": ingestion.as_dict(),
        "manifest": manifest,
        "snapshots": {
            "case_total": len(case_snapshots),
            "image_total": len(image_snapshots),
            "historical_queryable": checks["historical_selected"],
            "current_queryable": checks["current_selected"],
            # `items` is the generic authoritative-context artifact contract's flat
            # snapshot list (case_items + image_items combined); the split lists
            # below are kept for this scenario's own text/image-specific reporting.
            "items": [{
                **{key: value for key, value in item.items() if key not in {"case_text", "effective_from", "effective_until"}},
                "effective_from": item["effective_from"].isoformat() if item["effective_from"] else None,
                "effective_until": item["effective_until"].isoformat() if item["effective_until"] else None,
                "preview": item["case_text"][:160],
            } for item in case_snapshots] + [{
                **{key: value for key, value in item.items() if key not in {"effective_from", "effective_until"}},
                "effective_from": item["effective_from"].isoformat() if item["effective_from"] else None,
                "effective_until": item["effective_until"].isoformat() if item["effective_until"] else None,
            } for item in image_snapshots],
            "case_items": [{
                **{key: value for key, value in item.items() if key not in {"case_text", "effective_from", "effective_until"}},
                "effective_from": item["effective_from"].isoformat() if item["effective_from"] else None,
                "effective_until": item["effective_until"].isoformat() if item["effective_until"] else None,
                "preview": item["case_text"][:160],
            } for item in case_snapshots],
            "image_items": [{
                **{key: value for key, value in item.items() if key not in {"effective_from", "effective_until"}},
                "effective_from": item["effective_from"].isoformat() if item["effective_from"] else None,
                "effective_until": item["effective_until"].isoformat() if item["effective_until"] else None,
            } for item in image_snapshots],
        },
        "retrieval": {
            "query": query_text,
            "query_identity": query.query_id,
            "query_vector_identity": stable_identity(query_vector),
            "results": [dict(item) if isinstance(item, Mapping) else item.model_dump(mode="json") for item in facts],
            "index_facts": list(index_facts),
            "execution": retrieval_plan,
            "retrieved_case_count": len(retrieved_case_ids),
            "retrieved_image_count": len(retrieved_image_ids),
        },
        "resolution": {
            "current_as_of": current_as_of.isoformat(),
            "historical_as_of": historical_as_of.isoformat(),
            "current_selected": checks["current_selected"],
            "historical_selected": checks["historical_selected"],
            "decisions": current_decisions,
            "historical_decisions": historical_decisions,
        },
        "contract": {
            "allowed_count": allowed_count,
            "denied_count": denied_count,
            "abstained_count": abstained_count,
            "prohibited_task_count": prohibited_task_denied_or_abstained,
            "allowed_package_count": len(allowed.packages),
            "denied_package_count": len(denied.packages),
            "abstained_package_count": len(abstained.packages),
            "prohibited_task_package_count": len(prohibited_task.packages),
        },
        "packages": [item.model_dump(mode="json") for item in allowed.packages],
        "lineage": {
            "node_count": len(lineage_nodes),
            "edge_count": len(lineage_rows),
            "readable": checks["readable_lineage"],
        },
        "mutation": {
            "invalidated_count": len(original_ids - mutated_ids),
            "reused_count": len(reused_other_case_images & mutated_ids),
            "invalidated": sorted(original_ids - mutated_ids),
            "reused_other_case_images": sorted(reused_other_case_images & mutated_ids),
        },
        "providers": {
            "embedding": provider_outcomes,
            "completion": enrichment_outcomes,
            "expectations": {
                "embedding": len(case_snapshots) + 1 if mode == "live" else 0,
                "completion": len(case_snapshots) if mode == "live" else 0,
            },
        },
        "relation_counts": allowed.relation_counts,
        "timing_seconds": round(perf_counter() - started, 3),
    }
    (output / "qualification.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.html").write_text(_report(artifact), encoding="utf-8")
    return artifact


def qualify_multicare(scenario: str | Path, output: str | Path) -> dict[str, Any]:
    scenario_path, output_path = Path(scenario), Path(output)
    payload = json.loads((scenario_path / "fixtures.json").read_text(encoding="utf-8"))
    cases_payload = {"cases_dataset_revision": payload.get("cases_dataset_revision"), "cases": payload.get("cases", [])}
    images_payload = {"images_dataset_revision": payload.get("images_dataset_revision"), "images": payload.get("images", [])}
    return _run(scenario_path, output_path, cases_payload=cases_payload, images_payload=images_payload)


def _flatten_live_case_records(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the live OpenMed/multicare-cases row shape.

    The Dataset Viewer returns one row per PubMed Central *article*, where each row's
    ``cases`` field is a list of one-or-more per-patient case dicts (an article can
    report more than one case). ``RowSelection``/``acquire_rows`` only projects
    top-level scalar fields, so the raw acquisition is requested as
    ``fields=["article_id", "cases"]`` and flattened here into one flat record per
    case (matching the shape ``fixtures.json`` already uses) before the shared
    ``_run`` pipeline processes it.
    """
    flattened: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        article_id = str(record.get("article_id", ""))
        for case in record.get("cases", []) or []:
            flattened.append({
                "case_id": case.get("case_id"),
                "article_id": article_id,
                "case_text": case.get("case_text"),
                "age": case.get("age"),
                "gender": case.get("gender"),
            })
    return {"cases_dataset_revision": payload.get("revision"), "cases": flattened}


def qualify_multicare_live(
    scenario: str | Path,
    profile_path: str | Path,
    case_acquisition: str | Path,
    image_acquisition: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    case_path = Path(case_acquisition)
    if case_path.is_dir():
        case_path = case_path / "acquisition.json"
    image_path = Path(image_acquisition)
    if image_path.is_dir():
        image_path = image_path / "acquisition.json"
    raw_cases_payload = json.loads(case_path.read_text(encoding="utf-8"))
    cases_payload = _flatten_live_case_records(raw_cases_payload)
    images_payload = json.loads(image_path.read_text(encoding="utf-8"))
    return _run(
        Path(scenario), Path(output),
        cases_payload=cases_payload, images_payload=images_payload,
        embedding_profile_path=Path(profile_path),
    )


def _report(artifact: Mapping[str, Any]) -> str:
    checks = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in artifact["checks"].items()
    )
    cases = "".join(
        "<tr>"
        f"<td>{html.escape(item['case_id'])}</td><td>{html.escape(str(item['effective_from']))}</td>"
        f"<td>{html.escape(item['preview'])}</td>"
        "</tr>"
        for item in artifact["snapshots"]["case_items"]
    )
    images = "".join(
        "<tr>"
        f"<td>{html.escape(item['case_id'])}</td><td>{html.escape(item['image_id'])}</td>"
        f"<td>{html.escape(item['media_type'])}</td><td>{html.escape(item['license'])}</td>"
        "</tr>"
        for item in artifact["snapshots"]["image_items"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>IGOR multimodal clinical-context qualification</title>
<style>body{{font:15px/1.5 system-ui;max-width:1180px;margin:40px auto;padding:0 20px;color:#17313b}}table{{width:100%;border-collapse:collapse;margin:18px 0}}th,td{{border-bottom:1px solid #d8e5e8;padding:9px;text-align:left;vertical-align:top}}.pass{{color:#167042}}.fail{{color:#9e2d2d}}code{{background:#eef3f4;padding:2px 5px}}</style></head><body>
<h1>Bounded multimodal clinical-case context qualification</h1>
<p class='{'pass' if artifact['valid'] else 'fail'}'><strong>{'Passed' if artifact['valid'] else 'Failed'}</strong> — {artifact['mode']} mode, {artifact['cases']} cases.</p>
<h2>Trust checks</h2><table><tr><th>Check</th><th>Result</th></tr>{checks}</table>
<h2>Case snapshots</h2><table><tr><th>Case</th><th>Effective from</th><th>Narrative preview</th></tr>{cases}</table>
<h2>CC-BY image snapshots</h2><table><tr><th>Case</th><th>Image</th><th>Media type</th><th>License</th></tr>{images}</table>
<h2>Multimodal semantic retrieval</h2><p><code>{html.escape(artifact['retrieval']['query'])}</code></p><pre>{html.escape(str(artifact['retrieval']['execution'].get('plan', '')))}</pre>
<h2>Context Contract</h2><p>Allowed: {artifact['contract']['allowed_count']}; denied: {artifact['contract']['denied_count']}; abstained: {artifact['contract']['abstained_count']}; prohibited-task blocked: {artifact['contract']['prohibited_task_count']}.</p>
<h2>Lineage</h2><p>{artifact['lineage']['node_count']} readable nodes and {artifact['lineage']['edge_count']} edges.</p>
</body></html>"""
