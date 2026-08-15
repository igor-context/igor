"""Bounded authoritative-context qualification driven by an IGOR project."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
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
    canonical_json,
    load_reference_profile,
    stable_identity,
    validate_schema_payload,
)
from igor_dlt import IngestConfig, ingest_bound_records
from igor_lancedb import LanceRetrievalAdapter, LanceStore

from .compose import RunnerError
from .context_model_lifecycle import materialize_context_model_lifecycle
from .project import compile_project, deps_project, validate_project


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected a mapping in {path}")
    return value


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


def _source_seam(project: Path, runtime: Mapping[str, Any]) -> tuple[ContextSourceContract, ConnectorBinding]:
    document = _load_yaml(project / "sources" / "legal_amendments.yml")
    source = dict(runtime["source"])
    contract = ContextSourceContract.model_validate(document["source_contracts"][source["source_contract"]])
    raw_binding = document["connector_bindings"][source["connector_binding"]]
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
    row_numbers: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = {item.resource_id: item for item in contract.resources}
    bindings = {item.resource_id: item for item in binding.resources}
    requirement = requirements[resource_id]
    resource_binding = bindings[resource_id]
    field_by_concept = {field.concept_id: field.source_field for field in resource_binding.fields}
    required = {
        field.concept_id for field in requirement.fields
        if field.required
    } | set(requirement.identity_concepts)
    complete: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        missing = sorted(
            concept for concept in required
            if not str(record.get(field_by_concept.get(concept, ""), "") or "").strip()
        )
        if not missing:
            complete.append(record)
            continue
        issues.append({
            "record_index": index,
            "row_number": row_numbers[index] if index < len(row_numbers) else None,
            "event_id": str(record.get("amendment_law_id", "")),
            "law_id": str(record.get("law_id", "")),
            "reason_code": "missing_required_source_concepts",
            "missing_concepts": missing,
            "action": "excluded_from_context_ingestion",
        })
    return complete, issues


def _normalize_records(payload: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    result: list[dict[str, Any]] = []
    for parent in payload.get(str(mapping["event_collection"]), []):
        for child in parent.get(str(mapping["child_collection"]), []):
            row = {field: parent.get(field, "") for field in mapping.get("inherited_fields", [])}
            row.update(child)
            for logical in ("article_number", "caption"):
                config = dict(mapping[logical])
                fallback = child.get(config["fixture_fallback"], "")
                row.setdefault(config["before"], fallback)
                row.setdefault(config["after"], fallback)
            result.append(row)
    return result


def _revision_time(value: Any, config: Mapping[str, Any]) -> datetime | None:
    match = re.search(str(config["pattern"]), str(value or ""))
    if match is None:
        return None
    parsed = datetime.strptime(match.group("date"), str(config["format"]))
    return parsed.replace(tzinfo=timezone.utc)


def _snapshots(records: list[dict[str, Any]], mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    time_config = dict(mapping["revision_time"])
    for record in records:
        before_time = _revision_time(record.get(mapping["revision"]["before"]), time_config)
        after_time = _revision_time(record.get(mapping["revision"]["after"]), time_config)
        article_number = str(
            record.get(mapping["article_number"]["after"])
            or record.get(mapping["article_number"]["before"])
            or ""
        )
        article_identity = stable_identity({"law_id": record.get("law_id", ""), "article_number": article_number})
        for side, effective_from, effective_until in (
            ("before", before_time, after_time),
            ("after", after_time, None),
        ):
            text = str(record.get(mapping["text"][side], ""))
            source_identity = str(record.get(mapping["source_identity"][side], ""))
            identity = stable_identity({
                "article_identity": article_identity,
                "source_identity": source_identity,
                "text": text,
                "revision": record.get(mapping["revision"][side], ""),
            })
            snapshots.append({
                "snapshot_identity": identity,
                "article_identity": article_identity,
                "event_id": str(record.get("amendment_law_id", "")),
                "law_id": str(record.get("law_id", "")),
                "article_number": article_number,
                "caption": str(record.get(mapping["caption"][side], "")),
                "text": text,
                "amendment_reason": str(record.get("amendment_reason", "")),
                "revision_id": str(record.get(mapping["revision"][side], "")),
                "source_identity": source_identity,
                "version": side,
                "effective_from": effective_from,
                "effective_until": effective_until,
                "superseded_by": None,
                "temporal_evidence": effective_from is not None and (side == "after" or effective_until is not None),
            })
        snapshots[-2]["superseded_by"] = snapshots[-1]["snapshot_identity"]
    return snapshots


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


def _deterministic_vector(text: str, dimension: int) -> list[float]:
    values = [0.0] * dimension
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        values[int.from_bytes(digest[:2], "big") % dimension] += 1.0
    norm = sum(item * item for item in values) ** 0.5 or 1.0
    return [item / norm for item in values]


def _representations(snapshots: list[dict[str, Any]], schema: SchemaDescriptor) -> list[QualifiedRepresentation]:
    result = []
    for snapshot in snapshots:
        part = ContentPart(
            kind="text",
            media_type="text/plain",
            text=snapshot["text"],
            content_sha256=stable_identity(snapshot["text"]),
        )
        representation = Representation(
            ir_version="0.1",
            representation_type="text",
            schema_ref=schema,
            source_snapshot_ids=(snapshot["snapshot_identity"],),
            payload=snapshot["text"],
        )
        result.append(QualifiedRepresentation(representation=representation, parts=(part,)))
    return result


def _runtime_rows(
    snapshots: list[dict[str, Any]],
    vectors: list[list[float]],
    *,
    observed_at: datetime,
    source_system: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations, outputs, retrieval_index = [], [], []
    for snapshot, vector in zip(snapshots, vectors, strict=True):
        display_name = f"{snapshot['law_id']} Article {snapshot['article_number']} ({snapshot['version']})"
        observations.append({
            "object_role": "amendment_article",
            "snapshot_role": "article_snapshot",
            "source_system": source_system,
            "source_key": snapshot["source_identity"] or snapshot["snapshot_identity"],
            "snapshot_identity": snapshot["snapshot_identity"],
            "observed_at": observed_at.isoformat(),
            "content_ref": f"{source_system}://{snapshot['source_identity'] or snapshot['snapshot_identity']}",
            "content_sha256": stable_identity(snapshot["text"]),
            "media_type": "text/plain",
            "display_name": display_name,
            "preview": snapshot["text"][:240],
            "schema_id": "legal.article",
            "schema_revision": "1",
            "operation_name": "ingest",
            "status": "observed",
        })
        outputs.append({
            "identity": snapshot["snapshot_identity"],
            "role": "article_version",
            "kind": "context_representation",
            "object_kind": "article_version",
            "derived_from": [snapshot["snapshot_identity"]],
            "schema_id": "legal.article",
            "schema_revision": "1",
            "display_name": display_name,
            "payload": {
                "law_id": snapshot["law_id"],
                "article_number": snapshot["article_number"],
                "text": snapshot["text"],
                "effective_from": snapshot["effective_from"].isoformat() if snapshot["effective_from"] else None,
            },
            "vector": vector,
            "text": snapshot["text"],
            "source_system": source_system,
            "source_key": snapshot["source_identity"] or snapshot["snapshot_identity"],
            "media_type": "text/plain",
            "content_ref": f"{source_system}://{snapshot['source_identity'] or snapshot['snapshot_identity']}",
            "status": "active",
            "authority_status": "authoritative",
            "authority_level": "source-authoritative" if snapshot["temporal_evidence"] else "",
            "temporal_status": "known" if snapshot["temporal_evidence"] else "unknown",
            "policy_id": "legal.official-source-policy:1",
            "as_of": observed_at.isoformat(),
            "evidence_preview": snapshot["text"][:240],
        })
        retrieval_index.append({
            "identity": snapshot["snapshot_identity"],
            "context_identity": snapshot["snapshot_identity"],
            "retrieval_identity": stable_identity({"indexed-context": snapshot["snapshot_identity"]}),
            "vector": vector,
            "text": snapshot["text"],
            "context_model_id": "recare.legal.context",
            "status": "active",
        })
    return observations, outputs, retrieval_index


def _enrichment_input_text(event_snapshots: list[dict[str, Any]]) -> str:
    rationale = event_snapshots[0]["amendment_reason"]
    article_context = "\n\n".join(
        f"Article {item['article_number']} {item['version']} version:\n{item['text']}"
        for item in event_snapshots
    )
    return (
        "Return only a JSON object matching the schema.\n"
        "Keep rationale_summary to one concise sentence under 160 characters.\n"
        "Do not copy the source text verbatim.\n\n"
        f"Amendment rationale:\n{rationale}\n\n{article_context}"
    )


def _retrieval_rows(
    facts: tuple[Any, ...],
    snapshots: list[dict[str, Any]],
    *,
    as_of: datetime,
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_identity = {item["snapshot_identity"]: item for item in snapshots}
    evidence, authority, temporal = [], [], []
    rows = []
    seen_contexts: set[str] = set()
    request_identity = stable_identity({"query": runtime["requests"]["query"], "as_of": as_of.isoformat()})
    for fact in facts:
        value = fact if isinstance(fact, Mapping) else fact.model_dump(mode="json")
        context_identity = str(value["context_identity"])
        if context_identity in seen_contexts:
            continue
        seen_contexts.add(context_identity)
        snapshot = by_identity[context_identity]
        evidence_item = Evidence(
            ir_version="0.1",
            subject_identity=snapshot["snapshot_identity"],
            source_snapshot_id=snapshot["snapshot_identity"],
            locator=f"source:{snapshot['source_identity'] or snapshot['snapshot_identity']}",
            excerpt=snapshot["text"][:240],
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
            "retrieval_role": "article_search",
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


def _evaluate_lifecycle(
    *,
    output: Path,
    name: str,
    manifest: Mapping[str, Any],
    observations: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    retrieval_index: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    contract_context: Mapping[str, Any],
):
    store = LanceStore(output / name)
    return materialize_context_model_lifecycle(
        store=store,
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval=retrieval_index,
        resolution_inputs=_resolution_inputs(retrieval_rows),
        resolution_port=_AuthorityTemporalResolution(),
        contract_context=contract_context,
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
    return {
        "consumer": request["consumer"],
        "task": request["task"],
        "purpose": request["purpose"],
        "authority_level": "" if outcome == "abstain" else runtime["authority"]["level"],
        "evaluated_at": as_of.isoformat(),
    }


def _run(
    scenario: Path,
    output: Path,
    *,
    records_payload: Mapping[str, Any],
    embedding_profile_path: Path | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    manifest, project = _compiled_project(scenario, output)
    runtime = _load_yaml(project / "runtime.yml")
    records = _normalize_records(records_payload, runtime["mapping"])
    limits = runtime["limits"]
    if len(records) > int(limits["max_articles"]):
        raise RunnerError("authoritative-context article ceiling exceeded")
    contract, binding = _source_seam(project, runtime)
    records, source_issues = _partition_source_records(
        records,
        contract=contract,
        binding=binding,
        resource_id=str(runtime["source"]["resource_id"]),
        row_numbers=list(records_payload.get("row_numbers", [])),
    )
    if not records:
        raise RunnerError("authoritative-context qualification has no complete source records")
    event_ids = sorted({str(item.get("amendment_law_id", "")) for item in records})
    if len(event_ids) > int(limits["max_events"]):
        raise RunnerError("authoritative-context event ceiling exceeded")
    snapshots = _snapshots(records, runtime["mapping"])
    if not snapshots:
        raise RunnerError("authoritative-context qualification has no snapshots")

    observed_at = datetime(2026, 8, 14, tzinfo=timezone.utc)
    raw = [{
        "_resource_id": runtime["source"]["resource_id"],
        "_media_type": "text/plain",
        "_content_ref": f"source://{item.get('article_id_after', stable_identity(item))}",
        "_content_sha256": stable_identity(item),
        "_observed_at": observed_at.isoformat(),
        **item,
    } for item in records]
    ingestion = ingest_bound_records(
        raw,
        IngestConfig("qualification", "legal", str(output / "ingestion-lance"), "qualification/legal"),
        contract,
        binding,
    )

    schema_document = _load_yaml(project / "schemas" / "legal.yml")
    article_schema = SchemaDescriptor.model_validate(schema_document["schemas"]["legal.article.v1"])
    representations = _representations(snapshots, article_schema)
    query_text = str(runtime["requests"]["query"])
    query_part = ContentPart(kind="text", media_type="text/plain", text=query_text, content_sha256=stable_identity(query_text))
    query_representation = QualifiedRepresentation(
        representation=Representation(
            ir_version="0.1",
            representation_type="text",
            schema_ref=article_schema,
            source_snapshot_ids=(stable_identity({"query": query_text}),),
            payload=query_text,
        ),
        parts=(query_part,),
    )
    provider_outcomes: list[dict[str, Any]] = []
    enrichment_outcomes: list[dict[str, Any]] = []
    if embedding_profile_path is None:
        dimension = int(limits["deterministic_dimension"])
        vectors = [_deterministic_vector(item.text, dimension) for item in representations]
        query_vector = _deterministic_vector(query_text, dimension)
        embedding_space_identity = stable_identity({"provider": "deterministic", "dimension": dimension})
        mode = "deterministic"
    else:
        profile = load_reference_profile(embedding_profile_path)
        embedding_profile = profile.embedding
        dimension = int(embedding_profile.parameters["dimensions"])
        space = EmbeddingSpace(
            ir_version="0.1",
            provider=embedding_profile.provider,
            model=embedding_profile.model,
            model_revision=embedding_profile.revision,
            dimension=dimension,
            dtype="float32",
            metric="cosine",
            normalized=False,
            input_schema_identity=article_schema.identity,
        )
        all_representations = (*representations, query_representation)
        requests = tuple(
            EmbeddingRequest(
                output_identity=stable_identity({"embedding": item.identity, "profile": embedding_profile.identity}),
                input=item,
                space=space,
                profile=embedding_profile,
            )
            for item in all_representations
        )
        outcomes = embedding_adapter_for(embedding_profile).embed_batch(requests)
        if not all(item.status == "succeeded" for item in outcomes):
            raise RunnerError("one or more configured embedding requests failed")
        provider_outcomes = [item.model_dump(mode="json", exclude_none=True) for item in outcomes]
        values = [list(item.value or []) for item in outcomes]
        vectors, query_vector = values[:-1], values[-1]
        embedding_space_identity = space.identity
        mode = "live"

        enrichment_schema = SchemaDescriptor.model_validate(schema_document["schemas"]["legal.enrichment.v1"])
        recipe_document = _load_yaml(project / "enrichments" / "legal.yml")["recipes"]["legal.amendment-recipe.v1"]
        recipe = EnrichmentRecipe(
            recipe_id=recipe_document["recipe_id"],
            revision=str(recipe_document["revision"]),
            accepted_representation_types=tuple(recipe_document["accepted_representation_types"]),
            accepted_media_types=tuple(recipe_document["accepted_media_types"]),
            output_schema_identity=enrichment_schema.identity,
            prompt_version=str(recipe_document["prompt_version"]),
            taxonomy_version=str(recipe_document["taxonomy_version"]),
            evidence_required=bool(recipe_document["evidence_required"]),
            abstention_allowed=bool(recipe_document["abstention_allowed"]),
        )
        completion_adapter = completion_adapter_for(profile.completion)
        by_event: dict[str, list[dict[str, Any]]] = {}
        for snapshot in snapshots:
            by_event.setdefault(snapshot["event_id"], []).append(snapshot)
        for event_id, event_snapshots in sorted(by_event.items()):
            event_text = _enrichment_input_text(event_snapshots)
            event_part = ContentPart(
                kind="text", media_type="text/plain", text=event_text,
                content_sha256=stable_identity(event_text),
            )
            event_input = QualifiedRepresentation(
                representation=Representation(
                    ir_version="0.1", representation_type="text", schema_ref=article_schema,
                    source_snapshot_ids=tuple(item["snapshot_identity"] for item in event_snapshots),
                    payload=event_text,
                ),
                parts=(event_part,),
            )
            request = CompletionRequest(
                output_identity=stable_identity({"enrichment": event_id, "profile": profile.completion.identity}),
                inputs=(event_input,),
                output_schema=enrichment_schema,
                recipe=recipe,
                prompt_version=recipe.prompt_version,
                taxonomy_version=recipe.taxonomy_version,
                profile=profile.completion,
            )
            outcome = completion_adapter.enrich(request)
            if outcome.status != "succeeded":
                enrichment_outcomes.append({"event_id": event_id, **outcome.model_dump(mode="json", exclude_none=True)})
                continue
            validate_schema_payload(enrichment_schema, outcome.value)
            enrichment_outcomes.append({"event_id": event_id, **outcome.model_dump(mode="json", exclude_none=True)})

    observations, outputs, retrieval_index = _runtime_rows(
        snapshots,
        vectors,
        observed_at=observed_at,
        source_system="huggingface-recare" if mode == "live" else "recare-fixture",
    )
    # Vector-bearing context outputs are the canonical source for context_retrieval.
    # Supplying a second hand-authored index row would duplicate each candidate.
    retrieval_index = []
    index_store = LanceStore(output / "retrieval-lance")
    materialize_context_model_lifecycle(
        store=index_store,
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval=retrieval_index,
    )
    retrieval = LanceRetrievalAdapter(index_store)
    index_facts = retrieval.ensure_vector_index()
    query = compile_context_model_retrieval_queries(manifest, {
        "article_search": {
            "text": query_text,
            "vector": query_vector,
            "limit": int(limits["retrieval_limit"]),
            "space_identity": embedding_space_identity,
        }
    })[0]
    facts = retrieval.retrieve(query)
    if not facts:
        raise RunnerError("semantic retrieval returned no candidates")
    retrieval_plan = dict(retrieval.last_execution)

    effective_times = sorted(item["effective_from"] for item in snapshots if item["effective_from"] is not None)
    current_as_of = effective_times[-1] + timedelta(days=1)
    historical_as_of = effective_times[0] + timedelta(days=1)
    current_rows = _retrieval_rows(facts, snapshots, as_of=current_as_of, runtime=runtime)
    historical_rows = _retrieval_rows(facts, snapshots, as_of=historical_as_of, runtime=runtime)

    allowed = _evaluate_lifecycle(
        output=output,
        name="allowed-relations",
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval_index=retrieval_index,
        retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="allow", as_of=current_as_of),
    )
    denied = _evaluate_lifecycle(
        output=output,
        name="denied-relations",
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval_index=retrieval_index,
        retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="deny", as_of=current_as_of),
    )
    abstain_outputs = [{**item, "authority_level": "", "authority_status": "unknown"} for item in outputs]
    abstained = _evaluate_lifecycle(
        output=output,
        name="abstained-relations",
        manifest=manifest,
        observations=observations,
        outputs=abstain_outputs,
        retrieval_index=retrieval_index,
        retrieval_rows=current_rows,
        contract_context=_contract_context(runtime, outcome="abstain", as_of=current_as_of),
    )
    historical = _evaluate_lifecycle(
        output=output,
        name="historical-relations",
        manifest=manifest,
        observations=observations,
        outputs=outputs,
        retrieval_index=retrieval_index,
        retrieval_rows=historical_rows,
        contract_context=_contract_context(runtime, outcome="allow", as_of=historical_as_of),
    )

    def decisions(lifecycle: Any) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for result in lifecycle.resolution_results for item in result.decisions]

    current_decisions = decisions(allowed)
    historical_decisions = decisions(historical)
    mutation_records = [dict(item) for item in records]
    mutation_records[0][runtime["mapping"]["text"]["before"]] = str(
        mutation_records[0].get(runtime["mapping"]["text"]["before"], "")
    ) + " [bounded mutation]"
    mutated_snapshots = _snapshots(mutation_records, runtime["mapping"])
    original_ids = {item["snapshot_identity"] for item in snapshots}
    mutated_ids = {item["snapshot_identity"] for item in mutated_snapshots}

    allowed_count = sum(item["decision"] == "allow" for item in allowed.contract_evaluations)
    denied_count = sum(item["decision"] == "deny" for item in denied.contract_evaluations)
    abstained_count = sum(item["decision"] == "abstain" for item in abstained.contract_evaluations)
    ann_active = any("IVF" in str(item.get("index_type", "")).upper() for item in index_facts)
    lineage_rows = allowed.materialization.relations.get("lineage_edges", [])
    lineage_nodes = allowed.materialization.relations.get("lineage_nodes", [])
    checks = {
        "project_compiled": bool(manifest.get("manifest_identity")),
        "source_pinned": records_payload.get("revision", records_payload.get("dataset_revision"))
        == "6281b8c718fa9f6a2ac44fda2498a92ccf47825b",
        "temporal_grounded": all(item["temporal_evidence"] for item in snapshots),
        "native_ann": ann_active and "ANN" in str(retrieval_plan.get("plan", "")).upper(),
        "current_selected": any(item["outcome"] == "selected" for item in current_decisions),
        "superseded_rejected": any(item["temporal_basis"] == "expired" for item in current_decisions),
        "historical_selected": any(item["outcome"] == "selected" for item in historical_decisions),
        "allowed_published": allowed_count > 0 and bool(allowed.packages),
        "denied_failed_closed": denied_count > 0 and not denied.packages,
        "abstained_failed_closed": abstained_count > 0 and not abstained.packages,
        "readable_lineage": bool(lineage_rows) and all(item.get("display_name") for item in lineage_nodes),
        "mutation_invalidation": bool(original_ids - mutated_ids),
        "unaffected_reuse": bool(original_ids & mutated_ids),
        "live_enrichment": mode != "live" or (
            len(enrichment_outcomes) == len(event_ids)
            and all(item.get("status") == "succeeded" for item in enrichment_outcomes)
        ),
    }
    artifact = {
        "artifact_type": "authoritative-context-qualification",
        "schema_version": "0.1",
        "qualification": "bounded-authoritative-context-v1",
        "mode": mode,
        "valid": all(checks.values()),
        "checks": checks,
        "dataset": {
            "repository": "kasys/ReCaRe",
            "revision": records_payload.get("revision", records_payload.get("dataset_revision")),
            "event_ids": event_ids,
            "row_sha256": list(records_payload.get("row_sha256", [])),
            "acquired_rows": len(_normalize_records(records_payload, runtime["mapping"])),
            "processed_rows": len(records),
        },
        "source_issues": source_issues,
        "amendment_events": len(event_ids),
        "amendments": [{"event_id": item, "display_name": f"Amendment {item}"} for item in event_ids],
        "ingestion": ingestion.as_dict(),
        "manifest": manifest,
        "snapshots": {
            "total": len(snapshots),
            "historical_queryable": checks["historical_selected"],
            "current_queryable": checks["current_selected"],
            "items": [{
                **{key: value for key, value in item.items() if key not in {"text", "effective_from", "effective_until"}},
                "effective_from": item["effective_from"].isoformat() if item["effective_from"] else None,
                "effective_until": item["effective_until"].isoformat() if item["effective_until"] else None,
                "preview": item["text"][:160],
            } for item in snapshots],
        },
        "retrieval": {
            "query": query_text,
            "query_identity": query.query_id,
            "query_vector_identity": stable_identity(query_vector),
            "results": [dict(item) if isinstance(item, Mapping) else item.model_dump(mode="json") for item in facts],
            "index_facts": list(index_facts),
            "execution": retrieval_plan,
        },
        "resolution": {
            "current_as_of": current_as_of.isoformat(),
            "historical_as_of": historical_as_of.isoformat(),
            "current_selected": checks["current_selected"],
            "superseded_rejected": checks["superseded_rejected"],
            "historical_selected": checks["historical_selected"],
            "decisions": current_decisions,
            "historical_decisions": historical_decisions,
        },
        "contract": {
            "allowed_count": allowed_count,
            "denied_count": denied_count,
            "abstained_count": abstained_count,
            "allowed_package_count": len(allowed.packages),
            "denied_package_count": len(denied.packages),
            "abstained_package_count": len(abstained.packages),
        },
        "packages": [item.model_dump(mode="json") for item in allowed.packages],
        "lineage": {
            "node_count": len(lineage_nodes),
            "edge_count": len(lineage_rows),
            "readable": checks["readable_lineage"],
        },
        "mutation": {
            "invalidated_count": len(original_ids - mutated_ids),
            "reused_count": len(original_ids & mutated_ids),
            "invalidated": sorted(original_ids - mutated_ids),
            "reused": sorted(original_ids & mutated_ids),
        },
        "providers": {
            "embedding": provider_outcomes,
            "completion": enrichment_outcomes,
            "expectations": {
                "embedding": len(snapshots) + 1 if mode == "live" else 0,
                "completion": len(event_ids) if mode == "live" else 0,
            },
        },
        "relation_counts": allowed.relation_counts,
        "timing_seconds": round(perf_counter() - started, 3),
    }
    (output / "qualification.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.html").write_text(_report(artifact), encoding="utf-8")
    return artifact


def qualify_recare(scenario: str | Path, output: str | Path) -> dict[str, Any]:
    scenario_path, output_path = Path(scenario), Path(output)
    payload = json.loads((scenario_path / "fixtures.json").read_text(encoding="utf-8"))
    return _run(scenario_path, output_path, records_payload=payload)


def qualify_recare_live(
    scenario: str | Path,
    profile_path: str | Path,
    acquisition: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    acquisition_path = Path(acquisition)
    if acquisition_path.is_dir():
        acquisition_path = acquisition_path / "acquisition.json"
    payload = json.loads(acquisition_path.read_text(encoding="utf-8"))
    return _run(Path(scenario), Path(output), records_payload=payload, embedding_profile_path=Path(profile_path))


def _report(artifact: Mapping[str, Any]) -> str:
    checks = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in artifact["checks"].items()
    )
    snapshots = "".join(
        "<tr>"
        f"<td>{html.escape(item['law_id'])}</td><td>{html.escape(item['article_number'])}</td>"
        f"<td>{html.escape(item['version'])}</td><td>{html.escape(str(item['effective_from']))}</td>"
        f"<td>{html.escape(str(item['effective_until']))}</td><td>{html.escape(item['preview'])}</td>"
        "</tr>"
        for item in artifact["snapshots"]["items"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>IGOR authoritative-context qualification</title>
<style>body{{font:15px/1.5 system-ui;max-width:1180px;margin:40px auto;padding:0 20px;color:#17313b}}table{{width:100%;border-collapse:collapse;margin:18px 0}}th,td{{border-bottom:1px solid #d8e5e8;padding:9px;text-align:left;vertical-align:top}}.pass{{color:#167042}}.fail{{color:#9e2d2d}}code{{background:#eef3f4;padding:2px 5px}}</style></head><body>
<h1>Bounded authoritative-context qualification</h1>
<p class='{'pass' if artifact['valid'] else 'fail'}'><strong>{'Passed' if artifact['valid'] else 'Failed'}</strong> — {artifact['mode']} mode, {artifact['amendment_events']} amendment events.</p>
<h2>Trust checks</h2><table><tr><th>Check</th><th>Result</th></tr>{checks}</table>
<h2>Snapshot history</h2><table><tr><th>Law</th><th>Article</th><th>Version</th><th>Effective from</th><th>Effective until</th><th>Evidence preview</th></tr>{snapshots}</table>
<h2>Semantic retrieval</h2><p><code>{html.escape(artifact['retrieval']['query'])}</code></p><pre>{html.escape(str(artifact['retrieval']['execution']['plan']))}</pre>
<h2>Context Contract</h2><p>Allowed: {artifact['contract']['allowed_count']}; denied: {artifact['contract']['denied_count']}; abstained: {artifact['contract']['abstained_count']}.</p>
<h2>Lineage</h2><p>{artifact['lineage']['node_count']} readable nodes and {artifact['lineage']['edge_count']} edges.</p>
</body></html>"""
