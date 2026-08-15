"""Small live qualification for the DataFusion -> retrieval port -> Lance path."""

from __future__ import annotations

import json
import base64
import hashlib
import os
import sys
from pathlib import Path
from time import perf_counter

from igor_context_compiler import (
    ContextModelCompileRequest,
    ContextModelReference,
    EmbeddingRequest,
    QualifiedRepresentation,
    VoyageMultimodalEmbeddingAdapter,
    compile_context_model,
)
from igor_core import ContentPart, EmbeddingSpace, Representation, SchemaDescriptor, load_reference_profile, stable_identity
from igor_lancedb import LanceRetrievalAdapter, LanceStore

from .context_model_lifecycle import materialize_context_model_lifecycle


def _semantic_sql_context_manifest(schema: SchemaDescriptor, embedding_profile):
    return compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {
                "id": "support",
                "revision": "semantic-sql-live-v0",
                "title": "Support semantic SQL live qualification",
            },
            "sources": {
                "live_support_messages": {
                    "source_contract": "support.messages.live.v1",
                    "connector_binding": "semantic-sql-live.v1",
                    "identity_fields": ["ticket_id"],
                },
            },
            "objects": {
                "support_ticket": {
                    "kind": "business_object",
                    "source": "live_support_messages",
                    "display_name": "Support ticket {{ ticket_id }}",
                },
                "support_message_snapshot": {
                    "kind": "source_snapshot",
                    "source": "live_support_messages",
                    "display_name": "{{ ticket_id }} observed at {{ observed_at }}",
                },
                "support_message": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_message_snapshot"],
                    "operation": "representation.v1",
                    "schema": "semantic-sql.qualification.v1",
                    "display_name": "Support semantic SQL message",
                },
                "support_message_embedding": {
                    "kind": "retrieval_projection",
                    "derived_from": ["support_message"],
                    "operation": "embedding.multimodal.v1",
                    "schema": "semantic-sql.qualification.v1",
                    "model_profile": "semantic-sql.embedding-profile.v1",
                    "display_name": "Support semantic SQL embedding",
                },
                "invoice_preview": {
                    "kind": "semantic_derivation",
                    "operation": "representation.image.v1",
                    "schema": "semantic-sql.qualification.v1",
                    "display_name": "Invoice preview",
                },
            },
            "authority": {
                "support.authority-temporal": {
                    "target": "support_message",
                    "policy": "support.authority-temporal.v1",
                    "active_only": True,
                    "valid_at_task_time": True,
                },
            },
            "retrievals": {
                "support_messages": {
                    "search": "vector",
                    "candidate_limit": 3,
                    "resolution": {"policy": "support.authority-temporal", "accepted_outcomes": ["selected"]},
                },
            },
        },
        references=(
            ContextModelReference(
                kind="source_contract",
                ref="support.messages.live.v1",
                identity=stable_identity({"source_contract": "support.messages.live.v1"}),
            ),
            ContextModelReference(
                kind="connector_binding",
                ref="semantic-sql-live.v1",
                identity=stable_identity({"connector_binding": "semantic-sql-live.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.v1",
                identity=stable_identity({"operation": "representation.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="embedding.multimodal.v1",
                identity=stable_identity({"operation": "embedding.multimodal.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.image.v1",
                identity=stable_identity({"operation": "representation.image.v1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="semantic-sql.qualification.v1",
                identity=schema.identity,
            ),
            ContextModelReference(
                kind="model_profile",
                ref="semantic-sql.embedding-profile.v1",
                identity=embedding_profile.identity,
            ),
            ContextModelReference(
                kind="authority_policy",
                ref="support.authority-temporal.v1",
                identity=stable_identity({"policy": "support.authority-temporal", "revision": "1"}),
            ),
        ),
    ))


def qualify_live(profile_path: str | Path, output: str | Path, lance_root: str | Path) -> dict:
    profile = load_reference_profile(profile_path)
    if profile.embedding.provider != "voyage":
        raise ValueError("live semantic SQL qualification requires the qualified Voyage embedding profile")
    schema = SchemaDescriptor(schema_version="0.1", schema_id="semantic-sql.qualification", revision="1", domain="support")
    texts = (
        ("ticket-001", "Customer was charged twice for the same invoice."),
        ("ticket-002", "The account cannot reset its password."),
        ("ticket-003", "A refund is still missing after seven days."),
        ("ticket-004", "Customer was charged twice for the same invoice."),
    )
    representations = []
    for key, text in texts:
        representation = Representation(
            ir_version="0.1", representation_type="text", schema_ref=schema,
            source_snapshot_ids=(stable_identity({"source": key}),), payload=text,
        )
        representations.append(QualifiedRepresentation(
            representation=representation,
            parts=(ContentPart(kind="text", media_type="text/plain", text=text, content_sha256=stable_identity(text)),),
        ))
    query_text = "customer charged twice"
    query_representation = Representation(
        ir_version="0.1", representation_type="text", schema_ref=schema,
        source_snapshot_ids=(stable_identity("query:" + query_text),), payload=query_text,
    )
    query_input = QualifiedRepresentation(
        representation=query_representation,
        parts=(ContentPart(kind="text", media_type="text/plain", text=query_text, content_sha256=stable_identity(query_text)),),
    )
    space = EmbeddingSpace(
        ir_version="0.1", provider=profile.embedding.provider, model=profile.embedding.model,
        model_revision=profile.embedding.revision, dimension=int(profile.embedding.parameters["dimensions"]),
        dtype="float32", metric="cosine", normalized=False, input_schema_identity=schema.identity,
    )
    requests = tuple(
        EmbeddingRequest(output_identity=stable_identity({"embedding": item.identity}), input=item, space=space, profile=profile.embedding)
        for item in (*representations, query_input)
    )
    started = perf_counter()
    outcomes = VoyageMultimodalEmbeddingAdapter().embed_batch(requests)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    if not all(item.status == "succeeded" for item in outcomes):
        raise RuntimeError(json.dumps([item.model_dump(mode="json", exclude_none=True) for item in outcomes], sort_keys=True))
    vectors = [item.value for item in outcomes]
    # Keep this qualification schema-isolated so repeated runs cannot reuse an older
    # Lance schema that predates the complete authority/time presentation fields.
    store = LanceStore(Path(lance_root) / "run-v2")
    outputs = [{
        "identity": item.representation.identity,
        "role": "support_message",
        "kind": "context_representation",
        "object_kind": "support_message",
        "derived_from": list(item.representation.source_snapshot_ids),
        "schema_id": schema.schema_id,
        "schema_revision": schema.revision,
        "display_name": key,
        "payload": item.representation.model_dump(mode="json"),
        "value": json.dumps(item.representation.model_dump(mode="json"), sort_keys=True),
        "vector": vector,
        "text": text,
        "source_system": "live-qualification",
        "source_key": key,
        "media_type": "text/plain",
        "evidence_preview": text[:96],
        "content_ref": f"qualification://{key}",
        "status": "inactive" if key == "ticket-004" else "active",
        "authority_status": "rejected" if key == "ticket-004" else "authoritative",
        "temporal_status": "stale" if key == "ticket-004" else "valid",
        "policy_id": "support.authority-temporal:1",
        "as_of": "2026-08-13T00:00:00Z",
    } for (key, text), item, vector in zip(texts, representations, vectors[:4])]
    image_identity = stable_identity("asset:invoice-preview")
    image_fixture = Path(lance_root) / "run-v2" / "invoice-preview.png"
    image_fixture.parent.mkdir(parents=True, exist_ok=True)
    image_fixture.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    outputs.append({
        "identity": image_identity,
        "role": "invoice_preview",
        "kind": "context_representation",
        "object_kind": "invoice_image",
        "derived_from": [],
        "schema_id": schema.schema_id,
        "schema_revision": schema.revision,
        "display_name": "invoice-preview.png",
        "payload": {"asset_id": "asset-001"},
        "value": json.dumps({"asset_id": "asset-001"}, sort_keys=True),
        "vector": [1.01 * left - 0.01 * right for left, right in zip(vectors[0], vectors[-1])],
        "text": "duplicate charge invoice image",
        "source_system": "live-qualification",
        "source_key": "asset-001",
        "media_type": "image/png",
        "evidence_preview": "PNG invoice preview: duplicate charge evidence",
        "content_ref": f"file://{image_fixture}",
        "status": "active",
        "authority_status": "authoritative",
        "temporal_status": "valid",
        "policy_id": "support.authority-temporal:1",
        "as_of": "2026-08-13T00:00:00Z",
    })
    observations = [{
        "object_role": "support_ticket",
        "snapshot_role": "support_message_snapshot",
        "source_system": "live-qualification",
        "source_key": key,
        "snapshot_identity": item.representation.source_snapshot_ids[0],
        "content_ref": f"qualification://{key}",
        "content_sha256": stable_identity(text),
        "media_type": "text/plain",
        "observed_at": "2026-08-13T00:00:00Z",
        "display_name": key,
        "preview": text[:96],
    } for (key, text), item in zip(texts, representations)]
    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest=_semantic_sql_context_manifest(schema, profile.embedding),
        observations=observations,
        outputs=outputs,
    )
    catalog_rows = [
        row for row in lifecycle.materialization.relations["context_catalog"]
        if row["physical_relation"] == "context_outputs"
    ]
    retrieval = LanceRetrievalAdapter(store)
    index_facts = retrieval.ensure_vector_index()
    query_vector_ref = stable_identity({"query": query_text, "embedding_space": space.identity})
    sql = (
        "SELECT canonical_identity, display_name, source_key, media_type, evidence_preview, content_ref, "
        "semantic_score, authority_status, temporal_status, policy_id, as_of "
        "FROM semantic_search('context_catalog', 'customer charged twice', 'vector', 3, '" + query_vector_ref + "', 'status = ''active''') "
        "WHERE context_model_id = 'support' AND status = 'active' "
        "ORDER BY semantic_score DESC, canonical_identity ASC"
    )
    vector_registry = {query_vector_ref: tuple(vectors[-1])}
    query_started = perf_counter()
    rows = lifecycle.query_standard_sql(
        sql,
        table_name="context_catalog",
        retrieval=retrieval,
        query_vectors=vector_registry,
    ).to_pylist()
    query_ms = round((perf_counter() - query_started) * 1000, 2)
    if not rows or rows[0]["source_key"] != "ticket-001":
        raise RuntimeError(f"unexpected semantic ranking: {rows}")
    artifact = {
        "qualification": "semantic-sql-live-v0",
        "bounded_records": len(catalog_rows),
        "query": sql,
        "rows": rows,
        "path": "DataFusion TableFunction -> semantic retrieval port -> LanceDB native vector search",
        "profiles": {"embedding": profile.embedding.identity},
        "timing_ms": {"embedding_batch": elapsed_ms, "semantic_sql": query_ms},
        "lance_tables": sorted(store.names()),
        "index_facts": {"retrieval": "Lance native vector search", "ann_index": index_facts},
        "query_vector": {"ref": query_vector_ref, "embedding_space": space.identity,
                         "dimension": space.dimension, "metric": space.metric},
        "runtime_trace": {"engine": "DataFusion", "table_function": "semantic_search",
                           "adapter": retrieval.last_execution.get("adapter"),
                           "prefilter": retrieval.last_execution.get("prefilter"),
                           "plan": retrieval.last_execution.get("plan", "")},
        "behavioral_cases": {"excluded_inactive": "ticket-004", "non_text_fixture": "asset-001",
                              "excluded_from_rows": "ticket-004" not in {row["source_key"] for row in rows}},
        "lineage": [{"canonical_identity": row["canonical_identity"], "source_key": row["source_key"],
                      "content_ref": row["content_ref"], "media_type": row["media_type"]} for row in rows],
        "network": {"embedding_provider": profile.embedding.provider, "api_key_recorded": bool(os.environ.get("IGOR_EMBEDDING_API_KEY"))},
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    artifact["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"qualification": artifact["qualification"], "rows": len(rows), "top_result": rows[0]["source_key"], "artifact": str(output_path)}


if __name__ == "__main__":
    raise SystemExit(json.dumps(qualify_live(sys.argv[1], sys.argv[2], sys.argv[3]), sort_keys=True))
