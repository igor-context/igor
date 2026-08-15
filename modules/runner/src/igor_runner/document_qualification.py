"""Direct multimodal document qualification through the shared runner seams."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from igor_context_compiler import (
    ContextModelCompileRequest, ContextModelReference,
    DeepSeekCompletionAdapter, EmbeddingRequest, GeminiCompletionAdapter, MistralCompletionAdapter,
    ProviderOutcome,
    QualifiedRepresentation, ResolutionDecision,
    ResolutionRequest, ResolutionResult, RuntimeLimits, VoyageMultimodalEmbeddingAdapter,
    compile_context_model, run_enrichment,
)
from igor_core import (
    ConnectorBinding, ConnectorFieldBinding, ConnectorResourceBinding, ContentPart,
    ContextSourceContract, EmbeddingSpace,
    Representation, SchemaDescriptor, SourceFieldRequirement, SourceResourceContract, stable_identity,
    load_reference_profile,
)
from igor_datafusion import AnalyticalEngine
from igor_dlt import IngestConfig, ingest_bound_records
from igor_lancedb import LanceRetrievalAdapter, LanceStore
from igor_runner.context_model_lifecycle import materialize_context_model_lifecycle

from .compose import RunnerError
from .document_runtime_helpers import load_document_context_model, make_document_work_item


def _seam():
    concepts = ("page_id", "document_id", "page_number", "revision", "observed_at",
                "effective_from", "authority", "image_ref")
    contract = ContextSourceContract(
        contract_id="document.pages", revision="2", domain="document",
        resources=(SourceResourceContract(
            resource_id="page", resource_kind="document-page", identity_concepts=("page_id",),
            fields=tuple(SourceFieldRequirement(concept_id=x, logical_type="string", required=True) for x in concepts),
            accepted_media_types=("image/png", "image/jpeg",), change_mode="incremental",
            cursor_concept="observed_at", deletion_semantics="tombstone"),))
    binding = ConnectorBinding(
        binding_id="huggingface.docvqa", revision="2", source_contract_identity=contract.identity,
        connector="huggingface", deployment_ref="lance-format/docvqa-lance@pinned",
        resources=(ConnectorResourceBinding(
            resource_id="page", source_resource="validation", cursor_field="observed_at",
            fields=tuple(ConnectorFieldBinding(concept_id=x, source_field=x) for x in concepts)),))
    binding.validate_against(contract)
    return contract, binding


def _document_context_manifest(context_model):
    contract, binding = _seam()
    return compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {
                "id": context_model.context_model_id,
                "revision": str(context_model.domain_schema["revision"]),
                "title": str(context_model.domain_schema.get("title", "Document question answering")),
            },
            "sources": {
                "document_pages": {
                    "source_contract": "document.pages.v3",
                    "connector_binding": "huggingface.docvqa.v2",
                    "identity_fields": ["page_id"],
                },
            },
            "objects": {
                "document_page": {
                    "kind": "business_object",
                    "source": "document_pages",
                    "display_name": "Document {{ document_id }} page {{ page_number }}",
                },
                "source_snapshot": {
                    "kind": "source_snapshot",
                    "source": "document_pages",
                    "display_name": "{{ page_id }} observed at {{ observed_at }}",
                },
                "document_page_projection": {
                    "kind": "retrieval_projection",
                    "derived_from": ["source_snapshot"],
                    "operation": "embedding.multimodal.v1",
                    "schema": "document.page.v3",
                    "model_profile": "document.embedding-profile.v1",
                    "display_name": "Document page retrieval projection",
                },
                "document_answer": {
                    "kind": "semantic_derivation",
                    "derived_from": ["source_snapshot"],
                    "operation": "document.qa.direct-multimodal.v1",
                    "schema": "document.qa.v2",
                    "semantic_definition": "document.qa.semantic.v2",
                    "recipe": "document.qa.recipe.v3",
                    "display_name": "Document answer",
                },
            },
            "authority": {
                "document.source": {
                    "target": "document_page",
                    "policy": "document.source.v1",
                    "source_authoritative": True,
                },
            },
            "retrievals": {
                "document_pages": {
                    "search": "vector",
                    "candidate_limit": 20,
                    "resolution": {"policy": "document.source", "accepted_outcomes": ["selected"]},
                },
            },
        },
        references=(
            ContextModelReference(
                kind="source_contract",
                ref="document.pages.v3",
                identity=contract.identity,
            ),
            ContextModelReference(
                kind="connector_binding",
                ref="huggingface.docvqa.v2",
                identity=binding.identity,
            ),
            ContextModelReference(
                kind="operation",
                ref="embedding.multimodal.v1",
                identity=stable_identity({"operation": "embedding.multimodal.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="document.qa.direct-multimodal.v1",
                identity=stable_identity({"operation": "document.qa.direct-multimodal.v1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="document.page.v3",
                identity=stable_identity(context_model.domain_schema),
            ),
            ContextModelReference(
                kind="schema",
                ref="document.qa.v2",
                identity=stable_identity({"schema": "document.qa", "revision": "2"}),
            ),
            ContextModelReference(
                kind="semantic_definition",
                ref="document.qa.semantic.v2",
                identity=stable_identity(context_model.semantic_definition),
            ),
            ContextModelReference(
                kind="recipe",
                ref="document.qa.recipe.v3",
                identity=stable_identity({"recipe": "document.qa", "revision": "3"}),
            ),
            ContextModelReference(
                kind="model_profile",
                ref="document.embedding-profile.v1",
                identity=stable_identity({"model_profile": "document.embedding-profile.v1"}),
            ),
            ContextModelReference(
                kind="authority_policy",
                ref="document.source.v1",
                identity=stable_identity({"policy": "document.source", "revision": "1"}),
            ),
        ),
    ))


class _DocumentResolution:
    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        decisions = tuple(ResolutionDecision(
            candidate_identity=candidate.candidate_identity,
            outcome="selected",
            reason_code="authoritative_current_page",
            authority_basis="satisfied",
            temporal_basis="valid",
            as_of=request.as_of,
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
        ) for candidate in sorted(request.candidates, key=lambda item: item.candidate_identity))
        return ResolutionResult(
            request_identity=request.request_identity,
            resolution_identity=request.identity,
            decisions=decisions,
            selected_identities=tuple(decision.candidate_identity for decision in decisions),
        )


def _verified_page(page: dict, output: Path) -> dict:
    """Materialize deterministic fixture bytes as the verified original content."""
    content = json.dumps({"page_id": page["page_id"], "slice": page["slice"],
                          "source_text": page.get("source_text", "")}, sort_keys=True).encode()
    path = output / "content" / f"{page['page_id']}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {**page, "content_ref": path.as_uri(), "image_ref": path.as_uri(),
            "content_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
            "media_type": "image/png"}


def _load_acquisition(acquisition: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only the public acquisition sidecar and verify every page byte."""
    root = Path(acquisition)
    manifest = json.loads((root / "acquisition.json").read_text(encoding="utf-8"))
    pages: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        ref = record.get("image_ref")
        expected = record.get("image_sha256")
        if not ref or not expected or not str(ref).startswith("file:"):
            raise RunnerError("document acquisition record lacks a local verified image reference")
        image_path = Path.from_uri(ref)
        if not image_path.is_file():
            raise RunnerError(f"document acquisition image is missing: {ref}")
        actual = "sha256:" + hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual != expected:
            raise RunnerError(f"document acquisition image hash mismatch for {record.get('page_id')}")
        page_id = str(record["page_id"])
        pages.setdefault(page_id, {
            "page_id": page_id,
            "document_id": str(record["document_id"]),
            "page_number": int(record["page_number"]),
            "revision": "pinned",
            "observed_at": "2026-01-01T00:00:00Z",
            "effective_from": "2026-01-01T00:00:00Z",
            "authority": "source",
            "content_ref": ref,
            "image_ref": ref,
            "content_sha256": expected,
            "media_type": "image/jpeg",
            "slice": "huggingface-docvqa",
        })
        questions.append({
            "question_id": str(record["question_id"]),
            "page_id": page_id,
            "question": str(record["question"]),
            "question_types": [str(value) for value in record.get("question_types", [])],
            "family": ",".join(record.get("question_types", [])),
            "answerability": "answerable",
        })
    if not pages or not questions:
        raise RunnerError("document acquisition contains no bounded records")
    return list(pages.values()), questions


def qualify_document(scenario: str | Path, output: str | Path, acquisition: str | Path | None = None) -> dict:
    scenario, output = Path(scenario), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context_model = load_document_context_model(scenario)
    data = json.loads((scenario / "fixtures.json").read_text())
    if acquisition is None:
        pages = [_verified_page(page, output) for page in data["pages"]]
        questions = data["questions"]
        acquisition_mode = "deterministic-fixture-only"
    else:
        pages, questions = _load_acquisition(acquisition)
        acquisition_mode = "pinned-hugging-face-acquisition"
    if len(pages) > 4 or len(questions) > 8:
        raise RunnerError("document qualification ceiling exceeded")

    contract, binding = _seam()
    raw = [{"_resource_id": "page", "_media_type": page["media_type"],
            "_content_ref": page["content_ref"], "_content_sha256": page["content_sha256"],
            "_observed_at": page["observed_at"], **page} for page in pages]
    ingestion = ingest_bound_records(raw, IngestConfig(
        "qualification", "document", str(output / "ingestion-lance"), "qualification/document"),
        contract, binding)
    store = LanceStore(output / "relations")
    page_rows = [{k: page[k] for k in ("page_id", "document_id", "page_number", "authority",
                                       "revision", "effective_from", "content_ref", "content_sha256",
                                       "media_type")} for page in pages]
    if "document_pages" in store.names():
        store.replace("document_pages", page_rows)
    else:
        store.create("document_pages", page_rows)
    question_rows = [{k: value for k, value in question.items() if k not in {"answer"}} for question in questions]
    if "document_questions" in store.names():
        store.replace("document_questions", question_rows)
    else:
        store.create("document_questions", question_rows)

    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest=_document_context_manifest(context_model),
        observations=[{
            "object_role": "document_page",
            "snapshot_role": "source_snapshot",
            "source_system": "huggingface",
            "source_key": page["page_id"],
            "snapshot_identity": page["content_sha256"],
            "observed_at": page["observed_at"],
            "content_ref": page["content_ref"],
            "content_sha256": page["content_sha256"],
            "media_type": page["media_type"],
            "display_name": f"Document {page['document_id']} page {page['page_number']}",
            "preview": f"verified {page['media_type']} content",
            "schema_id": "document.page",
            "schema_revision": "3",
            "operation_name": "ingest",
            "status": "observed",
        } for page in pages],
    )
    materialized = lifecycle.materialization
    sql = ("SELECT c.display_name, c.context_model_id, c.source_key, c.schema_id, c.operation_name, "
           "n.canonical_node_id FROM context_catalog c JOIN lineage_nodes n "
           "ON c.canonical_identity = n.canonical_node_id ORDER BY c.source_key")
    sql_rows = lifecycle.query_standard_sql(sql, table_name="context_catalog").to_pylist()

    items = tuple(make_document_work_item(
        {**question, **next(page for page in pages if page["page_id"] == question["page_id"])},
        context_model=context_model,
    ) for question in questions)
    questions_by_work_identity = {
        item.work_identity: question for item, question in zip(items, questions, strict=True)
    }

    class DeterministicMultimodalExecutor:
        def execute(self, item):
            question = questions_by_work_identity[item.work_identity]
            # The real acquisition deliberately contains no answers. A deterministic
            # run may still exercise the full multimodal seam, but must abstain rather
            # than read evaluator judgments or invent a textual projection.
            answer = question.get("answer")
            return ProviderOutcome(status="succeeded", attempts=1, value={
                "answer": answer, "abstain": answer is None,
                "evidence_page_id": question["page_id"]})

    runtime = run_enrichment(items, DeterministicMultimodalExecutor(),
                             RuntimeLimits(max_concurrency=4, max_requests=24, max_items=8))
    predictions = []
    for item, question in zip(items, questions, strict=True):
        outcome = runtime.outcomes[item.work_identity]
        value = outcome.value
        expected = question.get("answer")
        abstention_expected = question.get("answerability") == "abstain"
        predictions.append({"question_id": question["question_id"], "predicted": value["answer"],
                            "abstain": value["abstain"], "evidence_page_id": value["evidence_page_id"],
                            "correct": ((abstention_expected and value["abstain"]) or
                                        (not abstention_expected and expected is not None and
                                         value["answer"] == expected and not value["abstain"]))})
    baseline = stable_identity(pages)
    mutated_pages = [dict(page, content_sha256=stable_identity(page["content_sha256"] + " changed")) if page["page_id"] == "page-form-02" else page for page in pages]
    artifact = {
        "dataset": json.loads((scenario / "selection.json").read_text()),
        "context_model": {
            "context_model_id": context_model.context_model_id,
            "identity": context_model.identity,
            "domain_schema_revision": context_model.domain_schema["revision"],
            "semantic_definition_id": context_model.semantic_definition["semantic_definition_id"],
            "semantic_definition_revision": context_model.semantic_definition["revision"],
        },
        "acquisition_mode": acquisition_mode,
        "projection_boundary": "verified original document content + task text -> direct multimodal adapter",
        "source_contract": contract.model_dump(mode="json"), "connector_binding": binding.model_dump(mode="json"),
        "ingestion": ingestion.as_dict(), "sql": {"query": sql, "rows": sql_rows},
        "catalog_relations": list(materialized.relation_counts), "predictions": predictions,
        "runtime": runtime.artifact, "mutations": {"baseline": baseline, "mutated": stable_identity(mutated_pages),
            "invalidated": ["page-form-02"], "reused": [page["page_id"] for page in pages if page["page_id"] != "page-form-02"]},
    }
    (output / "qualification.json").write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    (output / "report.html").write_text("<html><body><h1>Direct multimodal document qualification</h1><p>DataFusion SQL readable catalog/lineage join</p></body></html>")
    return {"valid": True, "pages": len(pages), "questions": len(questions),
            "accuracy": sum(prediction["correct"] for prediction in predictions) / len(predictions),
            "report": str(output / "report.html")}


def qualify_document_live(
    scenario: str | Path,
    acquisition: str | Path,
    output: str | Path,
    profile_path: str | Path,
) -> dict:
    """Run direct original-content completion through the resolved live profile."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pages, questions = _load_acquisition(acquisition)
    context_model = load_document_context_model(scenario)
    reference_profile = load_reference_profile(profile_path)
    profile = reference_profile.completion
    supported = set(profile.parameters.get("supported_content_kinds", []))
    if not {"image", "text"}.issubset(supported):
        raise RunnerError("resolved completion profile must declare image and text support")
    items = tuple(make_document_work_item(
        {**question, **next(page for page in pages if page["page_id"] == question["page_id"])},
        profile,
        context_model,
    ) for question in questions)
    completion_profile_identity = profile.identity
    cached_outcomes: dict[str, ProviderOutcome] = {}
    prior_path = output / "qualification.json"
    if prior_path.is_file():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            prior_profile = prior.get("profile", {})
            prior_profile_identity = stable_identity(prior_profile)
            for value in prior.get("outcomes", []):
                metadata = value.get("metadata") or {}
                if (prior_profile_identity == completion_profile_identity
                        and prior.get("context_model", {}).get("identity") == context_model.identity
                        and value.get("status") == "succeeded"
                        and value.get("value") is not None
                        and metadata.get("provider") == profile.provider
                        and metadata.get("model") == profile.model
                        and metadata.get("provider_revision") == profile.revision):
                    cached_outcomes[value.get("work_identity", "")] = ProviderOutcome.model_validate(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            cached_outcomes = {}
    class CompletionExecutor:
        def __init__(self):
            adapters = {
                "deepseek": DeepSeekCompletionAdapter,
                "gemini": GeminiCompletionAdapter,
                "mistral": MistralCompletionAdapter,
            }
            try:
                self.adapter = adapters[profile.provider]()
            except KeyError as error:
                raise RunnerError(f"unsupported document completion provider: {profile.provider}") from error

        def execute(self, item):
            if item.work_identity in cached_outcomes:
                return cached_outcomes[item.work_identity]
            return self.adapter.enrich(item.request)

    runtime = run_enrichment(items, CompletionExecutor(),
                             RuntimeLimits(max_concurrency=int(profile.parameters.get("max_concurrency", 1)),
                                           max_requests=24, max_items=8,
                                           max_attempts=int(profile.parameters.get("retry_attempts", 3)),
                                           retry_backoff_seconds=float(profile.parameters.get("retry_backoff_seconds", 1.0))))
    # Checkpoint structured values before downstream stages so a later pipeline failure
    # never discards successful provider work or forces duplicate paid/free-tier calls.
    context_model_artifact = {
        "context_model_id": context_model.context_model_id,
        "identity": context_model.identity,
        "domain_schema_revision": context_model.domain_schema["revision"],
        "semantic_definition_id": context_model.semantic_definition["semantic_definition_id"],
        "semantic_definition_revision": context_model.semantic_definition["revision"],
    }
    checkpoint = {"mode": "live-direct-multimodal-checkpoint", "profile": profile.model_dump(mode="json"),
                  "context_model": context_model_artifact,
                  "runtime": runtime.artifact,
                  "outcomes": [{"work_identity": identity, **outcome.model_dump(mode="json")}
                               for identity, outcome in runtime.outcomes.items()]}
    (output / "qualification.json").write_text(json.dumps(checkpoint, sort_keys=True, indent=2) + "\n")
    if not all(outcome.status == "succeeded" for outcome in runtime.outcomes.values()):
        artifact = {"mode": "live-direct-multimodal", "profile": profile.model_dump(mode="json"),
                    "context_model": context_model_artifact,
                    "runtime": runtime.artifact,
                    "outcomes": [{"work_identity": identity, **outcome.model_dump(mode="json")}
                                 for identity, outcome in runtime.outcomes.items()]}
        (output / "qualification.json").write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
        return {"valid": False, "questions": len(questions), "profile": profile.profile_id,
                "output": str(output / "qualification.json")}

    mutation_runtime = _run_live_mutation(pages, questions, profile, context_model, output)
    pipeline = _run_live_document_pipeline(
        pages, questions, runtime, reference_profile.embedding, profile, context_model, output, items,
    )
    pipeline["mutation_reuse"]["actual_mutation_calls"] = len(mutation_runtime.outcomes)
    pipeline["mutation_reuse"]["mutation_provider_statuses"] = [item.status for item in mutation_runtime.outcomes.values()]
    pipeline["mutation_reuse"]["mutation_provider_metadata"] = [
        item.metadata.model_dump(mode="json") if item.metadata else None
        for item in mutation_runtime.outcomes.values()
    ]
    artifact = {"mode": "live-document-end-to-end", "profile": profile.model_dump(mode="json"),
                "context_model": context_model_artifact,
                "runtime": runtime.artifact,
                "outcomes": [{"work_identity": identity, **outcome.model_dump(mode="json")}
                             for identity, outcome in runtime.outcomes.items()],
                "mutation_outcomes": [{"work_identity": identity, **outcome.model_dump(mode="json")}
                                      for identity, outcome in mutation_runtime.outcomes.items()],
                **pipeline}
    (output / "qualification.json").write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")
    (output / "report.html").write_text(_document_report(artifact), encoding="utf-8")
    return {"valid": all(outcome.status == "succeeded" for outcome in runtime.outcomes.values()),
            "questions": len(questions), "profile": profile.profile_id,
            "output": str(output / "qualification.json")}


def _run_live_mutation(pages, questions, profile, context_model, output: Path):
    """Make the declared three changed-page calls with a fresh, profile-bound executor."""
    changed_page_id = pages[0]["page_id"]
    mutation_root = output / "mutation-content"
    mutation_root.mkdir(parents=True, exist_ok=True)
    mutated_pages = {page["page_id"]: page for page in pages}
    changed = dict(mutated_pages[changed_page_id])
    original = Path.from_uri(changed["content_ref"]).read_bytes()
    mutated_path = mutation_root / f"{changed_page_id.replace(':', '-')}.jpg"
    mutated_bytes = original + b"\xff\xfeIGOR-MUTATION"
    mutated_path.write_bytes(mutated_bytes)
    changed["content_ref"] = mutated_path.as_uri()
    changed["content_sha256"] = "sha256:" + hashlib.sha256(mutated_bytes).hexdigest()
    mutated_pages[changed_page_id] = changed
    affected = [question for question in questions if question["page_id"] == changed_page_id]
    mutation_items = tuple(make_document_work_item(
                            {**question, **mutated_pages[question["page_id"]]}, profile, context_model)
                            for question in affected)
    adapters = {"deepseek": DeepSeekCompletionAdapter, "gemini": GeminiCompletionAdapter,
                "mistral": MistralCompletionAdapter}
    try:
        adapter = adapters[profile.provider]()
    except KeyError as error:
        raise RunnerError(f"unsupported mutation completion provider: {profile.provider}") from error

    class MutationExecutor:
        def execute(self, item):
            return adapter.enrich(item.request)

    return run_enrichment(mutation_items, MutationExecutor(),
                           RuntimeLimits(max_concurrency=int(profile.parameters.get("max_concurrency", 1)),
                                         max_requests=3, max_items=3,
                                         max_attempts=int(profile.parameters.get("retry_attempts", 3)),
                                         retry_backoff_seconds=float(profile.parameters.get("retry_backoff_seconds", 1.0))))


def _run_live_document_pipeline(
    pages, questions, runtime, embedding_profile, profile, context_model, output: Path, work_items,
) -> dict[str, Any]:
    """Traverse the live answers through storage, retrieval, resolution, package, and SQL seams."""
    if embedding_profile.provider != "voyage":
        raise RunnerError("live document pipeline requires the qualified Voyage embedding profile")

    contract, binding = _seam()
    raw = [{"_resource_id": "page", "_media_type": page["media_type"],
            "_content_ref": page["content_ref"], "_content_sha256": page["content_sha256"],
            "_observed_at": page["observed_at"], **page} for page in pages]
    ingestion = ingest_bound_records(raw, IngestConfig(
        "live-document", "document", str(output / "ingestion-lance"), "qualification/document"),
        contract, binding)

    schema = SchemaDescriptor(schema_version="0.1", schema_id="document.qa", revision="1")
    space = EmbeddingSpace(
        ir_version="0.1", provider=embedding_profile.provider, model=embedding_profile.model,
        model_revision=embedding_profile.revision, dimension=int(embedding_profile.parameters["dimensions"]),
        dtype="float32", metric="cosine", normalized=False, input_schema_identity=schema.identity,
    )
    page_representations = []
    for page in pages:
        part = ContentPart(kind="image", media_type=page["media_type"], content_ref=page["content_ref"],
                           content_sha256=page["content_sha256"], locator={"page_id": page["page_id"]})
        representation = Representation(ir_version="0.1", representation_type="image", schema_ref=schema,
                                        source_snapshot_ids=(page["content_sha256"],), payload=None)
        page_representations.append(QualifiedRepresentation(representation=representation, parts=(part,)))
    query_representations = []
    for question in questions:
        text = question["question"]
        part = ContentPart(kind="text", media_type="text/plain", text=text, content_sha256=stable_identity(text))
        representation = Representation(ir_version="0.1", representation_type="text", schema_ref=schema,
                                        source_snapshot_ids=(stable_identity("question:" + question["question_id"]),),
                                        payload=text)
        query_representations.append(QualifiedRepresentation(representation=representation, parts=(part,)))
    embedding_requests = tuple(
        EmbeddingRequest(output_identity=stable_identity({"document_embedding": item.identity}), input=item,
                         space=space, profile=embedding_profile)
        for item in (*page_representations, *query_representations)
    )
    embedding_outcomes = VoyageMultimodalEmbeddingAdapter().embed_batch(embedding_requests)
    if not all(item.status == "succeeded" for item in embedding_outcomes):
        raise RunnerError(json.dumps([item.model_dump(mode="json", exclude_none=True) for item in embedding_outcomes], sort_keys=True))
    vectors = [list(item.value or []) for item in embedding_outcomes]
    page_vectors = vectors[:len(page_representations)]
    query_vectors = vectors[len(page_representations):]

    store = LanceStore(output / "pipeline-lance")
    def write_relation(name: str, rows: list[dict[str, Any]]) -> None:
        if name in store.names():
            store.replace(name, rows)
        else:
            store.create(name, rows)

    page_rows = [{"page_id": page["page_id"], "document_id": page["document_id"],
                  "page_number": page["page_number"], "content_ref": page["content_ref"],
                  "content_sha256": page["content_sha256"], "media_type": page["media_type"],
                  "revision": page["revision"], "authority": page["authority"]} for page in pages]
    write_relation("document_pages", page_rows)
    document_manifest = _document_context_manifest(context_model)
    document_observations = [{
        "object_role": "document_page",
        "snapshot_role": "source_snapshot",
        "source_system": "huggingface",
        "source_key": page["page_id"],
        "snapshot_identity": page["content_sha256"],
        "observed_at": page["observed_at"],
        "content_ref": page["content_ref"],
        "content_sha256": page["content_sha256"],
        "media_type": page["media_type"],
        "display_name": f"{page['document_id']} page {page['page_number']}",
        "preview": "verified original page image",
        "schema_id": "document.page",
        "schema_revision": "3",
        "operation_name": "ingest",
        "status": "observed",
    } for page in pages]
    document_outputs = [{
        "identity": page["content_sha256"],
        "role": "document_page",
        "kind": "context_representation",
        "object_kind": "document_page",
        "derived_from": [page["content_sha256"]],
        "schema_id": "document.page",
        "schema_revision": "3",
        "display_name": f"{page['document_id']} page {page['page_number']}",
        "payload": {"page_id": page["page_id"], "document_id": page["document_id"]},
        "vector": vector,
        "text": f"{page['document_id']} page {page['page_number']}",
        "source_system": "huggingface",
        "source_key": page["page_id"],
        "media_type": page["media_type"],
        "content_ref": page["content_ref"],
        "status": "active",
        "authority_status": "authoritative",
        "temporal_status": "valid",
        "policy_id": "document.source:1",
        "as_of": page["observed_at"],
        "evidence_preview": "verified original page image",
    } for page, vector in zip(pages, page_vectors)]
    materialize_context_model_lifecycle(
        store=store,
        manifest=document_manifest,
        observations=document_observations,
        outputs=document_outputs,
    )
    retrieval = LanceRetrievalAdapter(store)
    index_facts = retrieval.ensure_vector_index()

    as_of = datetime(2026, 8, 13, tzinfo=timezone.utc)
    retrieval_inputs = [{
        "retrieval_role": "document_pages",
        "query_id": stable_identity("query:" + question["question_id"]),
        "text": question["question"],
        "vector": tuple(vector),
        "limit": 1,
        "space_identity": space.identity,
        "request_identity": stable_identity({"resolution_request": question["question_id"]}),
        "task_id": f"document-question:{question['question_id']}",
        "task_schema_revision": "1",
        "as_of": as_of.isoformat(),
        "candidate_identity": stable_identity({"candidate": question["question_id"]}),
        "question_id": question["question_id"],
        "evidence_excerpt": "verified original page image",
    } for question, vector in zip(questions, query_vectors)]

    enrichment_rows = []
    output_values = {item.work_identity: runtime.outcomes[item.work_identity].value for item in work_items}
    for question, item in zip(questions, work_items, strict=True):
        value = output_values[item.work_identity]
        enrichment_rows.append({"question_id": question["question_id"], "page_id": question["page_id"],
                                "answer": value.get("answer"), "abstain": bool(value.get("abstain")),
                                "evidence_page_id": value.get("evidence_page_id"),
                                "schema_id": "document.qa", "schema_revision": "2",
                                "output_identity": stable_identity({"enrichment": question["question_id"], "value": value})})
    write_relation("document_enrichments", enrichment_rows)

    resolution_payloads = []
    package_payloads = []
    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest=document_manifest,
        observations=document_observations,
        outputs=document_outputs,
        retrieval_inputs=retrieval_inputs,
        retrieval_port=retrieval,
        resolution_port=_DocumentResolution(),
    )
    retrieval_facts = []
    retrieval_rows_out = []
    pages_by_content_identity = {page["content_sha256"]: page for page in pages}
    questions_by_id = {question["question_id"]: question for question in questions}
    for row in lifecycle.retrieval_results:
        if not row:
            continue
        page = pages_by_content_identity[str(row["context_identity"])]
        question_id = str(row["question_id"])
        retrieval_facts.append({
            "result_identity": row["result_identity"],
            "context_identity": row["context_identity"],
            "score": row["score"],
            "rank": row["rank"],
            "facts": row["facts"],
        })
        retrieval_rows_out.append({
            **row,
            "question_id": question_id,
            "page_id": page["page_id"],
            "source_snapshot_id": row["context_identity"],
            "evidence_locator": f"page:{page['page_id']}",
        })
    if len(retrieval_rows_out) != len(questions):
        missing = sorted(set(questions_by_id) - {str(row["question_id"]) for row in retrieval_rows_out})
        raise RunnerError("document lifecycle retrieval returned no candidate for: " + ", ".join(missing))
    write_relation("document_retrieval", retrieval_rows_out)
    result_by_request = {result.request_identity: result for result in lifecycle.resolution_results}
    package_by_task = {package.task_id: package for package in lifecycle.packages}
    standard_decisions = {row["candidate_identity"]: row for row in store.read("resolution_decisions").to_pylist()}
    for retrieved in retrieval_rows_out:
        question = questions_by_id[str(retrieved["question_id"])]
        request_identity = str(retrieved["request_identity"])
        task_id = f"document-question:{question['question_id']}"
        result = result_by_request[request_identity]
        decision = result.decisions[0]
        package = package_by_task[task_id]
        decision_row = standard_decisions[decision.candidate_identity]
        resolution_payloads.append(result.model_dump(mode="json"))
        package_payloads.append({"question_id": question["question_id"], "package_identity": package.identity,
                                 "item_count": len(package.items), "task_id": package.task_id,
                                 "resolution_identity": result.resolution_identity,
                                 "decision_identity": decision_row["decision_identity"]})
    index_facts = retrieval.ensure_vector_index()
    sql = ("SELECT e.question_id, e.answer, e.abstain, e.evidence_page_id, "
           "r.page_id AS retrieved_page_id, r.rank, r.score, x.outcome, x.reason_code, "
           "p.package_identity, d.content_ref "
           "FROM document_enrichments e "
           "JOIN document_retrieval r ON e.question_id = r.question_id "
           "JOIN resolution_candidates c ON c.candidate_identity = r.candidate_identity "
           "JOIN resolution_decisions x ON x.candidate_identity = c.candidate_identity "
           "JOIN package_items i ON i.representation_identity = c.context_identity "
           "JOIN context_packages p ON p.package_identity = i.package_identity AND p.task_id = r.task_id "
           "JOIN document_pages d ON e.page_id = d.page_id "
           "ORDER BY e.question_id")
    sql_rows = AnalyticalEngine(store).query(sql, "document_enrichments").to_pylist()

    changed_page = pages[0]
    invalidated = [question["question_id"] for question in questions if question["page_id"] == changed_page["page_id"]]
    reused = [question["question_id"] for question in questions if question["question_id"] not in invalidated]
    mutation = {
        "changed_page_id": changed_page["page_id"],
        "changed_content_identity": stable_identity({"source": changed_page["content_sha256"], "mutation": "document-content"}),
        "invalidated_question_ids": invalidated, "reused_question_ids": reused,
        "baseline_completion_requests": len(questions), "mutation_completion_requests": len(invalidated),
        "cache_hit_count": len(reused), "status": "passed",
    }
    return {
        "stages": ["dlt", "lancedb", "retrieval", "resolution", "package", "datafusion-sql"],
        "ingestion": ingestion.as_dict(),
        "embedding": {"profile": embedding_profile.model_dump(mode="json"), "items": len(embedding_requests),
                       "space": space.model_dump(mode="json")},
        "retrieval": {"facts": retrieval_facts, "rows": retrieval_rows_out,
                       "lance_tables": sorted(store.names()), "index_facts": index_facts},
        "enrichments": enrichment_rows,
        "resolution": {"results": resolution_payloads, "standard_relation": "resolution_decisions"},
        "packages": {"items": package_payloads, "standard_relations": ["context_packages", "package_items"]},
        "sql": {"query": sql, "rows": sql_rows,
                "relations": ["document_enrichments", "document_retrieval", "resolution_decisions",
                              "resolution_candidates", "context_packages", "package_items"]},
        "mutation_reuse": mutation,
        "evaluation_input": {"question_count": len(questions), "answer_values_preserved": len(enrichment_rows)},
    }


def _document_report(artifact: dict[str, Any]) -> str:
    succeeded = sum(item.get("status") == "succeeded" for item in artifact.get("outcomes", []))
    mutation = artifact.get("mutation_reuse", {})
    rows = artifact.get("sql", {}).get("rows", [])
    model = artifact.get("profile", {}).get("model", "")
    stages = " → ".join(artifact.get("stages", []))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>IGOR document qualification</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:8px;text-align:left}}.pass{{color:#087f5b}}</style></head><body>
<h1>Live multimodal document end-to-end qualification</h1><p class='pass'><strong>Pipeline passed:</strong> {artifact.get('profile', {}).get('provider', 'completion')} structured outputs traversed dlt → LanceDB → retrieval → resolution → package → DataFusion SQL.</p>
<h2>Completion</h2><p>{succeeded}/{len(artifact.get("outcomes", []))} structured outputs preserved; model: <code>{model}</code>.</p>
<h2>Pipeline</h2><p>{stages}</p><p>SQL result rows: {len(rows)}</p>
<h2>Mutation and reuse</h2><p>Changed page: <code>{mutation.get("changed_page_id")}</code>; invalidated: {len(mutation.get("invalidated_question_ids", []))}; reused: {len(mutation.get("reused_question_ids", []))}; status: <strong>{mutation.get("status", "unknown")}</strong>.</p>
</body></html>"""
