import json
import hashlib
from pathlib import Path
from igor_context_compiler import ProviderOutcome, RuntimeResult
from igor_core import ModelProfile
from igor_lancedb import STANDARD_CONTEXT_RELATIONS, LanceStore
from igor_runner import document_qualification
from igor_runner.document_qualification import qualify_document
from igor_runner.document_runtime_helpers import load_document_context_model, make_document_work_item

def test_document_qualification_is_bounded_and_artifact_complete(tmp_path):
    scenario = Path("/opt/scenarios/document-huggingface/v0.1")
    result = qualify_document(scenario, tmp_path)
    assert result["pages"] == 4 and result["questions"] == 8 and result["accuracy"] == 1.0
    artifact = json.loads((tmp_path / "qualification.json").read_text())
    assert artifact["ingestion"]["load_metadata"]
    assert len(artifact["sql"]["rows"]) == 4
    assert artifact["mutations"]["invalidated"] == ["page-form-02"]
    assert len(artifact["mutations"]["reused"]) == 3
    assert "DataFusion SQL" in (tmp_path / "report.html").read_text()
    store = LanceStore(tmp_path / "relations")
    assert set(("context_models", "context_objects", "context_snapshots", "context_catalog", "lineage_nodes")).issubset(store.names())
    assert artifact["catalog_relations"] == list(STANDARD_CONTEXT_RELATIONS)
    assert store.read("context_objects").num_rows == 4


def test_document_qualification_consumes_verified_acquisition_without_judgments(tmp_path):
    image = tmp_path / "images" / "page.jpg"
    image.parent.mkdir()
    image.write_bytes(b"original page bytes")
    image_hash = "sha256:" + hashlib.sha256(image.read_bytes()).hexdigest()
    (tmp_path / "acquisition.json").write_text(json.dumps({
        "repository": "lance-format/docvqa-lance",
        "revision": "43245f42abf5a2e7acd68617061497f3bb67048c",
        "records": [{
            "page_id": "doc:d1:page:1", "document_id": "d1", "page_number": "1",
            "question_id": "q1", "question": "What is shown?", "question_types": ["figure"],
            "image_ref": image.as_uri(), "image_sha256": image_hash,
        }],
    }))
    result = qualify_document(Path("/opt/scenarios/document-huggingface/v0.1"), tmp_path / "run", tmp_path)
    artifact = json.loads((tmp_path / "run" / "qualification.json").read_text())
    assert result["questions"] == 1
    assert artifact["acquisition_mode"] == "pinned-hugging-face-acquisition"
    assert artifact["predictions"][0]["abstain"] is True


def test_live_document_pipeline_uses_standard_resolution_and_package_relations(tmp_path, monkeypatch):
    scenario = Path("/opt/scenarios/document-huggingface/v0.1")
    context_model = load_document_context_model(scenario)
    page_content = tmp_path / "page.jpg"
    page_content.write_bytes(b"verified document page")
    page_hash = "sha256:" + hashlib.sha256(page_content.read_bytes()).hexdigest()
    pages = [{
        "page_id": "doc:test:page:1",
        "document_id": "doc-test",
        "page_number": 1,
        "revision": "pinned",
        "observed_at": "2026-01-01T00:00:00Z",
        "effective_from": "2026-01-01T00:00:00Z",
        "authority": "source",
        "content_ref": page_content.as_uri(),
        "image_ref": page_content.as_uri(),
        "content_sha256": page_hash,
        "media_type": "image/jpeg",
    }]
    questions = [{
        "question_id": "q1",
        "page_id": pages[0]["page_id"],
        "question": "What is on the page?",
        "question_types": ["figure"],
        "answerability": "answerable",
    }]
    completion_profile = ModelProfile(
        schema_version="0.1",
        profile_id="test-completion",
        capability="completion",
        provider="deterministic",
        model="deterministic",
        revision="test",
        parameters={"supported_content_kinds": ["image", "text"]},
    )
    embedding_profile = ModelProfile(
        schema_version="0.1",
        profile_id="test-embedding",
        capability="embedding",
        provider="voyage",
        model="deterministic-voyage",
        revision="test",
        parameters={"dimensions": 3},
    )
    work_items = tuple(make_document_work_item(
        {**questions[0], **pages[0]},
        completion_profile,
        context_model,
    ) for _ in questions)
    runtime = RuntimeResult(
        outcomes={
            work_items[0].work_identity: ProviderOutcome(
                status="succeeded",
                attempts=1,
                value={"answer": "verified document page", "abstain": False, "evidence_page_id": pages[0]["page_id"]},
            ),
        },
        batches=(),
        attempts=(),
        artifact={"mode": "test"},
    )

    class FakeVoyage:
        def embed_batch(self, requests):
            return tuple(
                ProviderOutcome(status="succeeded", attempts=1, value=[1.0, 0.0, 0.0])
                for _ in requests
            )

    monkeypatch.setattr(document_qualification, "VoyageMultimodalEmbeddingAdapter", FakeVoyage)
    result = document_qualification._run_live_document_pipeline(
        pages, questions, runtime, embedding_profile, completion_profile, context_model, tmp_path / "run", work_items,
    )
    store = LanceStore(tmp_path / "run" / "pipeline-lance")
    names = set(store.names())
    assert {"resolution_decisions", "resolution_candidates", "context_packages", "package_items"}.issubset(names)
    assert "document_resolutions" not in names
    assert "document_packages" not in names
    assert result["sql"]["rows"][0]["package_identity"] == store.read("context_packages").to_pylist()[0]["package_identity"]
    assert result["sql"]["relations"] == [
        "document_enrichments",
        "document_retrieval",
        "resolution_decisions",
        "resolution_candidates",
        "context_packages",
        "package_items",
    ]
