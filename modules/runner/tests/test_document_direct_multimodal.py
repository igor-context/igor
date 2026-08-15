from pathlib import Path
from copy import deepcopy
import hashlib
import pytest

from igor_runner.document_runtime_helpers import (
    DocumentContextModel,
    load_document_context_model,
    make_document_work_item,
    require_supported_parts,
)


SCENARIO = Path("/opt/scenarios/document-huggingface/v0.1")


def _question(content: Path) -> dict:
    return {
        "question_id": "q1", "question": "To whom is the document sent?", "page_id": "page-1",
        "question_types": ["handwritten", "form"],
        "content_ref": content.as_uri(),
        "content_sha256": "sha256:" + hashlib.sha256(content.read_bytes()).hexdigest(),
        "media_type": "image/png",
    }


def test_document_work_item_contains_original_verified_content_and_question(tmp_path: Path) -> None:
    content = tmp_path / "page.png"
    content.write_bytes(b"verified page bytes")
    item = make_document_work_item(_question(content), context_model=load_document_context_model(SCENARIO))
    assert any(part.kind == "image" and part.content_ref == content.as_uri() for part in item.request.inputs[0].parts)
    prompt = next(part.text for part in item.request.inputs[0].parts if part.kind == "text")
    assert "To whom is the document sent?" in prompt
    assert "handwritten text as valid page evidence" in prompt
    assert "form structure" in prompt
    assert "Paul" not in prompt
    assert item.request.output_schema.json_schema["properties"]["answer"]["description"]


def test_context_model_mirrors_source_schema_and_forbids_reference_prompt_fields() -> None:
    model = load_document_context_model(SCENARIO)
    fields = model.domain_schema["source_dataset"]["fields"]
    assert [field["name"] for field in fields] == [
        "id", "image", "image_id", "question_id", "question", "answers", "answer", "doc_id",
        "ucsf_document_id", "ucsf_document_page_no", "data_split", "question_types",
        "image_emb", "question_emb",
    ]
    by_name = {field["name"]: field for field in fields}
    assert all(by_name[name]["prompt_allowed"] is False
               for name in ("answers", "answer", "image_emb", "question_emb"))


def test_context_model_change_invalidates_work_identity(tmp_path: Path) -> None:
    content = tmp_path / "page.png"
    content.write_bytes(b"verified page bytes")
    original = load_document_context_model(SCENARIO)
    changed_definition = deepcopy(original.semantic_definition)
    changed_definition["revision"] = "99"
    changed = DocumentContextModel(original.domain_schema, changed_definition)
    original_item = make_document_work_item(_question(content), context_model=original)
    changed_item = make_document_work_item(_question(content), context_model=changed)
    assert original_item.work_identity != changed_item.work_identity
    assert original_item.request.output_identity != changed_item.request.output_identity


def test_unsupported_document_modality_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="does not support"):
        require_supported_parts(("image",), {"text"})
