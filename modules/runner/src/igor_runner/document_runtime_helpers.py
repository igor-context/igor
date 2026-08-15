"""Document context-model loading and direct multimodal work-item construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from igor_context_compiler import CompletionRequest, EnrichmentWorkItem, QualifiedRepresentation
from igor_core import ContentPart, EnrichmentRecipe, ModelProfile, Representation, SchemaDescriptor, stable_identity


_EXPECTED_SOURCE_FIELDS = (
    "id", "image", "image_id", "question_id", "question", "answers", "answer",
    "doc_id", "ucsf_document_id", "ucsf_document_page_no", "data_split",
    "question_types", "image_emb", "question_emb",
)
_FORBIDDEN_PROMPT_FIELDS = {"answers", "answer", "image_emb", "question_emb"}


@dataclass(frozen=True)
class DocumentContextModel:
    """Approved, identity-bearing domain meaning for the pinned document workload."""

    domain_schema: dict[str, Any]
    semantic_definition: dict[str, Any]

    @property
    def identity(self) -> str:
        return stable_identity({
            "domain_schema": self.domain_schema,
            "semantic_definition": self.semantic_definition,
        })

    @property
    def context_model_id(self) -> str:
        return str(self.domain_schema["context_model_id"])

    @property
    def taxonomy_version(self) -> str:
        return f"{self.semantic_definition['semantic_definition_id']}@{self.semantic_definition['revision']}"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"document context-model file must contain a YAML mapping: {path}")
    return value


def load_document_context_model(scenario: str | Path) -> DocumentContextModel:
    """Load and validate the human-approved schema and semantic taxonomy."""
    root = Path(scenario)
    domain_schema = _load_yaml_mapping(root / "domain-schema.yaml")
    semantic_definition = _load_yaml_mapping(root / "semantic-definition.yaml")
    source_fields = domain_schema.get("source_dataset", {}).get("fields", [])
    names = tuple(field.get("name") for field in source_fields if isinstance(field, dict))
    if names != _EXPECTED_SOURCE_FIELDS:
        raise ValueError("document domain schema does not mirror the pinned Hugging Face source fields")
    by_name = {field["name"]: field for field in source_fields}
    unsafe = sorted(name for name in _FORBIDDEN_PROMPT_FIELDS
                    if by_name[name].get("prompt_allowed") is not False)
    if unsafe:
        raise ValueError(f"evaluation or supplied-derivation fields must forbid prompt use: {unsafe}")
    guidance = semantic_definition.get("question_type_guidance")
    if not isinstance(guidance, dict) or not guidance:
        raise ValueError("document semantic definition requires question_type_guidance")
    answerability = semantic_definition.get("answerability", {})
    if not answerability.get("answer_when") or not answerability.get("abstain_when"):
        raise ValueError("document semantic definition requires answerability and abstention rules")
    return DocumentContextModel(domain_schema=domain_schema, semantic_definition=semantic_definition)


def _question_types(question: dict[str, Any]) -> tuple[str, ...]:
    values = question.get("question_types")
    if isinstance(values, list):
        return tuple(str(value) for value in values)
    # Deterministic fixtures predate the source taxonomy and use semantic families.
    family = str(question.get("family", "")).strip()
    return tuple(value.strip() for value in family.split(",") if value.strip())


def document_task_prompt(question: dict[str, Any], context_model: DocumentContextModel) -> str:
    """Project only approved, relevant semantic guidance—not evaluator judgments."""
    definition = context_model.semantic_definition
    guidance = definition["question_type_guidance"]
    guidance_by_casefold = {str(key).casefold(): str(value) for key, value in guidance.items()}
    tags = _question_types(question)
    selected = [
        f"- {tag}: {guidance_by_casefold[tag.casefold()]}"
        for tag in tags if tag.casefold() in guidance_by_casefold
    ]
    if not selected and "others" in guidance_by_casefold:
        selected = [f"- others: {guidance_by_casefold['others']}"]
    answer_when = "\n".join(f"- {rule}" for rule in definition["answerability"]["answer_when"])
    abstain_when = "\n".join(f"- {rule}" for rule in definition["answerability"]["abstain_when"])
    selected_text = "\n".join(selected)
    tag_text = ", ".join(tags) if tags else "others"
    return (
        f"Question: {question['question']}\n"
        f"Source question types: {tag_text}\n"
        "Approved question-type guidance:\n"
        f"{selected_text}\n"
        "Answer when:\n"
        f"{answer_when}\n"
        "Abstain only when:\n"
        f"{abstain_when}\n"
        "Use only the associated page image as answer evidence. "
        "Never use reference answers or supplied embeddings."
    )


def make_work_item(question):
    schema = SchemaDescriptor(schema_version="0.1", schema_id="document.qa", revision="1", json_schema={"type":"object"})
    text = question["question"]
    rep = Representation(ir_version="0.1", representation_type="text", schema_ref=schema, source_snapshot_ids=(stable_identity(question["question_id"]),), payload=text)
    qualified = QualifiedRepresentation(representation=rep, parts=(ContentPart(kind="text", media_type="text/plain", text=text, content_sha256=stable_identity(text)),))
    recipe = EnrichmentRecipe(recipe_id="document.qa", revision="1", accepted_representation_types=("text",), accepted_media_types=("text/plain",), output_schema_identity=schema.identity, prompt_version="document-v1", taxonomy_version="semantic-definition-v1", evidence_required=True)
    request = CompletionRequest(output_identity=stable_identity("output:" + question["question_id"]), inputs=(qualified,), output_schema=schema, recipe=recipe, prompt_version="document-v1", taxonomy_version="semantic-definition-v1", profile=ModelProfile(schema_version="0.1", profile_id="deterministic-document", capability="completion", provider="deterministic", model="qualification", revision="1"))
    return EnrichmentWorkItem(work_identity=stable_identity("work:" + question["question_id"]), request=request, capability_profile_identity="deterministic-document", modality="text", recipe_identity=recipe.identity, schema_identity=schema.identity)


def require_supported_parts(parts, supported: set[str]) -> None:
    unsupported = sorted(set(parts) - supported)
    if unsupported:
        raise ValueError(f"completion profile does not support {', '.join(unsupported)} content")


def make_document_work_item(
    question: dict[str, Any],
    profile: ModelProfile | None = None,
    context_model: DocumentContextModel | None = None,
):
    """Build identity-bearing direct multimodal input from source bytes and approved meaning."""
    if context_model is None:
        raise ValueError("document work item requires an approved context model")
    schema = SchemaDescriptor(
        schema_version="0.1", schema_id="document.qa", revision="2",
        semantic_annotations={
            "context_model_id": context_model.context_model_id,
            "context_model_identity": context_model.identity,
        },
        json_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": ["string", "null"],
                    "description": "Exact concise value visible on the associated page, or null only when abstaining.",
                },
                "abstain": {
                    "type": "boolean",
                    "description": "False when the requested page value is legible and unambiguous; true otherwise.",
                },
                "evidence_page_id": {
                    "type": "string",
                    "description": "The supplied page identifier used as evidence, not an answer quotation.",
                },
            },
            "required": ["answer", "abstain", "evidence_page_id"],
            "additionalProperties": False,
        },
    )
    image = ContentPart(kind="image", media_type=question["media_type"],
                        content_ref=question["content_ref"], content_sha256=question["content_sha256"],
                        locator={"page_id": question["page_id"]})
    prompt = document_task_prompt(question, context_model)
    text = ContentPart(kind="text", media_type="text/plain", text=prompt,
                       content_sha256=stable_identity(prompt))
    representation = Representation(ir_version="0.1", representation_type="image", schema_ref=schema,
                                    source_snapshot_ids=(question["content_sha256"],), payload=None)
    qualified = QualifiedRepresentation(representation=representation, parts=(image, text))
    recipe = EnrichmentRecipe(
        recipe_id="document.qa", revision="3", accepted_representation_types=("image",),
        accepted_media_types=(question["media_type"], "text/plain"),
        output_schema_identity=schema.identity, prompt_version="document-context-v2",
        taxonomy_version=context_model.taxonomy_version, evidence_required=True,
    )
    profile = profile or ModelProfile(
        schema_version="0.1", profile_id="deterministic-document", capability="completion",
        provider="deterministic", model="multimodal-qualification", revision="2",
        parameters={"supported_content_kinds": ["image", "text"]},
    )
    require_supported_parts(tuple(part.kind for part in qualified.parts),
                            set(profile.parameters["supported_content_kinds"]))
    output_identity = stable_identity({
        "question_id": question["question_id"],
        "qualified_input": qualified.model_dump(mode="json"),
        "schema_identity": schema.identity,
        "recipe_identity": recipe.identity,
        "context_model_identity": context_model.identity,
        "profile_identity": profile.identity,
    })
    request = CompletionRequest(
        output_identity=output_identity, inputs=(qualified,), output_schema=schema, recipe=recipe,
        prompt_version=recipe.prompt_version, taxonomy_version=recipe.taxonomy_version, profile=profile,
    )
    return EnrichmentWorkItem(
        work_identity=stable_identity({"output_identity": output_identity}), request=request,
        capability_profile_identity=profile.identity, modality="image+text",
        recipe_identity=recipe.identity, schema_identity=schema.identity,
    )
