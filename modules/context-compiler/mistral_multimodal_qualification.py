"""Bounded live Mistral image and PDF qualification with redacted artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from igor_context_compiler import CompletionRequest, MistralCompletionAdapter, QualifiedRepresentation
from igor_core import (
    ContentPart, EnrichmentRecipe, Representation, SchemaDescriptor, load_model_profile,
    stable_identity, validate_schema_payload,
)


def _request(path: Path, *, kind: str, media_type: str, prompt: str,
             schema: SchemaDescriptor, profile) -> CompletionRequest:
    content = path.read_bytes()
    content_sha256 = "sha256:" + hashlib.sha256(content).hexdigest()
    source_identity = stable_identity({"path": path.name, "content_sha256": content_sha256})
    representation_type = "custom" if kind == "document" else kind
    representation = Representation(
        ir_version="0.1", representation_type=representation_type, schema_ref=schema,
        source_snapshot_ids=(source_identity,), payload=None,
    )
    parts = (
        ContentPart(kind=kind, media_type=media_type, content_ref=path.as_uri(),
                    content_sha256=content_sha256),
        ContentPart(kind="text", media_type="text/plain", text=prompt,
                    content_sha256=stable_identity(prompt)),
    )
    qualified = QualifiedRepresentation(representation=representation, parts=parts)
    recipe = EnrichmentRecipe(
        recipe_id=f"mistral-smoke.{kind}", revision="1",
        accepted_representation_types=(representation_type,), accepted_media_types=(media_type, "text/plain"),
        output_schema_identity=schema.identity, prompt_version="mistral-smoke-v1",
        taxonomy_version="none", evidence_required=False,
    )
    return CompletionRequest(
        output_identity=stable_identity({"input": content_sha256, "schema": schema.identity}),
        inputs=(qualified,), output_schema=schema, recipe=recipe,
        prompt_version=recipe.prompt_version, taxonomy_version=recipe.taxonomy_version,
        profile=profile,
    )


def main(image_path: str, pdf_path: str, image_profile_path: str,
         document_profile_path: str, output_path: str) -> int:
    image_schema = SchemaDescriptor(
        schema_version="0.1", schema_id="mistral-smoke.image", revision="1",
        json_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": ["title", "lifecycle_stages"],
            "properties": {
                "title": {"type": "string"},
                "lifecycle_stages": {"type": "array", "items": {"type": "string"}},
            },
        },
    )
    pdf_schema = SchemaDescriptor(
        schema_version="0.1", schema_id="mistral-smoke.pdf", revision="1",
        json_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": ["invoice_id", "total_eur", "approved_by"],
            "properties": {
                "invoice_id": {"type": "string"},
                "total_eur": {"type": "integer"},
                "approved_by": {"type": "string"},
            },
        },
    )
    image_profile = load_model_profile(image_profile_path, expected_capability="completion")
    document_profile = load_model_profile(document_profile_path, expected_capability="completion")
    requests = {
        "image": _request(
            Path(image_path), kind="image", media_type="image/png",
            prompt="Read the diagram. Return its exact title and the five numbered Context Compiler lifecycle stages.",
            schema=image_schema, profile=image_profile,
        ),
        "pdf": _request(
            Path(pdf_path), kind="document", media_type="application/pdf",
            prompt="Read this PDF invoice. Return the invoice ID, integer total in EUR, and approver name.",
            schema=pdf_schema, profile=document_profile,
        ),
    }
    schemas = {"image": image_schema, "pdf": pdf_schema}
    adapter = MistralCompletionAdapter()
    results = {}
    all_valid = True
    for name, request in requests.items():
        outcome = adapter.enrich(request)
        valid = False
        validation_error = None
        if outcome.status == "succeeded":
            try:
                validate_schema_payload(schemas[name], outcome.value)
                valid = True
            except Exception as error:  # artifact records a bounded validation message
                validation_error = str(error)[:500]
        all_valid = all_valid and valid
        results[name] = {
            "status": outcome.status,
            "schema_valid": valid,
            "value": outcome.value,
            "metadata": outcome.metadata.model_dump(mode="json", exclude_none=True) if outcome.metadata else None,
            "error": outcome.error or validation_error,
        }
    payload = {
        "qualification": "mistral-image-pdf-smoke-v0",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "request_count": 2,
        "profiles": {
            "image": image_profile.model_dump(mode="json"),
            "document": document_profile.model_dump(mode="json"),
        },
        "pdf_path_note": "Mistral Document QnA internally combines Document AI/OCR with model understanding.",
        "results": results,
        "valid": all_valid,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"qualification": payload["qualification"], "valid": all_valid,
                      "results": {key: {"status": value["status"], "schema_valid": value["schema_valid"]}
                                  for key, value in results.items()}}, sort_keys=True))
    return 0 if all_valid else 1


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: qualification IMAGE PDF IMAGE_PROFILE DOCUMENT_PROFILE OUTPUT")
    raise SystemExit(main(*sys.argv[1:]))
