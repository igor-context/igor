"""Opt-in live provider qualification; writes only redacted metadata."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from igor_context_compiler import (CompletionRequest, DeepSeekCompletionAdapter, EmbeddingRequest,
                                   QualifiedRepresentation, VoyageMultimodalEmbeddingAdapter)
from igor_core import ContentPart, EmbeddingSpace, EnrichmentRecipe, ModelProfile, Representation, SchemaDescriptor, stable_identity


def main(output: str = "/tmp/live-provider-qualification.json") -> int:
    schema = SchemaDescriptor(schema_version="0.1", schema_id="qualification", revision="1", json_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "additionalProperties": False, "required": ["status"],
        "properties": {"status": {"type": "string", "enum": ["accepted"]}},
    })
    representation = Representation(ir_version="0.1", representation_type="text", schema_ref=schema,
                                    source_snapshot_ids=("sha256:" + "1" * 64,), payload="IGOR live provider qualification")
    qualified = QualifiedRepresentation(representation=representation, parts=(ContentPart(
        kind="text", media_type="text/plain", text="IGOR live provider qualification. Return status accepted.",
        content_sha256=stable_identity("IGOR live provider qualification. Return status accepted."),
    ),))
    recipe = EnrichmentRecipe(recipe_id="qualification", revision="1", accepted_representation_types=("text",),
        accepted_media_types=("text/plain",), output_schema_identity=schema.identity,
        prompt_version="enrichment-v1", taxonomy_version="context-v1", evidence_required=False)
    embedding_profile = ModelProfile(schema_version="0.1", profile_id="voyage-multimodal-3-v0", capability="embedding",
        provider="voyage", model="voyage-multimodal-3", revision="api-alias-selected-2026-08-13",
        parameters={"endpoint": "https://api.voyageai.com/v1/multimodalembeddings", "dimensions": 1024,
                    "request_chunk_size": 32, "request_pause_seconds": 0})
    completion_profile = ModelProfile(schema_version="0.1", profile_id="deepseek-v4-flash-v0", capability="completion",
        provider="deepseek", model="deepseek-v4-flash", revision="v4-flash-api-2026-04-24",
        parameters={"endpoint": "https://api.deepseek.com", "temperature": 0, "response_format": "json_object"})
    space = EmbeddingSpace(ir_version="0.1", provider="voyage", model="voyage-multimodal-3",
        model_revision=embedding_profile.revision, dimension=1024, dtype="float32", metric="cosine",
        normalized=False, input_schema_identity=schema.identity)
    results = {}
    try:
        results["embedding"] = VoyageMultimodalEmbeddingAdapter().embed(
            EmbeddingRequest(output_identity="sha256:" + "2" * 64, input=qualified, space=space, profile=embedding_profile)).model_dump(mode="json", exclude_none=True)
    except Exception as error:
        results["embedding"] = {"status": "configuration_error", "error": str(error)}
    try:
        results["completion"] = DeepSeekCompletionAdapter().enrich(
            CompletionRequest(output_identity="sha256:" + "3" * 64, inputs=(qualified,), output_schema=schema,
                             recipe=recipe, prompt_version="enrichment-v1", taxonomy_version="context-v1",
                             profile=completion_profile)).model_dump(mode="json", exclude_none=True)
    except Exception as error:
        results["completion"] = {"status": "configuration_error", "error": str(error)}
    payload = {"qualification": "live-context-providers-v0", "observed_at": datetime.now(timezone.utc).isoformat(),
               "profiles": {"embedding": embedding_profile.model_dump(mode="json"), "completion": completion_profile.model_dump(mode="json")},
               "results": results}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"qualification": payload["qualification"], "results": {
        key: {"status": value.get("status"), "attempts": value.get("attempts")} for key, value in results.items()}}))
    return 0 if all(value.get("status") == "succeeded" for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/live-provider-qualification.json"))
