from __future__ import annotations

import io
import json
from urllib.error import HTTPError

from igor_context_compiler import (
    CompletionRequest, DeepSeekCompletionAdapter, EmbeddingRequest, GeminiCompletionAdapter,
    MistralCompletionAdapter, QualifiedRepresentation, VoyageMultimodalEmbeddingAdapter,
)
from igor_core import ContentPart, EmbeddingSpace, EnrichmentRecipe, ModelProfile, Representation, SchemaDescriptor, stable_identity


class Response:
    def __init__(self, value):
        self.value = value

    def read(self):
        return io.BytesIO(json.dumps(self.value).encode()).read()


def transport(value):
    def send(request, timeout):
        assert request.headers["Content-type"] == "application/json"
        assert timeout > 0
        return Response(value)
    return send


def fixture():
    schema = SchemaDescriptor(schema_version="0.1", schema_id="provider-test", revision="1", json_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "additionalProperties": False, "required": ["status"],
        "properties": {"status": {"type": "string", "enum": ["accepted"]}},
    })
    representation = Representation(ir_version="0.1", representation_type="text", schema_ref=schema,
                                    source_snapshot_ids=("sha256:" + "1" * 64,), payload="hello")
    qualified = QualifiedRepresentation(representation=representation, parts=(
        ContentPart(kind="text", media_type="text/plain", text="hello", content_sha256=stable_identity("hello")),
    ))
    embedding_profile = ModelProfile(schema_version="0.1", profile_id="voyage-test", capability="embedding",
                                     provider="voyage", model="voyage-multimodal-3", revision="test-1",
                                     parameters={"endpoint": "https://api.voyageai.com/v1/multimodalembeddings", "dimensions": 3})
    completion_profile = ModelProfile(schema_version="0.1", profile_id="deepseek-test", capability="completion",
                                      provider="deepseek", model="deepseek-v4-flash", revision="test-1",
                                     parameters={"endpoint": "https://api.deepseek.com", "temperature": 0, "response_format": "json_object",
                                                 "supported_content_kinds": ["text", "image", "document"]})
    space = EmbeddingSpace(ir_version="0.1", provider="voyage", model="voyage-multimodal-3", model_revision="test-1",
                           dimension=3, dtype="float32", metric="cosine", normalized=False,
                           input_schema_identity=schema.identity)
    return schema, qualified, embedding_profile, completion_profile, space


def test_voyage_maps_documented_response_and_revision():
    schema, qualified, profile, _, space = fixture()
    adapter = VoyageMultimodalEmbeddingAdapter(api_key="canary", transport=transport({
        "data": [{"embedding": [0.1, 0.2, 0.3]}], "model": "voyage-multimodal-3", "id": "response-1"}))
    outcome = adapter.embed(EmbeddingRequest(output_identity="sha256:" + "2" * 64, input=qualified, space=space, profile=profile))
    assert outcome.status == "succeeded"
    assert outcome.value == [0.1, 0.2, 0.3]
    assert outcome.metadata.provider_revision == "test-1"
    assert outcome.metadata.response_id == "response-1"


def test_deepseek_maps_json_object_and_revision():
    schema, qualified, _, profile, _ = fixture()
    adapter = DeepSeekCompletionAdapter(api_key="canary", transport=transport({
        "id": "response-2", "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": '{"status":"accepted"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
    recipe = EnrichmentRecipe(recipe_id="test", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain",), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    outcome = adapter.enrich(CompletionRequest(output_identity="sha256:" + "3" * 64, inputs=(qualified,),
                                                output_schema=schema, recipe=recipe, prompt_version="p1", taxonomy_version="t1",
                                                profile=profile))
    assert outcome.status == "succeeded"
    assert outcome.value == {"status": "accepted"}
    assert outcome.metadata.usage == {"prompt_tokens": 3, "completion_tokens": 2}


def test_deepseek_sends_original_image_as_multimodal_content(tmp_path):
    schema, qualified, _, profile, _ = fixture()
    image_path = tmp_path / "page.jpg"
    image_bytes = b"original document page"
    image_path.write_bytes(image_bytes)
    image = ContentPart(kind="image", media_type="image/jpeg", content_ref=image_path.as_uri(),
                        content_sha256="sha256:" + __import__("hashlib").sha256(image_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (image, qualified.parts[0])})
    captured = {}

    def send(request, timeout):
        captured.update(json.loads(request.data))
        return Response({"id": "response-image", "model": "deepseek-v4-flash",
                         "choices": [{"message": {"content": '{"status":"accepted"}'}}]})

    recipe = EnrichmentRecipe(recipe_id="test-image", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain", "image/jpeg"), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    outcome = DeepSeekCompletionAdapter(api_key="canary", transport=send).enrich(
        CompletionRequest(output_identity="sha256:" + "6" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))
    assert outcome.status == "succeeded"
    parts = captured["messages"][1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1] == {"type": "text", "text": "hello"}


def test_gemini_sends_original_image_and_json_schema(tmp_path):
    schema, qualified, _, _, _ = fixture()
    image_path = tmp_path / "page.jpg"
    image_bytes = b"original document page"
    image_path.write_bytes(image_bytes)
    image = ContentPart(kind="image", media_type="image/jpeg", content_ref=image_path.as_uri(),
                        content_sha256="sha256:" + __import__("hashlib").sha256(image_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (image, qualified.parts[0])})
    profile = ModelProfile(schema_version="0.1", profile_id="gemini-test", capability="completion",
                           provider="gemini", model="gemini-3.1-flash-lite", revision="test-1",
                           parameters={"endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"})
    captured = {}
    def send(request, timeout):
        captured.update(json.loads(request.data))
        return Response({"candidates": [{"content": {"parts": [{"text": '{"status":"accepted"}'}]}}],
                         "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}})
    recipe = EnrichmentRecipe(recipe_id="test-image", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain", "image/jpeg"), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    outcome = GeminiCompletionAdapter(api_key="canary", transport=send).enrich(
        CompletionRequest(output_identity="sha256:" + "7" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))
    assert outcome.status == "succeeded"
    parts = captured["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[1] == {"text": "hello"}


def test_mistral_sends_original_image_and_json_mode(tmp_path):
    schema, qualified, _, _, _ = fixture()
    image_bytes = b"verified image bytes"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(image_bytes)
    image = ContentPart(kind="image", media_type="image/png", content_ref=image_path.as_uri(),
                        content_sha256="sha256:" + __import__("hashlib").sha256(image_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (image, qualified.parts[0])})
    profile = ModelProfile(
        schema_version="0.1", profile_id="mistral-test", capability="completion",
        provider="mistral", model="mistral-small-2603", revision="test-1",
        parameters={"endpoint": "https://api.mistral.ai/v1/chat/completions", "temperature": 0,
                    "response_format": "json_object", "supported_content_kinds": ["text", "image"]},
    )
    captured = {}

    def send(request, timeout):
        captured.update(json.loads(request.data))
        return Response({"id": "mistral-response", "model": "mistral-small-2603",
                         "choices": [{"message": {"content": '{"status":"accepted"}'}}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    recipe = EnrichmentRecipe(recipe_id="test-image", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain", "image/png"), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    outcome = MistralCompletionAdapter(api_key="canary", transport=send).enrich(
        CompletionRequest(output_identity="sha256:" + "8" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))

    assert outcome.status == "succeeded"
    assert outcome.value == {"status": "accepted"}
    assert captured["response_format"] == {"type": "json_object"}
    parts = captured["messages"][1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"].startswith("data:image/png;base64,")
    assert parts[1] == {"type": "text", "text": "hello"}


def test_mistral_sends_verified_pdf_as_document_url(tmp_path):
    schema, qualified, _, _, _ = fixture()
    document_bytes = b"%PDF-1.4\nverified fixture"
    document_path = tmp_path / "test.pdf"
    document_path.write_bytes(document_bytes)
    document = ContentPart(kind="document", media_type="application/pdf", content_ref=document_path.as_uri(),
                           content_sha256="sha256:" + __import__("hashlib").sha256(document_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (document, qualified.parts[0])})
    profile = ModelProfile(
        schema_version="0.1", profile_id="mistral-test", capability="completion",
        provider="mistral", model="ministral-8b-2512", revision="test-1",
        parameters={"endpoint": "https://api.mistral.ai/v1/chat/completions", "temperature": 0,
                    "response_format": "json_object", "supported_content_kinds": ["text", "image", "document"]},
    )
    captured = {}

    def send(request, timeout):
        captured.update(json.loads(request.data))
        return Response({"id": "mistral-pdf", "model": "ministral-8b-2512",
                         "choices": [{"message": {"content": '{"status":"accepted"}'}}]})

    recipe = EnrichmentRecipe(recipe_id="test-document", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("application/pdf",), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    outcome = MistralCompletionAdapter(api_key="canary", transport=send).enrich(
        CompletionRequest(output_identity="sha256:" + "a" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))
    assert outcome.status == "succeeded"
    parts = captured["messages"][1]["content"]
    assert parts[0]["type"] == "document_url"
    assert parts[0]["document_url"].startswith("data:application/pdf;base64,")
    assert parts[1] == {"type": "text", "text": "hello"}


def test_mistral_profile_rejects_content_outside_its_workload(tmp_path):
    schema, qualified, _, _, _ = fixture()
    image_bytes = b"verified image bytes"
    image_path = tmp_path / "page.png"
    image_path.write_bytes(image_bytes)
    image = ContentPart(kind="image", media_type="image/png", content_ref=image_path.as_uri(),
                        content_sha256="sha256:" + __import__("hashlib").sha256(image_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (image, qualified.parts[0])})
    profile = ModelProfile(
        schema_version="0.1", profile_id="mistral-text-test", capability="completion",
        provider="mistral", model="mistral-small-2603", revision="test-1",
        parameters={"endpoint": "https://api.mistral.ai/v1/chat/completions", "temperature": 0,
                    "response_format": "json_object", "workload": "structured_text",
                    "supported_content_kinds": ["text"]},
    )
    recipe = EnrichmentRecipe(recipe_id="test-image", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain", "image/png"), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)

    def must_not_send(request, timeout):
        raise AssertionError("unsupported modality must fail before provider execution")

    outcome = MistralCompletionAdapter(api_key="canary", transport=must_not_send).enrich(
        CompletionRequest(output_identity="sha256:" + "b" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))
    assert outcome.status == "permanent_rejection"
    assert outcome.error == "Mistral profile does not support image content"


def test_provider_retryable_http_status_is_typed():
    def send(request, timeout):
        raise HTTPError(request.full_url, 429, "throttle", {}, None)

    _, qualified, profile, _, space = fixture()
    outcome = VoyageMultimodalEmbeddingAdapter(transport=send).embed(
        EmbeddingRequest(output_identity="sha256:" + "4" * 64, input=qualified, space=space, profile=profile))
    assert outcome.status == "transient_failure"
    assert outcome.error == "provider returned HTTP 429"
    assert outcome.attempts == 3


def test_voyage_batches_and_retries_transient_failures():
    schema, qualified, profile, _, space = fixture()
    profile = profile.model_copy(update={"parameters": {
        **profile.parameters, "max_batch_size": 2, "max_attempts": 2,
    }})
    requests = tuple(
        EmbeddingRequest(output_identity="sha256:" + str(index) * 64, input=qualified, space=space, profile=profile)
        for index in range(1, 6)
    )
    calls = []

    def send(request, timeout):
        body = json.loads(request.data)
        calls.append(len(body["inputs"]))
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "throttle", {}, None)
        return Response({"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in body["inputs"]]})

    outcomes = VoyageMultimodalEmbeddingAdapter(api_key="canary", transport=send).embed_batch(requests)

    assert [item.status for item in outcomes] == ["succeeded"] * 5
    assert [item.attempts for item in outcomes] == [2, 2, 1, 1, 1]
    assert calls == [2, 2, 2, 1]


def test_mistral_retries_transient_completion_failure():
    schema, qualified, _, _, _ = fixture()
    profile = ModelProfile(
        schema_version="0.1", profile_id="mistral-test", capability="completion",
        provider="mistral", model="mistral-small-2603", revision="test-1",
        parameters={"endpoint": "https://api.mistral.ai/v1/chat/completions", "temperature": 0,
                    "response_format": "json_object", "supported_content_kinds": ["text"], "max_attempts": 2},
    )
    recipe = EnrichmentRecipe(recipe_id="test", revision="1", accepted_representation_types=("text",),
                              accepted_media_types=("text/plain",), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    calls = 0

    def send(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)
        return Response({"id": "mistral-response", "model": "mistral-small-2603",
                         "choices": [{"message": {"content": '{"status":"accepted"}'}}]})

    outcome = MistralCompletionAdapter(api_key="canary", transport=send).enrich(
        CompletionRequest(output_identity="sha256:" + "c" * 64, inputs=(qualified,), output_schema=schema,
                          recipe=recipe, prompt_version="p1", taxonomy_version="t1", profile=profile))

    assert outcome.status == "succeeded"
    assert outcome.attempts == 2
    assert calls == 2


def test_voyage_sends_real_image_bytes_as_documented_base64():
    schema, qualified, profile, _, space = fixture()
    image_bytes = b"\x89PNG\r\n\x1a\nfixture"
    image = ContentPart(kind="image", media_type="image/png", content_ref="memory://image",
                        content_sha256="sha256:" + __import__("hashlib").sha256(image_bytes).hexdigest())
    qualified = qualified.model_copy(update={"parts": (image,)})
    captured = {}

    def send(request, timeout):
        captured.update(json.loads(request.data))
        return Response({"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    outcome = VoyageMultimodalEmbeddingAdapter(
        api_key="canary", transport=send, content_resolver=lambda part: image_bytes,
    ).embed(EmbeddingRequest(output_identity="sha256:" + "5" * 64, input=qualified, space=space, profile=profile))
    assert outcome.status == "succeeded"
    part = captured["inputs"][0]["content"][0]
    assert part["type"] == "image_base64"
    assert part["image_base64"].startswith("data:image/png;base64,")
