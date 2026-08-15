"""Provider adapters owned by the Context Compiler.

The public compiler ports stay provider-neutral.  These adapters own HTTP,
credential discovery, response classification, and profile validation.
"""

from __future__ import annotations

import json
import os
import socket
import base64
import hashlib
import time
from collections.abc import Callable
from http.client import HTTPResponse
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from pathlib import Path

from igor_core import ContentPart, ModelProfile

from .compiler import CompletionRequest, EmbeddingRequest, ProviderMetadata, ProviderOutcome


Transport = Callable[[Request, float], HTTPResponse]
ContentResolver = Callable[[ContentPart], bytes]


def _open(request: Request, timeout: float) -> HTTPResponse:
    return urlopen(request, timeout=timeout)  # noqa: S310 - endpoint is profile-controlled and validated


def _resolve_local(part: ContentPart) -> bytes:
    if part.content_ref is None:
        raise ValueError("content part has no reference")
    parsed = urlparse(part.content_ref)
    if parsed.scheme != "file":
        raise ValueError("default content resolver supports file references only")
    content = Path(parsed.path).read_bytes()
    if "sha256:" + hashlib.sha256(content).hexdigest() != part.content_sha256:
        raise ValueError("resolved content hash mismatch")
    return content


def _failure(error: Exception, *, metadata: ProviderMetadata) -> ProviderOutcome:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return ProviderOutcome(status="timeout", attempts=1, error="provider request timed out", metadata=metadata)
    if isinstance(error, HTTPError):
        if error.code == 408 or error.code == 429 or error.code >= 500:
            return ProviderOutcome(status="transient_failure", attempts=1,
                                   error=f"provider returned HTTP {error.code}", metadata=metadata)
        if error.code == 400:
            try:
                detail = error.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                detail = "unavailable"
            return ProviderOutcome(status="permanent_rejection", attempts=1,
                                   error=f"provider returned HTTP 400: {detail}", metadata=metadata)
        return ProviderOutcome(status="permanent_rejection", attempts=1,
                               error=f"provider returned HTTP {error.code}", metadata=metadata)
    if isinstance(error, URLError) and isinstance(error.reason, (TimeoutError, socket.timeout)):
        return ProviderOutcome(status="timeout", attempts=1, error="provider request timed out", metadata=metadata)
    if isinstance(error, URLError):
        return ProviderOutcome(status="transient_failure", attempts=1,
                               error="provider transport failed", metadata=metadata)
    return ProviderOutcome(status="permanent_rejection", attempts=1,
                           error="provider request failed", metadata=metadata)


def _retryable(outcome: ProviderOutcome) -> bool:
    return outcome.status in {"timeout", "transient_failure"}


def _max_attempts(profile: ModelProfile) -> int:
    value = profile.parameters.get("max_attempts", profile.parameters.get("retry_attempts", 3))
    if not isinstance(value, int) or value < 1:
        raise ValueError("model profile retry attempts must be a positive integer")
    return value


def _batch_size(profile: ModelProfile, default: int) -> int:
    value = profile.parameters.get("max_batch_size", profile.parameters.get("request_chunk_size", default))
    if not isinstance(value, int) or value < 1:
        raise ValueError("model profile batch size must be a positive integer")
    return value


def _retry_delay(profile: ModelProfile, attempt: int) -> float:
    value = profile.parameters.get("retry_backoff_seconds", 0)
    if not isinstance(value, int | float) or value < 0:
        raise ValueError("model profile retry_backoff_seconds must be a non-negative number")
    return float(value) * attempt


def _with_attempts(outcome: ProviderOutcome, attempts: int) -> ProviderOutcome:
    if outcome.attempts == attempts:
        return outcome
    return outcome.model_copy(update={"attempts": attempts})


def _json(response: HTTPResponse) -> Any:
    return json.loads(response.read().decode("utf-8"))


def _validate(profile: ModelProfile, *, capability: str, provider: str, model: str) -> None:
    if profile.capability != capability:
        raise ValueError(f"profile capability must be {capability}")
    if profile.provider != provider or profile.model != model:
        raise ValueError(f"profile must resolve to {provider}/{model}")


class VoyageMultimodalEmbeddingAdapter:
    """Voyage multimodal embeddings over the provider's documented REST API."""

    def __init__(self, api_key: str | None = None, *, transport: Transport = _open,
                 content_resolver: ContentResolver = _resolve_local, timeout: float = 30.0):
        self._api_key = api_key
        self._transport = transport
        self._content_resolver = content_resolver
        self._timeout = timeout

    def embed(self, request: EmbeddingRequest) -> ProviderOutcome:
        return self.embed_batch((request,))[0]

    def embed_batch(self, requests: Sequence[EmbeddingRequest]) -> tuple[ProviderOutcome, ...]:
        if not requests:
            return ()
        request = requests[0]
        profile = request.profile
        _validate(profile, capability="embedding", provider="voyage", model="voyage-multimodal-3")
        size = _batch_size(profile, 8)
        outcomes: list[ProviderOutcome] = []
        for start in range(0, len(requests), size):
            batch = tuple(requests[start:start + size])
            attempts = 0
            while True:
                attempts += 1
                batch_outcomes = self._embed_batch_once(batch)
                if not all(_retryable(item) for item in batch_outcomes) or attempts >= _max_attempts(profile):
                    outcomes.extend(_with_attempts(item, attempts) for item in batch_outcomes)
                    break
                delay = _retry_delay(profile, attempts)
                if delay:
                    time.sleep(delay)
        return tuple(outcomes)

    def _embed_batch_once(self, requests: Sequence[EmbeddingRequest]) -> tuple[ProviderOutcome, ...]:
        request = requests[0]
        profile = request.profile
        _validate(profile, capability="embedding", provider="voyage", model="voyage-multimodal-3")
        endpoint = profile.parameters.get("endpoint")
        dimensions = profile.parameters.get("dimensions")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://") or dimensions != request.space.dimension:
            raise ValueError("resolved Voyage profile endpoint/dimensions do not match the embedding space")
        metadata = ProviderMetadata(provider=profile.provider, model=profile.model,
                                    provider_revision=profile.revision)
        inputs: list[dict[str, list[dict[str, str]]]] = []
        try:
            for item in requests:
                content_parts: list[dict[str, str]] = []
                for part in item.input.parts:
                    if part.kind == "text":
                        content_parts.append({"type": "text", "text": part.text or ""})
                    elif part.kind == "image":
                        content = self._content_resolver(part)
                        encoded = base64.b64encode(content).decode("ascii")
                        content_parts.append({"type": "image_base64", "image_base64": f"data:{part.media_type};base64,{encoded}"})
                    else:
                        return tuple(ProviderOutcome(status="permanent_rejection", attempts=1, error=f"Voyage adapter does not support {part.kind} content", metadata=metadata) for _ in requests)
                inputs.append({"content": content_parts})
        except (OSError, ValueError) as error:
            return tuple(ProviderOutcome(status="permanent_rejection", attempts=1, error=str(error), metadata=metadata) for _ in requests)
        body = {"inputs": inputs, "model": profile.model}
        try:
            response = self._transport(Request(endpoint, data=json.dumps(body).encode(), method="POST",
                                               headers={"Authorization": f"Bearer {self._api_key or os.environ.get('IGOR_EMBEDDING_API_KEY', '')}",
                                                        "Content-Type": "application/json"}), self._timeout)
            payload = _json(response)
            usage = payload.get("usage", {})
            response_id = payload.get("id")
            model_fingerprint = payload.get("model")
            metadata = metadata.model_copy(update={"response_id": response_id, "model_fingerprint": model_fingerprint,
                                       "usage": {k: v for k, v in usage.items() if isinstance(v, int)}})
            values = payload["data"]
            if not isinstance(values, list) or len(values) != len(requests):
                return tuple(ProviderOutcome(status="malformed_output", attempts=1, error="embedding batch size mismatch", metadata=metadata) for _ in requests)
            outcomes = []
            for value in values:
                vector = value["embedding"]
                if not isinstance(vector, list) or len(vector) != request.space.dimension:
                    outcomes.append(ProviderOutcome(status="malformed_output", attempts=1, error="embedding dimension mismatch", metadata=metadata))
                else:
                    outcomes.append(ProviderOutcome(status="succeeded", attempts=1, value=vector, metadata=metadata))
            return tuple(outcomes)
        except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
            return tuple(_failure(error, metadata=metadata) for _ in requests)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return tuple(ProviderOutcome(status="malformed_output", attempts=1, error="invalid Voyage response", metadata=metadata) for _ in requests)


class DeepSeekCompletionAdapter:
    """DeepSeek JSON-mode multimodal chat completion over the provider API."""

    def __init__(self, api_key: str | None = None, *, transport: Transport = _open, timeout: float = 60.0):
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    def _content(self, part: ContentPart) -> dict[str, Any]:
        if part.kind == "text":
            return {"type": "text", "text": part.text or ""}
        if part.kind in {"image", "document"}:
            content = _resolve_local(part)
            encoded = base64.b64encode(content).decode("ascii")
            return {"type": "image_url", "image_url": {
                "url": f"data:{part.media_type};base64,{encoded}",
            }}
        raise ValueError(f"DeepSeek completion does not support {part.kind} content")

    def enrich(self, request: CompletionRequest) -> ProviderOutcome:
        profile = request.profile
        if profile.capability != "completion" or profile.provider != "deepseek" or profile.model not in {"deepseek-chat", "deepseek-v4-flash"}:
            raise ValueError("profile must resolve to deepseek/deepseek-chat or deepseek/deepseek-v4-flash")
        endpoint = profile.parameters.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("resolved DeepSeek profile endpoint is invalid")
        temperature = profile.parameters.get("temperature")
        response_format = profile.parameters.get("response_format")
        if temperature != 0 or response_format != "json_object":
            raise ValueError("resolved DeepSeek profile must select zero temperature and JSON output")
        metadata = ProviderMetadata(provider=profile.provider, model=profile.model,
                                    provider_revision=profile.revision)
        schema = request.output_schema.json_schema
        if schema is None:
            raise ValueError("DeepSeek completion requires a registered JSON Schema")
        content = [self._content(part) for item in request.inputs for part in item.parts]
        body = {"model": profile.model, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content":
                    f"Return only a JSON object matching this JSON Schema. Do not add fields. "
                    f"Recipe {request.recipe.recipe_id}@{request.recipe.revision}. {json.dumps(schema)}"},
                             {"role": "user", "content": content}]}
        try:
            response = self._transport(Request(endpoint.rstrip("/") + "/chat/completions",
                                               data=json.dumps(body).encode(), method="POST",
                                               headers={"Authorization": f"Bearer {self._api_key or os.environ.get('IGOR_COMPLETION_API_KEY', '')}",
                                                        "Content-Type": "application/json"}), self._timeout)
            payload = _json(response)
            message = payload["choices"][0]["message"]["content"]
            value = json.loads(message) if isinstance(message, str) else message
            if not isinstance(value, dict):
                raise ValueError("completion payload is not a JSON object")
            usage = payload.get("usage", {})
            return ProviderOutcome(status="succeeded", attempts=1, value=value,
                                   metadata=metadata.model_copy(update={"response_id": payload.get("id"),
                                       "model_fingerprint": payload.get("model"),
                                       "usage": {k: v for k, v in usage.items() if isinstance(v, int)}}))
        except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
            return _failure(error, metadata=metadata)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return ProviderOutcome(status="malformed_output", attempts=1, error="invalid DeepSeek response", metadata=metadata)


class GeminiCompletionAdapter:
    """Gemini multimodal JSON completion over the generateContent API."""

    def __init__(self, api_key: str | None = None, *, transport: Transport = _open, timeout: float = 60.0):
        self._api_key = api_key
        self._transport = transport
        self._timeout = timeout

    def enrich(self, request: CompletionRequest) -> ProviderOutcome:
        profile = request.profile
        _validate(profile, capability="completion", provider="gemini", model="gemini-3.1-flash-lite")
        endpoint = profile.parameters.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("resolved Gemini profile endpoint is invalid")
        schema = request.output_schema.json_schema
        if schema is None:
            raise ValueError("Gemini completion requires a registered JSON Schema")
        parts: list[dict[str, Any]] = []
        try:
            for item in request.inputs:
                for part in item.parts:
                    if part.kind == "text":
                        parts.append({"text": part.text or ""})
                    elif part.kind in {"image", "document"}:
                        content = _resolve_local(part)
                        parts.append({"inline_data": {"mime_type": part.media_type,
                                                        "data": base64.b64encode(content).decode("ascii")}})
                    else:
                        raise ValueError(f"Gemini completion does not support {part.kind} content")
            parts.append({"text": f"Return only JSON matching this schema: {json.dumps(schema)}"})
        except (OSError, ValueError) as error:
            return ProviderOutcome(status="permanent_rejection", attempts=1, error=str(error))
        metadata = ProviderMetadata(provider=profile.provider, model=profile.model,
                                    provider_revision=profile.revision)
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseMimeType": "application/json"}}
        try:
            response = self._transport(Request(endpoint, data=json.dumps(body).encode(), method="POST",
                                               headers={"x-goog-api-key": self._api_key or os.environ.get("GEMINI_API_KEY", ""),
                                                        "Content-Type": "application/json"}), self._timeout)
            payload = _json(response)
            message = payload["candidates"][0]["content"]["parts"][0]["text"]
            value = json.loads(message)
            if not isinstance(value, dict):
                raise ValueError("Gemini completion payload is not an object")
            usage = payload.get("usageMetadata", {})
            return ProviderOutcome(status="succeeded", attempts=1, value=value,
                                   metadata=metadata.model_copy(update={"usage": {k: v for k, v in usage.items() if isinstance(v, int)}}))
        except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
            return _failure(error, metadata=metadata)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ProviderOutcome(status="malformed_output", attempts=1, error="invalid Gemini response", metadata=metadata)


class MistralCompletionAdapter:
    """Mistral text-and-vision JSON completion over the Chat Completions API."""

    _MODELS = {"ministral-8b-2512", "mistral-small-2603"}

    def __init__(self, api_key: str | None = None, *, transport: Transport = _open,
                 content_resolver: ContentResolver = _resolve_local, timeout: float = 60.0):
        self._api_key = api_key
        self._transport = transport
        self._content_resolver = content_resolver
        self._timeout = timeout

    def _content(self, part: ContentPart) -> dict[str, Any]:
        if part.kind == "text":
            return {"type": "text", "text": part.text or ""}
        if part.kind == "image":
            content = self._content_resolver(part)
            encoded = base64.b64encode(content).decode("ascii")
            return {"type": "image_url", "image_url": f"data:{part.media_type};base64,{encoded}"}
        if part.kind == "document" and part.media_type == "application/pdf":
            content = self._content_resolver(part)
            encoded = base64.b64encode(content).decode("ascii")
            return {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded}"}
        raise ValueError(f"Mistral completion does not support {part.kind} content")

    def enrich(self, request: CompletionRequest) -> ProviderOutcome:
        profile = request.profile
        if (profile.capability != "completion" or profile.provider != "mistral"
                or profile.model not in self._MODELS):
            raise ValueError("profile must resolve to a qualified Mistral completion model")
        endpoint = profile.parameters.get("endpoint")
        if endpoint != "https://api.mistral.ai/v1/chat/completions":
            raise ValueError("resolved Mistral profile endpoint is invalid")
        if profile.parameters.get("temperature") != 0:
            raise ValueError("resolved Mistral profile must select zero temperature")
        if profile.parameters.get("response_format") != "json_object":
            raise ValueError("resolved Mistral profile must select JSON output")
        schema = request.output_schema.json_schema
        if schema is None:
            raise ValueError("Mistral completion requires a registered JSON Schema")
        metadata = ProviderMetadata(provider=profile.provider, model=profile.model,
                                    provider_revision=profile.revision)
        configured_kinds = profile.parameters.get("supported_content_kinds")
        if (not isinstance(configured_kinds, list)
                or not all(isinstance(kind, str) for kind in configured_kinds)):
            raise ValueError("resolved Mistral profile must declare supported content kinds")
        requested_kinds = {part.kind for item in request.inputs for part in item.parts}
        unsupported_kinds = sorted(requested_kinds - set(configured_kinds))
        if unsupported_kinds:
            return ProviderOutcome(
                status="permanent_rejection", attempts=1,
                error=f"Mistral profile does not support {', '.join(unsupported_kinds)} content",
                metadata=metadata,
            )
        try:
            content = [self._content(part) for item in request.inputs for part in item.parts]
        except (OSError, ValueError) as error:
            return ProviderOutcome(status="permanent_rejection", attempts=1,
                                   error=str(error), metadata=metadata)
        body = {
            "model": profile.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content":
                    f"Return only a JSON object matching this JSON Schema. Do not add fields. "
                    f"Recipe {request.recipe.recipe_id}@{request.recipe.revision}. {json.dumps(schema)}"},
                {"role": "user", "content": content},
            ],
        }
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._transport(Request(
                    endpoint, data=json.dumps(body).encode(), method="POST",
                    headers={"Authorization": f"Bearer {self._api_key or os.environ.get('IGOR_COMPLETION_API_KEY', '')}",
                             "Content-Type": "application/json"}), self._timeout)
                payload = _json(response)
                message = payload["choices"][0]["message"]["content"]
                value = json.loads(message) if isinstance(message, str) else message
                if not isinstance(value, dict):
                    raise ValueError("completion payload is not a JSON object")
                usage = payload.get("usage", {})
                return ProviderOutcome(
                    status="succeeded", attempts=attempts, value=value,
                    metadata=metadata.model_copy(update={
                        "response_id": payload.get("id"),
                        "model_fingerprint": payload.get("model"),
                        "usage": {key: value for key, value in usage.items() if isinstance(value, int)},
                    }),
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
                outcome = _failure(error, metadata=metadata)
                if not _retryable(outcome) or attempts >= _max_attempts(profile):
                    return _with_attempts(outcome, attempts)
                delay = _retry_delay(profile, attempts)
                if delay:
                    time.sleep(delay)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return ProviderOutcome(status="malformed_output", attempts=attempts,
                                       error="invalid Mistral response", metadata=metadata)


def embedding_adapter_for(profile: ModelProfile, **kwargs: Any):
    """Resolve a configured embedding profile inside the provider-owning module."""
    if profile.capability != "embedding":
        raise ValueError("embedding adapter requires an embedding profile")
    if profile.provider == "voyage":
        return VoyageMultimodalEmbeddingAdapter(**kwargs)
    raise ValueError(f"unsupported embedding provider: {profile.provider}")


def completion_adapter_for(profile: ModelProfile, **kwargs: Any):
    """Resolve a configured completion profile inside the provider-owning module."""
    if profile.capability != "completion":
        raise ValueError("completion adapter requires a completion profile")
    adapters = {
        "deepseek": DeepSeekCompletionAdapter,
        "gemini": GeminiCompletionAdapter,
        "mistral": MistralCompletionAdapter,
    }
    try:
        adapter = adapters[profile.provider]
    except KeyError as error:
        raise ValueError(f"unsupported completion provider: {profile.provider}") from error
    return adapter(**kwargs)
