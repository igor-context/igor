"""Reference-stage composition and project interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ArtifactPayload",
    "CompositionRoot",
    "ContextModelLifecycleResult",
    "DomainConfig",
    "RunConfig",
    "RunnerError",
    "StageContext",
    "StageSpec",
    "StructuredStorageConfig",
    "compile_context",
    "compile_context_model_lifecycle",
    "materialize_context_model_lifecycle",
    "qualify_document",
    "run_support_smoke",
    "run_support_huggingface_smoke",
    "run_support_huggingface_e2e",
    "run_context_e2e",
    "run_image_context",
    "semantic_sql_query",
]

_COMPOSE_EXPORTS = {
    "ArtifactPayload",
    "CompositionRoot",
    "DomainConfig",
    "RunConfig",
    "RunnerError",
    "StageContext",
    "StageSpec",
    "StructuredStorageConfig",
    "compile_context",
    "run_support_smoke",
    "run_support_huggingface_smoke",
    "run_support_huggingface_e2e",
    "run_context_e2e",
    "run_image_context",
    "semantic_sql_query",
}
_LIFECYCLE_EXPORTS = {
    "ContextModelLifecycleResult",
    "compile_context_model_lifecycle",
    "materialize_context_model_lifecycle",
}
_DOCUMENT_EXPORTS = {"qualify_document"}


def __getattr__(name: str):
    if name in _COMPOSE_EXPORTS:
        module = import_module("igor_runner.compose")
        return getattr(module, name)
    if name in _LIFECYCLE_EXPORTS:
        module = import_module("igor_runner.context_model_lifecycle")
        return getattr(module, name)
    if name in _DOCUMENT_EXPORTS:
        module = import_module("igor_runner.document_qualification")
        return getattr(module, name)
    raise AttributeError(name)
