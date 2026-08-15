"""Tool-neutral storage seam backed by LanceDB."""

from igor_lancedb.store import LanceStore, LanceStoreConfig
from igor_lancedb.context import (
    STANDARD_CONTEXT_RELATIONS,
    ContextModelConformanceResult,
    ContextModelDeclaredTestResult,
    LanceContextModelMaterializer,
    LanceContextOutputStore,
    LanceReadableContextCatalog,
    LanceRetrievalAdapter,
    ContextModelTestExecutionResult,
    execute_context_model_tests,
    validate_standard_context_relations,
)

__all__ = [
    "LanceStore",
    "LanceStoreConfig",
    "LanceContextModelMaterializer",
    "LanceContextOutputStore",
    "LanceReadableContextCatalog",
    "LanceRetrievalAdapter",
    "ContextModelConformanceResult",
    "ContextModelDeclaredTestResult",
    "ContextModelTestExecutionResult",
    "STANDARD_CONTEXT_RELATIONS",
    "execute_context_model_tests",
    "validate_standard_context_relations",
]
