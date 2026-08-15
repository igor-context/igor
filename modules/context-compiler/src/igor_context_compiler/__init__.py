"""Public, provider-neutral Context Compiler interface."""

from .compiler import (
    BudgetPort,
    CompilerFailure,
    CompilationPlan,
    CompilationRequest,
    CompilationResult,
    CompilerPorts,
    DependencyInventory,
    DerivationSpec,
    DeterministicProviders,
    CompletionRequest, EmbeddingRequest, ProviderMetadata, ProviderOutcome, QualifiedRepresentation,
    RetryPolicy,
    EnrichmentPort,
    EmbeddingPort,
    OutputStore,
    AuthorityAssertion, TemporalAssertion, ResolutionCandidate, ResolutionRequest,
    ResolutionDecision, ResolutionResult,
    ResolutionPort,
    RetrievalFact,
    RetrievalPort,
    RetrievalQuery,
    plan,
    execute,
)
from .context_model import (
    ContextModelCompileRequest, ContextModelReference, ResolvedAuthorityPolicy,
    ResolvedContextContract, ResolvedContextEvaluation, ResolvedContextModelManifest,
    ResolvedContextModelTest, ResolvedContextObject, ResolvedContextOutput,
    ResolvedPresentation, ResolvedRetrievalDefinition, ResolvedSourceBinding,
    compile_context_model, compile_context_model_derivation_instances, compile_context_model_derivations,
    compile_context_model_resolution_request, compile_context_model_retrieval_queries,
)
from .providers import (
    DeepSeekCompletionAdapter, GeminiCompletionAdapter, MistralCompletionAdapter,
    VoyageMultimodalEmbeddingAdapter, completion_adapter_for, embedding_adapter_for,
)
from .runtime import (
    AgenticEnrichmentExecutor, BatchEnrichmentExecutor, BatchPlan, DirectEnrichmentExecutor,
    EnrichmentExecutor, EnrichmentWorkItem, InProcessQuotaCoordinator, Reservation,
    RuntimeAttempt, RuntimeLimits, RuntimeResult, plan_batches, run_enrichment,
)

__all__ = [
    "BudgetPort", "CompilerFailure", "CompilationPlan", "CompilationRequest", "CompilationResult",
    "CompilerPorts", "DependencyInventory", "DerivationSpec", "DeterministicProviders",
    "EnrichmentPort", "EmbeddingPort", "OutputStore", "RetrievalQuery", "RetrievalFact",
    "CompletionRequest", "EmbeddingRequest", "ProviderMetadata", "ProviderOutcome",
    "QualifiedRepresentation", "RetryPolicy",
    "RetrievalPort", "AuthorityAssertion", "TemporalAssertion", "ResolutionCandidate", "ResolutionRequest",
    "ResolutionDecision", "ResolutionResult", "ResolutionPort", "plan", "execute",
    "DeepSeekCompletionAdapter", "GeminiCompletionAdapter", "MistralCompletionAdapter",
    "VoyageMultimodalEmbeddingAdapter",
    "completion_adapter_for", "embedding_adapter_for",
    "AgenticEnrichmentExecutor", "BatchEnrichmentExecutor", "BatchPlan", "DirectEnrichmentExecutor",
    "EnrichmentExecutor", "EnrichmentWorkItem", "InProcessQuotaCoordinator", "Reservation",
    "RuntimeAttempt", "RuntimeLimits", "RuntimeResult", "plan_batches", "run_enrichment",
    "ContextModelCompileRequest", "ContextModelReference", "ResolvedAuthorityPolicy",
    "ResolvedContextContract", "ResolvedContextEvaluation", "ResolvedContextModelManifest",
    "ResolvedContextModelTest", "ResolvedContextObject", "ResolvedContextOutput",
    "ResolvedPresentation", "ResolvedRetrievalDefinition", "ResolvedSourceBinding",
    "compile_context_model", "compile_context_model_derivation_instances",
    "compile_context_model_derivations", "compile_context_model_resolution_request",
    "compile_context_model_retrieval_queries",
]
