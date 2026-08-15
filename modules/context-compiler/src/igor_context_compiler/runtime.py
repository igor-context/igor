"""Provider-neutral, rate-aware enrichment execution.

This module contains the functional scheduling seam.  Provider adapters only need
to implement ``EnrichmentExecutor`` (or the optional batch method); quotas,
retries, ordering, and operational accounting remain compiler-owned.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from typing import Callable, Protocol, Sequence
import resource

from pydantic import BaseModel, ConfigDict, Field

from .compiler import CompletionRequest, EnrichmentPort, ProviderOutcome


class RuntimeLimits(BaseModel):
    """Hard per-run and per-reservation ceilings; zero means unlimited."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_requests: int = Field(default=0, ge=0)
    max_input_tokens: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=0, ge=0)
    max_items: int = Field(default=0, ge=0)
    max_bytes: int = Field(default=0, ge=0)
    max_cost_micros: int = Field(default=0, ge=0)
    max_concurrency: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=1, ge=1, le=8)
    retry_backoff_seconds: float = Field(default=0.0, ge=0, le=60)
    deadline_seconds: float | None = Field(default=None, gt=0)


class EnrichmentWorkItem(BaseModel):
    """Immutable, independently retryable and invalidatable work identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    work_identity: str
    request: CompletionRequest
    capability_profile_identity: str
    modality: str
    recipe_identity: str
    schema_identity: str
    governance_boundary: str = "default"
    request_shape: str = "default"
    priority: int = 0
    estimated_input_tokens: int = Field(default=0, ge=0)
    estimated_output_tokens: int = Field(default=0, ge=0)
    estimated_bytes: int = Field(default=0, ge=0)
    estimated_cost_micros: int = Field(default=0, ge=0)


class BatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_identity: str
    work_identities: tuple[str, ...]
    compatibility_key: tuple[str, ...]
    reason: str


class Reservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    items: int = 0
    bytes: int = 0
    cost_micros: int = 0


class RuntimeAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    work_identity: str
    attempt: int
    status: str
    queue_wait_seconds: float = 0
    provider_seconds: float = 0
    validation_seconds: float = 0
    error: str | None = None


class RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    outcomes: dict[str, ProviderOutcome]
    batches: tuple[BatchPlan, ...]
    attempts: tuple[RuntimeAttempt, ...]
    artifact: dict[str, object]


class EnrichmentExecutor(Protocol):
    def execute(self, item: EnrichmentWorkItem) -> ProviderOutcome: ...


class BatchEnrichmentExecutor(Protocol):
    def execute_batch(self, items: Sequence[EnrichmentWorkItem]) -> Sequence[ProviderOutcome]: ...


class DirectEnrichmentExecutor:
    """Adapt the existing one-request port to the runtime seam."""

    def __init__(self, port: EnrichmentPort):
        self.port = port

    def execute(self, item: EnrichmentWorkItem) -> ProviderOutcome:
        return self.port.enrich(item.request)


class AgenticEnrichmentExecutor:
    """Adapt a bounded agent runner without exposing its state or tool types."""

    def __init__(self, run: Callable[[CompletionRequest], ProviderOutcome]):
        self._run = run

    def execute(self, item: EnrichmentWorkItem) -> ProviderOutcome:
        return self._run(item.request)


class InProcessQuotaCoordinator:
    """Atomic quota reservations shared by all runtime workers in a process."""

    def __init__(self, limits: RuntimeLimits):
        self.limits = limits
        self._used = Reservation()
        self._lock = Lock()

    def reserve(self, reservation: Reservation) -> bool:
        with self._lock:
            candidate = Reservation(**{
                field: getattr(self._used, field) + getattr(reservation, field)
                for field in Reservation.model_fields
            })
            for field, limit_field in (("requests", "max_requests"), ("input_tokens", "max_input_tokens"),
                                       ("output_tokens", "max_output_tokens"), ("items", "max_items"),
                                       ("bytes", "max_bytes"), ("cost_micros", "max_cost_micros")):
                limit = getattr(self.limits, limit_field)
                if limit and getattr(candidate, field) > limit:
                    return False
            self._used = candidate
            return True

    @property
    def used(self) -> Reservation:
        with self._lock:
            return self._used


def _compatibility_key(item: EnrichmentWorkItem) -> tuple[str, ...]:
    return (item.capability_profile_identity, item.modality, item.recipe_identity,
            item.schema_identity, item.governance_boundary, item.request_shape)


def plan_batches(items: Sequence[EnrichmentWorkItem], limits: RuntimeLimits) -> tuple[BatchPlan, ...]:
    """Pack only compatible items, preserving stable identity order."""
    grouped: dict[tuple[str, ...], list[EnrichmentWorkItem]] = defaultdict(list)
    for item in items:
        grouped[_compatibility_key(item)].append(item)
    result: list[BatchPlan] = []
    max_items = limits.max_items or len(items) or 1
    for key in sorted(grouped):
        current: list[EnrichmentWorkItem] = []
        input_tokens = output_tokens = byte_count = 0
        for item in sorted(grouped[key], key=lambda value: value.work_identity):
            would_exceed = (current and (len(current) >= max_items or
                (limits.max_input_tokens and input_tokens + item.estimated_input_tokens > limits.max_input_tokens) or
                (limits.max_output_tokens and output_tokens + item.estimated_output_tokens > limits.max_output_tokens) or
                (limits.max_bytes and byte_count + item.estimated_bytes > limits.max_bytes)))
            if would_exceed:
                result.append(BatchPlan(batch_identity=_batch_identity(key, current),
                                        work_identities=tuple(value.work_identity for value in current),
                                        compatibility_key=key, reason="provider-and-run-limit packing"))
                current, input_tokens, output_tokens, byte_count = [], 0, 0, 0
            current.append(item)
            input_tokens += item.estimated_input_tokens
            output_tokens += item.estimated_output_tokens
            byte_count += item.estimated_bytes
        if current:
            result.append(BatchPlan(batch_identity=_batch_identity(key, current),
                                    work_identities=tuple(value.work_identity for value in current),
                                    compatibility_key=key, reason="provider-and-run-limit packing"))
    return tuple(result)


def _batch_identity(key: tuple[str, ...], items: Sequence[EnrichmentWorkItem]) -> str:
    import hashlib
    material = "|".join((*key, *(item.work_identity for item in items)))
    return "sha256:" + hashlib.sha256(material.encode()).hexdigest()


def run_enrichment(items: Sequence[EnrichmentWorkItem], executor: EnrichmentExecutor,
                   limits: RuntimeLimits = RuntimeLimits(),
                   coordinator: InProcessQuotaCoordinator | None = None,
                   batch_executor: BatchEnrichmentExecutor | None = None,
                   clock: Callable[[], float] = monotonic,
                   sleeper: Callable[[float], None] = sleep) -> RuntimeResult:
    """Execute independent work concurrently with hard, observable limits."""
    if len({item.work_identity for item in items}) != len(items):
        raise ValueError("duplicate work identity")
    ordered = tuple(sorted(items, key=lambda item: (-item.priority, item.work_identity)))
    plans = plan_batches(ordered, limits)
    by_id = {item.work_identity: item for item in ordered}
    quota = coordinator or InProcessQuotaCoordinator(limits)
    started = clock()
    attempts: list[RuntimeAttempt] = []
    outcomes: dict[str, ProviderOutcome] = {}
    lock = Lock()

    def execute_one(item: EnrichmentWorkItem) -> tuple[str, ProviderOutcome, list[RuntimeAttempt]]:
        local: list[RuntimeAttempt] = []
        first = clock()
        for number in range(1, limits.max_attempts + 1):
            if limits.deadline_seconds is not None and clock() - started > limits.deadline_seconds:
                outcome = ProviderOutcome(status="timeout", attempts=number, error="runtime deadline exceeded")
                local.append(RuntimeAttempt(work_identity=item.work_identity, attempt=number, status=outcome.status,
                                            queue_wait_seconds=first - started, error=outcome.error))
                return item.work_identity, outcome, local
            reservation = Reservation(requests=1, input_tokens=item.estimated_input_tokens,
                                      output_tokens=item.estimated_output_tokens, items=1,
                                      bytes=item.estimated_bytes, cost_micros=item.estimated_cost_micros)
            if not quota.reserve(reservation):
                outcome = ProviderOutcome(status="retry_exhausted", attempts=number, error="declared quota exceeded")
                local.append(RuntimeAttempt(work_identity=item.work_identity, attempt=number, status=outcome.status,
                                            queue_wait_seconds=first - started, error=outcome.error))
                return item.work_identity, outcome, local
            call_started = clock()
            try:
                outcome = executor.execute(item)
            except Exception as error:  # adapters cannot crash sibling work
                outcome = ProviderOutcome(status="permanent_rejection", attempts=number, error=str(error))
            elapsed = clock() - call_started
            local.append(RuntimeAttempt(work_identity=item.work_identity, attempt=number, status=outcome.status,
                                        queue_wait_seconds=first - started, provider_seconds=elapsed,
                                        error=outcome.error))
            if outcome.status not in ("transient_failure", "timeout") or number >= limits.max_attempts:
                if outcome.status in ("transient_failure", "timeout") and number >= limits.max_attempts:
                    outcome = outcome.model_copy(update={"status": "retry_exhausted", "attempts": number})
                return item.work_identity, outcome.model_copy(update={"attempts": number}), local
            if limits.retry_backoff_seconds:
                sleeper(limits.retry_backoff_seconds * (2 ** (number - 1)))
        raise AssertionError("unreachable")

    def execute_batch(batch: BatchPlan) -> tuple[list[tuple[str, ProviderOutcome, list[RuntimeAttempt]]], bool]:
        batch_items = tuple(by_id[identity] for identity in batch.work_identities)
        reservation = Reservation(
            requests=1, input_tokens=sum(item.estimated_input_tokens for item in batch_items),
            output_tokens=sum(item.estimated_output_tokens for item in batch_items),
            items=len(batch_items), bytes=sum(item.estimated_bytes for item in batch_items),
            cost_micros=sum(item.estimated_cost_micros for item in batch_items),
        )
        if not quota.reserve(reservation):
            return [(
                item.work_identity,
                ProviderOutcome(status="retry_exhausted", attempts=1, error="declared quota exceeded"),
                [RuntimeAttempt(work_identity=item.work_identity, attempt=1, status="retry_exhausted",
                                error="declared quota exceeded")],
            ) for item in batch_items], True
        try:
            values = tuple(batch_executor.execute_batch(batch_items))  # type: ignore[union-attr]
        except Exception as error:
            values = tuple(ProviderOutcome(status="permanent_rejection", attempts=1, error=str(error))
                           for _ in batch_items)
        if len(values) != len(batch_items):
            raise ValueError("batch executor returned wrong result count")
        results = []
        for item, outcome in zip(batch_items, values):
            if outcome.status in ("transient_failure", "timeout"):
                results.append(execute_one(item))
            else:
                results.append((item.work_identity, outcome.model_copy(update={"attempts": 1}),
                                [RuntimeAttempt(work_identity=item.work_identity, attempt=1, status=outcome.status,
                                                error=outcome.error)]))
        return results, False

    # Batch providers receive a batch, but failed members are retried individually
    # so a successful member is never unnecessarily reissued.
    if batch_executor is not None:
        with ThreadPoolExecutor(max_workers=limits.max_concurrency) as pool:
            futures = [pool.submit(execute_batch, batch) for batch in plans]
            for future in as_completed(futures):
                results, _ = future.result()
                with lock:
                    for identity, outcome, local in results:
                        outcomes[identity] = outcome
                        attempts.extend(local)
    else:
        with ThreadPoolExecutor(max_workers=limits.max_concurrency) as pool:
            futures = [pool.submit(execute_one, item) for item in ordered]
            for future in as_completed(futures):
                identity, outcome, local = future.result()
                with lock:
                    outcomes[identity], attempts[:] = outcome, [*attempts, *local]

    attempts.sort(key=lambda value: (value.work_identity, value.attempt))
    ordered_outcomes = {item.work_identity: outcomes[item.work_identity] for item in sorted(ordered, key=lambda value: value.work_identity)}
    total = quota.used
    artifact = {
        "limits": limits.model_dump(mode="json"),
        "batches": [batch.model_dump(mode="json") for batch in plans],
        "reservations": total.model_dump(mode="json"),
        "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        "ordering": list(ordered_outcomes),
        "process_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "wall_seconds": clock() - started,
    }
    return RuntimeResult(outcomes=ordered_outcomes, batches=plans, attempts=tuple(attempts), artifact=artifact)
