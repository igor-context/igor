from __future__ import annotations

from time import sleep

from igor_context_compiler import (
    AgenticEnrichmentExecutor, CompletionRequest, DirectEnrichmentExecutor,
    EnrichmentWorkItem, ProviderOutcome, QualifiedRepresentation,
    RuntimeLimits, InProcessQuotaCoordinator,
    plan_batches, run_enrichment,
)
from igor_core import ContentPart, EnrichmentRecipe, ModelProfile, Representation, SchemaDescriptor, stable_identity


def make_item(name: str, *, modality: str = "text", profile: str = "profile") -> EnrichmentWorkItem:
    schema = SchemaDescriptor(schema_version="0.1", schema_id="runtime", revision="1", json_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    })
    representation = Representation(ir_version="0.1", representation_type=modality,
                                    schema_ref=schema, source_snapshot_ids=(stable_identity(name),), payload=name)
    qualified = QualifiedRepresentation(representation=representation, parts=(ContentPart(
        kind="text", media_type="text/plain", text=name,
        content_sha256=stable_identity(name)),))
    recipe = EnrichmentRecipe(recipe_id="recipe", revision="1", accepted_representation_types=(modality,),
                              accepted_media_types=("text/plain",), output_schema_identity=schema.identity,
                              prompt_version="p1", taxonomy_version="t1", evidence_required=False)
    request = CompletionRequest(output_identity=stable_identity("output-" + name), inputs=(qualified,),
                                output_schema=schema, recipe=recipe, prompt_version="p1", taxonomy_version="t1",
                                profile=ModelProfile(schema_version="0.1", profile_id=profile, capability="completion",
                                                     provider="test", model="test", revision="1"))
    return EnrichmentWorkItem(work_identity=stable_identity("work-" + name), request=request,
                              capability_profile_identity=profile, modality=modality,
                              recipe_identity=recipe.identity, schema_identity=schema.identity)


class FakeExecutor:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []

    def execute(self, item):
        self.calls.append(item.work_identity)
        if self.delay:
            sleep(self.delay)
        return ProviderOutcome(status="succeeded", attempts=1, value={"ok": True})


def test_batches_are_compatible_and_stable():
    items = (make_item("b"), make_item("a"), make_item("other", profile="other"))
    plans = plan_batches(items, RuntimeLimits(max_items=2))
    assert sorted(plan.work_identities for plan in plans) == sorted([
        tuple(sorted((make_item("a").work_identity, make_item("b").work_identity))),
        (make_item("other", profile="other").work_identity,),
    ])


def test_parallel_runtime_preserves_identity_order_and_reports_metrics():
    items = tuple(make_item(str(i)) for i in range(4))
    result = run_enrichment(items, FakeExecutor(0.03), RuntimeLimits(max_concurrency=4))
    assert list(result.outcomes) == sorted(item.work_identity for item in items)
    assert result.artifact["ordering"] == list(result.outcomes)
    assert result.artifact["peak_memory_bytes"] >= 0


def test_direct_and_agentic_adapters_share_outcome_shape():
    item = make_item("one")
    direct = DirectEnrichmentExecutor(type("Port", (), {"enrich": lambda self, request: ProviderOutcome(
        status="succeeded", attempts=1, value={"mode": "direct"})})())
    agentic = AgenticEnrichmentExecutor(lambda request: ProviderOutcome(status="succeeded", attempts=1,
                                                                          value={"mode": "agentic"}))
    assert run_enrichment((item,), direct).outcomes[item.work_identity].status == "succeeded"
    assert run_enrichment((item,), agentic).outcomes[item.work_identity].status == "succeeded"


def test_batch_failure_is_isolated_and_shared_quota_is_atomic():
    items = tuple(make_item(str(i)) for i in range(3))

    class Batch:
        def execute_batch(self, values):
            return tuple(ProviderOutcome(status="transient_failure", attempts=1, error="throttle")
                         if value.work_identity == items[1].work_identity else
                         ProviderOutcome(status="succeeded", attempts=1, value={"ok": True}) for value in values)

    class Retry:
        def execute(self, value):
            return ProviderOutcome(status="succeeded", attempts=1, value={"recovered": True})

    result = run_enrichment(items, Retry(), RuntimeLimits(max_items=3, max_concurrency=2, max_requests=2),
                            coordinator=InProcessQuotaCoordinator(RuntimeLimits(max_requests=2)),
                            batch_executor=Batch())
    assert all(outcome.status == "succeeded" for outcome in result.outcomes.values())
    assert result.outcomes[items[1].work_identity].value == {"recovered": True}
    assert result.artifact["reservations"]["requests"] == 2
