from __future__ import annotations

import hashlib
import json
import os
import tempfile
import struct
import zlib
import yaml
from time import perf_counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping
from datetime import datetime, timezone, timedelta

from igor_core import (
    ArtifactReference,
    ProducerIdentity,
    RunIdentityLinks,
    RunManifest,
    LineageGraph,
    StageResult,
    canonical_json,
    stable_identity,
    ModelProfile,
    load_reference_profile,
    validate_run_directory,
)
from igor_datafusion import AnalyticalEngine
from igor_dlt import IngestConfig, ingest_bound_records, ingest_records
from igor_fixtures import validate_scenario_pack
from igor_lancedb import (
    LanceContextOutputStore,
    LanceRetrievalAdapter,
    LanceStore,
    LanceStoreConfig,
)
from igor_relational import ModelManifest, ModelSpec, RelationalRunner
from igor_context_compiler import (
    CompilationRequest, CompilerPorts, DependencyInventory, DeterministicProviders, RetryPolicy,
    ContextModelCompileRequest, ContextModelReference,
    QualifiedRepresentation, ResolutionRequest,
    ResolutionDecision, ResolutionResult, AuthorityAssertion, TemporalAssertion, execute, plan,
    CompilerFailure,
    DeepSeekCompletionAdapter, GeminiCompletionAdapter, MistralCompletionAdapter,
    VoyageMultimodalEmbeddingAdapter, compile_context_model,
)
from igor_runner.context_model_lifecycle import compile_context_model_lifecycle, materialize_context_model_lifecycle
from igor_core import (
    ConnectorBinding, ConnectorFieldBinding, ConnectorResourceBinding, ContentPart,
    ContextSourceContract, Derivation, Embedding,
    EmbeddingSpace, EnrichmentOutput, EnrichmentRecipe, Evidence, Representation,
    SchemaDescriptor, SchemaField, SourceFieldRequirement, SourceResourceContract,
    SourceSnapshot, LineageNode, LineageEdge, LineageGraph,
)


class RunnerError(ValueError):
    """Deterministic error raised for invalid composition or stage execution."""


def semantic_sql_query(store: LanceStore, sql: str, *, query_vectors=None, retrieval=None, table_name=None):
    """Run vendor-neutral semantic SQL through the configured retrieval adapter."""
    default_table = table_name or ("events" if "events" in store.names() else store.names()[0])
    return AnalyticalEngine(store, retrieval or LanceRetrievalAdapter(store), query_vectors=query_vectors).query(sql, table_name=default_table)


def compile_context(request: CompilationRequest, inventory: DependencyInventory, store) :
    """Runner composition seam: wire the compiler to a storage-owned port."""
    compiler_plan = plan(request, inventory)
    output_store = store if hasattr(store, "has_valid") else LanceContextOutputStore(store)
    return execute(compiler_plan, CompilerPorts(store=output_store, embedding=DeterministicProviders(), enrichment=DeterministicProviders(), retry=RetryPolicy(max_attempts=3)), request)


def _support_context_manifest():
    return compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {
                "id": "support",
                "revision": "0.1",
                "title": "Support ticket context",
            },
            "sources": {
                "support_messages": {
                    "source_contract": "support.messages.v1",
                    "connector_binding": "huggingface.support-ticket-text.v1",
                    "identity_fields": ["record_id"],
                },
            },
            "objects": {
                "support_ticket": {
                    "kind": "business_object",
                    "source": "support_messages",
                    "display_name": "Support ticket {{ record_id }}",
                },
                "support_message_snapshot": {
                    "kind": "source_snapshot",
                    "source": "support_messages",
                    "display_name": "{{ record_id }} observed at {{ observed_at }}",
                },
                "support_message": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_message_snapshot"],
                    "operation": "representation.v1",
                    "schema": "support.message.v1",
                    "display_name": "Support message {{ record_id }}",
                },
                "support_embedding": {
                    "kind": "retrieval_projection",
                    "derived_from": ["support_message"],
                    "operation": "embedding.multimodal.v1",
                    "schema": "support.embedding.v1",
                    "model_profile": "support.embedding-profile.v1",
                    "display_name": "Support message embedding",
                },
                "support_enrichment": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_message"],
                    "operation": "enrichment.structured.v1",
                    "schema": "support.enrichment.v1",
                    "recipe": "support.enrichment-recipe.v1",
                    "display_name": "Support message enrichment",
                },
            },
            "authority": {
                "support.authority-temporal": {
                    "target": "support_message",
                    "policy": "support.authority-temporal.v1",
                    "policy_id": "support.authority-temporal",
                    "policy_revision": "1",
                    "accepted_outcomes": ["selected"],
                    "active_only": True,
                    "valid_at_task_time": True,
                },
            },
            "retrievals": {
                "support_messages": {
                    "search": "vector",
                    "candidate_limit": 1000,
                    "resolution": {"policy": "support.authority-temporal", "accepted_outcomes": ["selected"]},
                },
            },
        },
        references=(
            ContextModelReference(
                kind="source_contract",
                ref="support.messages.v1",
                identity=stable_identity({"source_contract": "support.messages.v1"}),
            ),
            ContextModelReference(
                kind="connector_binding",
                ref="huggingface.support-ticket-text.v1",
                identity=stable_identity({"connector_binding": "huggingface.support-ticket-text.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.v1",
                identity=stable_identity({"operation": "representation.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="embedding.multimodal.v1",
                identity=stable_identity({"operation": "embedding.multimodal.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="enrichment.structured.v1",
                identity=stable_identity({"operation": "enrichment.structured.v1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="support.message.v1",
                identity=stable_identity({"schema": "support.message.v1", "revision": "1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="support.embedding.v1",
                identity=stable_identity({"schema": "support.embedding", "revision": "1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="support.enrichment.v1",
                identity=stable_identity({"schema": "support.enrichment", "revision": "1"}),
            ),
            ContextModelReference(
                kind="model_profile",
                ref="support.embedding-profile.v1",
                identity=stable_identity({"model_profile": "support.embedding-profile.v1"}),
            ),
            ContextModelReference(
                kind="recipe",
                ref="support.enrichment-recipe.v1",
                identity=stable_identity({"recipe": "support.enrichment-recipe.v1"}),
            ),
            ContextModelReference(
                kind="authority_policy",
                ref="support.authority-temporal.v1",
                identity=stable_identity({"policy": "support.authority-temporal", "revision": "1"}),
            ),
        ),
    ))


def _source_snapshot_observation(snapshot: SourceSnapshot, *, object_role: str, snapshot_role: str,
                                 display_name: str, preview: str = "") -> dict[str, object]:
    return {
        "object_role": object_role,
        "snapshot_role": snapshot_role,
        "source_system": snapshot.source_system,
        "source_key": snapshot.source_key,
        "snapshot_identity": snapshot.identity,
        "observed_at": snapshot.observed_at.isoformat(),
        "content_ref": snapshot.content_ref,
        "content_sha256": snapshot.content_sha256,
        "media_type": snapshot.media_type,
        "display_name": display_name,
        "preview": preview,
        "status": "observed",
        "payload": snapshot.metadata,
    }


@dataclass(frozen=True)
class DomainConfig:
    """Stable domain and environment identity for one composed run."""

    environment: str
    domain: str

    def __post_init__(self) -> None:
        for name, value in (("environment", self.environment), ("domain", self.domain)):
            if not value.strip():
                raise RunnerError(f"{name} cannot be empty")
            if "/" in value or "\\" in value:
                raise RunnerError(f"{name} must be a path-safe identifier")

    @property
    def namespace(self) -> str:
        return f"{self.environment}/{self.domain}"


@dataclass(frozen=True)
class StructuredStorageConfig:
    """Provider-neutral structured-storage routing and lifecycle metadata."""

    root_uri: str
    credential_ref: str | None = None
    lifecycle_class: str = "standard"
    isolation_tier: str = "shared-prefix"

    def __post_init__(self) -> None:
        if not self.root_uri.strip():
            raise RunnerError("structured storage root_uri cannot be empty")
        if self.credential_ref is not None and not self.credential_ref.strip():
            raise RunnerError("structured storage credential_ref cannot be empty")
        if not self.lifecycle_class.strip():
            raise RunnerError("structured storage lifecycle_class cannot be empty")
        if not self.isolation_tier.strip():
            raise RunnerError("structured storage isolation_tier cannot be empty")

    def namespace_uri(self, domain: DomainConfig) -> str:
        return f"{self.root_uri.rstrip('/')}/{domain.namespace}"


@dataclass(frozen=True)
class ArtifactPayload:
    name: str
    value: object
    producer: ProducerIdentity
    media_type: str = "application/json"
    lineage: LineageGraph | None = None


@dataclass(frozen=True)
class StageContext:
    config: "RunConfig"
    artifacts: Mapping[str, ArtifactReference]


StageHandler = Callable[[StageContext], tuple[ArtifactPayload, ...]]


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    producer: ProducerIdentity
    handler: StageHandler


@dataclass(frozen=True)
class RunConfig:
    scenario_path: Path
    profile_path: Path
    implementation_revision: str = "runner:0.1"
    evaluator_revision: str = "igor-evaluator:0.1"
    domain: DomainConfig = field(default_factory=lambda: DomainConfig("local", "support"))
    storage: StructuredStorageConfig = field(default_factory=lambda: StructuredStorageConfig("local"))

    @classmethod
    def load(
        cls,
        scenario_path: str | Path,
        profile_path: str | Path,
        *,
        domain: DomainConfig | None = None,
        storage: StructuredStorageConfig | None = None,
    ) -> "RunConfig":
        scenario = Path(scenario_path)
        profile = Path(profile_path)
        if not scenario.is_dir():
            raise RunnerError(f"scenario pack does not exist: {scenario}")
        if not profile.is_file():
            raise RunnerError(f"compiler profile does not exist: {profile}")
        try:
            validate_scenario_pack(scenario)
            load_reference_profile(profile)
        except Exception as error:
            raise RunnerError(f"invalid runner input: {error}") from error
        return cls(scenario, profile, domain=domain or DomainConfig("local", "support"), storage=storage or StructuredStorageConfig("local"))

    def identity_links(self) -> RunIdentityLinks:
        scenario = json.loads((self.scenario_path / "scenario.json").read_text(encoding="utf-8"))
        profile = load_reference_profile(self.profile_path)
        return RunIdentityLinks(
            benchmark_contract=f"igor-benchmark-contract:{scenario['compatibility']['benchmark_contract']}",
            scenario_pack=f"{scenario['scenario_id']}:{scenario['version']}",
            scorecard=f"{scenario['scorecard']}:{scenario['version']}",
            implementation=self.implementation_revision,
            source_fixture=scenario["source_fixture_identity"],
            transformation_config=profile.identity,
            evaluator=self.evaluator_revision,
        )


class CompositionRoot:
    """Order stages and assemble a contract-valid run without owning tool logic."""

    def run(self, config: RunConfig, stages: tuple[StageSpec, ...], output: str | Path):
        if not stages:
            raise RunnerError("at least one stage is required")
        if len({stage.stage_id for stage in stages}) != len(stages):
            raise RunnerError("stage IDs must be unique")
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        links = config.identity_links()
        manifest = RunManifest(schema_version="0.1", identity_links=links)
        references: dict[str, ArtifactReference] = {}
        lineage_graphs: list[LineageGraph] = []
        stage_paths: list[str] = []
        for stage in stages:
            before = set(references)
            try:
                payloads = stage.handler(StageContext(config, references))
            except Exception as error:
                raise RunnerError(f"stage {stage.stage_id} failed: {error}") from error
            if len({payload.name for payload in payloads}) != len(payloads):
                raise RunnerError(f"stage {stage.stage_id} produced duplicate artifact names")
            produced: list[str] = []
            for payload in payloads:
                if payload.lineage is not None:
                    lineage_graphs.append(payload.lineage)
                relative = f"artifacts/{payload.name}.json"
                if payload.name in references:
                    raise RunnerError(f"artifact name already produced: {payload.name}")
                data = (canonical_json(payload.value) + "\n").encode("utf-8")
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                reference = ArtifactReference(
                    schema_version="0.1",
                    path=relative,
                    producer=payload.producer,
                    media_type=payload.media_type,
                    sha256=f"sha256:{hashlib.sha256(data).hexdigest()}",
                    size_bytes=len(data),
                )
                references[payload.name] = reference
                produced.append(reference.identity)
            stage_result = StageResult(
                schema_version="0.1",
                stage_id=stage.stage_id,
                status="succeeded",
                producer=stage.producer,
                input_artifact_identities=[references[name].identity for name in sorted(before)],
                output_artifact_identities=produced,
            )
            stage_path = f"stages/{stage.stage_id}.json"
            stage_file = root / stage_path
            stage_file.parent.mkdir(parents=True, exist_ok=True)
            stage_file.write_text(stage_result.model_dump_json(), encoding="utf-8")
            stage_paths.append(stage_path)
        if lineage_graphs:
            lineage = LineageGraph.merge(manifest.identity, tuple(lineage_graphs))
            lineage_relative = "artifacts/lineage.json"
            lineage_data = (canonical_json(lineage) + "\n").encode("utf-8")
            lineage_path = root / lineage_relative
            lineage_path.parent.mkdir(parents=True, exist_ok=True)
            lineage_path.write_bytes(lineage_data)
            stored_lineage = lineage_path.read_bytes()
            lineage_reference = ArtifactReference(schema_version="0.1", path=lineage_relative, producer=ProducerIdentity(name="igor-runner", revision="0.1"), media_type="application/json", sha256=f"sha256:{hashlib.sha256(stored_lineage).hexdigest()}", size_bytes=len(stored_lineage))
            references["lineage"] = lineage_reference
        manifest = manifest.model_copy(update={"stage_result_paths": stage_paths, "lineage_path": "artifacts/lineage.json" if lineage_graphs else None})
        (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        index = {"schema_version": "0.1", "run_identity": manifest.identity, "artifacts": [item.model_dump(mode="json") for item in references.values()]}
        (root / "artifact-index.json").write_text(json.dumps(index, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        try:
            return validate_run_directory(root)
        except Exception as error:
            raise RunnerError(f"runner produced an invalid run: {error}") from error


def _provider_pair(profile):
    embedding_profile = profile.embedding
    completion_profile = profile.completion
    if embedding_profile.provider == "deterministic" and completion_profile.provider == "deterministic":
        provider = DeterministicProviders()
        return embedding_profile, completion_profile, provider, provider
    if embedding_profile.provider == "voyage" and completion_profile.provider in {"deepseek", "gemini", "mistral"}:
        adapters = {
            "deepseek": DeepSeekCompletionAdapter,
            "gemini": GeminiCompletionAdapter,
            "mistral": MistralCompletionAdapter,
        }
        return (embedding_profile, completion_profile, VoyageMultimodalEmbeddingAdapter(),
                adapters[completion_profile.provider]())
    raise RunnerError(
        "unsupported compiler provider composition; expected deterministic/deterministic "
        "or voyage with a qualified completion adapter"
    )


def _retry_policy(profile) -> RetryPolicy:
    attempts = int(profile.embedding.parameters.get("retry_attempts", profile.completion.parameters.get("retry_attempts", 3)))
    backoff = float(profile.embedding.parameters.get("retry_backoff_seconds", profile.completion.parameters.get("retry_backoff_seconds", 0.25)))
    return RetryPolicy(max_attempts=max(1, min(8, attempts)), backoff_seconds=max(0.0, min(60.0, backoff)))


def _load_support_taxonomy(path: str | Path) -> dict:
    taxonomy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("labels"), dict):
        raise RunnerError("support taxonomy must declare a labels mapping")
    labels = taxonomy["labels"]
    for label, definition in labels.items():
        if not isinstance(label, str) or not isinstance(definition, dict) or not definition.get("description"):
            raise RunnerError(f"support taxonomy label {label!r} needs a description")
    return taxonomy


def _support_compilation(records: list[dict], units: dict, store: LanceStore,
                        inventory: DependencyInventory | None = None, profile=None,
                        taxonomy_path: str | Path | None = None):
    """Build the versioned support projection; policy is intentionally scenario-local."""
    schema = SchemaDescriptor(schema_version="0.1", schema_id="support.ticket", revision="1", domain="support",
                              fields=(SchemaField(field_id="record_id", name="record_id", logical_type="string", required=True),
                                      SchemaField(field_id="text", name="text", logical_type="string", required=True)))
    taxonomy = _load_support_taxonomy(taxonomy_path or Path("scenarios/support-huggingface/v0.1/taxonomy.yaml"))
    labels = list(taxonomy["labels"])
    action_by_label = {
        "cancel_transfer": "cancel_transfer", "card_payment_fee_charged": "explain_card_payment_fee",
        "supported_cards_and_currencies": "explain_supported_cards_and_currencies", "visa_or_mastercard": "explain_visa_or_mastercard",
        "lost_or_stolen_phone": "lost_phone_security_steps", "change_pin": "send_pin_change_steps",
        "beneficiary_not_allowed": "explain_beneficiary_restriction", "getting_spare_card": "explain_spare_card_request",
        "card_about_to_expire": "explain_card_expiry", "apple_pay_or_google_pay": "explain_apple_pay_or_google_pay",
    }
    taxonomy_description = "; ".join(f"{label}: {definition['description']}" for label, definition in taxonomy["labels"].items())
    taxonomy_examples = [
        {"text": example, "intent_label": label, "recommended_action": action_by_label[label]}
        for label, definition in taxonomy["labels"].items()
        for example in definition.get("examples", [])
    ]
    enrichment_schema = SchemaDescriptor(
        schema_version="0.1", schema_id="support.enrichment", revision="1", domain="support",
        semantic_annotations={
            "intent_label.allowed_values": ",".join(labels + ["unclassified"]),
            "urgency.allowed_values": "high,normal",
            "recommended_action.allowed_values": ",".join(action_by_label.values()) + ",request_more_information",
        },
        fields=(
            SchemaField(field_id="intent_label", name="intent_label", logical_type="string", required=True),
            SchemaField(field_id="urgency", name="urgency", logical_type="string", required=True),
            SchemaField(field_id="recommended_action", name="recommended_action", logical_type="string", required=True),
        ),
        json_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "description": f"Classify banking support text using taxonomy {taxonomy['taxonomy_id']}. {taxonomy_description} Return exactly one enum value and the matching action. Precedence: {', '.join(taxonomy.get('precedence', []))}.",
            "examples": taxonomy_examples,
            "required": ["intent_label", "urgency", "recommended_action"],
            "properties": {
                "intent_label": {"type": "string", "enum": labels + ["unclassified"], "description": "Choose one label. Label definitions: " + taxonomy_description + " Precedence: " + ", ".join(taxonomy.get("precedence", [])) + ". Generic top-up means supported_cards_and_currencies; Apple/Google must be explicit. A new/replacement card without an explicit extra-card request means card_about_to_expire."},
                "urgency": {"type": "string", "enum": ["high", "normal"]},
                "recommended_action": {"type": "string", "enum": list(action_by_label.values()) + ["request_more_information"]},
            },
        },
    )
    enrichment_recipe = EnrichmentRecipe(
        recipe_id="support.classify-and-route", revision="2",
        accepted_representation_types=("text",), accepted_media_types=("text/plain",),
        output_schema_identity=enrichment_schema.identity,
        prompt_version="support-enrichment-v3", taxonomy_version=str(taxonomy["taxonomy_id"]),
    )
    if profile is None:
        raise RunnerError("support compilation requires a resolved compiler profile")
    embedding_profile, completion_profile, embedding_provider, enrichment_provider = _provider_pair(profile)
    snapshots, representations, evidences, derivations = [], [], [], []
    by_record = {row["record_id"]: row for row in records}
    for unit in sorted(units["context_units"], key=lambda row: row["context_unit_id"]):
        row = by_record[unit["record_id"]]
        # Only declared source concepts participate in semantic identity; connector
        # metadata may change without invalidating text-derived outputs.
        identity_row = {"record_id": row["record_id"], "text": row["text"]}
        content_hash = hashlib.sha256(canonical_json(identity_row).encode()).hexdigest()
        snapshot = SourceSnapshot(ir_version="0.1", source_system="support-fixture", source_key=row["record_id"],
                                   observed_at="2026-08-12T00:00:00Z", content_ref=f"fixture://support/{row['record_id']}",
                                   content_sha256=content_hash, media_type="application/json", schema_ref=schema)
        representation = Representation(ir_version="0.1", representation_type="text", schema_ref=schema,
                                         source_snapshot_ids=(snapshot.identity,), payload=row["text"], metadata={"context_unit_id": unit["context_unit_id"]})
        derivation = Derivation(ir_version="0.1", operation="support.representation.v1", input_identities=(snapshot.identity,),
                                output_identity=representation.identity, code_revision="runner:0.1", configuration_identity="support-policy:0.1")
        evidence = Evidence(ir_version="0.1", subject_identity=representation.identity, source_snapshot_id=snapshot.identity,
                            locator=f"record:{row['record_id']}", excerpt=row["text"], observed_at=snapshot.observed_at)
        snapshots.append(snapshot); representations.append(representation); derivations.append(derivation); evidences.append(evidence)
    dimension = embedding_profile.parameters.get("dimensions", 8)
    space = EmbeddingSpace(ir_version="0.1", provider=embedding_profile.provider, model=embedding_profile.model,
                           model_revision=embedding_profile.revision, dimension=dimension,
                           dtype="float32" if embedding_profile.provider != "deterministic" else "float64",
                           metric="cosine", normalized=False, input_schema_identity=schema.identity)
    embedding_ids = tuple(stable_identity({"embedding": item.identity, "space": space.identity}) for item in representations)
    enrichment_ids = tuple(stable_identity({
        "enrichment": item.identity,
        "profile": completion_profile.identity,
        "schema": enrichment_schema.identity,
        "prompt_version": "support-enrichment-v3",
        "taxonomy_version": str(taxonomy["taxonomy_id"]),
    }) for item in representations)
    output_ids = tuple(item.identity for item in representations)
    all_output_ids = output_ids + embedding_ids + enrichment_ids
    qualified = tuple(QualifiedRepresentation(
        representation=item,
        parts=(ContentPart(kind="text", media_type="text/plain", text=str(item.payload),
                           content_sha256=stable_identity(str(item.payload))),),
        evidence=(evidences[i],),
    )
                      for i, item in enumerate(representations))
    as_of = datetime(2026, 8, 12, tzinfo=timezone.utc)
    policy_id, policy_revision = "support.authority-temporal", "1"
    authority_assertions = tuple(
        AuthorityAssertion(
            assertion_identity=stable_identity({"authority": item.identity, "policy": policy_id, "revision": policy_revision}),
            subject_identity=item.identity, issuer_ref="support-fixture", scope_ref="support-context",
            evidence_identities=(evidences[i].identity,),
        ) for i, item in enumerate(representations)
    )
    temporal_assertions = tuple(
        TemporalAssertion(
            assertion_identity=stable_identity({"temporal": item.identity, "policy": policy_id, "revision": policy_revision}),
            subject_identity=item.identity, observed_at=as_of, effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            effective_until=None, evidence_identities=(evidences[i].identity,),
        ) for i, item in enumerate(representations)
    )
    resolution_input = {
        "retrieval_role": "support_messages",
        "request_identity": stable_identity({"support-resolution": records}),
        "task_id": "support-intent-v0.1",
        "task_schema_revision": "support-task:1",
        "as_of": as_of,
        "evidence": tuple(evidences),
        "authority_assertions": authority_assertions,
        "temporal_assertions": temporal_assertions,
        "candidates": tuple(
            {
                "candidate_identity": stable_identity({"candidate": item.identity}),
                "context_identity": item.identity,
                "evidence_identities": (evidences[i].identity,),
                "authority_assertion_ids": (authority_assertions[i].assertion_identity,),
                "temporal_assertion_ids": (temporal_assertions[i].assertion_identity,),
            } for i, item in enumerate(representations)
        ),
    }
    derivation_inputs = tuple(
        {"role": "support_message", "input_identities": derivation.input_identities, "output_identity": derivation.output_identity}
        for derivation in derivations
    ) + tuple(
        {
            "role": "support_embedding",
            "input_identities": (item.identity,),
            "output_identity": embedding_ids[i],
            "configuration_identity": space.identity,
            "model_revision": embedding_profile.revision,
        }
        for i, item in enumerate(representations)
    ) + tuple(
        {
            "role": "support_enrichment",
            "input_identities": (item.identity,),
            "output_identity": enrichment_ids[i],
            "configuration_identity": profile.identity,
            "model_revision": completion_profile.revision,
        }
        for i, item in enumerate(representations)
    )
    semantic_records = [{"record_id": row["record_id"], "text": row["text"]} for row in records]
    run_identity = stable_identity({"support": semantic_records})
    task_id = "support-intent-v0.1"
    inv = inventory or DependencyInventory(known_output_identities=tuple(s.identity for s in snapshots), changed_identities=())
    # The retrieval table is Lance-owned; the support stage only supplies rows.
    observations = [
        _source_snapshot_observation(
            snapshot,
            object_role="support_ticket",
            snapshot_role="support_ticket_snapshot",
            display_name=f"Support ticket {snapshot.source_key}",
            preview=by_record[snapshot.source_key]["text"][:160],
        )
        for snapshot in snapshots
    ]
    bootstrap_lifecycle = compile_context_model_lifecycle(
        store=store,
        manifest=_support_context_manifest(),
        observations=observations,
        inventory=inv,
        ports=CompilerPorts(
            store=LanceContextOutputStore(store),
            embedding=embedding_provider,
            enrichment=enrichment_provider,
            retry=_retry_policy(profile),
        ),
        run_identity=run_identity,
        task_id=task_id,
        embedding_profile=embedding_profile,
        completion_profile=completion_profile,
        embedding_space=space,
        code_revision="runner:0.1",
        derivation_inputs=derivation_inputs,
        embedding_inputs=qualified,
        completion_inputs=qualified,
        completion_output_schema=enrichment_schema,
        enrichment_recipe=enrichment_recipe,
        prompt_version="support-enrichment-v3",
        taxonomy_version=str(taxonomy["taxonomy_id"]),
    )
    result = bootstrap_lifecycle.compilation
    if result is None:
        raise RunnerError("support Context Model lifecycle did not execute bootstrap compiler request")
    output_store = LanceContextOutputStore(store)
    embedding_values = [result.semantic_outputs.get(identity) or Embedding.model_validate(output_store.read(identity))
                        for identity in embedding_ids]
    enrichment_values = [
        result.semantic_outputs.get(identity) or EnrichmentOutput.model_validate(output_store.read(identity))
        for identity in enrichment_ids
    ]
    representation_outputs = []
    for item, vector, output_identity in zip(representations, embedding_values, output_ids, strict=True):
        active = item.metadata.get("context_unit_id") != units["context_units"][-1]["context_unit_id"]
        representation_outputs.append({
            "identity": output_identity,
            "role": "support_message",
            "kind": "context_representation",
            "object_kind": "support_message",
            "derived_from": list(item.source_snapshot_ids),
            "schema_id": schema.schema_id,
            "schema_revision": schema.revision,
            "display_name": str(item.metadata.get("context_unit_id", output_identity)),
            "payload": item.model_dump(mode="json"),
            "vector": vector.vector,
            "text": str(item.payload),
            "source_system": "huggingface.support-ticket-text",
            "source_key": next(
                unit["record_id"]
                for unit in units["context_units"]
                if unit["context_unit_id"] == item.metadata.get("context_unit_id")
            ),
            "media_type": "text/plain",
            "content_ref": f"hf://support-ticket-text/{item.metadata.get('context_unit_id')}",
            "preview": str(item.payload)[:160],
            "status": "active" if active else "inactive",
            "authority_status": "authoritative" if active else "rejected",
            "temporal_status": "valid" if active else "stale",
            "policy_id": "support.authority-temporal:1",
            "as_of": "2026-08-12T00:00:00Z",
        })
    semantic_outputs = list(representation_outputs)
    semantic_outputs.extend({
        "identity": identity,
        "role": "support_embedding",
        "kind": "semantic_derivation",
        "derived_from": [representation.identity],
        "schema_id": "support.embedding",
        "schema_revision": "1",
        "display_name": f"{representation.metadata.get('context_unit_id')} embedding",
        "payload": embedding.model_dump(mode="json"),
        "value": canonical_json(embedding.model_dump(mode="json")),
        "status": "active",
    } for identity, representation, embedding in zip(embedding_ids, representations, embedding_values, strict=True))
    semantic_outputs.extend({
        "identity": identity,
        "role": "support_enrichment",
        "kind": "semantic_derivation",
        "derived_from": [representation.identity],
        "schema_id": enrichment_schema.schema_id,
        "schema_revision": enrichment_schema.revision,
        "display_name": f"{representation.metadata.get('context_unit_id')} enrichment",
        "payload": enrichment.model_dump(mode="json") if hasattr(enrichment, "model_dump") else enrichment,
        "value": canonical_json(enrichment.model_dump(mode="json") if hasattr(enrichment, "model_dump") else enrichment),
        "status": "active",
    } for identity, representation, enrichment in zip(enrichment_ids, representations, enrichment_values, strict=True))
    inv = inv.model_copy(update={"valid_output_identities": all_output_ids})
    lifecycle = compile_context_model_lifecycle(
        store=store,
        manifest=_support_context_manifest(),
        observations=observations,
        inventory=inv,
        ports=CompilerPorts(
            store=LanceContextOutputStore(store),
            embedding=embedding_provider,
            enrichment=enrichment_provider,
            retrieval=LanceRetrievalAdapter(store),
            resolution=_SupportResolution(),
            retry=_retry_policy(profile),
        ),
        outputs=semantic_outputs,
        run_identity=run_identity,
        task_id=task_id,
        embedding_profile=embedding_profile,
        completion_profile=completion_profile,
        embedding_space=space,
        code_revision="runner:0.1",
        budget_tokens=12000,
        derivation_inputs=derivation_inputs,
        embedding_inputs=qualified,
        completion_inputs=qualified,
        completion_output_schema=enrichment_schema,
        enrichment_recipe=enrichment_recipe,
        prompt_version="support-enrichment-v3",
        taxonomy_version=str(taxonomy["taxonomy_id"]),
        retrieval_inputs={
            "support_messages": {
                "query_id": stable_identity({"query": "support"}),
                "vector": tuple(embedding_values[0].vector),
                "limit": max(1000, len(representations) * 2),
            },
        },
        resolution_inputs=(resolution_input,),
        required_context_identities=output_ids,
    )
    result = lifecycle.compilation
    if result is None:
        raise RunnerError("support Context Model lifecycle did not execute compiler request")
    resolution_record = result.artifact_records.get("resolution_request")
    if not resolution_record:
        raise RunnerError("support Context Model lifecycle did not emit a resolution request artifact")
    lifecycle_retrieval = _support_lifecycle_retrieval_rows(resolution_record)
    packages = []
    if result.package_identity:
        package_items = [
            item for item in store.read("package_items").to_pylist()
            if item["package_identity"] == result.package_identity
        ]
        packages.append({"identity": result.package_identity, "task_id": task_id, "items": package_items})
    return (
        result, snapshots, representations, evidences, derivations, embedding_values,
        enrichment_values, packages, lifecycle_retrieval,
    )


def _support_lifecycle_retrieval_rows(resolution_record: dict) -> tuple[dict, ...]:
    common = {
        "retrieval_role": "support_messages",
        "request_identity": resolution_record["request_identity"],
        "task_id": resolution_record["task_id"],
        "task_schema_revision": resolution_record["task_schema_revision"],
        "as_of": str(resolution_record["as_of"]),
    }
    rows = []
    evidence = tuple(resolution_record.get("evidence", ()))
    authority = tuple(resolution_record.get("authority_assertions", ()))
    temporal = tuple(resolution_record.get("temporal_assertions", ()))
    for index, candidate in enumerate(resolution_record.get("candidates", ())):
        row = {**common, **dict(candidate)}
        retrieval = row.pop("retrieval", None)
        if isinstance(retrieval, dict):
            row["retrieval_identity"] = str(retrieval.get("result_identity", ""))
            score = retrieval.get("score", 0.0)
            row["score"] = float(score if score is not None else 0.0)
            row["rank"] = int(retrieval.get("rank", index) or 0)
            row["facts"] = dict(retrieval.get("facts", {}))
        else:
            row.setdefault("retrieval_identity", stable_identity({
                "support-retrieval": row.get("context_identity", row.get("candidate_identity", index)),
                "rank": index,
            }))
            row.setdefault("score", 0.0)
            row.setdefault("rank", index)
        if index == 0:
            row["evidence"] = evidence
            row["authority_assertions"] = authority
            row["temporal_assertions"] = temporal
        rows.append(row)
    return tuple(rows)


class _SupportResolution:
    """Qualified support-domain policy; all authority/temporal meaning stays here."""

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        if (request.policy_id, request.policy_revision) != ("support.authority-temporal", "1"):
            raise CompilerFailure("unsupported_operation", "unknown support authority/temporal policy revision")
        authority = {item.assertion_identity: item for item in request.authority_assertions}
        temporal = {item.assertion_identity: item for item in request.temporal_assertions}
        decisions = []
        for candidate in sorted(request.candidates, key=lambda item: item.candidate_identity):
            authority_ids = tuple(candidate.authority_assertion_ids)
            temporal_ids = tuple(candidate.temporal_assertion_ids)
            authority_items = [authority[item] for item in authority_ids if item in authority]
            auth_ok = bool(authority_items) and len(authority_items) == len(authority_ids)
            authority_conflict = len({item.issuer_ref for item in authority_items}) > 1
            temporal_items = [temporal[item] for item in temporal_ids if item in temporal]
            valid_temporal = False
            temporal_basis = "unknown"
            reason = "support-policy:missing-basis"
            if temporal_items:
                value = temporal_items[0]
                if value.effective_from is None or (value.effective_until is not None and value.effective_until <= request.as_of):
                    temporal_basis, reason = "expired", "support-policy:expired"
                elif value.effective_from > request.as_of:
                    temporal_basis, reason = "future", "support-policy:future"
                elif value.effective_from <= request.as_of and (value.effective_until is None or request.as_of < value.effective_until):
                    valid_temporal, temporal_basis, reason = True, "valid", "support-policy:authoritative-valid"
            if authority_conflict:
                outcome, authority_basis = "conflict", "conflicting"
                reason = "support-policy:conflicting-authority"
            elif auth_ok and valid_temporal:
                outcome, authority_basis = "selected", "satisfied"
            elif not auth_ok:
                outcome, authority_basis = "abstained", "unknown"
            else:
                outcome, authority_basis = "rejected", "satisfied"
            decision_kwargs = {
                "candidate_identity": candidate.candidate_identity,
                "outcome": outcome,
                "reason_code": reason,
                "authority_basis_ids": authority_ids,
                "temporal_basis_ids": temporal_ids,
                "authority_basis": authority_basis,
                "temporal_basis": temporal_basis,
                "as_of": request.as_of,
                "policy_id": request.policy_id,
                "policy_revision": request.policy_revision,
            }
            if outcome == "conflict":
                decision_kwargs["conflict_set_id"] = stable_identity({
                    "conflict": candidate.candidate_identity,
                    "authority": authority_ids,
                })
            decisions.append(ResolutionDecision(**decision_kwargs))
        provisional = ResolutionResult(request_identity=request.request_identity, resolution_identity=request.identity, decisions=tuple(decisions))
        return provisional


def _support_stages(store: LanceStore, inventory: DependencyInventory | None = None, records_override: list[dict] | None = None, source_transfer_bytes: int = 0) -> tuple[StageSpec, ...]:
    producer = ProducerIdentity(name="igor-runner", revision="0.1")
    showcase_state: dict[str, object] = {}

    def units_for(source_records: list[dict], configured: dict) -> dict:
        if records_override is None:
            return configured
        return {"schema_version": "0.1", "context_units": [
            {"context_unit_id": f"ctx:support:{index:03d}", "record_id": row["record_id"]}
            for index, row in enumerate(source_records, start=1)
        ]}

    def support_source_contract() -> tuple[ContextSourceContract, ConnectorBinding]:
        contract = ContextSourceContract(
            contract_id="support.huggingface-ticket-text", revision="1", domain="support",
            resources=(SourceResourceContract(
                resource_id="tickets", resource_kind="dataset-row", identity_concepts=("ticket_id",),
                fields=(
                    SourceFieldRequirement(concept_id="ticket_id", logical_type="string", required=True),
                    SourceFieldRequirement(concept_id="text", logical_type="string", required=True),
                    SourceFieldRequirement(concept_id="channel", logical_type="string", required=True),
                ), accepted_media_types=("text/csv",), selection={"revision": "immutable"},
                change_mode="snapshot", deletion_semantics="ignore",
            ),),
        )
        binding = ConnectorBinding(
            binding_id="huggingface.support-ticket-text", revision="1",
            source_contract_identity=contract.identity, connector="huggingface",
            deployment_ref="datasets-server/immutable-revision",
            resources=(ConnectorResourceBinding(
                resource_id="tickets", source_resource="train.csv",
                fields=(ConnectorFieldBinding(concept_id="ticket_id", source_field="record_id"),
                        ConnectorFieldBinding(concept_id="text", source_field="text"),
                        ConnectorFieldBinding(concept_id="channel", source_field="channel")),
            ),),
        )
        binding.validate_against(contract)
        return contract, binding

    def prepare(context: StageContext) -> tuple[ArtifactPayload, ...]:
        root = context.config.scenario_path
        source = json.loads((root / "fixtures/source.json").read_text(encoding="utf-8"))
        if records_override is not None:
            source["records"] = records_override
        units = units_for(source["records"], json.loads((root / "fixtures/context-units.json").read_text(encoding="utf-8")))
        contract, binding = support_source_contract()
        raw = [{"_resource_id": "tickets", "_media_type": "text/csv",
                "_content_ref": f"hf://immutable/support-ticket-text/{row['record_id']}",
                "_content_sha256": "sha256:" + hashlib.sha256(canonical_json(row).encode()).hexdigest(),
                "_observed_at": "2026-08-12T00:00:00Z", **row} for row in source["records"]]
        ingested = ingest_bound_records(
            raw,
            IngestConfig(
                context.config.domain.environment,
                context.config.domain.domain,
                context.config.storage.root_uri,
                context.config.domain.namespace,
            ),
            contract,
            binding,
        )
        if "events" in store.names():
            store.replace("events", list(ingested.records))
        else:
            store.create("events", list(ingested.records))
        return (ArtifactPayload("context", {"source": source, "context_units": units, "ingestion": ingested.as_dict(),
                                             "source_contract": contract.model_dump(mode="json"),
                                             "connector_binding": binding.model_dump(mode="json")}, producer,
                                      lineage=ingested.lineage(stable_identity(context.config.identity_links()))),)

    def relational(context: StageContext) -> tuple[ArtifactPayload, ...]:
        model_manifest = ModelManifest("0.1", (ModelSpec("source_summary", "SELECT COUNT(*) AS record_count FROM events"),))
        result = RelationalRunner(AnalyticalEngine(store), store).run(model_manifest)
        return (ArtifactPayload("provenance", {"materialized": result.materialized, "lineage": result.lineage, "rows": result.rows}, producer, lineage=result.lineage_graph(stable_identity(context.config.identity_links()), model_manifest)),)

    def outputs(context: StageContext) -> tuple[ArtifactPayload, ...]:
        source = json.loads((context.config.scenario_path / "fixtures/source.json").read_text(encoding="utf-8"))
        if records_override is not None:
            source["records"] = records_override
        units = units_for(source["records"], json.loads((context.config.scenario_path / "fixtures/context-units.json").read_text(encoding="utf-8")))
        profile = load_reference_profile(context.config.profile_path)
        started = perf_counter()
        (
            result, snapshots, representations, evidences, derivations, embeddings,
            enrichments, packages, lifecycle_retrieval,
        ) = _support_compilation(source["records"], units, store, inventory, profile, context.config.scenario_path / "taxonomy.yaml")
        showcase_state.update({
            "result": result, "snapshots": snapshots, "representations": representations,
            "evidences": evidences, "derivations": derivations, "embeddings": embeddings, "enrichments": enrichments,
            "packages": packages, "lifecycle_retrieval": lifecycle_retrieval,
            "records": source["records"], "units": units,
            "compilation_seconds": perf_counter() - started,
        })
        predictions = []
        records_by_id = {row["record_id"]: row for row in source["records"]}
        representations_by_context = {
            item.metadata["context_unit_id"]: item
            for item in representations
        }
        enrichments_by_representation = {
            value.representation_identity: value
            for value in enrichments
        }
        for unit in sorted(units["context_units"], key=lambda item: item["context_unit_id"]):
            representation = representations_by_context[unit["context_unit_id"]]
            enrichment = enrichments_by_representation.get(representation.identity)
            if enrichment is None:
                raise RunnerError(f"missing enrichment for {unit['context_unit_id']}")
            payload = enrichment.payload
            predictions.append({
                "context_unit_id": unit["context_unit_id"],
                "record_id": unit["record_id"],
                "intent_label": payload["intent_label"],
                "urgency": payload["urgency"],
                "recommended_action": payload["recommended_action"],
                "source_text": records_by_id[unit["record_id"]]["text"],
                "enrichment_identity": enrichment.identity,
                "source_label": records_by_id[unit["record_id"]].get("label", ""),
            })
        run_id = stable_identity(context.config.identity_links())
        lineage_nodes = {}
        def node(node_type, key):
            value = LineageNode(schema_version="0.1", node_type=node_type, key=key, run_identity=run_id,
                                producer=producer, configuration_revision="support-context-e2e:0.1")
            lineage_nodes[value.identity] = value
            return value
        lineage_edges = []
        for derivation in derivations:
            output_node = node("derivation_output", derivation.output_identity)
            for source_id in derivation.input_identities:
                source_node = node("source_record", source_id)
                lineage_edges.append(LineageEdge(schema_version="0.1", edge_type="derived_from", source_node_id=output_node.identity, target_node_id=source_node.identity, run_identity=run_id, producer=producer, configuration_revision="support-context-e2e:0.1"))
        compiler_lineage = LineageGraph(schema_version="0.1", run_identity=run_id, nodes=tuple(lineage_nodes.values()), edges=tuple(lineage_edges))
        return (
            ArtifactPayload("semantic", {"source_snapshots": [item.model_dump(mode="json") for item in snapshots], "representations": [item.model_dump(mode="json") for item in representations], "evidences": [item.model_dump(mode="json") for item in evidences], "embeddings": [item.model_dump(mode="json") for item in embeddings], "enrichments": [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in enrichments]}, producer, lineage=compiler_lineage),
            ArtifactPayload("retrieval", {"facts": [item.model_dump(mode="json") for item in result.retrieval_facts], "resolution": [item.model_dump(mode="json") for item in result.resolution]}, producer),
            ArtifactPayload("context/resolution-request", result.artifact_records.get("resolution_request", {}), producer),
            ArtifactPayload("context/resolution-decisions", result.artifact_records.get("resolution_result", {}), producer),
            ArtifactPayload("metrics", {"predictions": predictions}, producer),
            ArtifactPayload("context/compiler-plan", result.artifact_records["plan"], producer),
            ArtifactPayload("context/derivations", {"records": result.artifact_records["derivations"]}, producer),
            ArtifactPayload("context/context-packages", {"packages": packages}, producer),
            ArtifactPayload("context/compiler-execution", {
                "plan_identity": result.plan_identity,
                "attempts": dict(result.attempts),
                "provider_mode": "deterministic" if profile.embedding.provider == "deterministic" else "live",
                "provider_profiles": {"embedding": profile.embedding.identity, "completion": profile.completion.identity},
            }, producer),
        )

    def showcase(context: StageContext) -> tuple[ArtifactPayload, ...]:
        profile = load_reference_profile(context.config.profile_path)
        result = showcase_state["result"]
        snapshots = showcase_state["snapshots"]
        representations = showcase_state["representations"]
        evidences = showcase_state["evidences"]
        records = showcase_state["records"]
        units = showcase_state["units"]
        enrichments = showcase_state["enrichments"]
        embeddings = showcase_state["embeddings"]
        run_id = stable_identity(context.config.identity_links())

        # Materialize portable relations through the existing DataFusion/Lance owner.
        relation_rows = {
            "support_tickets": [{"context_unit_id": unit["context_unit_id"], "record_id": unit["record_id"], "text": next(row["text"] for row in records if row["record_id"] == unit["record_id"])} for unit in units["context_units"]],
            "support_enrichments": [{"representation_identity": item.representation_identity, "intent_label": item.payload.get("intent_label", ""), "recommended_action": item.payload.get("recommended_action", "")} for item in enrichments],
            "retrieval_candidates": [item.model_dump(mode="json") for item in result.retrieval_facts],
            "package_facts": [{"package_identity": package["identity"], "task_id": package["task_id"], "item_count": len(package["items"])} for package in showcase_state["packages"]],
        }
        for name, rows in relation_rows.items():
            if rows:
                if name in store.names():
                    store.replace(name, rows)
                else:
                    store.create(name, rows)

        records_by_id = {row["record_id"]: row for row in records}
        record_by_context_unit = {
            unit["context_unit_id"]: records_by_id[unit["record_id"]]
            for unit in units["context_units"]
        }
        observations = [
            _source_snapshot_observation(
                snapshot,
                object_role="support_ticket",
                snapshot_role="support_ticket_snapshot",
                display_name=f"Support ticket {snapshot.source_key}",
                preview=records_by_id[snapshot.source_key]["text"][:160],
            )
            for snapshot in snapshots
        ]
        context_outputs = []
        for unit, representation, embedding in zip(units["context_units"], representations, embeddings, strict=True):
            row = records_by_id[unit["record_id"]]
            active = unit["record_id"] != records[-1]["record_id"]
            context_outputs.append({
                "identity": representation.identity,
                "role": "support_message",
                "kind": "context_representation",
                "object_kind": "support_message",
                "derived_from": list(representation.source_snapshot_ids),
                "schema_id": "support.ticket",
                "schema_revision": "1",
                "display_name": unit["context_unit_id"],
                "payload": representation.model_dump(mode="json"),
                "vector": embedding.vector,
                "text": str(representation.payload),
                "source_system": "huggingface.support-ticket-text",
                "source_key": unit["record_id"],
                "media_type": "text/plain",
                "content_ref": f"hf://support-ticket-text/{unit['record_id']}",
                "preview": row["text"][:160],
                "status": "active" if active else "inactive",
                "authority_status": "authoritative" if active else "rejected",
                "temporal_status": "valid" if active else "stale",
                "policy_id": "support.authority-temporal:1",
                "as_of": "2026-08-12T00:00:00Z",
            })
        context_outputs.extend({
            "identity": embedding.identity,
            "role": "support_embedding",
            "kind": "semantic_derivation",
            "derived_from": [embedding.representation_identity],
            "schema_id": "support.embedding",
            "schema_revision": "1",
            "display_name": f"{record_by_context_unit[representation.metadata['context_unit_id']]['record_id']} embedding",
            "payload": embedding.model_dump(mode="json"),
            "value": canonical_json(embedding.model_dump(mode="json")),
            "status": "active",
        } for representation, embedding in zip(representations, embeddings, strict=True))
        context_outputs.extend({
            "identity": enrichment.identity,
            "role": "support_enrichment",
            "kind": "semantic_derivation",
            "derived_from": [enrichment.representation_identity],
            "schema_id": "support.enrichment",
            "schema_revision": "1",
            "display_name": f"{record_by_context_unit[representation.metadata['context_unit_id']]['record_id']} enrichment",
            "payload": enrichment.model_dump(mode="json") if hasattr(enrichment, "model_dump") else enrichment,
            "value": canonical_json(enrichment.model_dump(mode="json") if hasattr(enrichment, "model_dump") else enrichment),
            "status": "active",
        } for representation, enrichment in zip(representations, enrichments, strict=True))
        resolution_results = (result.resolution_result,) if result.resolution_result is not None else ()
        lifecycle = materialize_context_model_lifecycle(
            store=store,
            manifest=_support_context_manifest(),
            observations=observations,
            outputs=context_outputs,
            retrieval=showcase_state["lifecycle_retrieval"],
            resolution_results=resolution_results,
        )
        materialized = lifecycle.materialization
        context_catalog = [
            row for row in materialized.relations["context_catalog"]
            if row["physical_relation"] == "context_outputs" and row["object_kind"] == "support_message"
        ]
        # The semantic-SQL artifact is deliberately assembled here, after the
        # support compiler has produced its real embeddings, so it proves the
        # pinned 100-row path rather than the bounded synthetic qualification.
        semantic_adapter = LanceRetrievalAdapter(store)
        index_facts = semantic_adapter.ensure_vector_index()
        query_text = "why was my card charged"
        query_vector_ref = stable_identity({
            "query": query_text,
            "embedding_space": embeddings[0].space_identity,
        })
        semantic_sql = (
            "SELECT canonical_identity, display_name, context_model_id, object_kind, "
            "source_system, source_key, media_type, evidence_preview, content_ref, "
            "semantic_score, authority_status, temporal_status, policy_id, as_of "
            "FROM semantic_search('context_catalog', 'why was my card charged', 'vector', 10, '"
            + query_vector_ref
            + "', 'status = ''active''') "
            "WHERE context_model_id = 'support' AND status = 'active' "
            "ORDER BY semantic_score DESC, canonical_identity ASC"
        )
        query_started = perf_counter()
        semantic_result = lifecycle.query_standard_sql(
            semantic_sql,
            table_name="context_catalog",
            query_vectors={query_vector_ref: tuple(embeddings[0].vector)},
            retrieval=semantic_adapter,
        )
        semantic_query_ms = round((perf_counter() - query_started) * 1000, 2)
        semantic_rows = semantic_result.to_pylist()
        semantic_lineage = [{
            "canonical_identity": row["canonical_identity"],
            "source_key": row["source_key"],
            "content_ref": row["content_ref"],
            "media_type": row["media_type"],
        } for row in semantic_rows]
        semantic_artifact = {
            "qualification": "support-huggingface-semantic-sql-v0",
            "bounded_records": len(context_catalog),
            "query": semantic_sql,
            "rows": semantic_rows,
            "relation_facts": {
                "context_catalog": len(context_catalog),
                "context_retrieval": len(store.read("context_retrieval").to_pylist()),
                "candidate_count": len(store.read("context_retrieval").to_pylist()),
                "returned_count": len(semantic_rows),
            },
            "timing_ms": {"semantic_sql": semantic_query_ms},
            "lance_tables": sorted(store.names()),
            "modality_scope": ["text/plain"],
            "index_facts": {"retrieval": "Lance native vector search", "ann_index": index_facts},
            "query_vector": {
                "ref": query_vector_ref,
                "embedding_space": embeddings[0].space_identity,
                "dimension": len(embeddings[0].vector),
                "metric": "cosine",
            },
            "runtime_trace": {
                "engine": "DataFusion",
                "table_function": "semantic_search",
                "adapter": semantic_adapter.last_execution.get("adapter"),
                "prefilter": semantic_adapter.last_execution.get("prefilter"),
                "plan": semantic_adapter.last_execution.get("plan", ""),
            },
            "behavioral_cases": {
                "predicate": "context_model_id = 'support' AND status = 'active'",
                "excluded_inactive": records[-1]["record_id"],
            },
            "lineage": semantic_lineage,
        }
        semantic_unsigned = json.dumps(semantic_artifact, sort_keys=True, separators=(",", ":"))
        semantic_artifact["artifact_sha256"] = hashlib.sha256(semantic_unsigned.encode("utf-8")).hexdigest()
        sql_manifest = ModelManifest("support-hf-sql-0.1", (
            ModelSpec("ticket_concepts", "SELECT context_unit_id, record_id FROM support_tickets ORDER BY context_unit_id"),
            ModelSpec("enrichment_facts", "SELECT representation_identity, intent_label, recommended_action FROM support_enrichments ORDER BY representation_identity"),
            ModelSpec("retrieval_facts", "SELECT context_identity, rank, score, facts FROM retrieval_candidates ORDER BY rank, context_identity"),
            ModelSpec("resolution_facts", "SELECT candidate_identity, outcome, reason_code, authority_basis, temporal_basis FROM resolution_decisions ORDER BY candidate_identity"),
            ModelSpec("package_facts", "SELECT package_identity, task_id, item_count FROM package_facts ORDER BY package_identity"),
        ))
        sql_result = RelationalRunner(AnalyticalEngine(store), store).run(sql_manifest)

        as_of = datetime.fromisoformat(str(records[0].get("_as_of", "2026-08-12T00:00:00+00:00")))
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        subject = representations[0].identity
        evidence = evidences[0]
        def auth(issuer: str) -> AuthorityAssertion:
            return AuthorityAssertion(assertion_identity=stable_identity({"showcase-auth": issuer}), subject_identity=subject, issuer_ref=issuer, scope_ref="support-context", evidence_identities=(evidence.identity,))
        def temporal(label: str, start: datetime, end: datetime | None) -> TemporalAssertion:
            return TemporalAssertion(assertion_identity=stable_identity({"showcase-time": label}), subject_identity=subject, observed_at=as_of, effective_from=start, effective_until=end, evidence_identities=(evidence.identity,))
        auth_primary, auth_alt = auth("support-kb-primary"), auth("support-kb-alternate")
        policy_now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        time_valid = temporal("valid", policy_now - timedelta(days=30), None)
        time_stale = temporal("stale", policy_now - timedelta(days=90), policy_now - timedelta(days=1))
        request_identity = stable_identity("support-hf-resolution-showcase")
        showcase_cases = (
            ("showcase-winner", (auth_primary.assertion_identity,), (time_valid.assertion_identity,)),
            ("showcase-nonauthoritative", (), (time_valid.assertion_identity,)),
            ("showcase-stale", (auth_primary.assertion_identity,), (time_stale.assertion_identity,)),
            ("showcase-conflict", (auth_primary.assertion_identity, auth_alt.assertion_identity), (time_valid.assertion_identity,)),
            ("showcase-abstain", (), ()),
        )
        showcase_retrieval = []
        for index, (case, authority_ids, temporal_ids) in enumerate(showcase_cases):
            row = {
                "retrieval_role": "support_messages",
                "request_identity": request_identity,
                "task_id": "support-resolution-showcase-v0.1",
                "task_schema_revision": "1",
                "as_of": as_of.isoformat(),
                "candidate_identity": stable_identity(case),
                "context_identity": subject,
                "retrieval_identity": stable_identity({"support-showcase-retrieval": case}),
                "rank": index,
                "score": 1.0 - (index / 10),
                "facts": {"case": case},
                "evidence_identities": (evidence.identity,),
                "authority_assertion_ids": authority_ids,
                "temporal_assertion_ids": temporal_ids,
                "evidence_locator": evidence.locator,
            }
            if index == 0:
                row["evidence"] = (evidence.model_dump(mode="json"),)
                row["authority_assertions"] = (
                    auth_primary.model_dump(mode="json"),
                    auth_alt.model_dump(mode="json"),
                )
                row["temporal_assertions"] = (
                    time_valid.model_dump(mode="json"),
                    time_stale.model_dump(mode="json"),
                )
            showcase_retrieval.append(row)
        showcase_lifecycle = materialize_context_model_lifecycle(
            store=store,
            manifest=_support_context_manifest(),
            observations=observations,
            outputs=context_outputs,
            retrieval=showcase_retrieval,
            resolution_port=_SupportResolution(),
        )
        if not showcase_lifecycle.resolution_requests or not showcase_lifecycle.resolution_results:
            raise RunnerError("support showcase lifecycle did not compile resolution")
        resolution_request = showcase_lifecycle.resolution_requests[0]
        resolution_result = showcase_lifecycle.resolution_results[0]
        operation = {
            "limits": {"deterministic_rows": 100, "live_rows": 100, "completion_executions": 100, "embedding_requests": 100, "completion_input_tokens": 32000, "completion_output_tokens": 8000, "source_transfer_bytes": 512 * 1024 * 1024},
            "actual": {"rows": len(records), "deterministic_rows": len(records), "live_rows": len(records), "completion_executions": len(enrichments), "embedding_items": len(embeddings), "embedding_batch_size": max(1, int(profile.embedding.parameters.get("request_chunk_size", 1))), "embedding_requests": (len(embeddings) + max(1, int(profile.embedding.parameters.get("request_chunk_size", 1))) - 1) // max(1, int(profile.embedding.parameters.get("request_chunk_size", 1))), "completion_concurrency": int(profile.completion.parameters.get("max_concurrency", 1)), "completion_input_tokens": sum(max(1, len(str(item.payload)) // 4) for item in representations), "completion_output_tokens": sum(max(1, len(str(item.payload)) // 4) for item in enrichments), "source_transfer_bytes": source_transfer_bytes, "retries": sum(max(0, int(attempts) - 1) for attempts in result.attempts.values()), "retry_attempt_limit": _retry_policy(profile).max_attempts, "cache_reuse": 0, "compilation_seconds": showcase_state.get("compilation_seconds", 0.0)},
            "provider_mode": "live" if load_reference_profile(context.config.profile_path).embedding.provider != "deterministic" else "deterministic",
            "source_transfer_bytes": source_transfer_bytes,
        }
        return (
            ArtifactPayload("sql", {"manifest": asdict(sql_manifest), "materialized": sql_result.materialized, "rows": sql_result.rows, "lineage": sql_result.lineage}, producer),
            ArtifactPayload("semantic-sql", semantic_artifact, producer),
            ArtifactPayload("showcase", {"retrieval": [item.model_dump(mode="json") for item in result.retrieval_facts], "resolution_request": resolution_request.model_dump(mode="json"), "resolution_result": resolution_result.model_dump(mode="json"), "relation_tables": {name: store.metadata(name) for name in relation_rows if name in store.names()}, "lineage": json.loads((context.config.scenario_path / "scenario.json").read_text(encoding="utf-8")).get("artifacts", [])}, producer),
            ArtifactPayload("operational", operation, producer),
        )

    return (StageSpec("prepare", producer, prepare), StageSpec("relational", producer, relational), StageSpec("outputs", producer, outputs), StageSpec("showcase", producer, showcase))


def run_support_smoke(
    scenario_path: str | Path,
    profile_path: str | Path,
    output: str | Path,
    *,
    domain: DomainConfig | None = None,
    storage: StructuredStorageConfig | None = None,
):
    configured_storage = storage or StructuredStorageConfig(
        os.environ.get("IGOR_LANCE_ROOT", "/tmp/igor-lance")
    )
    config = RunConfig.load(
        scenario_path,
        profile_path,
        domain=domain,
        storage=configured_storage,
    )
    store = LanceStore(LanceStoreConfig(
        configured_storage.root_uri,
        config.domain.environment,
        config.domain.domain,
    ))
    return CompositionRoot().run(config, _support_stages(store), output)


def run_support_huggingface_smoke(scenario_path: str | Path, profile_path: str | Path, acquisition_path: str | Path, output: str | Path):
    """Consume a portable Hugging Face acquisition artifact through the support seam."""
    acquired = json.loads(Path(acquisition_path).read_text(encoding="utf-8"))
    records = [{"channel": "web", "record_id": f"ticket-{index:03d}", "text": str(item["text"])}
               for index, item in enumerate(acquired.get("records", []), start=1)]
    if len(records) < 20:
        raise RunnerError("Hugging Face acquisition must contain at least twenty text records")
    source_bytes = int(acquired.get("file_bytes", 0))
    profile = load_reference_profile(profile_path)
    limits = {"rows": 100, "live_rows": 100, "completion_executions": 100, "embedding_requests": 100, "completion_input_tokens": 32000, "completion_output_tokens": 8000, "source_transfer_bytes": 512 * 1024 * 1024}
    if source_bytes > limits["source_transfer_bytes"] or len(records) > limits["live_rows"] and profile.embedding.provider != "deterministic":
        raise RunnerError("support Hugging Face preflight budget exceeded before provider execution")
    configured_storage = StructuredStorageConfig(os.environ.get("IGOR_LANCE_ROOT", "/tmp/igor-lance"))
    config = RunConfig.load(scenario_path, profile_path, storage=configured_storage)
    store = LanceStore(LanceStoreConfig(configured_storage.root_uri, config.domain.environment, config.domain.domain))
    return CompositionRoot().run(config, _support_stages(store, records_override=records, source_transfer_bytes=source_bytes), output)


def run_context_e2e(scenario_path: str | Path, profile_path: str | Path, output: str | Path, *, records_override: list[dict] | None = None, source_transfer_bytes: int = 0):
    """Run baseline and one declared source mutation against one persistent inventory."""
    scenario = Path(scenario_path)
    source = records_override or json.loads((scenario / "fixtures/source.json").read_text(encoding="utf-8"))["records"]
    units = ({"schema_version": "0.1", "context_units": [
        {"context_unit_id": f"ctx:support:{index:03d}", "record_id": row["record_id"]}
        for index, row in enumerate(source, start=1)
    ]} if records_override is not None else json.loads((scenario / "fixtures/context-units.json").read_text(encoding="utf-8")))
    mutation = json.loads((scenario / "mutations/source-change.json").read_text(encoding="utf-8"))
    mutated = [dict(row, text=(row["text"] + " Please help urgently.") if row["record_id"] == mutation["changed_record_id"] else row["text"]) for row in source]
    expected_affected = [f"ctx:support:{index:03d}" for index, row in enumerate(source, start=1) if row["record_id"] == mutation["changed_record_id"]]
    expected_unaffected = [f"ctx:support:{index:03d}" for index, row in enumerate(source, start=1) if row["record_id"] != mutation["changed_record_id"]]
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="igor-context-e2e-lance-") as state:
        lance = LanceStore(state)
        config = RunConfig.load(scenario, profile_path, storage=StructuredStorageConfig(str(Path(state) / "dlt")))
        baseline = CompositionRoot().run(config, _support_stages(lance, records_override=source, source_transfer_bytes=source_transfer_bytes), root / "baseline")
        # The changed snapshot is included in the persistent inventory, while unchanged outputs remain valid.
        schema = SchemaDescriptor(schema_version="0.1", schema_id="support.ticket", revision="1", domain="support",
                                  fields=(SchemaField(field_id="record_id", name="record_id", logical_type="string", required=True),
                                          SchemaField(field_id="text", name="text", logical_type="string", required=True)))
        changed_row = next(row for row in mutated if row["record_id"] == mutation["changed_record_id"])
        changed_identity_row = {"record_id": changed_row["record_id"], "text": changed_row["text"]}
        changed_snapshot = SourceSnapshot(ir_version="0.1", source_system="support-fixture", source_key=changed_row["record_id"], observed_at="2026-08-12T00:00:00Z", content_ref=f"fixture://support/{changed_row['record_id']}", content_sha256=hashlib.sha256(canonical_json(changed_identity_row).encode()).hexdigest(), media_type="application/json", schema_ref=schema)
        baseline_semantic = json.loads((root / "baseline/artifacts/semantic.json").read_text(encoding="utf-8"))
        valid = tuple(item["output_identity"] for item in json.loads((root / "baseline/artifacts/context/derivations.json").read_text(encoding="utf-8"))["records"])
        known_snapshots = tuple(SourceSnapshot.model_validate(item).identity for item in baseline_semantic["source_snapshots"])
        inventory = DependencyInventory(known_output_identities=known_snapshots + (changed_snapshot.identity,), valid_output_identities=valid, changed_identities=(changed_snapshot.identity,))
        mutated_config = config.__class__(config.scenario_path, config.profile_path, implementation_revision="runner:0.1:source-mutation-support-source-change-001", evaluator_revision=config.evaluator_revision, domain=config.domain, storage=config.storage)
        mutated_package = CompositionRoot().run(mutated_config, _support_stages(lance, inventory, mutated, source_transfer_bytes), root / "mutated")
        if baseline.manifest.identity == mutated_package.manifest.identity:
            raise RunnerError("mutation must produce a distinct run identity")
        mutation_plan = json.loads((root / "mutated/artifacts/context/compiler-plan.json").read_text(encoding="utf-8"))
        unaffected_representation_ids = {
            Representation.model_validate(item).identity for item in baseline_semantic["representations"]
        }
        reused_representations = set(mutation_plan["cache_hits"]) & unaffected_representation_ids
        if len(reused_representations) != len(expected_unaffected):
            raise RunnerError("mutation cache reuse does not match expected unaffected set")
        irrelevant = [dict(row, channel="email" if row["record_id"] == "ticket-004" else row["channel"]) for row in source]
        irrelevant_config = config.__class__(config.scenario_path, config.profile_path, implementation_revision="runner:0.1:irrelevant-source-mutation-001", evaluator_revision=config.evaluator_revision, domain=config.domain, storage=config.storage)
        irrelevant_inventory = DependencyInventory(known_output_identities=known_snapshots, valid_output_identities=valid, changed_identities=())
        CompositionRoot().run(irrelevant_config, _support_stages(lance, irrelevant_inventory, irrelevant, source_transfer_bytes), root / "irrelevant")
        baseline_semantic_ids = {stable_identity(item) for item in baseline_semantic["representations"]}
        irrelevant_semantic = json.loads((root / "irrelevant/artifacts/semantic.json").read_text(encoding="utf-8"))
        irrelevant_ids = {stable_identity(item) for item in irrelevant_semantic["representations"]}
        authority_time = [dict(row, _as_of="2025-01-01T00:00:00+00:00") for row in source]
        authority_config = config.__class__(config.scenario_path, config.profile_path, implementation_revision="runner:0.1:authority-time-mutation-001", evaluator_revision=config.evaluator_revision, domain=config.domain, storage=config.storage)
        authority_inventory = DependencyInventory(known_output_identities=known_snapshots, valid_output_identities=valid, changed_identities=())
        CompositionRoot().run(authority_config, _support_stages(lance, authority_inventory, authority_time, source_transfer_bytes), root / "authority-time")
        authority_showcase = json.loads((root / "authority-time/artifacts/showcase.json").read_text(encoding="utf-8"))
        authority_outcomes = {item["outcome"] for item in authority_showcase["resolution_result"]["decisions"]}
        comparison = {
            "baseline": baseline.manifest.identity,
            "mutated": mutated_package.manifest.identity,
            "expected_affected": expected_affected,
            "expected_unaffected": expected_unaffected,
            "reused_representation_identities": sorted(reused_representations),
            "cache_hit_count": len(reused_representations),
            "invalidation_plan": mutation_plan,
            "mutation_dimensions": {
                "semantic": {"status": "passed", "affected": expected_affected, "unaffected": expected_unaffected},
                "irrelevant": {"status": "passed" if irrelevant_ids == baseline_semantic_ids else "failed", "preserved_representation_identities": sorted(irrelevant_ids & baseline_semantic_ids)},
                "authority_time": {"status": "passed" if "rejected" in authority_outcomes and authority_outcomes != {"selected", "abstained", "rejected", "conflict"} else "failed", "observed_outcomes": sorted(authority_outcomes), "as_of": "2025-01-01T00:00:00+00:00"},
            },
        }
        (root / "comparison.json").write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return comparison


def run_support_huggingface_e2e(scenario_path: str | Path, profile_path: str | Path, acquisition_path: str | Path, output: str | Path):
    acquired = json.loads(Path(acquisition_path).read_text(encoding="utf-8"))
    records = [{"channel": "web", "record_id": f"ticket-{index:03d}", "text": str(item["text"]), "label": str(item.get("label", ""))} for index, item in enumerate(acquired.get("records", []), start=1)]
    if len(records) < 20:
        raise RunnerError("Hugging Face acquisition must contain at least twenty text records")
    profile = load_reference_profile(profile_path)
    return run_context_e2e(scenario_path, profile_path, output, records_override=records, source_transfer_bytes=int(acquired.get("file_bytes", 0)))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _solid_red_png(width: int = 64, height: int = 64) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\xff\x00\x00" * width)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(
        b"IDAT", zlib.compress(scanline * height),
    ) + _png_chunk(b"IEND", b"")


_RED_PIXEL_PNG = _solid_red_png()


class _RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.outcomes = []

    def embed(self, request):
        outcome = self.provider.embed(request)
        self.outcomes.append(outcome)
        return outcome

    def enrich(self, request):
        outcome = self.provider.enrich(request)
        self.outcomes.append(outcome)
        return outcome


def _image_source_contract() -> tuple[ContextSourceContract, ConnectorBinding]:
    contract = ContextSourceContract(
        contract_id="media.asset-source", revision="1", domain="media",
        resources=(SourceResourceContract(
            resource_id="asset", resource_kind="image", identity_concepts=("asset_id",),
            fields=(
                SourceFieldRequirement(concept_id="asset_id", logical_type="string", required=True),
                SourceFieldRequirement(concept_id="caption", logical_type="string", required=True),
                SourceFieldRequirement(concept_id="updated_at", logical_type="datetime", required=True),
            ),
            accepted_media_types=("image/png", "image/jpeg"), selection={"status": "approved"},
            change_mode="incremental", cursor_concept="updated_at", deletion_semantics="tombstone",
        ),),
    )
    binding = ConnectorBinding(
        binding_id="fixture.media-assets", revision="1", source_contract_identity=contract.identity,
        connector="fixture", deployment_ref="local/image-domain",
        resources=(ConnectorResourceBinding(
            resource_id="asset", source_resource="approved_images",
            fields=(
                ConnectorFieldBinding(concept_id="asset_id", source_field="id"),
                ConnectorFieldBinding(concept_id="caption", source_field="description"),
                ConnectorFieldBinding(concept_id="updated_at", source_field="modified"),
            ), cursor_field="modified",
        ),),
    )
    binding.validate_against(contract)
    return contract, binding


def _media_context_manifest(
    contract: ContextSourceContract,
    binding: ConnectorBinding,
    source_schema: SchemaDescriptor,
    output_schema: SchemaDescriptor,
    recipe: EnrichmentRecipe,
    embedding_profile: ModelProfile,
):
    return compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {
                "id": "media.asset-context",
                "revision": "0.1",
                "title": "Media asset context",
            },
            "sources": {
                "media_assets": {
                    "source_contract": "media.asset-source.v1",
                    "connector_binding": "fixture.media-assets.v1",
                    "identity_fields": ["asset_id"],
                },
            },
            "objects": {
                "media_asset": {
                    "kind": "business_object",
                    "source": "media_assets",
                    "display_name": "Media asset {{ asset_id }}",
                },
                "image_source_snapshot": {
                    "kind": "source_snapshot",
                    "source": "media_assets",
                    "display_name": "{{ asset_id }} image",
                },
                "image_metadata_snapshot": {
                    "kind": "source_snapshot",
                    "source": "media_assets",
                    "display_name": "{{ asset_id }} metadata",
                },
                "image_representation": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_source_snapshot"],
                    "operation": "representation.image.v1",
                    "schema": "media.asset.v1",
                    "display_name": "Original image",
                },
                "image_text_projection": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_source_snapshot", "image_metadata_snapshot"],
                    "operation": "representation.image-text-projection.v1",
                    "schema": "media.asset.v1",
                    "recipe": "media.classify-asset.v1",
                    "display_name": "Image text projection",
                },
                "image_embedding": {
                    "kind": "retrieval_projection",
                    "derived_from": ["image_representation"],
                    "operation": "embedding.multimodal.v1",
                    "schema": "media.asset.v1",
                    "model_profile": "media.embedding-profile.v1",
                    "display_name": "Image embedding",
                },
                "image_enrichment": {
                    "kind": "semantic_derivation",
                    "derived_from": ["image_representation"],
                    "operation": "enrichment.structured.v1",
                    "schema": "media.asset-enrichment.v1",
                    "recipe": "media.classify-asset.v1",
                    "display_name": "Image enrichment",
                },
            },
        },
        references=(
            ContextModelReference(
                kind="source_contract",
                ref="media.asset-source.v1",
                identity=contract.identity,
            ),
            ContextModelReference(
                kind="connector_binding",
                ref="fixture.media-assets.v1",
                identity=binding.identity,
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.image.v1",
                identity=stable_identity({"operation": "representation.image.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="representation.image-text-projection.v1",
                identity=stable_identity({"operation": "representation.image-text-projection.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="embedding.multimodal.v1",
                identity=stable_identity({"operation": "embedding.multimodal.v1"}),
            ),
            ContextModelReference(
                kind="operation",
                ref="enrichment.structured.v1",
                identity=stable_identity({"operation": "enrichment.structured.v1"}),
            ),
            ContextModelReference(
                kind="schema",
                ref="media.asset.v1",
                identity=source_schema.identity,
            ),
            ContextModelReference(
                kind="schema",
                ref="media.asset-enrichment.v1",
                identity=output_schema.identity,
            ),
            ContextModelReference(
                kind="recipe",
                ref="media.classify-asset.v1",
                identity=recipe.identity,
            ),
            ContextModelReference(
                kind="model_profile",
                ref="media.embedding-profile.v1",
                identity=embedding_profile.identity,
            ),
        ),
    ))


def _run_image_compilation(
    profile_path: str | Path,
    root: Path,
    lance: LanceStore,
    caption: str,
    *,
    inventory: DependencyInventory | None = None,
    source_record: dict | None = None,
    recipe_prompt_version: str = "media-asset-v1",
    recipe_taxonomy_version: str = "media-taxonomy-v1",
    dry_run: bool = False,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    if source_record is None:
        image_path = root.parent / "source" / "red-pixel.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_RED_PIXEL_PNG)
        image_hash = "sha256:" + hashlib.sha256(_RED_PIXEL_PNG).hexdigest()
        asset_id, image_ref, media_type = "asset-red-001", image_path.as_uri(), "image/png"
    else:
        image_ref = source_record["image_ref"]
        image_path = Path(image_ref.removeprefix("file://"))
        image_hash = source_record["image_sha256"]
        asset_id, media_type = source_record["asset_id"], source_record["media_type"]
    contract, binding = _image_source_contract()
    raw = [{
        "_resource_id": "asset", "_media_type": media_type, "_content_ref": image_ref,
        "_content_sha256": image_hash, "_observed_at": "2026-08-13T12:00:00Z",
        "id": asset_id, "description": caption, "modified": "2026-08-13T12:00:00Z",
    }]
    ingestion = ingest_bound_records(raw, IngestConfig("local", "media", str(root / "ingestion-lance"), "local/media"), contract, binding)
    source_schema = SchemaDescriptor(
        schema_version="0.1", schema_id="media.asset", revision="1", domain="media",
        fields=(
            SchemaField(field_id="asset_id", name="asset_id", logical_type="string", required=True),
            SchemaField(field_id="caption", name="caption", logical_type="string", required=True),
        ),
    )
    output_schema = SchemaDescriptor(
        schema_version="0.1", schema_id="media.asset-enrichment", revision="1", domain="media",
        json_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False,
            "required": ["asset_kind", "dominant_color", "review_action"],
            "properties": {
                "asset_kind": {"type": "string", "enum": ["status_illustration", "document_scan", "photograph", "other"]},
                "dominant_color": {"type": "string", "enum": ["red", "blue", "green", "mixed", "unknown"]},
                "review_action": {"type": "string", "enum": ["accept", "manual_review"]},
            },
        },
    )
    recipe = EnrichmentRecipe(
        recipe_id="media.classify-asset", revision="1", accepted_representation_types=("image",),
        accepted_media_types=("text/plain", "image/png", "image/jpeg"), output_schema_identity=output_schema.identity,
        prompt_version=recipe_prompt_version, taxonomy_version=recipe_taxonomy_version,
    )
    image_snapshot = SourceSnapshot(
        ir_version="0.1", source_system=binding.connector, source_key=f"{asset_id}:image",
        observed_at="2026-08-13T12:00:00Z", content_ref=image_ref, content_sha256=image_hash,
        media_type=media_type, schema_ref=source_schema,
    )
    metadata_hash = stable_identity(caption)
    metadata_snapshot = SourceSnapshot(
        ir_version="0.1", source_system=binding.connector, source_key=f"{asset_id}:metadata",
        observed_at="2026-08-13T12:00:00Z", content_ref=f"inline://media/{asset_id}/caption",
        content_sha256=metadata_hash, media_type="text/plain", schema_ref=source_schema,
    )
    image_representation = Representation(
        ir_version="0.1", representation_type="image", schema_ref=source_schema,
        source_snapshot_ids=(image_snapshot.identity,), payload={"asset_id": asset_id},
        metadata={"runtime_modality": "image"},
    )
    projection = Representation(
        ir_version="0.1", representation_type="image", schema_ref=source_schema,
        source_snapshot_ids=(image_snapshot.identity, metadata_snapshot.identity), payload=caption,
        metadata={"runtime_modality": "evidence-linked-text-projection", "source_asset": image_representation.identity},
    )
    image_evidence = Evidence(
        ir_version="0.1", subject_identity=image_representation.identity,
        source_snapshot_id=image_snapshot.identity, locator="pixel:0,0", observed_at=image_snapshot.observed_at,
    )
    projection_evidence = Evidence(
        ir_version="0.1", subject_identity=projection.identity, source_snapshot_id=metadata_snapshot.identity,
        locator="field:description", excerpt=caption, observed_at=metadata_snapshot.observed_at,
    )
    profile = load_reference_profile(profile_path)
    embedding_profile, completion_profile, embedding_provider, completion_provider = _provider_pair(profile)
    if completion_profile.provider == "deterministic":
        completion_provider = DeterministicProviders({"media.classify-asset": {
            "asset_kind": "status_illustration", "dominant_color": "red", "review_action": "accept",
        }})
    embedding_recorder = _RecordingProvider(embedding_provider)
    completion_recorder = _RecordingProvider(completion_provider)
    dimension = int(embedding_profile.parameters.get("dimensions", 8))
    space = EmbeddingSpace(
        ir_version="0.1", provider=embedding_profile.provider, model=embedding_profile.model,
        model_revision=embedding_profile.revision, dimension=dimension,
        dtype="float64" if embedding_profile.provider == "deterministic" else "float32",
        metric="cosine", normalized=False, input_schema_identity=source_schema.identity,
    )
    embedding_id = stable_identity({"embedding": image_representation.identity, "space": space.identity})
    enrichment_id = stable_identity({"enrichment": projection.identity, "recipe": recipe.identity,
                                     "profile": completion_profile.identity})
    derivation_inputs = (
        {
            "role": "image_representation",
            "input_identities": (image_snapshot.identity,),
            "output_identity": image_representation.identity,
            "configuration_identity": contract.identity,
        },
        {
            "role": "image_text_projection",
            "input_identities": (image_snapshot.identity, metadata_snapshot.identity),
            "output_identity": projection.identity,
            "configuration_identity": recipe.identity,
        },
        {
            "role": "image_embedding",
            "input_identities": (image_representation.identity,),
            "output_identity": embedding_id,
            "configuration_identity": space.identity,
            "model_revision": embedding_profile.revision,
        },
        {
            "role": "image_enrichment",
            "input_identities": (image_representation.identity,),
            "output_identity": enrichment_id,
            "configuration_identity": recipe.identity,
            "model_revision": completion_profile.revision,
        },
    )
    image_input = QualifiedRepresentation(
        representation=image_representation,
        parts=(
            ContentPart(kind="text", media_type="text/plain", text=caption,
                        content_sha256=stable_identity(caption)),
            ContentPart(kind="image", media_type=media_type, content_ref=image_ref, content_sha256=image_hash),
        ),
        evidence=(image_evidence,),
    )
    resolved_inventory = inventory if inventory is not None else DependencyInventory()
    resolved_inventory = resolved_inventory.model_copy(update={
        "known_output_identities": tuple(sorted(set(resolved_inventory.known_output_identities) | {
            image_snapshot.identity, metadata_snapshot.identity,
        })),
    })
    observations = (
        _source_snapshot_observation(
            image_snapshot,
            object_role="media_asset",
            snapshot_role="image_source_snapshot",
            display_name=f"{asset_id} image",
            preview=image_ref,
        ),
        _source_snapshot_observation(
            metadata_snapshot,
            object_role="media_asset",
            snapshot_role="image_metadata_snapshot",
            display_name=f"{asset_id} metadata",
            preview=caption[:160],
        ),
    )
    manifest = _media_context_manifest(contract, binding, source_schema, output_schema, recipe, embedding_profile)
    lifecycle = compile_context_model_lifecycle(
        store=lance,
        manifest=manifest,
        observations=observations,
        inventory=resolved_inventory,
        ports=CompilerPorts(
            store=LanceContextOutputStore(lance),
            embedding=embedding_recorder,
            enrichment=completion_recorder,
        ),
        run_identity=stable_identity({"media-image": caption, "profile": profile.identity}),
        task_id="media-asset-context-v0.1",
        embedding_profile=embedding_profile,
        completion_profile=completion_profile,
        embedding_space=space,
        code_revision="runner:0.1",
        budget_tokens=100,
        derivation_inputs=derivation_inputs,
        embedding_inputs=(image_input,),
        completion_inputs=(image_input,),
        completion_output_schema=output_schema,
        enrichment_recipe=recipe,
        prompt_version=recipe.prompt_version,
        taxonomy_version=recipe.taxonomy_version,
        required_context_identities=(image_representation.identity, projection.identity),
        plan_only=dry_run,
    )
    compiled_plan = lifecycle.compilation_plan
    if compiled_plan is None:
        raise RunnerError("image Context Model lifecycle did not build a compiler plan")
    if dry_run:
        return {
            "source_contract": contract.model_dump(mode="json"), "connector_binding": binding.model_dump(mode="json"),
            "source_record": source_record, "snapshots": [image_snapshot.model_dump(mode="json"), metadata_snapshot.model_dump(mode="json")],
            "representations": [image_representation.model_dump(mode="json"), projection.model_dump(mode="json")],
            "representation_identities": [image_representation.identity, projection.identity],
            "embedding_output_identity": embedding_id, "enrichment_output_identity": enrichment_id,
            "package_identity": None, "plan": compiled_plan.model_dump(mode="json"), "lineage": [], "attempts": {},
            "provider_outcomes": {"embedding": [], "completion": []}, "dry_run": True,
            "inventory_valid_output_identities": list(resolved_inventory.valid_output_identities),
        }
    result = lifecycle.compilation
    if result is None:
        raise RunnerError("image Context Model lifecycle did not execute compiler request")
    if result.failures or result.package_identity is None:
        provider_status = {
            "embedding": [item.status for item in embedding_recorder.outcomes],
            "completion": [item.status for item in completion_recorder.outcomes],
        }
        raise RunnerError(f"image compilation failed: {list(result.failures)}; providers={provider_status}")
    output_store = LanceContextOutputStore(lance)
    embedding = result.semantic_outputs.get(embedding_id) or Embedding.model_validate(output_store.read(embedding_id))
    enrichment = result.semantic_outputs.get(enrichment_id) or EnrichmentOutput.model_validate(output_store.read(enrichment_id))
    return {
        "source_contract": contract.model_dump(mode="json"), "connector_binding": binding.model_dump(mode="json"),
        "source_record": source_record,
        "ingestion": ingestion.as_dict(), "recipe": recipe.model_dump(mode="json"),
        "snapshots": [image_snapshot.model_dump(mode="json"), metadata_snapshot.model_dump(mode="json")],
        "representations": [image_representation.model_dump(mode="json"), projection.model_dump(mode="json")],
        "representation_identities": [image_representation.identity, projection.identity],
        "embedding_output_identity": embedding_id, "enrichment_output_identity": enrichment_id,
        "embedding": {**embedding.model_dump(mode="json"), "identity": embedding.identity},
        "enrichment": {**enrichment.model_dump(mode="json"), "identity": enrichment.identity},
        "package_identity": result.package_identity, "plan": result.artifact_records["plan"],
        "inventory_valid_output_identities": list(resolved_inventory.valid_output_identities),
        "lineage": result.artifact_records["lineage"], "attempts": dict(result.attempts),
        "provider_outcomes": {
            "embedding": [item.model_dump(mode="json", exclude={"value"}) for item in embedding_recorder.outcomes],
            "completion": [item.model_dump(mode="json", exclude={"value"}) for item in completion_recorder.outcomes],
        },
    }


def run_image_context(profile_path: str | Path, output: str | Path, *, mutation: bool = False) -> dict:
    """Qualify one image domain through source selection, compile, validation, package, and Lance persistence."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    lance_root = os.environ.get("IGOR_LANCE_ROOT", str(root / "compiler-lance"))
    lance = LanceStore(LanceStoreConfig(lance_root, "local", "media"))
    caption = "A red status illustration. Return asset_kind status_illustration, dominant_color red, and review_action accept."
    baseline = _run_image_compilation(profile_path, root / "baseline", lance, caption)
    payload = {"mode": "live" if baseline["provider_outcomes"]["embedding"][0]["metadata"]["provider"] == "voyage" else "mock",
               "baseline": baseline}
    if mutation:
        previous = baseline["snapshots"]
        old_image, old_metadata = (SourceSnapshot.model_validate(item) for item in previous)
        valid = tuple(baseline["plan"]["expected_output_identities"])
        inventory = DependencyInventory(
            known_output_identities=(old_image.identity, old_metadata.identity),
            valid_output_identities=valid,
            changed_identities=(old_metadata.identity,),
        )
        mutated = _run_image_compilation(
            profile_path, root / "mutated", lance,
            caption + " Editorial note updated without changing image bytes.", inventory=inventory,
        )
        reused = set(mutated["plan"]["cache_hits"])
        expected_reuse = {baseline["representation_identities"][0], baseline["embedding_output_identity"]}
        if not expected_reuse.issubset(reused):
            raise RunnerError("image metadata mutation did not reuse the unchanged image representation and embedding")
        payload["mutation"] = mutated
        payload["selective_invalidation"] = {"reused": sorted(reused), "expected_reused": sorted(expected_reuse)}
    (root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"valid": True, "mode": payload["mode"], "package_identity": baseline["package_identity"],
            "result": str(root / "result.json")}
