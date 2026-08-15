import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from igor_core import (
    ContextItem,
    EmbeddingSpace,
    LineageEdge,
    LineageGraph,
    LineageNode,
    ModelProfile,
    ProducerIdentity,
    ReferenceProfile,
    SchemaDescriptor,
    stable_identity,
)
from igor_context_compiler import (
    CompilationRequest,
    CompilerPorts,
    ContextModelCompileRequest,
    ContextModelReference,
    DeepSeekCompletionAdapter,
    DependencyInventory,
    DerivationSpec,
    MistralCompletionAdapter,
    ResolutionDecision,
    ResolutionRequest,
    ResolutionResult,
    RetrievalQuery,
    VoyageMultimodalEmbeddingAdapter,
    compile_context_model,
)
from igor_lancedb import LanceContextOutputStore, LanceStore
from igor_runner import compile_context_model_lifecycle, materialize_context_model_lifecycle
from igor_runner.compose import _provider_pair
from igor_runner import ArtifactPayload, CompositionRoot, DomainConfig, RunConfig, RunnerError, StageSpec, StructuredStorageConfig


def inputs(root: Path) -> tuple[Path, Path]:
    scenario = root / "scenario"
    (scenario / "fixtures").mkdir(parents=True)
    (scenario / "fixtures/source.json").write_text("{}", encoding="utf-8")
    (scenario / "fixtures/context-units.json").write_text("{}", encoding="utf-8")
    embeddings = root / "providers/embeddings"
    completions = root / "providers/completions"
    embeddings.mkdir(parents=True)
    completions.mkdir(parents=True)
    embeddings.joinpath("test.yaml").write_text(
        "schema_version: '0.1'\nprofile_id: test-embedding\ncapability: embedding\n"
        "provider: test\nmodel: e\nrevision: '1'\n",
        encoding="utf-8",
    )
    completions.joinpath("test.yaml").write_text(
        "schema_version: '0.1'\nprofile_id: test-completion\ncapability: completion\n"
        "provider: test\nmodel: x\nrevision: '1'\n",
        encoding="utf-8",
    )
    profile = root / "profile.yaml"
    profile.write_text(
        "schema_version: '0.1'\nprofile_id: test\nprompt_version: p\ntaxonomy_version: t\n"
        "embedding_profile: providers/embeddings/test.yaml\n"
        "completion_profile: providers/completions/test.yaml\n",
        encoding="utf-8",
    )
    return scenario, profile


def config(root: Path) -> RunConfig:
    scenario, profile = inputs(root)
    scenario.joinpath("scenario.json").write_text(
        json.dumps(
            {
                "compatibility": {"benchmark_contract": "0.1"},
                "scenario_id": "support",
                "version": "0.1.0",
                "scorecard": "scorecard",
                "source_fixture_identity": "source",
            }
        ),
        encoding="utf-8",
    )
    return RunConfig(scenario, profile)


def test_composition_tracks_stage_inputs_and_validates_run() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run_config = config(root)
        producer = ProducerIdentity(name="test", revision="1")
        stages = (
            StageSpec("first", producer, lambda context: (ArtifactPayload("context", {"ok": True}, producer),)),
            StageSpec("second", producer, lambda context: (ArtifactPayload("metrics", {"ok": True}, producer),)),
        )
        package = CompositionRoot().run(run_config, stages, root / "run")
        assert [stage.stage_id for stage in package.stages] == ["first", "second"]
        assert package.stages[1].input_artifact_identities == [package.artifact_index.artifacts[0].identity]


def test_duplicate_stage_ids_are_rejected() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run_config = config(root)
        producer = ProducerIdentity(name="test", revision="1")
        stage = StageSpec("same", producer, lambda context: ())
        try:
            CompositionRoot().run(run_config, (stage, stage), root / "run")
        except RunnerError as error:
            assert "stage IDs must be unique" in str(error)
        else:
            raise AssertionError("duplicate stages should fail")


def test_domain_storage_configuration_is_available_to_stages_without_changing_run_identity() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run_config = config(root)
        run_config = RunConfig(
            run_config.scenario_path,
            run_config.profile_path,
            domain=DomainConfig("qualification", "finance"),
            storage=StructuredStorageConfig("s3://deployment/lance", credential_ref="aws-runner"),
        )
        producer = ProducerIdentity(name="test", revision="1")

        def inspect(context):
            assert context.config.domain.namespace == "qualification/finance"
            assert context.config.storage.namespace_uri(context.config.domain) == "s3://deployment/lance/qualification/finance"
            assert context.config.storage.credential_ref == "aws-runner"
            return (ArtifactPayload("context", {"ok": True}, producer),)

        package = CompositionRoot().run(run_config, (StageSpec("inspect", producer, inspect),), root / "run")
        assert package.manifest.identity_links.scenario_pack == "support:0.1.0"


def test_storage_configuration_rejects_empty_values() -> None:
    try:
        StructuredStorageConfig(" ")
    except RunnerError as error:
        assert "root_uri" in str(error)
    else:
        raise AssertionError("empty storage roots should fail")


def test_composition_merges_lineage_into_one_run_artifact() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run_config = config(root)
        producer = ProducerIdentity(name="test", revision="1")
        run_identity = "sha256:" + "3" * 64
        source = LineageNode(schema_version="0.1", node_type="source_record", key="ticket-1", run_identity=run_identity, producer=producer, configuration_revision="config-1")
        target = LineageNode(schema_version="0.1", node_type="context_unit", key="ctx-1", run_identity=run_identity, producer=producer, configuration_revision="config-1")
        # The handler receives the actual run identity through the config, so build the graph there.
        def stage(context):
            actual = context.config.identity_links()
            from igor_core import stable_identity
            identity = stable_identity(actual)
            left = source.model_copy(update={"run_identity": identity})
            right = target.model_copy(update={"run_identity": identity})
            graph = LineageGraph(schema_version="0.1", run_identity=identity, nodes=(left, right), edges=(LineageEdge(schema_version="0.1", edge_type="canonicalized", source_node_id=left.identity, target_node_id=right.identity, run_identity=identity, producer=producer, configuration_revision="config-1"),))
            return (ArtifactPayload("context", {"ok": True}, producer, lineage=graph),)
        package = CompositionRoot().run(run_config, (StageSpec("lineage", producer, stage),), root / "run")
        assert package.lineage is not None
        assert package.manifest.lineage_path == "artifacts/lineage.json"
        assert any(artifact.path == "artifacts/lineage.json" for artifact in package.artifact_index.artifacts)


def test_profile_selects_live_provider_adapters_without_initializing_network_clients() -> None:
    profile = ReferenceProfile(
        schema_version="0.1", profile_id="live", prompt_version="p", taxonomy_version="t",
        embedding=ModelProfile(schema_version="0.1", profile_id="voyage", capability="embedding",
            provider="voyage", model="voyage-multimodal-3", revision="1",
            parameters={"endpoint": "https://api.voyageai.com/v1/multimodalembeddings", "dimensions": 1024}),
        completion=ModelProfile(schema_version="0.1", profile_id="deepseek", capability="completion",
            provider="deepseek", model="deepseek-v4-flash", revision="1",
            parameters={"endpoint": "https://api.deepseek.com", "temperature": 0, "response_format": "json_object"}),
    )
    _, _, embedding, enrichment = _provider_pair(profile)
    assert isinstance(embedding, VoyageMultimodalEmbeddingAdapter)
    assert isinstance(enrichment, DeepSeekCompletionAdapter)


def test_profile_selects_mistral_without_coupling_it_to_embedding_provider() -> None:
    profile = ReferenceProfile(
        schema_version="0.1", profile_id="live", prompt_version="p", taxonomy_version="t",
        embedding=ModelProfile(schema_version="0.1", profile_id="voyage", capability="embedding",
            provider="voyage", model="voyage-multimodal-3", revision="1",
            parameters={"endpoint": "https://api.voyageai.com/v1/multimodalembeddings", "dimensions": 1024}),
        completion=ModelProfile(schema_version="0.1", profile_id="mistral", capability="completion",
            provider="mistral", model="mistral-small-2603", revision="1",
            parameters={"endpoint": "https://api.mistral.ai/v1/chat/completions", "temperature": 0,
                        "response_format": "json_object", "supported_content_kinds": ["text", "image"]}),
    )
    _, _, embedding, enrichment = _provider_pair(profile)
    assert isinstance(embedding, VoyageMultimodalEmbeddingAdapter)
    assert isinstance(enrichment, MistralCompletionAdapter)


def test_context_model_lifecycle_executes_compiler_and_preserves_output_store(tmp_path: Path) -> None:
    class DeterministicRetrieval:
        def __init__(self):
            self.requests = []

        def retrieve(self, query):
            self.requests.append(query)
            return ({
                "result_identity": stable_identity({"retrieval": output_identity}),
                "context_identity": output_identity,
                "score": 0.77,
                "rank": 0,
                "facts": {"source": "test"},
            },)

    store = LanceStore(tmp_path / "lance")
    output_store = LanceContextOutputStore(store)
    source_identity = stable_identity({"source": "ticket-1"})
    output_identity = stable_identity({"representation": source_identity})
    schema = SchemaDescriptor(schema_version="0.1", schema_id="support.ticket", revision="1")
    embedding_profile = ModelProfile(
        schema_version="0.1", profile_id="deterministic-embedding",
        capability="embedding", provider="deterministic", model="test", revision="1",
    )
    completion_profile = ModelProfile(
        schema_version="0.1", profile_id="deterministic-completion",
        capability="completion", provider="deterministic", model="test", revision="1",
    )
    embedding_space = EmbeddingSpace(
        ir_version="0.1", provider="deterministic", model="test",
        model_revision="1", dimension=3, dtype="float64", metric="cosine",
        normalized=False, input_schema_identity=schema.identity,
    )
    manifest = compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {"id": "support.lifecycle", "revision": "1"},
            "sources": {
                "support_queue": {
                    "source_contract": "support.messages.v1",
                    "connector_binding": "fixture.support.v1",
                    "identity_fields": ["ticket_id"],
                },
            },
            "objects": {
                "support_ticket": {"kind": "business_object", "source": "support_queue"},
                "support_ticket_snapshot": {"kind": "source_snapshot", "source": "support_queue"},
                "support_representation": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_ticket_snapshot"],
                    "operation": "representation.v1",
                    "schema": "support.ticket.v1",
                },
            },
            "retrievals": {
                "support_lookup": {"search": "vector", "candidate_limit": 3},
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
                ref="fixture.support.v1",
                identity=stable_identity({"connector_binding": "fixture.support.v1"}),
            ),
            ContextModelReference(kind="schema", ref="support.ticket.v1", identity=schema.identity),
            ContextModelReference(
                kind="operation",
                ref="representation.v1",
                identity=stable_identity({"operation": "representation.v1"}),
            ),
        ),
    ))
    lifecycle = compile_context_model_lifecycle(
        store=store,
        manifest=manifest,
        observations=({
            "object_role": "support_ticket",
            "snapshot_role": "support_ticket_snapshot",
            "source_system": "test",
            "source_key": "ticket-1",
            "snapshot_identity": source_identity,
            "content_ref": "fixture://ticket-1",
            "content_sha256": source_identity,
            "media_type": "text/plain",
            "observed_at": "2026-08-14T00:00:00Z",
            "display_name": "Ticket 1",
        },),
        ports=CompilerPorts(store=output_store, retrieval=DeterministicRetrieval()),
        run_identity=stable_identity({"run": "lifecycle"}),
        task_id="support-lifecycle-test",
        embedding_profile=embedding_profile,
        completion_profile=completion_profile,
        embedding_space=embedding_space,
        code_revision="test",
        budget_tokens=64,
        output_role_identities={"support_representation": output_identity},
        required_context_identities=(output_identity,),
        retrieval_inputs={
            "support_lookup": {
                "text": "ticket",
                "vector": (0.1, 0.2, 0.3),
                "space_identity": embedding_space.identity,
            },
        },
    )

    assert lifecycle.compilation is not None
    assert lifecycle.compilation.package_identity
    assert set(("context_models", "context_snapshots", "context_outputs", "context_packages", "package_items")).issubset(store.names())
    assert lifecycle.relation_counts["context_packages"] == 1
    assert lifecycle.relation_counts["package_items"] == 1
    assert lifecycle.relation_counts["resolution_candidates"] == 1
    assert lifecycle.relation_counts["context_assertions"] >= 1
    package_items = store.read("package_items").to_pylist()
    assert package_items[0]["package_identity"] == lifecycle.compilation.package_identity
    assert package_items[0]["representation_identity"] == output_identity
    context_model = store.read("context_models").to_pylist()[0]
    assert context_model["context_model_identity"] == manifest.identity
    assert context_model["context_model_revision"] == manifest.revision
    assert output_store.read(output_identity) == {"inputs": [source_identity], "operation": "representation.v1"}
    candidate = store.read("resolution_candidates").to_pylist()[0]
    assert candidate["context_identity"] == output_identity
    assert candidate["semantic_score"] == 0.77
    assertions = store.read("context_assertions").to_pylist()
    assert output_identity in {row["assertion_subject"] for row in assertions}
    assert "context_catalog" in lifecycle.standard_sql_relations
    assert lifecycle.test_execution is not None
    assert lifecycle.test_execution.passed
    sql_rows = lifecycle.query_standard_sql(
        "SELECT m.context_model_id, o.output_role, p.package_kind "
        "FROM context_models m "
        "JOIN context_outputs o ON o.context_model_id = m.context_model_id "
        "JOIN context_packages p ON p.context_model_id = m.context_model_id "
        "WHERE o.output_role = 'representation.v1' "
        "ORDER BY o.identity"
    ).to_pylist()
    assert sql_rows == [{
        "context_model_id": "support.lifecycle",
        "output_role": "representation.v1",
        "package_kind": "task_context",
    }]


def test_context_model_lifecycle_attaches_resolution_to_supplied_compiler_request(tmp_path: Path) -> None:
    class DeterministicRetrieval:
        def retrieve(self, query):
            return ({
                "result_identity": stable_identity({"retrieval": output_identity}),
                "context_identity": output_identity,
                "score": 0.5,
                "rank": 0,
                "facts": {"source": "test"},
            },)

    class SelectAllResolution:
        def __init__(self):
            self.requests = []

        def resolve(self, request: ResolutionRequest) -> ResolutionResult:
            self.requests.append(request)
            decisions = tuple(ResolutionDecision(
                candidate_identity=candidate.candidate_identity,
                outcome="selected",
                reason_code="test:selected",
                authority_basis="satisfied",
                temporal_basis="valid",
                as_of=request.as_of,
                policy_id=request.policy_id,
                policy_revision=request.policy_revision,
            ) for candidate in request.candidates)
            return ResolutionResult(
                request_identity=request.request_identity,
                resolution_identity=request.identity,
                decisions=decisions,
                selected_identities=tuple(candidate.candidate_identity for candidate in request.candidates),
            )

    store = LanceStore(tmp_path / "lance")
    output_store = LanceContextOutputStore(store)
    source_identity = stable_identity({"source": "ticket-1", "resolution": "supplied-request"})
    output_identity = stable_identity({"representation": source_identity})
    candidate_identity = stable_identity({"candidate": output_identity})
    schema = SchemaDescriptor(schema_version="0.1", schema_id="support.ticket", revision="1")
    embedding_profile = ModelProfile(
        schema_version="0.1", profile_id="deterministic-embedding",
        capability="embedding", provider="deterministic", model="test", revision="1",
    )
    completion_profile = ModelProfile(
        schema_version="0.1", profile_id="deterministic-completion",
        capability="completion", provider="deterministic", model="test", revision="1",
    )
    embedding_space = EmbeddingSpace(
        ir_version="0.1", provider="deterministic", model="test",
        model_revision="1", dimension=3, dtype="float64", metric="cosine",
        normalized=False, input_schema_identity=schema.identity,
    )
    manifest = compile_context_model(ContextModelCompileRequest(
        declaration={
            "context_model": {"id": "support.lifecycle", "revision": "1"},
            "sources": {
                "support_queue": {
                    "source_contract": "support.messages.v1",
                    "connector_binding": "fixture.support.v1",
                    "identity_fields": ["ticket_id"],
                },
            },
            "objects": {
                "support_ticket": {"kind": "business_object", "source": "support_queue"},
                "support_ticket_snapshot": {"kind": "source_snapshot", "source": "support_queue"},
                "support_representation": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_ticket_snapshot"],
                    "operation": "representation.v1",
                    "schema": "support.ticket.v1",
                },
            },
            "authority": {
                "support_policy": {
                    "target": "support_representation",
                    "policy": "support.policy.v1",
                    "policy_revision": "1",
                },
            },
            "retrievals": {
                "support_lookup": {
                    "search": "vector",
                    "candidate_limit": 3,
                    "resolution": {"policy": "support_policy", "accepted_outcomes": ["selected"]},
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
                ref="fixture.support.v1",
                identity=stable_identity({"connector_binding": "fixture.support.v1"}),
            ),
            ContextModelReference(kind="schema", ref="support.ticket.v1", identity=schema.identity),
            ContextModelReference(
                kind="operation",
                ref="representation.v1",
                identity=stable_identity({"operation": "representation.v1"}),
            ),
            ContextModelReference(
                kind="authority_policy",
                ref="support.policy.v1",
                identity=stable_identity({"policy": "support.policy.v1"}),
            ),
        ),
    ))
    request = CompilationRequest(
        run_identity=stable_identity({"run": "supplied-request-resolution"}),
        task_id="support-lifecycle-test",
        required_output_identities=(output_identity,),
        derivations=(DerivationSpec(
            operation="representation.v1",
            input_identities=(source_identity,),
            output_identity=output_identity,
            code_revision="test",
            configuration_identity=schema.identity,
        ),),
        embedding_profile=embedding_profile,
        completion_profile=completion_profile,
        embedding_space=embedding_space,
        code_revision="test",
        budget_tokens=64,
        package_items=(ContextItem(
            representation_identity=output_identity,
            role="support-context",
            rank=0,
            token_estimate=8,
        ),),
        retrieval_queries=(RetrievalQuery(
            query_id=stable_identity({"query": "support"}),
            query_type="vector",
            vector=(0.1, 0.2, 0.3),
            limit=1,
            space_identity=embedding_space.identity,
        ),),
        required_context_identities=(output_identity,),
    )
    resolution = SelectAllResolution()

    lifecycle = compile_context_model_lifecycle(
        store=store,
        manifest=manifest,
        observations=({
            "object_role": "support_ticket",
            "snapshot_role": "support_ticket_snapshot",
            "source_system": "test",
            "source_key": "ticket-1",
            "snapshot_identity": source_identity,
            "content_ref": "fixture://ticket-1",
            "content_sha256": source_identity,
            "media_type": "text/plain",
            "observed_at": "2026-08-14T00:00:00Z",
            "display_name": "Ticket 1",
        },),
        request=request,
        inventory=DependencyInventory(known_output_identities=(source_identity,)),
        ports=CompilerPorts(store=output_store, retrieval=DeterministicRetrieval(), resolution=resolution),
        resolution_inputs=({
            "retrieval_role": "support_lookup",
            "request_identity": stable_identity({"request": output_identity}),
            "task_id": "support-lifecycle-test",
            "task_schema_revision": "1",
            "as_of": "2026-08-14T00:00:00Z",
            "candidates": ({
                "candidate_identity": candidate_identity,
                "context_identity": output_identity,
            },),
        },),
    )

    assert request.resolution is None
    assert lifecycle.compilation is not None
    assert lifecycle.compilation.resolution_result is not None
    assert resolution.requests[0].request_identity == stable_identity({"request": output_identity})
    assert resolution.requests[0].candidates[0].retrieval is not None
    assert resolution.requests[0].candidates[0].retrieval.score == 0.5
    assert lifecycle.compilation.package_identity


def test_context_model_lifecycle_resolves_and_compiles_default_packages(tmp_path: Path) -> None:
    class RetrieveSelected:
        def __init__(self):
            self.queries = []

        def retrieve(self, query):
            self.queries.append(query)
            return ({
                "result_identity": stable_identity({"retrieval": context_identity}),
                "context_identity": context_identity,
                "score": 0.91,
                "rank": 0,
                "facts": {"source": "test"},
            },)

    class SelectAllResolution:
        def __init__(self):
            self.requests = []

        def resolve(self, request: ResolutionRequest) -> ResolutionResult:
            self.requests.append(request)
            decisions = tuple(ResolutionDecision(
                candidate_identity=candidate.candidate_identity,
                outcome="selected",
                reason_code="test:selected",
                authority_basis="satisfied",
                temporal_basis="valid",
                as_of=request.as_of,
                policy_id=request.policy_id,
                policy_revision=request.policy_revision,
            ) for candidate in request.candidates)
            return ResolutionResult(
                request_identity=request.request_identity,
                resolution_identity=request.identity,
                decisions=decisions,
                selected_identities=tuple(candidate.candidate_identity for candidate in request.candidates),
            )

    store = LanceStore(tmp_path / "lance")
    context_identity = stable_identity({"context": "selected"})
    candidate_identity = stable_identity({"candidate": context_identity})
    as_of = datetime(2026, 8, 14, tzinfo=timezone.utc)
    retrieval = RetrieveSelected()
    resolution = SelectAllResolution()
    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest={
            "context_model_id": "support.lifecycle",
            "context_model_revision": "1",
            "context_model_identity": stable_identity({"context_model": "support.lifecycle", "resolution": "1"}),
            "compiler_version": "context-model-manifest.v0.1",
            "authority": [{
                "role": "support_message",
                "target_role": "support_message",
                "policy_ref": "support.policy",
                "policy_identity": stable_identity({"policy": "support.policy"}),
                "settings": {"policy_revision": "1"},
            }],
            "retrievals": [{
                "role": "support_lookup",
                "search": "hybrid",
                "candidate_limit": 3,
                "resolution_policy_role": "support_message",
                "accepted_outcomes": ["selected"],
            }],
            "tests": [{
                "test_type": "no_orphan_outputs",
                "parameters": {},
            }],
        },
        outputs=({
            "identity": context_identity,
            "role": "support_message",
            "kind": "context_representation",
            "display_name": "Selected context",
            "payload": {"ok": True},
            "status": "active",
        },),
        retrieval_inputs=({
            "request_identity": stable_identity({"request": context_identity}),
            "retrieval_role": "support_lookup",
            "query_id": stable_identity({"query": context_identity}),
            "text": "selected context",
            "vector": (0.1, 0.2, 0.3),
            "limit": 1,
            "task_id": "support-task",
            "task_schema_revision": "1",
            "as_of": as_of.isoformat(),
            "candidate_identity": candidate_identity,
            "evidence_locator": "fixture://selected-context",
        },),
        retrieval_port=retrieval,
        resolution_port=resolution,
    )

    assert retrieval.queries[0].query_type == "hybrid"
    assert lifecycle.retrieval_results[0]["context_identity"] == context_identity
    assert lifecycle.relation_counts["resolution_decisions"] == 1
    assert lifecycle.relation_counts["resolution_candidates"] == 1
    assert lifecycle.relation_counts["context_packages"] == 1
    assert lifecycle.relation_counts["package_items"] == 1
    assert lifecycle.relation_counts["context_assertions"] == 1
    assert lifecycle.resolution_results[0].selected_identities == (candidate_identity,)
    assert lifecycle.packages[0].task_id == "support-task"
    assert lifecycle.packages[0].items[0].role == "resolved-context"
    assert resolution.requests[0].candidates[0].retrieval is not None
    assert resolution.requests[0].candidates[0].retrieval.score == 0.91
    assert resolution.requests[0].evidence[0].locator == "fixture://selected-context"
    assert lifecycle.test_execution is not None
    assert lifecycle.test_execution.passed
    decision = store.read("resolution_decisions").to_pylist()[0]
    assert decision["context_identity"] == context_identity
    assert decision["selected_identity"] == candidate_identity
    sql_rows = lifecycle.query_standard_sql(
        "SELECT d.outcome, c.context_identity, p.task_id, i.representation_identity "
        "FROM resolution_decisions d "
        "JOIN resolution_candidates c ON c.candidate_identity = d.candidate_identity "
        "JOIN context_packages p ON p.context_model_id = d.context_model_id "
        "JOIN package_items i ON i.package_identity = p.package_identity"
    ).to_pylist()
    assert sql_rows == [{
        "outcome": "selected",
        "context_identity": context_identity,
        "task_id": "support-task",
        "representation_identity": context_identity,
    }]
    lineage_rows = lifecycle.query_standard_sql(
        "SELECT s.display_name AS source_name, e.edge_type, t.display_name AS target_name, t.node_type "
        "FROM lineage_edges e "
        "JOIN lineage_nodes s ON s.canonical_node_id = e.source_node_id "
        "JOIN lineage_nodes t ON t.canonical_node_id = e.target_node_id "
        "WHERE e.edge_type IN ('selected_by_resolution', 'included_in_package') "
        "ORDER BY e.edge_type"
    ).to_pylist()
    assert lineage_rows == [
        {
            "source_name": "Selected context",
            "edge_type": "included_in_package",
            "target_name": "Context package: support-task",
            "node_type": "context_package",
        },
        {
            "source_name": "Selected context",
            "edge_type": "selected_by_resolution",
            "target_name": "Resolution decision: selected",
            "node_type": "resolution_decision",
        },
    ]


def _contract_manifest(contract: dict) -> dict:
    return {
        "context_model_id": "contract.lifecycle",
        "context_model_revision": "1",
        "context_model_identity": stable_identity({"context_model": "contract.lifecycle"}),
        "compiler_version": "context-model-manifest.v0.1",
        "contracts": (contract,),
    }


def _base_contract(**overrides) -> dict:
    value = {
        "contract_id": "generic.context-package",
        "permitted_consumers": ["care-agent"],
        "permitted_tasks": ["answer-question"],
        "permitted_purposes": ["care-summary"],
        "allowed_evidence": ["source_snapshot"],
        "allowed_modalities": ["text"],
        "required_authority_level": "trusted",
        "valid_time_range": {"start": "2026-08-01T00:00:00+00:00", "end": "2026-08-31T00:00:00+00:00"},
        "historical_evidence_allowed": False,
        "max_items": 2,
        "max_tokens": 100,
        "max_bytes": 1000,
        "citation_required": True,
        "freshness": {"required": True},
        "abstention_conditions": ["missing_authoritative_evidence"],
        "prohibited_uses": ["marketing"],
    }
    value.update(overrides)
    return value


def _package_contract_run(tmp_path: Path, *, contract: dict | None = None,
                          context: dict | None = None,
                          output: dict | None = None,
                          package: dict | None = None):
    context_identity = stable_identity({"context": "contract-output"})
    package_identity = stable_identity({"package": "contract-package"})
    item = {
        "representation_identity": context_identity,
        "role": "resolved-context",
        "rank": 0,
        "token_estimate": 10,
        "evidence_identities": ["sha256:evidence"],
    }
    output_row = {
        "identity": context_identity,
        "role": "answer_fact",
        "kind": "context_representation",
        "display_name": "Answer fact",
        "payload": {"answer": "yes"},
        "status": "active",
        "authority_level": "trusted",
        "valid_until": "2026-08-20T00:00:00+00:00",
    }
    if output:
        output_row.update(output)
    package_row = {
        "package_identity": package_identity,
        "task_id": "answer-question",
        "package_kind": "task_context",
        "budget_tokens": 10,
        "items": [item],
    }
    if package:
        package_row.update(package)
    return materialize_context_model_lifecycle(
        store=LanceStore(tmp_path / "lance"),
        manifest=_contract_manifest(contract or _base_contract()),
        outputs=(output_row,),
        packages=(package_row,),
        contract_context={
            "consumer": "care-agent",
            "task_id": "answer-question",
            "purpose": "care-summary",
            "as_of": "2026-08-14T00:00:00+00:00",
            "evaluated_at": "2026-08-14T00:00:00+00:00",
            "evaluator_identity": "test.contract-evaluator",
            **dict(context or {}),
        },
    )


def test_context_contract_allows_and_publishes_package(tmp_path: Path) -> None:
    lifecycle = _package_contract_run(tmp_path)

    assert lifecycle.relation_counts["context_contracts"] == 1
    assert lifecycle.relation_counts["contract_evaluations"] == 1
    assert lifecycle.relation_counts["context_package_contracts"] == 1
    assert lifecycle.relation_counts["context_packages"] == 1
    assert lifecycle.contract_evaluations[0]["decision"] == "allow"
    assert lifecycle.test_execution is not None
    assert lifecycle.test_execution.passed
    rows = lifecycle.query_standard_sql(
        "SELECT p.package_identity, b.contract_identity, e.decision "
        "FROM context_packages p "
        "JOIN context_package_contracts b ON b.package_identity = p.package_identity "
        "JOIN contract_evaluations e ON e.evaluation_identity = b.evaluation_identity"
    ).to_pylist()
    assert rows[0]["decision"] == "allow"


def test_context_contract_denies_unauthorized_consumer_and_unsupported_task(tmp_path: Path) -> None:
    unauthorized = _package_contract_run(tmp_path / "consumer", context={"consumer": "unknown-agent"})
    unsupported = _package_contract_run(tmp_path / "task", context={"task_id": "marketing"})

    assert unauthorized.relation_counts["context_packages"] == 0
    assert unauthorized.contract_evaluations[0]["decision"] == "deny"
    assert unauthorized.contract_evaluations[0]["violated_rules"] == ["unauthorized_consumer"]
    assert unsupported.relation_counts["context_packages"] == 0
    assert unsupported.contract_evaluations[0]["decision"] == "deny"
    assert "unsupported_task" in unsupported.contract_evaluations[0]["violated_rules"]
    assert "prohibited_use" in unsupported.contract_evaluations[0]["violated_rules"]


def test_context_contract_abstains_when_authoritative_evidence_is_missing(tmp_path: Path) -> None:
    lifecycle = _package_contract_run(tmp_path, output={"authority_level": ""}, context={"authority_level": ""})

    assert lifecycle.relation_counts["context_packages"] == 0
    assert lifecycle.relation_counts["context_package_contracts"] == 0
    assert lifecycle.relation_counts["contract_evaluations"] == 1
    assert lifecycle.contract_evaluations[0]["decision"] == "abstain"
    assert lifecycle.contract_evaluations[0]["violated_rules"] == ["missing_authoritative_evidence"]


def test_context_contract_denies_freshness_superseded_budget_and_citation_violations(tmp_path: Path) -> None:
    expired = _package_contract_run(tmp_path / "expired", output={"valid_until": "2026-08-01T00:00:00+00:00"})
    superseded = _package_contract_run(tmp_path / "superseded", output={"status": "superseded"})
    historical = _package_contract_run(
        tmp_path / "historical",
        contract=_base_contract(historical_evidence_allowed=True),
        output={"status": "superseded"},
    )
    budget = _package_contract_run(tmp_path / "budget", package={"budget_tokens": 101})
    missing_citation = _package_contract_run(
        tmp_path / "citation",
        package={"items": [{
            "representation_identity": stable_identity({"context": "contract-output"}),
            "role": "resolved-context",
            "rank": 0,
            "token_estimate": 10,
            "evidence_identities": [],
        }]},
    )

    assert expired.contract_evaluations[0]["decision"] == "deny"
    assert expired.contract_evaluations[0]["violated_rules"] == ["freshness_expired"]
    assert superseded.contract_evaluations[0]["violated_rules"] == ["superseded_evidence_rejected"]
    assert historical.contract_evaluations[0]["decision"] == "allow"
    assert budget.contract_evaluations[0]["violated_rules"] == ["max_tokens_exceeded"]
    assert missing_citation.contract_evaluations[0]["violated_rules"] == ["missing_required_citations"]


def test_context_contract_identity_ignores_display_name_but_changes_for_semantic_rules(tmp_path: Path) -> None:
    first = _package_contract_run(tmp_path / "first", contract=_base_contract(display_name="Readable A"))
    renamed = _package_contract_run(tmp_path / "renamed", contract=_base_contract(display_name="Readable B"))
    changed = _package_contract_run(tmp_path / "changed", contract=_base_contract(max_items=1))

    first_contract = first.materialization.relations["context_contracts"][0]
    renamed_contract = renamed.materialization.relations["context_contracts"][0]
    changed_contract = changed.materialization.relations["context_contracts"][0]
    assert first_contract["contract_identity"] == renamed_contract["contract_identity"]
    assert first_contract["contract_version"] == renamed_contract["contract_version"]
    assert first_contract["contract_identity"] != changed_contract["contract_identity"]
    assert first_contract["contract_version"] != changed_contract["contract_version"]
    assert first.contract_evaluations[0]["evaluation_identity"] != changed.contract_evaluations[0]["evaluation_identity"]


def test_cached_package_cannot_bypass_newer_context_contract_evaluation(tmp_path: Path) -> None:
    store_path = tmp_path / "lance"
    allowed = _package_contract_run(store_path / "allowed")
    package_identity = allowed.materialization.relations["context_packages"][0]["package_identity"]

    denied = _package_contract_run(
        store_path / "allowed",
        contract=_base_contract(permitted_consumers=["new-agent"]),
    )

    assert allowed.relation_counts["context_packages"] == 1
    assert denied.contract_evaluations[0]["package_identity"] == package_identity
    assert denied.contract_evaluations[0]["decision"] == "deny"
    assert denied.relation_counts["context_packages"] == 0
