import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from igor_core import (
    ArtifactIndex,
    ArtifactReference,
    ProducerIdentity,
    RunManifest,
    StageResult,
    validate_run_directory,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

EXAMPLE_RUN = Path(__file__).parents[1] / "examples" / "run-v0.1"


def test_valid_example_run_has_stable_referential_integrity() -> None:
    package = validate_run_directory(EXAMPLE_RUN)

    assert package.manifest.identity == package.artifact_index.run_identity
    assert len(package.artifact_index.artifacts) == 1
    assert package.stages[0].output_artifact_identities == [
        package.artifact_index.artifacts[0].identity
    ]


def test_identity_links_change_run_identity() -> None:
    manifest_data = json.loads((EXAMPLE_RUN / "manifest.json").read_text(encoding="utf-8"))
    baseline = RunManifest.model_validate(manifest_data)
    changed = manifest_data.copy()
    changed["identity_links"] = {**changed["identity_links"], "evaluator": "igor-evaluator:0.2"}

    assert baseline.identity != RunManifest.model_validate(changed).identity


def test_invalid_example_rejects_mismatched_run_identity() -> None:
    with pytest.raises(ValidationError, match="run_identity"):
        validate_run_directory(EXAMPLE_RUN.parent / "invalid-run")


def test_run_rejects_integrity_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(EXAMPLE_RUN, run)
    (run / "artifacts/placeholder.json").write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_run_directory(run)


def test_run_rejects_missing_artifact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(EXAMPLE_RUN, run)
    (run / "artifacts/placeholder.json").unlink()

    with pytest.raises(ValueError, match="missing or unsafe artifact"):
        validate_run_directory(run)


def test_stage_state_requires_matching_error() -> None:
    producer = ProducerIdentity(name="implementation", revision="r1")

    with pytest.raises(ValidationError, match="failed stages must declare an error"):
        StageResult(
            schema_version="0.1",
            stage_id="stage",
            status="failed",
            producer=producer,
        )


def test_artifact_rejects_paths_that_escape_run_directory() -> None:
    with pytest.raises(ValidationError, match="artifact paths"):
        ArtifactReference(
            schema_version="0.1",
            path="../outside.json",
            producer={"name": "implementation", "revision": "r1"},
            media_type="application/json",
            sha256="sha256:" + "0" * 64,
            size_bytes=0,
        )


def test_artifact_index_rejects_duplicate_paths() -> None:
    artifact = {
        "schema_version": "0.1",
        "path": "artifact.json",
        "producer": {"name": "implementation", "revision": "r1"},
        "media_type": "application/json",
        "sha256": "sha256:" + "0" * 64,
        "size_bytes": 0,
    }

    with pytest.raises(ValidationError, match="duplicate artifact paths"):
        ArtifactIndex(
            schema_version="0.1",
            run_identity="sha256:" + "1" * 64,
            artifacts=[artifact, {**artifact, "producer": {"name": "other", "revision": "r1"}}],
        )


def test_contracts_export_json_schema() -> None:
    schema = ArtifactReference.model_json_schema()

    assert "properties" in schema
    assert schema["properties"]["sha256"]["type"] == "string"


def test_lineage_graph_has_stable_closed_nodes_and_edges() -> None:
    run_identity = "sha256:" + "2" * 64
    producer = ProducerIdentity(name="test", revision="1")
    source = LineageNode(schema_version="0.1", node_type="source_record", key="ticket-1", run_identity=run_identity, producer=producer, configuration_revision="config-1")
    context = LineageNode(schema_version="0.1", node_type="context_unit", key="ctx-1", run_identity=run_identity, producer=producer, configuration_revision="config-1")
    edge = LineageEdge(schema_version="0.1", edge_type="canonicalized", source_node_id=source.identity, target_node_id=context.identity, run_identity=run_identity, producer=producer, configuration_revision="config-1")
    graph = LineageGraph(schema_version="0.1", run_identity=run_identity, nodes=(source, context), edges=(edge,))
    assert graph.identity.startswith("sha256:")
    with pytest.raises(ValidationError, match="undeclared node"):
        LineageGraph(schema_version="0.1", run_identity=run_identity, nodes=(source,), edges=(edge,))


def test_lineage_presentation_is_not_part_of_canonical_identity() -> None:
    run_identity = "sha256:" + "3" * 64
    producer = ProducerIdentity(name="test", revision="1")
    first = LineageNode(
        schema_version="0.1", node_type="source_record", key="page-1",
        run_identity=run_identity, producer=producer, configuration_revision="config-1",
        presentation={"display_name": "Page 1", "preview": "first label"},
    )
    renamed = first.model_copy(update={"presentation": {"display_name": "Renamed page", "preview": "second label"}})
    assert first.identity == renamed.identity


def test_lineage_semantic_context_model_change_changes_identity() -> None:
    run_identity = "sha256:" + "4" * 64
    producer = ProducerIdentity(name="test", revision="1")
    first = LineageNode(
        schema_version="0.1", node_type="context_unit", key="ctx-1",
        run_identity=run_identity, producer=producer, configuration_revision="config-1",
        presentation={"context_model_id": "support.v1"},
    )
    changed = first.model_copy(update={"presentation": {"context_model_id": "support.v2"}})
    assert first.identity != changed.identity
