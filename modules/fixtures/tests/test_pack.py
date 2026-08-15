from pathlib import Path
from tempfile import TemporaryDirectory
import json

import pytest

from igor_fixtures.pack import _artifact_descriptor, generate_support_pack, resolve_context_model_reference, validate_scenario_pack


def test_generated_pack_is_deterministic() -> None:
    with TemporaryDirectory() as first, TemporaryDirectory() as second:
        generate_support_pack(first)
        generate_support_pack(second)
        assert sorted(Path(first).rglob("*"))
        for left in sorted(Path(first).rglob("*.json")):
            assert left.read_bytes() == (Path(second) / left.relative_to(first)).read_bytes()


def test_pack_validates_and_has_stable_identity() -> None:
    with TemporaryDirectory() as directory:
        generate_support_pack(directory)
        result = validate_scenario_pack(directory)
        assert result["valid"] is True
        assert result["artifact_count"] == 5
        assert result["scenario_identity"].startswith("sha256:")


def test_labels_are_isolated_from_source_fixture() -> None:
    with TemporaryDirectory() as directory:
        generate_support_pack(directory)
        source = json.loads((Path(directory) / "fixtures/source.json").read_text())
        assert all("label" not in record and "intent" not in record for record in source["records"])


def test_mutation_sets_are_disjoint() -> None:
    with TemporaryDirectory() as directory:
        generate_support_pack(directory)
        mutation = json.loads((Path(directory) / "mutations/source-change.json").read_text())
        assert set(mutation["expected_affected"]).isdisjoint(mutation["expected_unaffected"])


def test_incomplete_scorecard_is_rejected() -> None:
    with TemporaryDirectory() as directory:
        generate_support_pack(directory)
        scorecard_path = Path(directory) / "scorecard.json"
        scorecard = json.loads(scorecard_path.read_text())
        del scorecard["metrics"][0]["formula"]
        scorecard_path.write_text(json.dumps(scorecard))
        manifest_path = Path(directory) / "scenario.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][-1] = _artifact_descriptor(Path(directory), "scorecard.json")
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="scorecard metric missing formula"):
            validate_scenario_pack(directory)


def test_context_model_reference_resolves_to_content_identity(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text("context_model_id: support.v1\n")
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "scenario.json").write_text(json.dumps({"context_model_ref": "../canonical.yaml"}))
    resolved = resolve_context_model_reference(scenario)
    assert resolved is not None and resolved["content_sha256"].startswith("sha256:")
