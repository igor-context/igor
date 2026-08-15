from pathlib import Path
import json

from igor_core.cli import main

EXAMPLE_RUN = Path(__file__).parents[1] / "examples" / "run-v0.1"


def test_profile_validate_emits_machine_readable_identity(
    tmp_path: Path,
    capsys,
) -> None:
    embeddings = tmp_path / "providers/embeddings"
    completions = tmp_path / "providers/completions"
    embeddings.mkdir(parents=True)
    completions.mkdir(parents=True)
    embeddings.joinpath("hash.yaml").write_text(
        'schema_version: "0.1"\nprofile_id: hash-v0\ncapability: embedding\nprovider: deterministic\nmodel: hash\nrevision: "1"\n',
        encoding="utf-8",
    )
    completions.joinpath("fixture.yaml").write_text(
        'schema_version: "0.1"\nprofile_id: fixture-v0\ncapability: completion\nprovider: deterministic\nmodel: fixture\nrevision: "1"\n',
        encoding="utf-8",
    )
    path = tmp_path / "reference.yaml"
    path.write_text(
        """\
schema_version: "0.1"
profile_id: local-test
prompt_version: p1
taxonomy_version: t1
embedding_profile: providers/embeddings/hash.yaml
completion_profile: providers/completions/fixture.yaml
""",
        encoding="utf-8",
    )

    assert main(["profile", "validate", str(path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"profile_id": "local-test"' in output
    assert '"profile_identity": "sha256:' in output
    assert '"embedding_profile_id": "hash-v0"' in output
    assert '"completion_profile_id": "fixture-v0"' in output


def test_model_profile_validate_enforces_capability(tmp_path: Path, capsys) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        'schema_version: "0.1"\nprofile_id: hash-v0\ncapability: embedding\n'
        'provider: deterministic\nmodel: hash\nrevision: "1"\n',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "profile",
                "validate-model",
                str(path),
                "--capability",
                "embedding",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"capability": "embedding"' in output
    assert '"profile_identity": "sha256:' in output


def test_profile_validate_returns_two_for_invalid_profile(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "reference.yaml"
    path.write_text("profile_id: incomplete\n", encoding="utf-8")

    assert main(["profile", "validate", str(path)]) == 2
    assert "invalid IGOR contract" in capsys.readouterr().err


def test_contract_validate_emits_run_identity(capsys) -> None:
    assert main(["contract", "validate", str(EXAMPLE_RUN), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["artifact_count"] == 1
    assert output["stage_count"] == 1


def test_contract_schema_exports_artifact_reference(capsys) -> None:
    assert main(["contract", "schema"]) == 0

    schemas = json.loads(capsys.readouterr().out)
    assert "artifact-reference" in schemas
    assert "sha256" in schemas["artifact-reference"]["properties"]


def test_context_validate_emits_package_identity(capsys) -> None:
    path = EXAMPLE_RUN.parent / "context-ir" / "contract.json"
    assert main(["context", "validate", str(path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["item_count"] == 1
