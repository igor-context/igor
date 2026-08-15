from pathlib import Path

import pytest
from pydantic import ValidationError

from igor_core import ModelProfile, ReferenceProfile, load_model_profile, load_reference_profile


def model_profile_data(capability: str, profile_id: str, model: str) -> dict:
    return {
        "schema_version": "0.1",
        "profile_id": profile_id,
        "capability": capability,
        "provider": "deterministic",
        "model": model,
        "revision": "1",
        "parameters": {},
    }


def resolved_profile_data() -> dict:
    embedding = model_profile_data("embedding", "deterministic-hash-v0", "hash-embedding")
    embedding["parameters"] = {"dimensions": 8}
    return {
        "schema_version": "0.1",
        "profile_id": "local-deterministic-v0",
        "prompt_version": "scaffold-v1",
        "taxonomy_version": "scaffold-v1",
        "embedding": embedding,
        "completion": model_profile_data(
            "completion", "deterministic-completion-v0", "fixture-enricher"
        ),
    }


def write_model_profile(path: Path, capability: str, profile_id: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''schema_version: "0.1"
profile_id: {profile_id}
capability: {capability}
provider: deterministic
model: {model}
revision: "1"
parameters: {{}}
''',
        encoding="utf-8",
    )


def write_composition(path: Path) -> None:
    path.write_text(
        '''schema_version: "0.1"
profile_id: local-test
prompt_version: p1
taxonomy_version: t1
embedding_profile: providers/embeddings/hash.yaml
completion_profile: providers/completions/fixture.yaml
''',
        encoding="utf-8",
    )


def test_profile_identity_changes_with_model_configuration() -> None:
    baseline = ReferenceProfile.model_validate(resolved_profile_data())
    changed_data = resolved_profile_data()
    changed_data["embedding"]["parameters"]["dimensions"] = 16
    changed = ReferenceProfile.model_validate(changed_data)

    assert baseline.identity.startswith("sha256:")
    assert baseline.identity != changed.identity


def test_model_profile_rejects_unknown_fields() -> None:
    data = model_profile_data("embedding", "test", "hash")
    data["unexpected"] = True

    with pytest.raises(ValidationError):
        ModelProfile.model_validate(data)


def test_model_profile_rejects_secret_parameters() -> None:
    data = model_profile_data("completion", "test", "fixture")
    data["parameters"] = {"api-key": "must-not-live-here"}

    with pytest.raises(ValidationError, match="cannot contain secret field"):
        ModelProfile.model_validate(data)


def test_load_reference_profile_resolves_independent_components(tmp_path: Path) -> None:
    write_model_profile(
        tmp_path / "providers/embeddings/hash.yaml", "embedding", "hash-v0", "hash"
    )
    write_model_profile(
        tmp_path / "providers/completions/fixture.yaml",
        "completion",
        "fixture-v0",
        "fixture",
    )
    composition = tmp_path / "reference.yaml"
    write_composition(composition)

    profile = load_reference_profile(composition)

    assert profile.profile_id == "local-test"
    assert profile.embedding.profile_id == "hash-v0"
    assert profile.completion.profile_id == "fixture-v0"


def test_load_reference_profile_rejects_capability_mismatch(tmp_path: Path) -> None:
    write_model_profile(
        tmp_path / "providers/embeddings/hash.yaml", "completion", "wrong-v0", "hash"
    )
    write_model_profile(
        tmp_path / "providers/completions/fixture.yaml",
        "completion",
        "fixture-v0",
        "fixture",
    )
    composition = tmp_path / "reference.yaml"
    write_composition(composition)

    with pytest.raises(ValueError, match="expected embedding"):
        load_reference_profile(composition)


def test_load_reference_profile_rejects_legacy_inline_combination(tmp_path: Path) -> None:
    path = tmp_path / "reference.yaml"
    path.write_text(
        '''schema_version: "0.1"
profile_id: combined
prompt_version: p1
taxonomy_version: t1
embedding: {provider: deterministic, model: hash, revision: "1"}
completion: {provider: deterministic, model: fixture, revision: "1"}
''',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_reference_profile(path)


def test_resolved_identity_depends_on_components_not_reference_paths(tmp_path: Path) -> None:
    for root in (tmp_path / "first", tmp_path / "second"):
        write_model_profile(
            root / "providers/embeddings/hash.yaml", "embedding", "hash-v0", "hash"
        )
        write_model_profile(
            root / "providers/completions/fixture.yaml",
            "completion",
            "fixture-v0",
            "fixture",
        )
        write_composition(root / "reference.yaml")

    first = load_reference_profile(tmp_path / "first/reference.yaml")
    second = load_reference_profile(tmp_path / "second/reference.yaml")

    assert first.identity == second.identity


def test_load_model_profile_enforces_expected_capability(tmp_path: Path) -> None:
    path = tmp_path / "completion.yaml"
    write_model_profile(path, "completion", "completion-v0", "fixture")

    with pytest.raises(ValueError, match="expected embedding"):
        load_model_profile(path, expected_capability="embedding")
