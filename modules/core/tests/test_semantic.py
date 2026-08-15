from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from igor_core import (
    ContextPackage,
    Derivation,
    Embedding,
    EmbeddingSpace,
    Representation,
    SchemaDescriptor,
    SourceSnapshot,
    affected_outputs,
    validate_context_artifact,
)


ID1 = "sha256:" + "1" * 64
ID2 = "sha256:" + "2" * 64
ID3 = "sha256:" + "3" * 64


def test_context_examples_validate() -> None:
    from pathlib import Path

    for path in (Path(__file__).parents[1] / "examples" / "context-ir").glob("*.json"):
        package = validate_context_artifact(str(path))
        assert package.identity.startswith("sha256:")


def test_source_and_representation_identities_include_schema_and_payload() -> None:
    schema = SchemaDescriptor(
        schema_version="0.1", schema_id="document.text", revision="1",
    )
    source = SourceSnapshot(
        ir_version="0.1", source_system="files", source_key="doc-1",
        observed_at=datetime(2026, 8, 12, tzinfo=timezone.utc), content_ref="blob://opaque/doc-1",
        content_sha256="sha256:" + "a" * 64, media_type="text/plain", schema_ref=schema,
    )
    representation = Representation(
        ir_version="0.1", representation_type="text", schema_ref=schema,
        source_snapshot_ids=(source.identity,), payload="hello",
    )
    changed = representation.model_copy(update={"payload": "goodbye"})
    assert source.identity.startswith("sha256:")
    assert representation.identity != changed.identity


def test_embedding_requires_space_dimension() -> None:
    space = EmbeddingSpace(
        ir_version="0.1", provider="fixture", model="hash", model_revision="1",
        dimension=2, dtype="float32", metric="cosine", normalized=True,
        input_schema_identity="document.text:v1",
    )
    embedding = Embedding(
        ir_version="0.1", representation_identity=ID1, space_identity=space.identity,
        vector=[0.1, 0.2],
    )
    assert embedding.identity.startswith("sha256:")


def test_selective_invalidation_is_transitive() -> None:
    derivations = [
        Derivation(ir_version="0.1", operation="extract", input_identities=(ID1,), output_identity=ID2, code_revision="c1", configuration_identity="cfg1"),
        Derivation(ir_version="0.1", operation="compile", input_identities=(ID2,), output_identity=ID3, code_revision="c1", configuration_identity="cfg1"),
    ]
    assert affected_outputs({ID1}, derivations) == {ID1, ID2, ID3}


def test_context_package_rejects_budget_overrun() -> None:
    with pytest.raises(ValidationError, match="token budget"):
        ContextPackage(
            ir_version="0.1", task_id="task", schema_revision="v1",
            items=[{"representation_identity": ID1, "role": "context", "rank": 0, "token_estimate": 11}],
            budget_tokens=10, created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )
