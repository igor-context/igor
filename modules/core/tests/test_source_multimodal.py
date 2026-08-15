from __future__ import annotations

import pytest

from igor_core import (
    ConnectorBinding,
    ConnectorFieldBinding,
    ConnectorResourceBinding,
    ContentPart,
    ContextSourceContract,
    EnrichmentRecipe,
    SchemaDescriptor,
    SourceFieldRequirement,
    SourceResourceContract,
    stable_identity,
    validate_schema_payload,
)


def source_contract() -> ContextSourceContract:
    return ContextSourceContract(
        contract_id="media.asset-source",
        revision="1",
        domain="media",
        resources=(
            SourceResourceContract(
                resource_id="asset",
                resource_kind="image",
                identity_concepts=("asset_id",),
                fields=(
                    SourceFieldRequirement(concept_id="asset_id", logical_type="string", required=True),
                    SourceFieldRequirement(concept_id="caption", logical_type="string", required=True),
                    SourceFieldRequirement(concept_id="updated_at", logical_type="datetime", required=True),
                ),
                accepted_media_types=("image/png",),
                change_mode="incremental",
                cursor_concept="updated_at",
                deletion_semantics="tombstone",
            ),
        ),
    )


def test_connector_binding_must_cover_required_concepts_and_cursor() -> None:
    contract = source_contract()
    binding = ConnectorBinding(
        binding_id="fixture.media",
        revision="1",
        source_contract_identity=contract.identity,
        connector="fixture",
        deployment_ref="local-media",
        resources=(
            ConnectorResourceBinding(
                resource_id="asset",
                source_resource="images",
                fields=(
                    ConnectorFieldBinding(concept_id="asset_id", source_field="id"),
                    ConnectorFieldBinding(concept_id="caption", source_field="description"),
                    ConnectorFieldBinding(concept_id="updated_at", source_field="modified"),
                ),
                cursor_field="modified",
            ),
        ),
    )
    binding.validate_against(contract)

    invalid = binding.model_copy(update={
        "resources": (binding.resources[0].model_copy(update={"cursor_field": "wrong"}),),
    })
    with pytest.raises(ValueError, match="cursor binding mismatch"):
        invalid.validate_against(contract)


def test_content_part_hashes_text_and_keeps_image_reference_opaque() -> None:
    text = ContentPart(
        kind="text",
        media_type="text/plain",
        text="invoice total EUR 42",
        content_sha256=stable_identity("invoice total EUR 42"),
    )
    assert text.content_ref is None
    image = ContentPart(
        kind="image",
        media_type="image/png",
        content_ref="fixture://media/invoice.png",
        content_sha256="sha256:" + "a" * 64,
        locator={"page": "1"},
    )
    assert image.text is None
    with pytest.raises(ValueError, match="text content hash mismatch"):
        ContentPart.model_validate({**text.model_dump(), "content_sha256": "sha256:" + "b" * 64})


def test_registered_json_schema_rejects_types_enums_nested_and_extra_fields() -> None:
    schema = SchemaDescriptor(
        schema_version="0.1",
        schema_id="media.image-observation",
        revision="1",
        domain="media",
        json_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_kind", "quality", "region"],
            "properties": {
                "asset_kind": {"type": "string", "enum": ["invoice", "receipt"]},
                "quality": {"type": "number", "minimum": 0, "maximum": 1},
                "region": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["page"],
                    "properties": {"page": {"type": "integer", "minimum": 1}},
                },
            },
        },
    )
    valid = {"asset_kind": "invoice", "quality": 0.9, "region": {"page": 1}}
    validate_schema_payload(schema, valid)
    for invalid in (
        {"asset_kind": "memo", "quality": 0.9, "region": {"page": 1}},
        {"asset_kind": "invoice", "quality": "high", "region": {"page": 1}},
        {"asset_kind": "invoice", "quality": 0.9, "region": {"page": 0}},
        {**valid, "unexpected": True},
    ):
        with pytest.raises(ValueError, match="payload violates schema"):
            validate_schema_payload(schema, invalid)

    recipe = EnrichmentRecipe(
        recipe_id="media.classify-image",
        revision="1",
        accepted_representation_types=("image",),
        accepted_media_types=("image/png",),
        output_schema_identity=schema.identity,
        prompt_version="media-image-v1",
        taxonomy_version="media-taxonomy-v1",
    )
    assert recipe.identity.startswith("sha256:")
