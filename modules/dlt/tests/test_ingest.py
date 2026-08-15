from igor_dlt import IngestConfig, ingest_bound_records, ingest_records
from igor_core import (ConnectorBinding, ConnectorFieldBinding, ConnectorResourceBinding,
                       ContextSourceContract, SourceFieldRequirement, SourceResourceContract)


ROWS = [
    {"record_id": "ticket-001", "channel": "email", "text": "I cannot reset my password."},
    {"record_id": "ticket-002", "channel": "chat", "text": "I was charged twice."},
]


def test_ingest_emits_canonical_records_and_load_metadata(tmp_path) -> None:
    result = ingest_records(ROWS, IngestConfig("test", "support", str(tmp_path / "lance"), "test/support"))
    assert result.records[0]["context_unit_id"].startswith("ctx:support:")
    assert all(row["domain"] == "support" and row["environment"] == "test" for row in result.records)
    assert result.load_metadata["pipeline"]["pipeline_name"] == "igor_test_support"


def test_domains_use_distinct_namespaces(tmp_path) -> None:
    root = str(tmp_path / "lance")
    support = ingest_records(ROWS[:1], IngestConfig("test", "support", root, "test/support"))
    finance = ingest_records(ROWS[1:], IngestConfig("test", "finance", root, "test/finance"))
    assert support.config.namespace_name != finance.config.namespace_name
    assert {row["domain"] for row in support.records} == {"support"}
    assert {row["domain"] for row in finance.records} == {"finance"}
    assert (tmp_path / "lance" / "test" / "support").exists()
    assert not list((tmp_path / "lance" / "test" / "support").rglob("*finance*"))


def test_bound_ingestion_applies_domain_intent_to_connector_fields(tmp_path) -> None:
    contract = ContextSourceContract(contract_id="media.assets", revision="1", domain="media", resources=(
        SourceResourceContract(resource_id="asset", resource_kind="image", identity_concepts=("asset_id",), fields=(
            SourceFieldRequirement(concept_id="asset_id", logical_type="string", required=True),
            SourceFieldRequirement(concept_id="caption", logical_type="string", required=True),
            SourceFieldRequirement(concept_id="updated_at", logical_type="datetime", required=True),
        ), accepted_media_types=("image/png",), change_mode="incremental", cursor_concept="updated_at",
            deletion_semantics="tombstone"),
    ))
    binding = ConnectorBinding(binding_id="fixture.media", revision="1", source_contract_identity=contract.identity,
        connector="fixture", deployment_ref="test", resources=(ConnectorResourceBinding(
            resource_id="asset", source_resource="images", fields=(
                ConnectorFieldBinding(concept_id="asset_id", source_field="id"),
                ConnectorFieldBinding(concept_id="caption", source_field="description"),
                ConnectorFieldBinding(concept_id="updated_at", source_field="modified"),
            ), cursor_field="modified"),))
    result = ingest_bound_records([{
        "_resource_id": "asset", "_media_type": "image/png", "_content_ref": "file:///tmp/image.png",
        "_content_sha256": "sha256:" + "a" * 64, "_observed_at": "2026-08-13T00:00:00Z",
        "id": "asset-1", "description": "red status image", "modified": "2026-08-13T00:00:00Z",
    }], IngestConfig("test", "media", str(tmp_path / "lance"), "test/media"), contract, binding)
    row = result.records[0]
    assert row["payload"]["asset_id"] == "asset-1"
    assert row["source_contract_identity"] == contract.identity
    assert row["connector_binding_identity"] == binding.identity
