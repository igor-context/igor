from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import dlt
from igor_core import (
    ConnectorBinding,
    ContextSourceContract,
    LineageEdge,
    LineageGraph,
    LineageNode,
    ProducerIdentity,
    stable_identity,
)


@dataclass(frozen=True)
class IngestConfig:
    environment: str
    domain: str
    bucket_url: str
    namespace_name: str
    dataset_name: str = "ingestion"
    storage_options: Mapping[str, str] | None = None
    state_dir: str | None = None
    pipeline_dir: str | None = None

    def __post_init__(self) -> None:
        for field in ("environment", "domain", "bucket_url", "namespace_name", "dataset_name"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} cannot be empty")
        if "/" in self.domain or "/" in self.environment:
            raise ValueError("environment and domain must be path-safe identifiers")


@dataclass(frozen=True)
class IngestResult:
    records: tuple[dict[str, Any], ...]
    load_metadata: dict[str, Any]
    config: IngestConfig

    def as_dict(self) -> dict[str, Any]:
        return {"records": list(self.records), "load_metadata": self.load_metadata, "config": asdict(self.config)}

    def lineage(self, run_identity: str) -> LineageGraph:
        """Emit source → load → context-unit facts without exposing dlt types."""
        producer = ProducerIdentity(name="igor-dlt", revision="1.29.0")
        configuration_revision = stable_identity({
            "environment": self.config.environment,
            "domain": self.config.domain,
            "namespace_name": self.config.namespace_name,
            "dataset_name": self.config.dataset_name,
            "source_definition_identity": self.records[0].get("source_definition_identity") if self.records else None,
            "connector_binding_identity": self.records[0].get("connector_binding_identity") if self.records else None,
            "snapshot_identity": self.records[0].get("snapshot_identity") if self.records else None,
        })
        load = LineageNode(
            schema_version="0.1", node_type="ingestion_load",
            key=stable_identity(self.load_metadata), run_identity=run_identity,
            producer=producer, configuration_revision=configuration_revision,
            metadata={"namespace_name": self.config.namespace_name},
        )
        nodes = [load]
        edges = []
        for record in self.records:
            source = LineageNode(
                schema_version="0.1", node_type="source_record", key=str(record["record_id"]),
                run_identity=run_identity, source_identity=str(record["record_id"]),
                producer=producer, configuration_revision=configuration_revision,
            )
            context = LineageNode(
                schema_version="0.1", node_type="context_unit", key=str(record["context_unit_id"]),
                run_identity=run_identity, source_identity=str(record["record_id"]),
                producer=producer, configuration_revision=configuration_revision,
            )
            nodes.append(source)
            if record.get("content_sha256"):
                observation = LineageNode(
                    schema_version="0.1", node_type="source_observation", key=str(record["content_sha256"]),
                    run_identity=run_identity, source_identity=str(record["record_id"]),
                    producer=producer, configuration_revision=configuration_revision,
                    metadata={"content_ref": str(record.get("content_ref", "")), "media_type": str(record.get("media_type", ""))},
                )
                nodes.append(observation)
                edges.append(LineageEdge(schema_version="0.1", edge_type="observed", source_node_id=source.identity, target_node_id=observation.identity, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision))
                ingest_source_node = observation
            else:
                ingest_source_node = source
            nodes.append(context)
            edges.extend((
                LineageEdge(schema_version="0.1", edge_type="ingested", source_node_id=ingest_source_node.identity, target_node_id=load.identity, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision),
                LineageEdge(schema_version="0.1", edge_type="canonicalized", source_node_id=load.identity, target_node_id=context.identity, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision),
            ))
        return LineageGraph(schema_version="0.1", run_identity=run_identity, nodes=tuple(nodes), edges=tuple(edges))


def _canonical_records(records: Sequence[Mapping[str, Any]], config: IngestConfig) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for source in records:
        for field in ("record_id", "channel", "text"):
            if not str(source.get(field, "")).strip():
                raise ValueError(f"source record missing {field}")
        record_id = str(source["record_id"])
        result.append(
            {
                "record_id": record_id,
                "channel": str(source["channel"]),
                "text": str(source["text"]),
                "environment": config.environment,
                "domain": config.domain,
                "context_unit_id": f"ctx:{config.domain}:{stable_identity(record_id)[:12]}",
            }
        )
    if not result:
        raise ValueError("source records cannot be empty")
    if len({record["record_id"] for record in result}) != len(result):
        raise ValueError("source record IDs must be unique")
    return tuple(result)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "asdict"):
        return _jsonable(value.asdict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def ingest_records(records: Sequence[Mapping[str, Any]], config: IngestConfig) -> IngestResult:
    canonical = _canonical_records(records, config)
    destination = dlt.destinations.lance(
        storage={
            "bucket_url": config.bucket_url,
            "namespace_name": config.namespace_name,
            "options": dict(config.storage_options or {}),
        }
    )
    pipeline = dlt.pipeline(
        pipeline_name=f"igor_{config.environment}_{config.domain}",
        destination=destination,
        dataset_name=config.dataset_name,
        pipelines_dir="/tmp/igor-dlt-pipelines",
    )
    load_info = pipeline.run(list(canonical), table_name="ingested_records", write_disposition="replace")
    return IngestResult(canonical, _jsonable(load_info.asdict()), config)


def ingest_bound_records(
    records: Sequence[Mapping[str, Any]],
    config: IngestConfig,
    contract: ContextSourceContract,
    binding: ConnectorBinding,
) -> IngestResult:
    """Apply a validated connector binding to already-acquired raw observations."""
    binding.validate_against(contract)
    requirements = {item.resource_id: item for item in contract.resources}
    bindings = {item.resource_id: item for item in binding.resources}
    canonical: list[dict[str, Any]] = []
    for source in records:
        resource_id = str(source.get("_resource_id", ""))
        if resource_id not in requirements:
            raise ValueError(f"raw record has unknown resource: {resource_id}")
        requirement = requirements[resource_id]
        resource_binding = bindings[resource_id]
        payload = {
            field.concept_id: source.get(field.source_field)
            for field in resource_binding.fields
            if field.source_field in source
        }
        missing = {
            field.concept_id for field in requirement.fields
            if field.required and payload.get(field.concept_id) is None
        }
        if missing:
            raise ValueError(f"raw record missing required concepts: {sorted(missing)}")
        identity_values = [str(payload.get(concept, "")).strip() for concept in requirement.identity_concepts]
        if any(not value for value in identity_values):
            raise ValueError("raw record has empty identity concept")
        media_type = str(source.get("_media_type", ""))
        if media_type not in requirement.accepted_media_types:
            raise ValueError(f"raw record media type not accepted: {media_type}")
        source_key = ":".join(identity_values)
        content_hash = str(source.get("_content_sha256", ""))
        if not content_hash.startswith("sha256:"):
            raise ValueError("raw record requires a sha256 content identity")
        observed_at = str(source.get("_observed_at", ""))
        content_ref = str(source.get("_content_ref", ""))
        if not observed_at or not content_ref:
            raise ValueError("raw record requires observed time and content reference")
        canonical.append({
            "record_id": source_key,
            "source_system": binding.connector,
            "source_key": source_key,
            "resource_id": resource_id,
            "observed_at": observed_at,
            "content_ref": content_ref,
            "content_sha256": content_hash,
            "media_type": media_type,
            "payload": payload,
            "environment": config.environment,
            "domain": config.domain,
            "source_contract_identity": contract.identity,
            "connector_binding_identity": binding.identity,
            "context_unit_id": f"ctx:{config.domain}:{stable_identity({'resource': resource_id, 'key': source_key})[:12]}",
        })
    if not canonical:
        raise ValueError("source records cannot be empty")
    canonical.sort(key=lambda item: (item["resource_id"], item["source_key"]))
    if len({(item["resource_id"], item["source_key"]) for item in canonical}) != len(canonical):
        raise ValueError("bound source record identities must be unique")

    destination = dlt.destinations.lance(
        storage={
            "bucket_url": config.bucket_url,
            "namespace_name": config.namespace_name,
            "options": dict(config.storage_options or {}),
        }
    )
    pipeline = dlt.pipeline(
        pipeline_name=f"igor_{config.environment}_{config.domain}_bound",
        destination=destination,
        dataset_name=config.dataset_name,
        pipelines_dir="/tmp/igor-dlt-pipelines",
    )
    load_info = pipeline.run(canonical, table_name="ingested_records", write_disposition="replace")
    return IngestResult(tuple(canonical), _jsonable(load_info.asdict()), config)
