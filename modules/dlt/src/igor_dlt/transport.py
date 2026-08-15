from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import dlt

from igor_core import stable_identity

from .ingest import IngestConfig, IngestResult


class IngestionError(RuntimeError):
    """Redacted, typed failure at the transport-neutral ingestion seam."""


class SourceLimitError(IngestionError):
    pass


class SourceCapabilityError(IngestionError):
    pass


@dataclass(frozen=True)
class IngestionLimits:
    max_records: int = 10_000
    max_pages: int = 1_000
    max_bytes: int = 20_000_000
    max_retries: int = 2
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.max_records, self.max_pages, self.max_bytes, self.max_retries)):
            raise ValueError("ingestion limits cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RestSource:
    uri: str
    resource_id: str
    records_field: str = "records"
    key_field: str = "id"
    cursor_field: str | None = None
    cursor_param: str = "cursor"
    next_field: str | None = "next"
    page_param: str = "page"
    page_size_param: str | None = "page_size"
    page_size: int = 100
    headers: Mapping[str, str] = field(default_factory=dict)
    field_mapping: Mapping[str, str] = field(default_factory=dict)
    mode: str = "snapshot"
    source_contract_identity: str = ""
    connector_binding_identity: str = ""


@dataclass(frozen=True)
class SqlSource:
    database: str
    resource_id: str
    table: str
    key_column: str
    cursor_column: str | None = None
    columns: tuple[str, ...] = ()
    field_mapping: Mapping[str, str] = field(default_factory=dict)
    mode: str = "snapshot"
    source_contract_identity: str = ""
    connector_binding_identity: str = ""


class McpResourceClient(Protocol):
    def list_resources(self, cursor: str | None = None) -> tuple[Sequence[Mapping[str, Any]], str | None, Mapping[str, Any]]: ...

    def read_resource(self, uri: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class McpSource:
    resource_id: str
    client: McpResourceClient
    mode: str = "snapshot"
    allow_tools: bool = False
    source_contract_identity: str = ""
    connector_binding_identity: str = ""


SourceDefinition = RestSource | SqlSource | McpSource


@dataclass(frozen=True)
class Observation:
    resource_id: str
    source_key: str
    payload: Mapping[str, Any]
    source_uri: str
    observed_at: str
    cursor: str | None
    content_sha256: str
    media_type: str = "application/json"
    deleted: bool = False

    @property
    def observation_id(self) -> str:
        return stable_identity({"resource": self.resource_id, "key": self.source_key, "content": self.content_sha256})


@dataclass(frozen=True)
class IngestionCounters:
    records: int = 0
    bytes: int = 0
    pages_or_batches: int = 0
    retries: int = 0
    reused_records: int = 0
    changed_records: int = 0
    deleted_records: int = 0
    failures: int = 0


@dataclass(frozen=True)
class SourceRun:
    observations: tuple[Observation, ...]
    counters: IngestionCounters
    source_identity: str
    snapshot_identity: str
    capabilities: Mapping[str, Any]
    source_contract_identity: str = ""
    connector_binding_identity: str = ""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _safe_limit(count: int, limit: int, label: str) -> None:
    if count > limit:
        raise SourceLimitError(f"{label} limit exceeded")


def _mapped(payload: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    return {concept: payload.get(field_name) for concept, field_name in mapping.items()}


def _request_json(source: RestSource, uri: str, limits: IngestionLimits) -> tuple[Mapping[str, Any], int, int]:
    last_error: Exception | None = None
    for attempt in range(limits.max_retries + 1):
        try:
            request = Request(uri, headers=dict(source.headers), method="GET")
            with urlopen(request, timeout=limits.timeout_seconds) as response:  # noqa: S310 - endpoint is profile-owned
                body = response.read(limits.max_bytes + 1)
            _safe_limit(len(body), limits.max_bytes, "byte")
            value = json.loads(body)
            if not isinstance(value, Mapping):
                raise IngestionError("REST response must be an object")
            return value, len(body), attempt
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == limits.max_retries:
                break
            time.sleep(0)
    raise IngestionError("REST request failed after bounded retries") from last_error


def _rest_observations(source: RestSource, limits: IngestionLimits, cursor: str | None) -> SourceRun:
    observations: list[Observation] = []
    uri = source.uri
    if cursor and source.cursor_field:
        parts = urlsplit(uri)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query[source.cursor_param] = cursor
        uri = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    page = 0
    total_bytes = retries = 0
    while uri:
        page += 1
        _safe_limit(page, limits.max_pages, "page")
        document, size, retry_count = _request_json(source, uri, limits)
        total_bytes += size
        retries += retry_count
        rows = document.get(source.records_field, document.get("data", []))
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise IngestionError("REST records field must be an array")
        for row in rows:
            if not isinstance(row, Mapping) or str(row.get(source.key_field, "")).strip() == "":
                raise IngestionError("REST record is missing its configured key")
            raw = dict(row)
            key = str(raw[source.key_field])
            observed = str(raw.get(source.cursor_field, "")) if source.cursor_field else ""
            payload = _mapped(raw, source.field_mapping) if source.field_mapping else raw
            observations.append(Observation(source.resource_id, key, payload, source.uri, observed or "unknown", observed or None, "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest(), deleted=bool(raw.get("_deleted", False))))
            _safe_limit(len(observations), limits.max_records, "record")
        next_uri = document.get(source.next_field) if source.next_field else None
        if next_uri is None and rows and page == 1 and source.page_param:
            # A deterministic fixture can expose a next URL; absent that, one page is a complete snapshot.
            next_uri = None
        uri = str(next_uri) if next_uri else ""
    return SourceRun(tuple(observations), IngestionCounters(len(observations), total_bytes, page, retries), stable_identity(asdict(source)), stable_identity([item.observation_id for item in observations]), {"pagination": True, "authentication": bool(source.headers), "snapshot": source.mode == "snapshot"}, source.source_contract_identity, source.connector_binding_identity)


def _validate_identifier(value: str, label: str) -> str:
    if not value.replace("_", "").isalnum():
        raise IngestionError(f"invalid SQL {label}")
    return value


def _sql_observations(source: SqlSource, limits: IngestionLimits, cursor: str | None) -> SourceRun:
    table = _validate_identifier(source.table, "table")
    key = _validate_identifier(source.key_column, "key column")
    columns = tuple(source.columns)
    if columns:
        columns = tuple(_validate_identifier(item, "column") for item in columns)
        select = ", ".join(columns)
    else:
        select = "*"
    query = f"SELECT {select} FROM {table}"
    params: tuple[Any, ...] = ()
    if source.cursor_column and cursor is not None:
        cursor_column = _validate_identifier(source.cursor_column, "cursor column")
        query += f" WHERE {cursor_column} > ?"
        params = (cursor,)
    if source.cursor_column:
        query += f" ORDER BY {_validate_identifier(source.cursor_column, 'cursor column')} ASC"
    observations: list[Observation] = []
    with sqlite3.connect(source.database) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query, params):
            raw = dict(row)
            source_key = str(raw.get(key, "")).strip()
            if not source_key:
                raise IngestionError("SQL record is missing its configured key")
            payload = _mapped(raw, source.field_mapping) if source.field_mapping else raw
            cursor_value = str(raw.get(source.cursor_column, "")) if source.cursor_column else ""
            observations.append(Observation(source.resource_id, source_key, payload, f"sql://{table}", cursor_value or "unknown", cursor_value or None, "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest(), deleted=bool(raw.get("_deleted", False))))
            _safe_limit(len(observations), limits.max_records, "record")
    return SourceRun(tuple(observations), IngestionCounters(len(observations), 0, 1), stable_identity(asdict(source)), stable_identity([item.observation_id for item in observations]), {"predicate_pushdown": bool(cursor and source.cursor_column), "batching": True}, source.source_contract_identity, source.connector_binding_identity)


def _mcp_observations(source: McpSource, limits: IngestionLimits) -> SourceRun:
    if source.allow_tools:
        raise SourceCapabilityError("MCP tools are not enabled by the resource ingestion seam")
    resources: list[Mapping[str, Any]] = []
    cursor: str | None = None
    pages = 0
    capabilities: Mapping[str, Any] = {}
    while True:
        pages += 1
        _safe_limit(pages, limits.max_pages, "page")
        page, cursor, capabilities = source.client.list_resources(cursor)
        resources.extend(page)
        _safe_limit(len(resources), limits.max_records, "record")
        if not cursor:
            break
    observations: list[Observation] = []
    for resource in resources:
        uri = str(resource.get("uri", ""))
        if not uri:
            raise IngestionError("MCP resource is missing its URI")
        document = source.client.read_resource(uri)
        text = document.get("text")
        blob = document.get("blob")
        payload = {"uri": uri, "name": resource.get("name"), "text": text, "blob": blob, "mime_type": document.get("mimeType", resource.get("mimeType", "application/octet-stream"))}
        digest = "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest()
        observations.append(Observation(source.resource_id, uri, payload, uri, str(resource.get("revision", "unknown")), str(resource.get("revision", "")) or None, digest, str(payload["mime_type"]), deleted=bool(resource.get("deleted", False))))
    return SourceRun(tuple(observations), IngestionCounters(len(observations), sum(len(_json_bytes(item.payload)) for item in observations), pages), stable_identity({"resource_id": source.resource_id, "mode": source.mode, "source_contract_identity": source.source_contract_identity, "connector_binding_identity": source.connector_binding_identity}), stable_identity([item.observation_id for item in observations]), {"resources_list": True, "resource_read": True, **dict(capabilities)}, source.source_contract_identity, source.connector_binding_identity)


def _state_path(config: IngestConfig) -> Path:
    state_dir = Path(config.state_dir or "/tmp/igor-dlt-state")
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{config.environment}_{config.domain}_{config.dataset_name}.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"keys": {}, "cursor": None}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {"keys": {}, "cursor": None}
    except (OSError, ValueError):
        raise IngestionError("ingestion state is unreadable")


def _canonical_observations(run: SourceRun, config: IngestConfig, state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], IngestionCounters, dict[str, Any]]:
    prior = state.get("keys", {}) if isinstance(state.get("keys", {}), Mapping) else {}
    canonical: list[dict[str, Any]] = []
    reused = changed = deleted = 0
    seen: set[str] = set()
    for item in run.observations:
        seen.add(item.source_key)
        before = prior.get(item.source_key)
        if before and before.get("content_sha256") == item.content_sha256 and not item.deleted:
            reused += 1
            continue
        else:
            changed += 1
        if item.deleted:
            deleted += 1
        canonical.append({"record_id": item.source_key, "source_system": item.source_uri.split(":", 1)[0], "source_key": item.source_key, "resource_id": item.resource_id, "observed_at": item.observed_at, "content_ref": item.source_uri, "content_sha256": item.content_sha256, "media_type": item.media_type, "payload": dict(item.payload), "deleted": item.deleted, "environment": config.environment, "domain": config.domain, "source_contract_identity": run.source_contract_identity, "connector_binding_identity": run.connector_binding_identity, "source_definition_identity": run.source_identity, "snapshot_identity": run.snapshot_identity, "context_unit_id": f"ctx:{config.domain}:{stable_identity({'resource': item.resource_id, 'key': item.source_key})[:12]}"})
    if run.capabilities.get("snapshot"):
        for key, before in prior.items():
            if key not in seen:
                deleted += 1
                canonical.append({"record_id": key, "source_system": "snapshot", "source_key": key, "resource_id": before["resource_id"], "observed_at": "tombstone", "content_ref": before["content_ref"], "content_sha256": before["content_sha256"], "media_type": before["media_type"], "payload": before["payload"], "deleted": True, "environment": config.environment, "domain": config.domain, "source_contract_identity": run.source_contract_identity, "connector_binding_identity": run.connector_binding_identity, "source_definition_identity": run.source_identity, "snapshot_identity": run.snapshot_identity, "context_unit_id": before["context_unit_id"]})
    current_keys = {str(key): dict(value) for key, value in prior.items()}
    for item in canonical:
        if item["deleted"]:
            current_keys.pop(item["source_key"], None)
        else:
            current_keys[item["source_key"]] = {key: item[key] for key in ("resource_id", "content_sha256", "content_ref", "media_type", "payload", "context_unit_id")}
    new_state = {"cursor": max((item.cursor for item in run.observations if item.cursor), default=state.get("cursor")), "keys": current_keys}
    counters = IngestionCounters(run.counters.records, run.counters.bytes, run.counters.pages_or_batches, run.counters.retries, reused, changed, deleted, run.counters.failures)
    return canonical, counters, new_state


def ingest_source(source: SourceDefinition, config: IngestConfig, limits: IngestionLimits | None = None) -> IngestResult:
    limits = limits or IngestionLimits()
    state_file = _state_path(config)
    state = _load_state(state_file)
    if isinstance(source, RestSource):
        run = _rest_observations(source, limits, state.get("cursor"))
    elif isinstance(source, SqlSource):
        run = _sql_observations(source, limits, state.get("cursor"))
    else:
        run = _mcp_observations(source, limits)
    canonical, counters, new_state = _canonical_observations(run, config, state)
    if not canonical and not state.get("keys"):
        raise IngestionError("source produced no observations")
    state_file.write_text(json.dumps(new_state, sort_keys=True, default=str))
    if not canonical:
        return IngestResult((), {"dlt": {"skipped": True}, "source": {"identity": run.source_identity, "snapshot_identity": run.snapshot_identity, "capabilities": dict(run.capabilities)}, "counters": asdict(counters)}, config)
    destination = dlt.destinations.lance(storage={"bucket_url": config.bucket_url, "namespace_name": config.namespace_name, "options": dict(config.storage_options or {})})
    pipeline = dlt.pipeline(pipeline_name=f"igor_{config.environment}_{config.domain}_{config.dataset_name}", destination=destination, dataset_name=config.dataset_name, pipelines_dir=str(config.pipeline_dir or "/tmp/igor-dlt-pipelines"))
    resource = dlt.resource(canonical, name="ingested_records", primary_key="record_id", write_disposition="merge")
    load_info = pipeline.run(resource, write_disposition="merge")
    metadata = {"dlt": _jsonable(load_info.asdict()), "source": {"identity": run.source_identity, "snapshot_identity": run.snapshot_identity, "capabilities": dict(run.capabilities)}, "counters": asdict(counters)}
    return IngestResult(tuple(canonical), metadata, config)


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
