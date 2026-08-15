"""Tool-neutral ingestion port backed by dlt's Lance destination."""

from igor_dlt.ingest import IngestConfig, IngestResult, ingest_bound_records, ingest_records

__all__ = ["IngestConfig", "IngestResult", "ingest_bound_records", "ingest_records"]
from .transport import (
    IngestionError,
    IngestionLimits,
    IngestionCounters,
    McpResourceClient,
    McpSource,
    Observation,
    RestSource,
    SourceCapabilityError,
    SourceDefinition,
    SourceLimitError,
    SqlSource,
    ingest_source,
)

__all__ = [
    "IngestConfig", "IngestResult", "ingest_bound_records", "ingest_records",
    "IngestionError", "IngestionLimits", "IngestionCounters", "McpResourceClient",
    "McpSource", "Observation", "RestSource", "SourceCapabilityError",
    "SourceDefinition", "SourceLimitError", "SqlSource", "ingest_source",
]
