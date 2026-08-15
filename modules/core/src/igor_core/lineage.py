"""Portable, tool-neutral lineage records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from igor_core.canonical import stable_identity
from igor_core.contracts import ContractVersion, NonEmptyString, ProducerIdentity, Sha256

LineageNodeType = Literal[
    "source_record",
    "source_observation",
    "ingestion_load",
    "context_unit",
    "derivation_output",
    "stored_relation",
    "retrieval_result",
    "evaluation_artifact",
]
LineageEdgeType = Literal[
    "observed",
    "ingested",
    "canonicalized",
    "derived_from",
    "materialized_as",
    "retrieved_from",
    "evaluated_from",
]


class LineagePresentation(BaseModel):
    """Bounded human-readable metadata carried beside canonical lineage identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str | None = None
    context_model_id: str | None = None
    context_model_revision: str | None = None
    node_kind: str | None = None
    source_system: str | None = None
    source_key: str | None = None
    source_label: str | None = None
    schema_id: str | None = None
    schema_revision: str | None = None
    operation_name: str | None = None
    status: str | None = None
    preview: str | None = Field(default=None, max_length=512)


class LineageNode(BaseModel):
    """A stable run-scoped thing that can participate in lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    node_type: LineageNodeType
    key: NonEmptyString
    run_identity: Sha256
    source_identity: NonEmptyString | None = None
    producer: ProducerIdentity
    configuration_revision: NonEmptyString
    presentation: LineagePresentation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> str:
        value = self.model_dump(mode="json", exclude={"presentation"})
        if self.presentation is not None:
            presentation = self.presentation if isinstance(self.presentation, LineagePresentation) else LineagePresentation.model_validate(self.presentation)
            semantic = presentation.model_dump(mode="json", include={"context_model_id", "context_model_revision", "node_kind", "schema_id", "schema_revision", "operation_name"})
            value["presentation_identity"] = semantic
        return stable_identity(value)


class LineageEdge(BaseModel):
    """A directed, typed relationship between two lineage nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    edge_type: LineageEdgeType
    source_node_id: Sha256
    target_node_id: Sha256
    run_identity: Sha256
    producer: ProducerIdentity
    configuration_revision: NonEmptyString
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_self_edges(self) -> "LineageEdge":
        if self.source_node_id == self.target_node_id:
            raise ValueError("lineage edges cannot point to themselves")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class LineageGraph(BaseModel):
    """A portable, closed lineage graph for one benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    run_identity: Sha256
    nodes: tuple[LineageNode, ...] = ()
    edges: tuple[LineageEdge, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> "LineageGraph":
        node_ids = [node.identity for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("lineage graph contains duplicate node identities")
        if any(node.run_identity != self.run_identity for node in self.nodes):
            raise ValueError("lineage node run_identity does not match graph run_identity")
        edge_ids = [edge.identity for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("lineage graph contains duplicate edge identities")
        node_set = set(node_ids)
        for edge in self.edges:
            if edge.run_identity != self.run_identity:
                raise ValueError("lineage edge run_identity does not match graph run_identity")
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                raise ValueError("lineage edge references an undeclared node")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)

    @classmethod
    def merge(cls, run_identity: str, graphs: tuple["LineageGraph", ...]) -> "LineageGraph":
        nodes: dict[str, LineageNode] = {}
        edges: dict[str, LineageEdge] = {}
        for graph in graphs:
            if graph.run_identity != run_identity:
                raise ValueError("cannot merge lineage graphs from different runs")
            for node in graph.nodes:
                nodes[node.identity] = node
            for edge in graph.edges:
                edges[edge.identity] = edge
        return cls(schema_version="0.1", run_identity=run_identity, nodes=tuple(nodes.values()), edges=tuple(edges.values()))
