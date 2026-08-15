from __future__ import annotations

from dataclasses import asdict, dataclass

import pyarrow as pa
from igor_datafusion import AnalyticalEngine
from igor_core import LineageEdge, LineageGraph, LineageNode, ProducerIdentity, stable_identity
from igor_lancedb import LanceStore


class RelationalRunnerError(ValueError):
    """Deterministic error for invalid models or failed relational runs."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sql: str
    depends_on: tuple[str, ...] = ()
    materialization: str = "table"
    tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelManifest:
    version: str
    models: tuple[ModelSpec, ...]


@dataclass(frozen=True)
class RelationalRun:
    materialized: tuple[str, ...]
    lineage: dict[str, tuple[str, ...]]
    rows: dict[str, int]

    def lineage_graph(self, run_identity: str, manifest: ModelManifest) -> LineageGraph:
        producer = ProducerIdentity(name="igor-relational", revision="0.1")
        configuration_revision = stable_identity(asdict(manifest))
        nodes: dict[str, LineageNode] = {}
        edges: list[LineageEdge] = []
        for model in manifest.models:
            target = LineageNode(schema_version="0.1", node_type="stored_relation", key=model.name, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision, metadata={"materialization": model.materialization})
            nodes[target.identity] = target
            for dependency in model.depends_on:
                source = LineageNode(schema_version="0.1", node_type="stored_relation", key=dependency, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision)
                nodes[source.identity] = source
                edges.append(LineageEdge(schema_version="0.1", edge_type="derived_from", source_node_id=source.identity, target_node_id=target.identity, run_identity=run_identity, producer=producer, configuration_revision=configuration_revision))
        return LineageGraph(schema_version="0.1", run_identity=run_identity, nodes=tuple(nodes.values()), edges=tuple(edges))


class RelationalRunner:
    def __init__(self, engine: AnalyticalEngine, store: LanceStore):
        self.engine = engine
        self.store = store

    def run(self, manifest: ModelManifest) -> RelationalRun:
        models = {model.name: model for model in manifest.models}
        if len(models) != len(manifest.models):
            raise RelationalRunnerError("model names must be unique")
        order = self._order(models)
        rows: dict[str, int] = {}
        for name in order:
            model = models[name]
            if model.materialization != "table":
                raise RelationalRunnerError(f"unsupported materialization: {model.materialization}")
            try:
                result = self.engine.query(model.sql)
            except ValueError as error:
                raise RelationalRunnerError(f"model {name} failed: {error}") from error
            if result.num_rows == 0:
                raise RelationalRunnerError(f"model {name} returned no rows")
            self.store.replace(name, result)
            self._run_tests(name, result, model.tests)
            rows[name] = result.num_rows
        return RelationalRun(tuple(order), {name: models[name].depends_on for name in order}, rows)

    def _order(self, models: dict[str, ModelSpec]) -> list[str]:
        for model in models.values():
            missing = set(model.depends_on) - models.keys()
            if missing:
                raise RelationalRunnerError(f"model {model.name} depends on unknown models: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visiting:
                raise RelationalRunnerError(f"model dependency cycle includes: {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in models[name].depends_on:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in models:
            visit(name)
        return order

    @staticmethod
    def _run_tests(name: str, result: pa.Table, tests: tuple[str, ...]) -> None:
        for test in tests:
            if test == "non_empty":
                continue
            if test.startswith("unique:"):
                column = test.split(":", 1)[1]
                if column not in result.column_names:
                    raise RelationalRunnerError(f"model {name} test column missing: {column}")
                values = result[column].to_pylist()
                if len(values) != len(set(values)):
                    raise RelationalRunnerError(f"model {name} test failed: unique:{column}")
                continue
            raise RelationalRunnerError(f"unsupported model test: {test}")
