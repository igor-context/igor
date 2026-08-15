from pathlib import Path

import pytest

from igor_lancedb import LanceStore
from igor_relational import ModelManifest, ModelSpec, RelationalRunner, RelationalRunnerError
from igor_datafusion import AnalyticalEngine


def make_runner(tmp_path: Path) -> RelationalRunner:
    store = LanceStore(tmp_path / "db")
    store.create("events", [
        {"id": 1, "group_name": "a", "amount": 10},
        {"id": 2, "group_name": "a", "amount": 20},
        {"id": 3, "group_name": "b", "amount": 5},
    ])
    return RelationalRunner(AnalyticalEngine(store), store)


def test_runner_materializes_arrow_results_and_records_lineage(tmp_path: Path) -> None:
    run = make_runner(tmp_path).run(ModelManifest("0.1", (
        ModelSpec("group_totals", "SELECT group_name, SUM(amount) AS total FROM events GROUP BY group_name ORDER BY group_name", tests=("unique:group_name",)),
    )))
    assert run.materialized == ("group_totals",)
    assert run.lineage == {"group_totals": ()}
    assert run.rows == {"group_totals": 2}


def test_runner_orders_dependencies_and_rejects_cycles_or_failed_tests(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run = runner.run(ModelManifest("0.1", (
        ModelSpec("totals", "SELECT group_name, SUM(amount) AS total FROM events GROUP BY group_name", tests=("unique:group_name",)),
        ModelSpec("large", "SELECT * FROM totals WHERE total > 10", depends_on=("totals",)),
    )))
    assert run.materialized == ("totals", "large")
    with pytest.raises(RelationalRunnerError, match="dependency cycle"):
        runner.run(ModelManifest("0.1", (ModelSpec("a", "SELECT 1 AS id FROM events", depends_on=("b",)), ModelSpec("b", "SELECT 1 AS id FROM events", depends_on=("a",)))))
    with pytest.raises(RelationalRunnerError, match="unique:group_name"):
        runner.run(ModelManifest("0.1", (ModelSpec("duplicates", "SELECT group_name FROM events", tests=("unique:group_name",)),)))
