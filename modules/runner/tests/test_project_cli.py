from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from igor_lancedb import LanceStore
from igor_runner.project import (
    ProjectError,
    compile_project,
    deps_project,
    init_project,
    package_project,
    plan_project,
    validate_project,
)
from igor_runner.project_cli import main as cli_main


def _project_root(tmp_path: Path) -> Path:
    return init_project("starter-project", path=tmp_path)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _recare_project_source() -> Path:
    for candidate in (
        Path("/opt/scenarios/recare-legal/v0.1"),
        Path(__file__).resolve().parents[3] / "scenarios" / "recare-legal" / "v0.1",
    ):
        if (candidate / "igor_project.yml").is_file():
            return candidate
    pytest.skip("ReCaRe full IGOR project scenario is not available")


def test_init_creates_complete_expected_structure(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    expected = {
        "igor_project.yml",
        "packages.yml",
        ".env.example",
        ".gitignore",
        "README.md",
        "package_request.example.yml",
        "models/example_context.yml",
        "sources/example_source.yml",
        "schemas/example_enrichment.yml",
        "taxonomies/example_taxonomy.yml",
        "enrichments/example_recipe.yml",
        "policies/example_authority.yml",
        "contracts/example_agent.yml",
        "retrievals/example_search.yml",
        "outputs/example_delivery.yml",
        "profiles/embeddings.example.yml",
        "profiles/completions.example.yml",
        "evals/retrieval.yml",
        "evals/authority.yml",
        "evals/temporal.yml",
        "evals/contract.yml",
        "evals/package.yml",
        "evals/lineage.yml",
        "evals/quality.yml",
        "evals/operational.yml",
        "fixtures/sources/valid-case.yml",
        "fixtures/sources/conflicting-case.yml",
        "fixtures/sources/expired-case.yml",
        "fixtures/sources/missing-evidence-case.yml",
        "fixtures/sources/contract-denied-case.yml",
        "fixtures/sources/mutation-and-reuse-case.yml",
        "fixtures/expected/valid-case.yml",
        "fixtures/expected/conflicting-case.yml",
        "fixtures/expected/expired-case.yml",
        "fixtures/expected/missing-evidence-case.yml",
        "fixtures/expected/contract-denied-case.yml",
        "fixtures/expected/mutation-and-reuse-case.yml",
        "fixtures/mutations/mutation-and-reuse-case.yml",
    }
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert expected.issubset(actual)


def test_initialized_project_validates_compiles_and_exposes_evals_and_contracts(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    monkeypatch.chdir(root)
    deps_project(root)
    errors, warnings = validate_project(root)
    assert errors == []
    assert warnings == []
    compiled = compile_project(root)
    manifest = compiled.manifest["manifest"]
    assert manifest["context_contracts"][0]["contract_id"] == "example.agent"
    assert compiled.evaluation_plan["categories"]["retrieval"] == ["retrieval.relevance"]
    assert "valid-case" in compiled.evaluation_plan["fixture_inventory"]["sources"]
    assert Path(root / ".igor/target/manifest.json").is_file()
    assert Path(root / ".igor/target/compile-report.html").is_file()


def test_package_request_publishes_context_package(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    summary = package_project(root, request="package_request.example.yml")
    assert summary["published"] is True
    assert summary["decisions"] == ["allow"]
    assert len(summary["published_packages"]) == 1
    assert Path(root / ".igor/target/package_request.example-package-publication.json").is_file()


@pytest.mark.parametrize(
    "command",
    [
        "sync",
        "build",
        "diff",
        "resolve",
        "query",
        "explain",
        "test",
        "evaluate",
        "qualify",
        "observe",
        "tune",
        "serve",
        "status",
    ],
)
def test_project_command_chain_runs_on_a_fresh_project(tmp_path: Path, monkeypatch, capsys, command: str) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    exit_code = cli_main([command])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == command
    assert Path(root / ".igor" / "target" / f"{command}.json").is_file()
    if command in {"build", "explain"}:
        assert Path(root / ".igor" / "target" / "manifest.json").is_file()
        assert Path(root / ".igor" / "target" / "compile-report.html").is_file()


def test_project_command_chain_runs_on_full_recare_igor_project(tmp_path: Path, monkeypatch, capsys) -> None:
    source = _recare_project_source()
    root = tmp_path / "recare-project"
    shutil.copytree(source, root)
    monkeypatch.chdir(root)

    for command in ("sync", "resolve", "query", "qualify"):
        assert cli_main([command]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == command
        assert payload["valid"] is True
        assert Path(root / ".igor" / "target" / f"{command}.json").is_file()

    resolve_payload = json.loads((root / ".igor" / "target" / "resolve.json").read_text(encoding="utf-8"))
    query_payload = json.loads((root / ".igor" / "target" / "query.json").read_text(encoding="utf-8"))
    qualify_payload = json.loads((root / ".igor" / "target" / "qualify.json").read_text(encoding="utf-8"))

    assert resolve_payload["context_model_id"] == "recare.legal.context"
    assert "legal_context_delivery" in resolve_payload["outputs"]
    assert "article_search" in query_payload["retrieval_roles"]
    assert "authority" in qualify_payload["evaluation_plan"]["categories"]
    assert Path(root / ".igor" / "target" / "manifest.json").is_file()
    assert Path(root / ".igor" / "target" / "compile-report.html").is_file()


def test_query_sql_runs_through_datafusion_over_project_relation_store(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    store_path = root / ".igor" / "runtime" / "qualification" / "allowed-relations"
    store = LanceStore(store_path)
    store.create(
        "context_catalog",
        [
            {
                "canonical_identity": "ctx:1",
                "display_name": "First",
                "context_model_id": "example.context",
                "object_kind": "source_snapshot",
                "status": "active",
                "source_system": "fixture",
                "source_key": "A",
                "media_type": "text/plain",
                "evidence_preview": "hello",
                "content_ref": "memory://a",
                "authority_status": "accepted",
                "temporal_status": "current",
                "policy_id": "policy",
                "as_of": "2026-08-15T00:00:00Z",
            },
            {
                "canonical_identity": "ctx:2",
                "display_name": "Second",
                "context_model_id": "example.context",
                "object_kind": "source_snapshot",
                "status": "inactive",
                "source_system": "fixture",
                "source_key": "B",
                "media_type": "text/plain",
                "evidence_preview": "bye",
                "content_ref": "memory://b",
                "authority_status": "rejected",
                "temporal_status": "superseded",
                "policy_id": "policy",
                "as_of": "2026-08-15T00:00:00Z",
            },
        ],
    )
    monkeypatch.chdir(root)

    assert cli_main(["query", "--sql", "SELECT status, COUNT(*) AS records FROM context_catalog GROUP BY status ORDER BY status"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "query"
    assert payload["sql_result"]["engine"] == "datafusion"
    assert payload["sql_result"]["columns"] == ["status", "records"]
    assert payload["sql_result"]["rows"] == [
        {"records": 1, "status": "active"},
        {"records": 1, "status": "inactive"},
    ]
    assert Path(root / ".igor" / "target" / "query.json").is_file()


def test_live_project_commands_execute_declared_igor_tools_without_importing_adapters(tmp_path: Path, monkeypatch, capsys) -> None:
    source = _recare_project_source()
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "igor-huggingface":
            output = Path(args[2])
        elif args[0] == "igor-runner":
            output = Path(args[5]) / "qualification.json"
        else:
            output = Path(args[4])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"ok": True, "tool": args[0]}) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps({"valid": True}), stderr="")

    monkeypatch.setattr("igor_runner.project.subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    assert cli_main(["init", "recare-cli", "--template", str(source)]) == 0
    capsys.readouterr()
    root = tmp_path / "recare-cli"
    monkeypatch.chdir(root)

    for command in ("sync", "resolve", "query", "qualify", "evaluate"):
        args = [command, "--live"] if command in {"sync", "qualify", "evaluate"} else [command]
        assert cli_main(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == command
        assert payload["valid"] is True

    assert json.loads((root / ".igor/target/sources/acquisition.json").read_text(encoding="utf-8"))["tool"] == "igor-huggingface"
    assert json.loads((root / ".igor/runtime/qualification/qualification.json").read_text(encoding="utf-8"))["tool"] == "igor-runner"
    assert json.loads((root / ".igor/runtime/evaluation/evaluation.json").read_text(encoding="utf-8"))["tool"] == "igor-evaluator"
    assert [call[0] for call in calls] == ["igor-huggingface", "igor-runner", "igor-evaluator"]


def test_package_request_fails_closed_for_deny_and_abstain(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)

    denied = _load_yaml(root / "package_request.example.yml")
    denied["publication"]["consumer"] = "marketing-agent"
    _write_yaml(root / "package_request.denied.yml", denied)
    try:
        package_project(root, request="package_request.denied.yml")
    except ProjectError as error:
        assert "failed closed" in str(error)
        assert "deny" in str(error)
    else:
        raise AssertionError("deny publication must fail closed")
    denied_summary = json.loads((root / ".igor/target/package_request.denied-package-publication.json").read_text(encoding="utf-8"))
    assert denied_summary["decisions"] == ["deny"]
    assert denied_summary["published_packages"] == []

    abstain = _load_yaml(root / "package_request.example.yml")
    abstain["publication"]["authority_level"] = ""
    abstain["runtime_outputs"][0]["authority_level"] = ""
    abstain["packages"][0]["items"][0]["authority_level"] = ""
    _write_yaml(root / "package_request.abstain.yml", abstain)
    try:
        package_project(root, request="package_request.abstain.yml")
    except ProjectError as error:
        assert "failed closed" in str(error)
        assert "abstain" in str(error)
    else:
        raise AssertionError("abstain publication must fail closed")
    abstain_summary = json.loads((root / ".igor/target/package_request.abstain-package-publication.json").read_text(encoding="utf-8"))
    assert abstain_summary["decisions"] == ["abstain"]
    assert abstain_summary["published_packages"] == []


def test_generated_project_contains_no_vendor_names_or_secret_values(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    forbidden_terms = ("openai", "voyage", "deepseek", "gemini", "mistral", "sk-", "AIza", "ghp_")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        assert not any(term in content for term in forbidden_terms), path
    assert "IGOR_EMBEDDING_API_KEY=" in (root / ".env.example").read_text(encoding="utf-8")
    assert "IGOR_COMPLETION_API_KEY=" in (root / ".env.example").read_text(encoding="utf-8")


def test_init_twice_fails_without_overwriting(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    try:
        init_project("starter-project", path=tmp_path)
    except ProjectError as error:
        assert "already initialized" in str(error)
    else:
        raise AssertionError("second init must fail closed")
    assert (root / "igor_project.yml").is_file()


def test_dependency_lock_is_deterministic_and_changed_dependency_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    dependency_root = root / "deps" / "shared-taxonomy"
    init_project("shared-taxonomy", path=root / "deps")
    packages = _load_yaml(root / "packages.yml")
    packages["packages"] = [{"name": "shared-taxonomy", "path": "deps/shared-taxonomy", "compatibility": ["0.1"]}]
    _write_yaml(root / "packages.yml", packages)
    monkeypatch.chdir(root)
    first = deps_project(root)
    first_bytes = (root / "igor.lock").read_text(encoding="utf-8")
    second = deps_project(root)
    second_bytes = (root / "igor.lock").read_text(encoding="utf-8")
    assert first.lock_identity == second.lock_identity
    assert first_bytes == second_bytes

    taxonomy = _load_yaml(dependency_root / "taxonomies/example_taxonomy.yml")
    taxonomy["semantic_definitions"]["support.assessment-taxonomy.v1"]["description"] = "changed"
    _write_yaml(dependency_root / "taxonomies/example_taxonomy.yml", taxonomy)
    errors, _ = validate_project(root)
    assert any(item.code == "invalid_dependency_integrity" for item in errors)


def test_validation_reports_precise_file_and_field_paths_for_unknown_fields(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    schema = _load_yaml(root / "schemas/example_enrichment.yml")
    schema["schemas"]["support.message.v1"]["unknown_field"] = True
    _write_yaml(root / "schemas/example_enrichment.yml", schema)
    monkeypatch.chdir(root)
    errors, _ = validate_project(root)
    assert any(
        item.file.endswith("schemas/example_enrichment.yml")
        and ("unknown_field" in item.field_path or "unknown_field" in item.message)
        for item in errors
    )


def test_duplicate_identity_missing_reference_and_cycle_fail_closed(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)

    duplicate = _load_yaml(root / "schemas/example_enrichment.yml")
    duplicate["schemas"]["support.message.clone.v1"] = dict(duplicate["schemas"]["support.message.v1"])
    _write_yaml(root / "schemas/example_enrichment.yml", duplicate)
    monkeypatch.chdir(root)
    errors, _ = validate_project(root)
    assert any(item.code == "duplicate_semantic_identity" for item in errors)

    duplicate = _load_yaml(root / "schemas/example_enrichment.yml")
    duplicate["schemas"].pop("support.message.clone.v1")
    _write_yaml(root / "schemas/example_enrichment.yml", duplicate)

    models = _load_yaml(root / "models/example_context.yml")
    models["objects"]["support_message"]["schema"] = "missing.schema.v1"
    _write_yaml(root / "models/example_context.yml", models)
    errors, _ = validate_project(root)
    assert any(item.code in {"project_invalid", "missing_reference"} for item in errors)

    models["objects"]["support_message"]["schema"] = "support.message.v1"
    models["objects"]["support_ticket_snapshot"]["derived_from"] = ["support_message_assessment"]
    _write_yaml(root / "models/example_context.yml", models)
    errors, _ = validate_project(root)
    assert any("dependency cycle" in item.message for item in errors)


def test_malformed_policy_contract_and_evaluation_fail_closed(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)

    policy = _load_yaml(root / "policies/example_authority.yml")
    policy["authority_policies"]["support.authority-policy.v1"]["conflict_behavior"] = "maybe"
    _write_yaml(root / "policies/example_authority.yml", policy)
    errors, _ = validate_project(root)
    assert any(item.file.endswith("example_authority.yml") for item in errors)

    policy["authority_policies"]["support.authority-policy.v1"]["conflict_behavior"] = "abstain"
    _write_yaml(root / "policies/example_authority.yml", policy)
    contract = _load_yaml(root / "contracts/example_agent.yml")
    contract["context_contracts"]["example.agent.v1"]["budgets"]["max_tokens"] = 0
    _write_yaml(root / "contracts/example_agent.yml", contract)
    errors, _ = validate_project(root)
    assert any(item.file.endswith("example_agent.yml") for item in errors)

    contract["context_contracts"]["example.agent.v1"]["budgets"]["max_tokens"] = 2000
    _write_yaml(root / "contracts/example_agent.yml", contract)
    evaluation = _load_yaml(root / "evals/retrieval.yml")
    evaluation["evaluations"]["retrieval.relevance"]["category"] = "provider"
    _write_yaml(root / "evals/retrieval.yml", evaluation)
    errors, _ = validate_project(root)
    assert any(item.file.endswith("retrieval.yml") for item in errors)


def test_compile_is_deterministic_and_display_only_changes_do_not_change_identity(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    first = compile_project(root)
    first_bytes = (root / ".igor/target/manifest.json").read_text(encoding="utf-8")
    second = compile_project(root)
    second_bytes = (root / ".igor/target/manifest.json").read_text(encoding="utf-8")
    assert first.manifest["manifest_identity"] == second.manifest["manifest_identity"]
    assert first_bytes == second_bytes

    model = _load_yaml(root / "models/example_context.yml")
    model["context_model"]["title"] = "Renamed title"
    model["objects"]["support_message_assessment"]["display_name"] = "Renamed output"
    _write_yaml(root / "models/example_context.yml", model)
    renamed = compile_project(root)
    assert renamed.manifest["manifest_identity"] == first.manifest["manifest_identity"]


def test_semantic_contract_and_taxonomy_changes_drive_plan_invalidations(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    baseline = compile_project(root)
    baseline_identity = baseline.manifest["manifest_identity"]

    contract = _load_yaml(root / "contracts/example_agent.yml")
    contract["context_contracts"]["example.agent.v1"]["freshness_hours"] = 48
    _write_yaml(root / "contracts/example_agent.yml", contract)
    changed = compile_project(root)
    assert changed.manifest["manifest_identity"] != baseline_identity
    plan = plan_project(root)
    assert "support_delivery" in plan["affected_outputs"]
    assert "example.agent.v1" in plan["affected_nodes"]

    taxonomy = _load_yaml(root / "taxonomies/example_taxonomy.yml")
    taxonomy["semantic_definitions"]["support.assessment-taxonomy.v1"]["description"] = "Updated meaning"
    _write_yaml(root / "taxonomies/example_taxonomy.yml", taxonomy)
    compile_project(root)
    plan = plan_project(root)
    assert "support_message_assessment" in plan["affected_nodes"]
    assert "support_delivery" in plan["affected_outputs"]


def test_plan_redacts_secret_values_and_writes_only_under_igor(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    monkeypatch.setenv("IGOR_EMBEDDING_API_KEY", "secret-embedding-value")
    monkeypatch.setenv("IGOR_COMPLETION_API_KEY", "secret-completion-value")
    compile_project(root)
    plan = plan_project(root, live=True)
    plan_payload = json.dumps(plan, sort_keys=True)
    assert "secret-embedding-value" not in plan_payload
    assert "secret-completion-value" not in plan_payload
    assert plan["live_profile_checks"] == {
        "IGOR_COMPLETION_API_KEY": True,
        "IGOR_EMBEDDING_API_KEY": True,
    }
    generated = [path for path in root.rglob("*") if path.is_file() and ".igor" in path.parts]
    assert generated
    assert all(str(path).startswith(str(root / ".igor")) for path in generated)


def test_compile_rejects_targets_that_escape_igor(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    deps_project(root)
    monkeypatch.chdir(root)
    escaped = tmp_path / "escaped"
    try:
        compile_project(root, target="../../../escaped")
    except ProjectError as error:
        assert "target must stay within" in str(error)
    else:
        raise AssertionError("escaped target must fail closed")
    assert not escaped.exists()


def test_transitive_dependencies_lock_validate_and_merge(tmp_path: Path, monkeypatch) -> None:
    root = _project_root(tmp_path)
    dep_a = init_project("dep-a", path=root / "deps")
    dep_b = init_project("dep-b", path=root / "deps")

    for dependency_root in (dep_a, dep_b):
        for directory in (
            "models",
            "sources",
            "schemas",
            "taxonomies",
            "enrichments",
            "policies",
            "contracts",
            "retrievals",
            "outputs",
            "profiles",
            "evals",
        ):
            for path in (dependency_root / directory).glob("*.yml"):
                path.unlink()

    dep_b_schema = _load_yaml(root / "schemas/example_enrichment.yml")
    unique_schema = dict(dep_b_schema["schemas"]["support.message.v1"])
    unique_schema["schema_id"] = "dependency.support.message.v1"
    _write_yaml(
        dep_b / "schemas/dependency_schema.yml",
        {"schema_version": "0.1", "schemas": {"dependency.support.message.v1": unique_schema}},
    )
    _write_yaml(
        dep_a / "packages.yml",
        {"packages": [{"name": "dep-b", "path": "../dep-b", "compatibility": ["0.1"]}]},
    )
    _write_yaml(
        root / "packages.yml",
        {"packages": [{"name": "dep-a", "path": "deps/dep-a", "compatibility": ["0.1"]}]},
    )
    model = _load_yaml(root / "models/example_context.yml")
    model["objects"]["support_message"]["schema"] = "dependency.support.message.v1"
    _write_yaml(root / "models/example_context.yml", model)

    monkeypatch.chdir(root)
    lock = deps_project(root)
    assert {item.name for item in lock.packages} == {"dep-a", "dep-b"}

    errors, warnings = validate_project(root)
    assert errors == []
    assert warnings == []
    compiled = compile_project(root)
    support_message = next(item for item in compiled.manifest["manifest"]["objects"] if item["role"] == "support_message")
    assert support_message["schema_ref"] == "dependency.support.message.v1"


def test_deps_reports_malformed_dependency_yaml_without_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli_main(["init", "cli-project"]) == 0
    root = tmp_path / "cli-project"
    dep_a = init_project("dep-a", path=root / "deps")
    _write_yaml(
        root / "packages.yml",
        {"packages": [{"name": "dep-a", "path": "deps/dep-a", "compatibility": ["0.1"]}]},
    )
    (dep_a / "packages.yml").write_text("packages: invalid\n", encoding="utf-8")

    monkeypatch.chdir(root)
    assert cli_main(["deps"]) == 2
    captured = capsys.readouterr()
    assert "packages.yml" in captured.err
    assert "invalid_field" in captured.err
    assert "Traceback" not in captured.err


def test_cli_round_trip_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli_main(["init", "cli-project"]) == 0
    root = tmp_path / "cli-project"
    monkeypatch.chdir(root)
    assert cli_main(["deps"]) == 0
    assert cli_main(["validate"]) == 0
    assert cli_main(["compile"]) == 0
    assert cli_main(["plan"]) == 0
    assert cli_main(["package", "--request", "package_request.example.yml"]) == 0
