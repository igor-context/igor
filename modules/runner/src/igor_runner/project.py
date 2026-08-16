"""Deterministic IGOR project loading, validation, compilation, and planning."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from igor_core import (
    ConnectorBinding,
    ConnectorFieldBinding,
    ConnectorResourceBinding,
    ContextSourceContract,
    EnrichmentRecipe,
    ModelProfile,
    SchemaDescriptor,
    canonical_json,
    load_model_profile,
    stable_identity,
)
from igor_context_compiler import (
    CompilerFailure,
    ContextModelCompileRequest,
    ContextModelReference,
    ResolvedContextContract,
    ResolvedContextEvaluation,
    ResolvedContextModelManifest,
    ResolvedContextOutput,
    compile_context_model,
)
PROJECT_VERSION = "0.1"
LOCK_VERSION = "0.1"
TARGET_DIRNAME = "target"
SECRET_KEY_PARTS = ("api_key", "credential", "password", "secret", "token")
NON_SECRET_KEY_ALLOWLIST = {
    "max_tokens",
    "budget_tokens",
    "token_estimate",
}
REQUIRED_EVALUATION_CATEGORIES = {
    "retrieval",
    "authority",
    "temporal",
    "contract",
    "package",
    "lineage",
}
ALLOWED_TOP_LEVEL_FILES = {
    "igor_project.yml",
    "packages.yml",
    ".env.example",
    ".gitignore",
    "README.md",
    "package_request.example.yml",
    "igor.lock",
    "runtime.yml",
}
PROJECT_DIRECTORIES = (
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
    "fixtures",
)
FIXTURE_DIRECTORIES = ("sources", "expected", "mutations")
ALLOWED_ASSET_SUFFIXES = {".yml", ".yaml", ".md", ".json", ".txt", ".gitignore", ".example", ".png", ".jpg", ".jpeg"}


class ProjectError(ValueError):
    """Deterministic project-level error."""


class ProjectDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["error", "warning"]
    code: str
    file: str
    object_name: str
    field_path: str
    message: str
    suggestion: str | None = None


class ProjectRuntimeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    args: tuple[str, ...] = ()
    produces: str | None = None


class ProjectRuntimeFile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    commands: dict[str, ProjectRuntimeCommand | tuple[ProjectRuntimeCommand, ...]] = Field(default_factory=dict)


if TYPE_CHECKING:
    from igor_runner.project_runtime import PackagePublicationRequest


class ProjectDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    project_version: Literal["0.1"] = PROJECT_VERSION
    qualification_mode: Literal["development", "production"] = "development"
    default_context_model: str | None = None
    compatibility: tuple[str, ...] = ("0.1",)


class ProjectFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    project: ProjectDefinition


class PackageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: str
    compatibility: tuple[str, ...] = ("0.1",)


class PackagesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    packages: tuple[PackageSpec, ...] = ()


class ContextModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    revision: str
    title: str | None = None


class ModelUses(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[str, ...] = ()
    authority: tuple[str, ...] = ()
    retrievals: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    contracts: tuple[str, ...] = ()
    evaluations: tuple[str, ...] = ()


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    context_model: ContextModelMetadata
    objects: dict[str, dict[str, Any]]
    presentation: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tests: tuple[dict[str, Any], ...] = ()
    uses: ModelUses = Field(default_factory=ModelUses)


class ProjectSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_contract: str
    connector_binding: str
    identity_fields: tuple[str, ...] = ()


class ProjectConnectorBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_id: str
    revision: str
    source_contract_ref: str
    connector: str
    deployment_ref: str
    resources: tuple[ConnectorResourceBinding, ...]


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    source_contracts: dict[str, ContextSourceContract] = Field(default_factory=dict)
    connector_bindings: dict[str, ProjectConnectorBinding] = Field(default_factory=dict)
    sources: dict[str, ProjectSourceBinding] = Field(default_factory=dict)


class SchemaFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    schemas: dict[str, SchemaDescriptor] = Field(default_factory=dict)


class SemanticDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    definition_id: str
    revision: str
    scope: str
    description: str
    concepts: tuple[dict[str, Any], ...] = ()
    rules: dict[str, Any] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    display_name: str | None = None

    def semantic_material(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "revision": self.revision,
            "scope": self.scope,
            "description": self.description,
            "concepts": self.concepts,
            "rules": self.rules,
            "coverage": self.coverage,
        }

    @property
    def identity(self) -> str:
        return stable_identity(self.semantic_material())


class TaxonomyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    semantic_definitions: dict[str, SemanticDefinition] = Field(default_factory=dict)


class RecipeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe_id: str
    revision: str
    accepted_representation_types: tuple[str, ...]
    accepted_media_types: tuple[str, ...]
    output_schema: str
    prompt_version: str
    taxonomy_version: str
    evidence_required: bool = True
    abstention_allowed: bool = True


class EnrichmentFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    recipes: dict[str, RecipeDefinition] = Field(default_factory=dict)


class AuthorityPolicyDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str
    revision: str
    required_authority_level: str
    historical_evidence_allowed: bool = False
    freshness_hours: int = Field(ge=0)
    allowed_as_of_max_age_days: int = Field(ge=0)
    conflict_behavior: Literal["select", "reject", "abstain", "conflict"]
    prohibited_sources: tuple[str, ...] = ()
    description: str | None = None

    def semantic_material(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "revision": self.revision,
            "required_authority_level": self.required_authority_level,
            "historical_evidence_allowed": self.historical_evidence_allowed,
            "freshness_hours": self.freshness_hours,
            "allowed_as_of_max_age_days": self.allowed_as_of_max_age_days,
            "conflict_behavior": self.conflict_behavior,
            "prohibited_sources": self.prohibited_sources,
        }

    @property
    def identity(self) -> str:
        return stable_identity(self.semantic_material())


class AuthorityAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    policy: str
    policy_id: str | None = None
    policy_revision: str | None = None
    accepted_outcomes: tuple[str, ...] = ()
    active_only: bool = False
    valid_at_task_time: bool = False


class PolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    authority_policies: dict[str, AuthorityPolicyDefinition] = Field(default_factory=dict)
    authority: dict[str, AuthorityAssignment] = Field(default_factory=dict)


class ContractBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_package_items: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_bytes: int = Field(gt=0)


class ContextContractDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    version: str
    permitted_consumers: tuple[str, ...]
    permitted_tasks: tuple[str, ...]
    permitted_purposes: tuple[str, ...]
    allowed_modalities: tuple[str, ...]
    required_authority_level: str
    historical_evidence_allowed: bool = False
    allowed_as_of_max_age_days: int = Field(ge=0)
    budgets: ContractBudgets
    freshness_hours: int = Field(ge=0)
    citations_required: bool = True
    abstain_conditions: tuple[str, ...] = ()
    prohibited_uses: tuple[str, ...] = ()
    title: str | None = None
    description: str | None = None

    def semantic_material(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "permitted_consumers": self.permitted_consumers,
            "permitted_tasks": self.permitted_tasks,
            "permitted_purposes": self.permitted_purposes,
            "allowed_modalities": self.allowed_modalities,
            "required_authority_level": self.required_authority_level,
            "historical_evidence_allowed": self.historical_evidence_allowed,
            "allowed_as_of_max_age_days": self.allowed_as_of_max_age_days,
            "budgets": self.budgets.model_dump(mode="json"),
            "freshness_hours": self.freshness_hours,
            "citations_required": self.citations_required,
            "abstain_conditions": self.abstain_conditions,
            "prohibited_uses": self.prohibited_uses,
        }

    @property
    def identity(self) -> str:
        return stable_identity(self.semantic_material())


class ContractFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    context_contracts: dict[str, ContextContractDefinition] = Field(default_factory=dict)


class RetrievalDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search: Literal["vector", "full_text", "hybrid"]
    candidate_limit: int = Field(gt=0)
    target_roles: tuple[str, ...] = ()
    eligibility: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    display_name: str | None = None


class RetrievalFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    retrievals: dict[str, RetrievalDefinition] = Field(default_factory=dict)


class OutputDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_type: str
    retrieval: str | None = None
    contract: str | None = None
    target_roles: tuple[str, ...] = ()
    format: str | None = None
    destination: str | None = None
    display_name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class OutputFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    outputs: dict[str, OutputDefinition] = Field(default_factory=dict)


class EvaluationFixtures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_cases: tuple[str, ...] = ()
    expected_cases: tuple[str, ...] = ()
    mutation_cases: tuple[str, ...] = ()


class EvaluationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Literal[
        "retrieval",
        "schema",
        "taxonomy",
        "authority",
        "temporal",
        "contract",
        "package",
        "lineage",
        "quality",
        "operational",
    ]
    description: str
    target_roles: tuple[str, ...] = ()
    target_outputs: tuple[str, ...] = ()
    target_contracts: tuple[str, ...] = ()
    target_retrievals: tuple[str, ...] = ()
    fixtures: EvaluationFixtures = Field(default_factory=EvaluationFixtures)
    checks: tuple[str, ...] = ()
    required: bool = True
    hidden_expectations: bool = True


class EvalFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = PROJECT_VERSION
    evaluations: dict[str, EvaluationDefinition] = Field(default_factory=dict)


class LockedPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: str
    source: Literal["path"] = "path"
    revision: str
    compatibility: tuple[str, ...] = ()
    integrity_hash: str
    dependency_names: tuple[str, ...] = ()


class LockFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = LOCK_VERSION
    project_version: str
    packages: tuple[LockedPackage, ...] = ()
    dependency_graph: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    lock_identity: str


class ProjectCompileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: dict[str, Any]
    graph: dict[str, Any]
    catalog: dict[str, Any]
    evaluation_plan: dict[str, Any]
    compile_report_html: str


@dataclass(frozen=True)
class RegistryValue:
    ref: str
    path: Path
    value: Any


@dataclass(frozen=True)
class LoadedDependency:
    name: str
    root: Path
    project: ProjectFile
    packages: PackagesFile
    hash: str
    dependencies: tuple[str, ...]


@dataclass
class LoadedProject:
    root: Path
    project: ProjectFile
    packages: PackagesFile
    lock: LockFile | None
    models: dict[str, RegistryValue]
    sources: dict[str, RegistryValue]
    source_contracts: dict[str, RegistryValue]
    connector_bindings: dict[str, RegistryValue]
    schemas: dict[str, RegistryValue]
    semantic_definitions: dict[str, RegistryValue]
    recipes: dict[str, RegistryValue]
    authority_policies: dict[str, RegistryValue]
    authority_assignments: dict[str, RegistryValue]
    context_contracts: dict[str, RegistryValue]
    retrievals: dict[str, RegistryValue]
    outputs: dict[str, RegistryValue]
    evaluations: dict[str, RegistryValue]
    profiles: dict[str, RegistryValue]
    fixture_cases: dict[str, dict[str, Path]]
    dependencies: dict[str, LoadedDependency]
    diagnostics: list[ProjectDiagnostic]


def discover_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "igor_project.yml").is_file():
            return candidate
    raise ProjectError(f"no IGOR project found from {current}")


def init_project(project_name: str, *, path: str | Path | None = None, template: str | Path | None = None) -> Path:
    target_parent = Path(path or Path.cwd()).resolve()
    root = target_parent / _project_slug(project_name)
    if template is not None:
        return _init_project_from_template(root, Path(template))
    files = _starter_project_files(project_name)
    if root.exists():
        if not root.is_dir():
            raise ProjectError(f"target exists and is not a directory: {root}")
        existing = _existing_layout_state(root, files)
        if existing == "same":
            raise ProjectError(f"project already initialized at {root}")
        if any(root.iterdir()):
            raise ProjectError(f"target directory contains conflicting files: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return root


def deps_project(root: str | Path) -> LockFile:
    project_root = discover_project_root(root)
    project_file = _load_required_yaml_file(project_root / "igor_project.yml", ProjectFile)
    packages = _load_required_yaml_file(project_root / "packages.yml", PackagesFile)
    lock = _build_lock(project_root, project_file, packages)
    (project_root / "igor.lock").write_text(_yaml_text(lock.model_dump(mode="json")), encoding="utf-8")
    return lock


def validate_project(root: str | Path, *, select: str | None = None) -> tuple[list[ProjectDiagnostic], list[ProjectDiagnostic]]:
    loaded = _load_project(discover_project_root(root), select=select, require_lock=True)
    errors = [item for item in loaded.diagnostics if item.severity == "error"]
    warnings = [item for item in loaded.diagnostics if item.severity == "warning"]
    if errors:
        return errors, warnings
    try:
        _compile_selected_manifest(loaded, select=select)
    except (ProjectError, CompilerFailure) as error:
        errors.append(
            ProjectDiagnostic(
                severity="error",
                code="project_invalid",
                file=str(loaded.root / "igor_project.yml"),
                object_name=select or loaded.project.project.default_context_model or loaded.project.project.name,
                field_path="$",
                message=str(error),
                suggestion="Resolve the referenced definition or adjust the selected model.",
            )
        )
    return errors, warnings


def compile_project(root: str | Path, *, select: str | None = None, target: str | None = None) -> ProjectCompileOutput:
    loaded = _load_project(discover_project_root(root), select=select, require_lock=True)
    errors = [item for item in loaded.diagnostics if item.severity == "error"]
    if errors:
        raise ProjectError(_render_diagnostics(errors))
    artifact, _manifest = _compile_project_bundle(loaded, select=select)
    target_root = _target_root(loaded.root, target)
    _write_compile_artifacts(target_root, artifact)
    return artifact


def _compile_project_bundle(loaded: LoadedProject, *, select: str | None) -> tuple[ProjectCompileOutput, ResolvedContextModelManifest]:
    try:
        manifest = _compile_selected_manifest(loaded, select=select)
    except CompilerFailure as error:
        raise ProjectError(str(error)) from error
    graph = _build_graph(manifest)
    catalog = _build_catalog(loaded, manifest)
    evaluation_plan = _build_evaluation_plan(loaded, manifest)
    report = _build_compile_report(loaded, manifest, graph, evaluation_plan)
    artifact = ProjectCompileOutput(
        manifest=manifest.artifact(),
        graph=graph,
        catalog=catalog,
        evaluation_plan=evaluation_plan,
        compile_report_html=report,
    )
    return artifact, manifest


def project_command(
    root: str | Path,
    command: str,
    *,
    select: str | None = None,
    target: str | None = None,
    live: bool = False,
    request: str | Path | None = None,
    sql: str | None = None,
    store: str | Path | None = None,
) -> dict[str, Any]:
    project_root = discover_project_root(root)
    loaded = _load_project(project_root, select=select, require_lock=command not in {"sync", "status", "observe", "serve"})
    diagnostics = loaded.diagnostics
    errors = [item for item in diagnostics if item.severity == "error"]
    warnings = [item for item in diagnostics if item.severity == "warning"]
    if errors and command not in {"sync", "status", "observe", "serve"}:
        raise ProjectError(_render_diagnostics(errors, warnings))

    target_root = _target_root(project_root, target)
    target_root.mkdir(parents=True, exist_ok=True)

    base: dict[str, Any] = {
        "artifact_type": f"igor-project-{command}",
        "schema_version": PROJECT_VERSION,
        "command": command,
        "project_name": loaded.project.project.name,
        "project_version": loaded.project.project.version,
        "selected_context_model": _selected_model(loaded, select).ref,
        "default_context_model": loaded.project.project.default_context_model,
        "lock_identity": loaded.lock.lock_identity if loaded.lock is not None else None,
        "warnings": [item.model_dump(mode="json") for item in warnings],
        "errors": [item.model_dump(mode="json") for item in errors],
        "valid": not errors,
    }

    if command == "sync":
        runtime_result = _execute_project_runtime_command(project_root, target_root, "sync") if live else None
        payload = {
            **base,
            "sources": sorted(loaded.sources),
            "source_contracts": sorted(loaded.source_contracts),
            "connector_bindings": sorted(loaded.connector_bindings),
            "packages": [item.name for item in loaded.packages.packages],
            "dependency_graph": {name: list(values) for name, values in sorted((loaded.lock.dependency_graph if loaded.lock else {}).items())},
            "fixture_inventory": {name: sorted(values) for name, values in sorted((name, cases.keys()) for name, cases in loaded.fixture_cases.items())},
        }
        if runtime_result is not None:
            payload["runtime"] = runtime_result
    elif command in {"build", "resolve", "query", "explain", "evaluate", "qualify", "diff", "test", "tune", "serve", "status", "observe"}:
        artifact, manifest = _compile_project_bundle(loaded, select=select)
        graph = artifact.graph
        evaluation_plan = artifact.evaluation_plan
        previous = _previous_manifest(project_root, target)
        diff = _plan_diff(previous, manifest, graph)
        live_config = _live_configuration_status() if live else {}
        qualification_plan = plan_project(project_root, select=select, target=target, live=live) if command == "qualify" else None
        preview = {
            "manifest_identity": manifest.identity,
            "context_model_id": manifest.context_model_id,
            "context_model_revision": manifest.revision,
            "sources": [item.role for item in manifest.sources],
            "objects": [item.role for item in manifest.objects],
            "retrievals": [item.role for item in manifest.retrievals],
            "outputs": [item.role for item in manifest.outputs],
            "contracts": [item.ref for item in manifest.context_contracts],
            "evaluations": [item.evaluation_id for item in manifest.evaluations],
        }
        if command == "build":
            payload = {**base, **preview, "graph_nodes": len(graph["nodes"]), "graph_edges": len(graph["edges"]), "catalog": artifact.catalog, "evaluation_plan": evaluation_plan, "compile_report": str(target_root / "compile-report.html")}
        elif command == "resolve":
            payload = {**base, **preview, "resolved_references": len(manifest.sources) + len(manifest.objects) + len(manifest.authority) + len(manifest.retrievals) + len(manifest.context_contracts) + len(manifest.outputs) + len(manifest.evaluations), "resolved_outputs": len(manifest.outputs)}
        elif command == "query":
            query_result = _query_project_standard_relations(project_root, sql, store=store) if sql is not None else None
            payload = {
                **base,
                **preview,
                "catalog": artifact.catalog,
                "retrieval_roles": [item.role for item in manifest.retrievals],
                "queryable_sql": sorted(artifact.catalog.get("queryable_relations", [])),
            }
            if query_result is not None:
                payload["sql_result"] = query_result
        elif command == "explain":
            payload = {**base, **preview, "compile_report": str(target_root / "compile-report.html"), "summary": artifact.compile_report_html[:512]}
        elif command == "test":
            payload = {**base, **preview, "validation": {"errors": len(errors), "warnings": len(warnings)}, "evaluation_categories": sorted(evaluation_plan["categories"])}
        elif command == "evaluate":
            runtime_result = _execute_project_runtime_command(project_root, target_root, "evaluate") if live else None
            payload = {**base, **preview, "evaluation_plan": evaluation_plan, "evaluation_categories": sorted(evaluation_plan["categories"]), "required_evaluations": [item.evaluation_id for item in manifest.evaluations if item.required]}
            if runtime_result is not None:
                payload["runtime"] = runtime_result
        elif command == "qualify":
            runtime_result = _execute_project_runtime_command(project_root, target_root, "qualify") if live else None
            payload = {**base, **preview, "validation": {"errors": len(errors), "warnings": len(warnings)}, "plan": qualification_plan, "diff": diff, "evaluation_plan": evaluation_plan}
            if runtime_result is not None:
                payload["runtime"] = runtime_result
        elif command == "diff":
            payload = {**base, **preview, **diff}
        elif command == "tune":
            suggestions: list[str] = []
            if not live_config.get("IGOR_EMBEDDING_API_KEY", True):
                suggestions.append("set IGOR_EMBEDDING_API_KEY for live embedding checks")
            if not live_config.get("IGOR_COMPLETION_API_KEY", True):
                suggestions.append("set IGOR_COMPLETION_API_KEY for live completion checks")
            if warnings:
                suggestions.extend(item.suggestion for item in warnings if item.suggestion)
            payload = {**base, **preview, "suggestions": suggestions, "live_profile_checks": live_config, "estimated_items": _estimate_items(manifest), "estimated_tokens": _estimate_tokens(manifest), "estimated_requests": _estimate_requests(manifest), "estimated_concurrency": _estimate_concurrency(loaded)}
        elif command == "serve":
            payload = {**base, **preview, "serve_mode": "static-preview", "ready": not errors, "target_root": str(target_root)}
        elif command == "status":
            payload = {**base, **preview, "has_lock": loaded.lock is not None, "dependency_count": len(loaded.dependencies), "target_root": str(target_root), "artifact_paths": sorted(str(path.relative_to(project_root)) for path in target_root.glob("*") if path.is_file())}
        elif command == "observe":
            payload = {**base, **preview, "source_inventory": sorted(loaded.sources), "fixture_inventory": {name: sorted(values) for name, values in sorted((name, cases.keys()) for name, cases in loaded.fixture_cases.items())}, "profile_capabilities": sorted({entry.value.capability for entry in loaded.profiles.values()})}
        else:
            payload = {**base, **preview}
        if command in {"build", "resolve", "query", "explain", "evaluate", "qualify"}:
            _write_compile_artifacts(target_root, artifact)
    else:
        raise ProjectError(f"unknown project command: {command}")

    artifact_path = target_root / f"{command}.json"
    _write_json(artifact_path, payload)
    return payload


def plan_project(root: str | Path, *, select: str | None = None, target: str | None = None, live: bool = False) -> dict[str, Any]:
    project_root = discover_project_root(root)
    loaded = _load_project(project_root, select=select, require_lock=True)
    errors = [item for item in loaded.diagnostics if item.severity == "error"]
    if errors:
        raise ProjectError(_render_diagnostics(errors))
    try:
        manifest = _compile_selected_manifest(loaded, select=select)
    except CompilerFailure as error:
        raise ProjectError(str(error)) from error
    graph = _build_graph(manifest)
    previous = _previous_manifest(project_root, target)
    diff = _plan_diff(previous, manifest, graph)
    live_config = _live_configuration_status() if live else {}
    plan = {
        "artifact_type": "igor-project-plan",
        "schema_version": PROJECT_VERSION,
        "context_model_id": manifest.context_model_id,
        "context_model_revision": manifest.revision,
        "manifest_identity": manifest.identity,
        "selected_context_models": [manifest.context_model_id],
        "sources_to_synchronize": [item.role for item in manifest.sources],
        "expected_snapshots": [item.role for item in manifest.objects if item.kind == "source_snapshot"],
        "derivations_to_run": [item.role for item in manifest.objects if item.kind in {"semantic_derivation", "retrieval_projection", "authority_assertion"}],
        "retrieval_indexes_affected": diff["affected_retrievals"],
        "authority_policies_involved": [item.role for item in manifest.authority],
        "contracts_to_evaluate": [item.ref for item in manifest.context_contracts],
        "outputs_considered": [item.role for item in manifest.outputs],
        "evaluations_to_run": [item.evaluation_id for item in manifest.evaluations if item.required],
        "packages_expected_to_rebuild": [item.role for item in manifest.outputs],
        "outputs_expected_to_be_reused": diff["unchanged_outputs"] if previous else [],
        "outputs_expected_to_be_invalidated": diff["affected_outputs"],
        "affected_outputs": diff["affected_outputs"],
        "expected_provider_capabilities": sorted({item.value.capability for item in loaded.profiles.values()}),
        "estimated_items": _estimate_items(manifest),
        "estimated_bytes": None,
        "estimated_tokens": _estimate_tokens(manifest),
        "estimated_requests": _estimate_requests(manifest),
        "estimated_concurrency": _estimate_concurrency(loaded),
        "estimated_cost": None,
        "estimate_uncertainty": "No runtime inventory is consulted; byte, cost, and reuse values remain bounded estimates.",
        "missing_configuration": [key for key, value in live_config.items() if value is False] if live else [],
        "live_profile_checks": live_config,
        "reuse_estimate_available": previous is not None,
        "runtime_reuse_state": "unavailable" if previous is None else "manifest-diff-only",
        "affected_nodes": diff["affected_nodes"],
        "unchanged_nodes": diff["unchanged_nodes"],
        "affected_retrievals": diff["affected_retrievals"],
    }
    target_root = _target_root(project_root, target)
    target_root.mkdir(parents=True, exist_ok=True)
    _write_json(target_root / "execution-plan.json", plan)
    return plan


def package_project(
    root: str | Path,
    *,
    request: str | Path,
    select: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    from igor_runner.project_runtime import (
        PackagePublicationRequest,
        publish_project_package,
    )

    project_root = discover_project_root(root)
    loaded = _load_project(project_root, select=select, require_lock=True)
    errors = [item for item in loaded.diagnostics if item.severity == "error"]
    if errors:
        raise ProjectError(_render_diagnostics(errors))
    try:
        manifest = _compile_selected_manifest(loaded, select=select)
    except CompilerFailure as error:
        raise ProjectError(str(error)) from error
    request_path = Path(request)
    if not request_path.is_absolute():
        request_path = (project_root / request_path).resolve()
    try:
        request_path.relative_to(project_root)
    except ValueError as error:
        raise ProjectError(f"package request must stay within the project boundary: {request}") from error
    publication_request = _load_required_yaml_file(request_path, PackagePublicationRequest)
    contract = _manifest_contract_for_output(manifest, publication_request.publication.output_role)
    target_root = _target_root(project_root, target)
    target_root.mkdir(parents=True, exist_ok=True)
    result = publish_project_package(
        project_root=project_root,
        target_root=target_root,
        request_path=request_path,
        manifest=manifest,
        contract=contract,
        request=publication_request,
    )
    artifact_name = _package_artifact_name(request_path)
    _write_json(target_root / f"{artifact_name}-package-publication.json", result.summary)
    _write_json(
        target_root / f"{artifact_name}-package-request.json",
        publication_request.model_dump(mode="json"),
    )
    if not result.published:
        decisions = ", ".join(result.summary["decisions"]) or "no-decision"
        raise ProjectError(
            f"Context Package publication failed closed for {publication_request.publication.output_role}: {decisions}"
        )
    return result.summary


def render_diagnostics(errors: Sequence[ProjectDiagnostic], warnings: Sequence[ProjectDiagnostic] = ()) -> str:
    lines: list[str] = []
    for bucket in (errors, warnings):
        for item in bucket:
            line = f"{item.severity.upper()} {item.code} {item.file} {item.field_path}: {item.message}"
            if item.suggestion:
                line += f" Suggestion: {item.suggestion}"
            lines.append(line)
    return "\n".join(lines)


def _load_project(root: Path, *, select: str | None, require_lock: bool) -> LoadedProject:
    diagnostics: list[ProjectDiagnostic] = []
    project = _load_yaml_file(root / "igor_project.yml", ProjectFile, diagnostics)
    packages = _load_yaml_file(root / "packages.yml", PackagesFile, diagnostics)
    lock = _load_yaml_file(root / "igor.lock", LockFile, diagnostics) if (root / "igor.lock").is_file() else None
    if require_lock and lock is None:
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="missing_dependency_lock",
                file=str(root / "igor.lock"),
                object_name="igor.lock",
                field_path="$",
                message="dependency lock is missing",
                suggestion="Run `igor deps` before validate, compile, or plan.",
            )
        )
    models = _load_registry_directory(root / "models", ModelFile, "model", diagnostics)
    sources = _load_registry_directory(root / "sources", SourceFile, "source", diagnostics)
    schemas = _load_registry_directory(root / "schemas", SchemaFile, "schema", diagnostics)
    taxonomies = _load_registry_directory(root / "taxonomies", TaxonomyFile, "taxonomy", diagnostics)
    enrichments = _load_registry_directory(root / "enrichments", EnrichmentFile, "enrichment", diagnostics)
    policies = _load_registry_directory(root / "policies", PolicyFile, "policy", diagnostics)
    contracts = _load_registry_directory(root / "contracts", ContractFile, "contract", diagnostics)
    retrievals = _load_registry_directory(root / "retrievals", RetrievalFile, "retrieval", diagnostics)
    outputs = _load_registry_directory(root / "outputs", OutputFile, "output", diagnostics)
    evals = _load_registry_directory(root / "evals", EvalFile, "eval", diagnostics)
    profiles = _load_profiles(root / "profiles", diagnostics)
    fixture_cases = _load_fixtures(root / "fixtures", diagnostics)
    loaded = LoadedProject(
        root=root,
        project=project if project is not None else ProjectFile(project=ProjectDefinition(name="unknown", version="0")),
        packages=packages if packages is not None else PackagesFile(),
        lock=lock,
        models=_flatten_models(models, diagnostics),
        sources=_flatten_named_sections(sources, "sources", diagnostics),
        source_contracts=_flatten_named_sections(sources, "source_contracts", diagnostics),
        connector_bindings=_flatten_named_sections(sources, "connector_bindings", diagnostics),
        schemas=_flatten_named_sections(schemas, "schemas", diagnostics),
        semantic_definitions=_flatten_named_sections(taxonomies, "semantic_definitions", diagnostics),
        recipes=_flatten_named_sections(enrichments, "recipes", diagnostics),
        authority_policies=_flatten_named_sections(policies, "authority_policies", diagnostics),
        authority_assignments=_flatten_named_sections(policies, "authority", diagnostics),
        context_contracts=_flatten_named_sections(contracts, "context_contracts", diagnostics),
        retrievals=_flatten_named_sections(retrievals, "retrievals", diagnostics),
        outputs=_flatten_named_sections(outputs, "outputs", diagnostics),
        evaluations=_flatten_named_sections(evals, "evaluations", diagnostics),
        profiles=profiles,
        fixture_cases=fixture_cases,
        dependencies={},
        diagnostics=diagnostics,
    )
    _scan_project_secrets(loaded)
    _validate_profile_presence(loaded)
    _validate_selection(loaded, select)
    if loaded.lock is not None:
        loaded.dependencies = _validate_and_load_dependencies(loaded)
        _merge_dependency_definitions(loaded)
    _validate_project_content(loaded)
    return loaded


def _compile_selected_manifest(loaded: LoadedProject, *, select: str | None) -> ResolvedContextModelManifest:
    model_entry = _selected_model(loaded, select)
    model = model_entry.value
    declaration = {
        "context_model": model.context_model.model_dump(mode="json"),
        "objects": dict(model.objects),
        "presentation": dict(model.presentation),
        "tests": [dict(item) for item in model.tests],
    }
    selected_sources = _select_registry(loaded.sources, model.uses.sources, "source role")
    selected_authority = _select_registry(loaded.authority_assignments, model.uses.authority, "authority role")
    selected_retrievals = _select_registry(loaded.retrievals, model.uses.retrievals, "retrieval role")
    selected_outputs = _select_registry(loaded.outputs, model.uses.outputs, "output role")
    selected_contracts = _select_registry(loaded.context_contracts, model.uses.contracts, "contract ref")
    selected_evaluations = _select_registry(loaded.evaluations, model.uses.evaluations, "evaluation")
    declaration["sources"] = {role: entry.value.model_dump(mode="json") for role, entry in sorted(selected_sources.items())}
    declaration["authority"] = {role: entry.value.model_dump(mode="json") for role, entry in sorted(selected_authority.items())}
    declaration["retrievals"] = {role: entry.value.model_dump(mode="json") for role, entry in sorted(selected_retrievals.items())}
    declaration["outputs"] = {role: entry.value.model_dump(mode="json") for role, entry in sorted(selected_outputs.items())}
    declaration["context_contracts"] = {ref: entry.value.model_dump(mode="json") for ref, entry in sorted(selected_contracts.items())}
    declaration["evaluations"] = {ref: entry.value.model_dump(mode="json") for ref, entry in sorted(selected_evaluations.items())}
    references = _build_references(loaded, declaration)
    return compile_context_model(ContextModelCompileRequest(declaration=declaration, references=tuple(references)))


def _build_references(loaded: LoadedProject, declaration: Mapping[str, Any]) -> list[ContextModelReference]:
    references: dict[tuple[str, str], ContextModelReference] = {}
    selected_sources = declaration.get("sources", {})
    objects = declaration.get("objects", {})
    authority = declaration.get("authority", {})
    selected_contracts = declaration.get("context_contracts", {})
    selected_outputs = declaration.get("outputs", {})
    selected_evaluations = declaration.get("evaluations", {})
    for role, value in sorted(dict(selected_sources).items()):
        source = ProjectSourceBinding.model_validate(value)
        contract_entry = _registry_value(loaded.source_contracts, source.source_contract, "source_contract")
        binding_entry = _registry_value(loaded.connector_bindings, source.connector_binding, "connector_binding")
        contract = contract_entry.value
        binding = _resolved_connector_binding(binding_entry.value, contract)
        binding.validate_against(contract)
        references[("source_contract", source.source_contract)] = ContextModelReference(
            kind="source_contract",
            ref=source.source_contract,
            identity=contract.identity,
            revision=contract.revision,
            material=contract.model_dump(mode="json"),
        )
        references[("connector_binding", source.connector_binding)] = ContextModelReference(
            kind="connector_binding",
            ref=source.connector_binding,
            identity=binding.identity,
            revision=binding.revision,
            material=binding.model_dump(mode="json"),
        )
    for role, raw in sorted(dict(objects).items()):
        value = dict(raw)
        operation = value.get("operation")
        if isinstance(operation, str):
            references[("operation", operation)] = ContextModelReference(
                kind="operation",
                ref=operation,
                identity=stable_identity({"operation": operation}),
                material={"operation": operation},
            )
        schema_ref = value.get("schema")
        if isinstance(schema_ref, str):
            schema = _registry_value(loaded.schemas, schema_ref, "schema").value
            references[("schema", schema_ref)] = ContextModelReference(
                kind="schema",
                ref=schema_ref,
                identity=schema.identity,
                revision=schema.revision,
                material=schema.model_dump(mode="json"),
            )
        definition_ref = value.get("semantic_definition")
        if isinstance(definition_ref, str):
            definition = _registry_value(loaded.semantic_definitions, definition_ref, "semantic_definition").value
            references[("semantic_definition", definition_ref)] = ContextModelReference(
                kind="semantic_definition",
                ref=definition_ref,
                identity=definition.identity,
                revision=definition.revision,
                material=definition.model_dump(mode="json"),
            )
        recipe_ref = value.get("recipe")
        if isinstance(recipe_ref, str):
            recipe = _resolved_recipe(_registry_value(loaded.recipes, recipe_ref, "recipe").value, loaded)
            references[("recipe", recipe_ref)] = ContextModelReference(
                kind="recipe",
                ref=recipe_ref,
                identity=recipe.identity,
                revision=recipe.revision,
                material=recipe.model_dump(mode="json"),
            )
        profile_ref = value.get("model_profile")
        if isinstance(profile_ref, str):
            profile = _registry_value(loaded.profiles, profile_ref, "model_profile").value
            references[("model_profile", profile_ref)] = ContextModelReference(
                kind="model_profile",
                ref=profile_ref,
                identity=profile.identity,
                revision=profile.revision,
                material=profile.model_dump(mode="json"),
            )
    for role, raw in sorted(dict(authority).items()):
        policy_ref = str(raw["policy"])
        policy = _registry_value(loaded.authority_policies, policy_ref, "authority_policy").value
        references[("authority_policy", policy_ref)] = ContextModelReference(
            kind="authority_policy",
            ref=policy_ref,
            identity=policy.identity,
            revision=policy.revision,
            material=policy.model_dump(mode="json"),
        )
    for ref in sorted(selected_contracts):
        contract = _registry_value(loaded.context_contracts, ref, "context_contract").value
        references[("context_contract", ref)] = ContextModelReference(
            kind="context_contract",
            ref=ref,
            identity=contract.identity,
            revision=contract.version,
            material=contract.model_dump(mode="json"),
        )
    for role, raw in sorted(dict(selected_outputs).items()):
        output = OutputDefinition.model_validate(raw)
        if output.contract:
            contract = _registry_value(loaded.context_contracts, output.contract, "context_contract").value
            references[("context_contract", output.contract)] = ContextModelReference(
                kind="context_contract",
                ref=output.contract,
                identity=contract.identity,
                revision=contract.version,
                material=contract.model_dump(mode="json"),
            )
    for ref in sorted(selected_evaluations):
        evaluation = _registry_value(loaded.evaluations, ref, "evaluation").value
        references[("evaluation", ref)] = ContextModelReference(
            kind="evaluation",
            ref=ref,
            identity=stable_identity({"evaluation": ref, **evaluation.model_dump(mode="json")}),
            material=evaluation.model_dump(mode="json"),
        )
    return [references[key] for key in sorted(references)]


def _resolved_connector_binding(binding: ProjectConnectorBinding, contract: ContextSourceContract) -> ConnectorBinding:
    resources = tuple(
        ConnectorResourceBinding(
            resource_id=item.resource_id,
            source_resource=item.source_resource,
            fields=tuple(ConnectorFieldBinding(concept_id=field.concept_id, source_field=field.source_field) for field in item.fields),
            filters=dict(item.filters),
            cursor_field=item.cursor_field,
        )
        for item in binding.resources
    )
    return ConnectorBinding(
        binding_id=binding.binding_id,
        revision=binding.revision,
        source_contract_identity=contract.identity,
        connector=binding.connector,
        deployment_ref=binding.deployment_ref,
        resources=resources,
    )


def _resolved_recipe(recipe: RecipeDefinition, loaded: LoadedProject) -> EnrichmentRecipe:
    schema = _registry_value(loaded.schemas, recipe.output_schema, "schema").value
    return EnrichmentRecipe(
        recipe_id=recipe.recipe_id,
        revision=recipe.revision,
        accepted_representation_types=recipe.accepted_representation_types,
        accepted_media_types=recipe.accepted_media_types,
        output_schema_identity=schema.identity,
        prompt_version=recipe.prompt_version,
        taxonomy_version=recipe.taxonomy_version,
        evidence_required=recipe.evidence_required,
        abstention_allowed=recipe.abstention_allowed,
    )


def _validate_project_content(loaded: LoadedProject) -> None:
    model_roles = {
        role
        for entry in loaded.models.values()
        for role in entry.value.objects
    }
    for role, entry in loaded.sources.items():
        source = entry.value
        if source.source_contract not in loaded.source_contracts:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "source_contract", source.source_contract))
        if source.connector_binding not in loaded.connector_bindings:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "connector_binding", source.connector_binding))
    for ref, entry in loaded.recipes.items():
        recipe = entry.value
        if recipe.output_schema not in loaded.schemas:
            loaded.diagnostics.append(_missing_ref(entry.path, ref, "output_schema", recipe.output_schema))
    for role, entry in loaded.authority_assignments.items():
        value = entry.value
        if value.target not in model_roles:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "target", value.target))
        if value.policy not in loaded.authority_policies:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "policy", value.policy))
    for role, entry in loaded.retrievals.items():
        value = entry.value
        resolution_policy = value.resolution.get("policy")
        if resolution_policy and resolution_policy not in loaded.authority_assignments:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "resolution.policy", str(resolution_policy)))
        for target in value.target_roles:
            if target not in model_roles:
                loaded.diagnostics.append(_missing_ref(entry.path, role, "target_roles", target))
    for role, entry in loaded.outputs.items():
        value = entry.value
        if value.retrieval and value.retrieval not in loaded.retrievals:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "retrieval", value.retrieval))
        if value.contract and value.contract not in loaded.context_contracts:
            loaded.diagnostics.append(_missing_ref(entry.path, role, "contract", value.contract))
        for target in value.target_roles:
            if target not in model_roles:
                loaded.diagnostics.append(_missing_ref(entry.path, role, "target_roles", target))
    for evaluation_id, entry in loaded.evaluations.items():
        value = entry.value
        for role in value.target_roles:
            if role not in model_roles:
                loaded.diagnostics.append(_missing_ref(entry.path, evaluation_id, "target_roles", role))
        for role in value.target_outputs:
            if role not in loaded.outputs:
                loaded.diagnostics.append(_missing_ref(entry.path, evaluation_id, "target_outputs", role))
        for ref in value.target_contracts:
            if ref not in loaded.context_contracts:
                loaded.diagnostics.append(_missing_ref(entry.path, evaluation_id, "target_contracts", ref))
        for ref in value.target_retrievals:
            if ref not in loaded.retrievals:
                loaded.diagnostics.append(_missing_ref(entry.path, evaluation_id, "target_retrievals", ref))
        _validate_fixture_refs(loaded, entry.path, evaluation_id, value.fixtures)
    semantic_identities = defaultdict(list)
    for ref, entry in loaded.schemas.items():
        semantic_identities[entry.value.identity].append((ref, entry.path))
    for ref, entry in loaded.semantic_definitions.items():
        semantic_identities[entry.value.identity].append((ref, entry.path))
    for ref, entry in loaded.recipes.items():
        try:
            semantic_identities[_resolved_recipe(entry.value, loaded).identity].append((ref, entry.path))
        except ProjectError:
            continue
    for ref, entry in loaded.authority_policies.items():
        semantic_identities[entry.value.identity].append((ref, entry.path))
    for ref, entry in loaded.context_contracts.items():
        semantic_identities[entry.value.identity].append((ref, entry.path))
    for profile_id, entry in loaded.profiles.items():
        semantic_identities[entry.value.identity].append((profile_id, entry.path))
    for identity, values in semantic_identities.items():
        if len(values) > 1:
            for ref, path in values:
                loaded.diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="duplicate_semantic_identity",
                        file=str(path),
                        object_name=ref,
                        field_path="$",
                        message=f"semantic identity {identity} is produced by multiple definitions",
                        suggestion="Change a meaning-bearing field or remove the duplicate definition.",
                    )
                )
    selected_categories = {entry.value.category for entry in loaded.evaluations.values()}
    missing_required = sorted(REQUIRED_EVALUATION_CATEGORIES - selected_categories)
    if missing_required:
        severity: Literal["error", "warning"] = "error" if loaded.project.project.qualification_mode == "production" else "warning"
        loaded.diagnostics.append(
            ProjectDiagnostic(
                severity=severity,
                code="missing_required_evaluations",
                file=str(loaded.root / "evals"),
                object_name=loaded.project.project.name,
                field_path="evals",
                message="missing required evaluation categories: " + ", ".join(missing_required),
                suggestion="Add the missing evaluation declarations before production qualification.",
            )
        )


def _validate_fixture_refs(loaded: LoadedProject, path: Path, evaluation_id: str, fixtures: EvaluationFixtures) -> None:
    inventories = {
        "source_cases": loaded.fixture_cases["sources"],
        "expected_cases": loaded.fixture_cases["expected"],
        "mutation_cases": loaded.fixture_cases["mutations"],
    }
    for field, available in inventories.items():
        for case_id in getattr(fixtures, field):
            if case_id not in available:
                loaded.diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="missing_fixture_case",
                        file=str(path),
                        object_name=evaluation_id,
                        field_path=f"fixtures.{field}",
                        message=f"fixture case {case_id} is missing",
                        suggestion="Create the fixture file under the matching fixtures directory.",
                    )
                )


def _load_yaml_file(path: Path, model: type[BaseModel], diagnostics: list[ProjectDiagnostic] | None = None) -> BaseModel | None:
    if not path.is_file():
        if diagnostics is not None:
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="missing_file",
                    file=str(path),
                    object_name=path.name,
                    field_path="$",
                    message="required file is missing",
                    suggestion="Create the file or run `igor init` for a starter project.",
                )
            )
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("YAML document must be a mapping")
        return model.model_validate(payload)
    except (ValidationError, ValueError, yaml.YAMLError) as error:
        if diagnostics is not None:
            diagnostics.extend(_diagnostics_from_exception(path, error))
            return None
        raise


def _load_required_yaml_file(path: Path, model: type[BaseModel]) -> BaseModel:
    diagnostics: list[ProjectDiagnostic] = []
    document = _load_yaml_file(path, model, diagnostics)
    if document is None:
        raise ProjectError(_render_diagnostics(diagnostics))
    return document


def _load_registry_directory(root: Path, model: type[BaseModel], label: str, diagnostics: list[ProjectDiagnostic]) -> dict[str, RegistryValue]:
    if not root.is_dir():
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="missing_directory",
                file=str(root),
                object_name=label,
                field_path="$",
                message="required project directory is missing",
                suggestion="Create the directory or re-run `igor init` in a clean target.",
            )
        )
        return {}
    result: dict[str, RegistryValue] = {}
    for path in sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml")):
        document = _load_yaml_file(path, model, diagnostics)
        if document is None:
            continue
        result[path.stem] = RegistryValue(ref=path.stem, path=path, value=document)
    return result


def _load_profiles(root: Path, diagnostics: list[ProjectDiagnostic]) -> dict[str, RegistryValue]:
    result: dict[str, RegistryValue] = {}
    if not root.is_dir():
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="missing_directory",
                file=str(root),
                object_name="profiles",
                field_path="$",
                message="required project directory is missing",
                suggestion="Create the directory or re-run `igor init` in a clean target.",
            )
        )
        return result
    for path in sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml")):
        try:
            profile = load_model_profile(path)
        except (ValidationError, ValueError, yaml.YAMLError) as error:
            diagnostics.extend(_diagnostics_from_exception(path, error))
            continue
        result[profile.profile_id] = RegistryValue(ref=profile.profile_id, path=path, value=profile)
    return result


def _load_fixtures(root: Path, diagnostics: list[ProjectDiagnostic]) -> dict[str, dict[str, Path]]:
    inventory: dict[str, dict[str, Path]] = {name: {} for name in FIXTURE_DIRECTORIES}
    if not root.is_dir():
        diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="missing_directory",
                file=str(root),
                object_name="fixtures",
                field_path="$",
                message="required fixtures directory is missing",
                suggestion="Create the starter fixtures or run `igor init`.",
            )
        )
        return inventory
    for name in FIXTURE_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="missing_directory",
                    file=str(directory),
                    object_name=name,
                    field_path="$",
                    message="required fixtures subdirectory is missing",
                    suggestion="Create the missing fixtures subdirectory.",
                )
            )
            continue
        for path in sorted(directory.glob("*")):
            if path.is_file():
                inventory[name][path.stem] = path
    return inventory


def _flatten_models(documents: Mapping[str, RegistryValue], diagnostics: list[ProjectDiagnostic]) -> dict[str, RegistryValue]:
    result: dict[str, RegistryValue] = {}
    for entry in documents.values():
        model = entry.value
        key = model.context_model.id
        if key in result:
            diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="duplicate_model_id",
                    file=str(entry.path),
                    object_name=key,
                    field_path="context_model.id",
                    message="context model ID is duplicated",
                    suggestion="Rename one of the model IDs so the project can select deterministically.",
                )
            )
            continue
        result[key] = RegistryValue(ref=key, path=entry.path, value=model)
    return result


def _flatten_named_sections(documents: Mapping[str, RegistryValue], field: str, diagnostics: list[ProjectDiagnostic]) -> dict[str, RegistryValue]:
    result: dict[str, RegistryValue] = {}
    for entry in documents.values():
        section = getattr(entry.value, field)
        for key, value in sorted(section.items()):
            if key in result:
                diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="duplicate_definition",
                        file=str(entry.path),
                        object_name=key,
                        field_path=field,
                        message=f"{field} declares {key} more than once across project files",
                        suggestion="Keep one canonical definition for each role or reference.",
                    )
                )
                continue
            result[key] = RegistryValue(ref=key, path=entry.path, value=value)
    return result


def _registry_value(registry: Mapping[str, RegistryValue], key: str, kind: str) -> RegistryValue:
    try:
        return registry[key]
    except KeyError as error:
        raise ProjectError(f"missing {kind} reference: {key}") from error


def _select_registry(registry: Mapping[str, RegistryValue], selected: Sequence[str], kind: str) -> dict[str, RegistryValue]:
    return {item: _registry_value(registry, item, kind) for item in selected}


def _selected_model(loaded: LoadedProject, select: str | None) -> RegistryValue:
    if select is not None:
        return _registry_value(loaded.models, select, "context model")
    default = loaded.project.project.default_context_model
    if default is not None and default in loaded.models:
        return loaded.models[default]
    if len(loaded.models) == 1:
        return next(iter(loaded.models.values()))
    raise ProjectError("project contains multiple context models; pass --select <context-model>")


def _manifest_contract_for_output(manifest: ResolvedContextModelManifest, output_role: str) -> dict[str, Any]:
    output = next((item for item in manifest.outputs if item.role == output_role), None)
    if output is None:
        raise ProjectError(f"unknown manifest output role: {output_role}")
    if output.contract_ref is None:
        raise ProjectError(f"manifest output role has no Context Contract: {output_role}")
    contract = next((item for item in manifest.context_contracts if item.ref == output.contract_ref), None)
    if contract is None:
        raise ProjectError(f"manifest output contract is unresolved: {output.contract_ref}")
    return {
        "ref": contract.ref,
        "contract_id": contract.contract_id,
        "version": contract.version,
        "title": contract.title,
        "rules": dict(contract.rules),
        "identity": contract.identity,
    }


def _validate_selection(loaded: LoadedProject, select: str | None) -> None:
    try:
        _selected_model(loaded, select)
    except ProjectError as error:
        loaded.diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="selection_required",
                file=str(loaded.root / "igor_project.yml"),
                object_name=loaded.project.project.name,
                field_path="project.default_context_model",
                message=str(error),
                suggestion="Set project.default_context_model or pass --select.",
            )
        )


def _validate_profile_presence(loaded: LoadedProject) -> None:
    capabilities = {entry.value.capability for entry in loaded.profiles.values()}
    for capability in ("embedding", "completion"):
        if capability not in capabilities:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="missing_profile_capability",
                    file=str(loaded.root / "profiles"),
                    object_name=capability,
                    field_path="profiles",
                    message=f"project must declare at least one {capability} profile",
                    suggestion="Add a separate versioned model profile for that capability.",
                )
            )


def _validate_and_load_dependencies(loaded: LoadedProject) -> dict[str, LoadedDependency]:
    assert loaded.lock is not None
    direct_specs = {item.name: item for item in loaded.packages.packages}
    actual = {item.name: item for item in loaded.lock.packages}
    dependencies: dict[str, LoadedDependency] = {}
    reachable: set[str] = set()
    visiting: set[str] = set()

    missing_direct = sorted(set(direct_specs) - set(actual))
    if missing_direct:
        loaded.diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="unlocked_dependency",
                file=str(loaded.root / "igor.lock"),
                object_name="igor.lock",
                field_path="packages",
                message="lock file is missing dependency entries: " + ", ".join(missing_direct),
                suggestion="Run `igor deps` to regenerate the lock.",
            )
        )

    for name, spec in sorted(direct_specs.items()):
        lock = actual.get(name)
        if lock is None:
            continue
        expected_path = _dependency_root_from_path(loaded.root, loaded.root, spec.path)
        if expected_path is None:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="path_escapes_project",
                    file=str(loaded.root / "packages.yml"),
                    object_name=name,
                    field_path=f"packages.{name}.path",
                    message=f"dependency path escapes the project boundary: {spec.path}",
                    suggestion="Keep deterministic dependency packages inside the project tree.",
                )
            )
            continue
        if lock.path != str(expected_path.relative_to(loaded.root)):
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="stale_locked_dependency",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="packages.path",
                    message=f"lock file path for {name} no longer matches packages.yml",
                    suggestion="Run `igor deps` after updating packages.yml.",
                )
            )
        if lock.compatibility != spec.compatibility:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="stale_locked_dependency",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="packages.compatibility",
                    message=f"lock file compatibility for {name} no longer matches packages.yml",
                    suggestion="Run `igor deps` after updating packages.yml.",
                )
            )

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if name in dependencies:
            reachable.add(name)
            return
        if name in visiting:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="dependency_cycle",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="dependency_graph",
                    message="dependency cycle detected: " + " -> ".join((*chain, name)),
                    suggestion="Break the cycle so dependency locking stays deterministic.",
                )
            )
            return
        lock = actual.get(name)
        if lock is None:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="unlocked_dependency",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="dependency_graph",
                    message=f"lock file is missing dependency entry: {name}",
                    suggestion="Run `igor deps` to regenerate the lock.",
                )
            )
            return
        visiting.add(name)
        reachable.add(name)
        dependency_root = _dependency_root_from_path(loaded.root, loaded.root, lock.path)
        if dependency_root is None:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="path_escapes_project",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="packages.path",
                    message=f"locked dependency path escapes the project boundary: {lock.path}",
                    suggestion="Run `igor deps` after correcting the dependency path.",
                )
            )
            visiting.remove(name)
            return
        if not dependency_root.is_dir():
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="missing_dependency",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="packages.path",
                    message="locked dependency path does not exist",
                    suggestion="Restore the dependency directory or run `igor deps` after updating packages.yml.",
                )
            )
            visiting.remove(name)
            return
        dep_project = _load_yaml_file(dependency_root / "igor_project.yml", ProjectFile, loaded.diagnostics)
        dep_packages = _load_yaml_file(dependency_root / "packages.yml", PackagesFile, loaded.diagnostics)
        dependency_hash = _project_content_hash(dependency_root)
        if dependency_hash != lock.integrity_hash:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="invalid_dependency_integrity",
                    file=str(loaded.root / "igor.lock"),
                    object_name=name,
                    field_path="packages.integrity_hash",
                    message=f"dependency content hash changed for {name}",
                    suggestion="Run `igor deps` after reviewing the dependency changes.",
                )
            )
        declared_children = tuple(lock.dependency_names)
        if dep_packages is not None:
            package_children = tuple(sorted(item.name for item in dep_packages.packages))
            if package_children != tuple(sorted(declared_children)):
                loaded.diagnostics.append(
                    ProjectDiagnostic(
                        severity="error",
                        code="stale_locked_dependency",
                        file=str(loaded.root / "igor.lock"),
                        object_name=name,
                        field_path="dependency_graph",
                        message=f"lock file dependency graph for {name} no longer matches packages.yml",
                        suggestion="Run `igor deps` after updating dependency packages.",
                    )
                )
        if dep_project is not None and dep_packages is not None:
            dependencies[name] = LoadedDependency(
                name=name,
                root=dependency_root,
                project=dep_project,
                packages=dep_packages,
                hash=dependency_hash,
                dependencies=declared_children,
            )
        for child in declared_children:
            visit(child, (*chain, name))
        visiting.remove(name)

    for name in sorted(direct_specs):
        visit(name, ())

    extra = sorted(set(actual) - reachable)
    if extra:
        loaded.diagnostics.append(
            ProjectDiagnostic(
                severity="error",
                code="stale_locked_dependency",
                file=str(loaded.root / "igor.lock"),
                object_name="igor.lock",
                field_path="packages",
                message="lock file contains unreachable dependency entries: " + ", ".join(extra),
                suggestion="Run `igor deps` after updating packages.yml.",
            )
        )
    return dependencies


def _validate_dependency_cycles(loaded: LoadedProject, dependencies: Mapping[str, LoadedDependency]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, chain: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="dependency_cycle",
                    file=str(loaded.root / "packages.yml"),
                    object_name=name,
                    field_path="packages",
                    message="dependency cycle detected: " + " -> ".join((*chain, name)),
                    suggestion="Break the cycle so dependency locking stays deterministic.",
                )
            )
            return
        visiting.add(name)
        dependency = dependencies.get(name)
        if dependency is not None:
            for child in dependency.dependencies:
                visit(child, (*chain, name))
        visiting.remove(name)
        visited.add(name)

    for name in sorted(dependencies):
        visit(name, ())


def _merge_dependency_definitions(loaded: LoadedProject) -> None:
    if not loaded.dependencies:
        return
    for dependency in loaded.dependencies.values():
        documents = {
            "sources": _load_registry_directory(dependency.root / "sources", SourceFile, "source", loaded.diagnostics),
            "schemas": _load_registry_directory(dependency.root / "schemas", SchemaFile, "schema", loaded.diagnostics),
            "taxonomies": _load_registry_directory(dependency.root / "taxonomies", TaxonomyFile, "taxonomy", loaded.diagnostics),
            "enrichments": _load_registry_directory(dependency.root / "enrichments", EnrichmentFile, "enrichment", loaded.diagnostics),
            "policies": _load_registry_directory(dependency.root / "policies", PolicyFile, "policy", loaded.diagnostics),
            "contracts": _load_registry_directory(dependency.root / "contracts", ContractFile, "contract", loaded.diagnostics),
            "retrievals": _load_registry_directory(dependency.root / "retrievals", RetrievalFile, "retrieval", loaded.diagnostics),
            "outputs": _load_registry_directory(dependency.root / "outputs", OutputFile, "output", loaded.diagnostics),
            "evals": _load_registry_directory(dependency.root / "evals", EvalFile, "eval", loaded.diagnostics),
        }
        _merge_registry(loaded.source_contracts, _flatten_named_sections(documents["sources"], "source_contracts", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.connector_bindings, _flatten_named_sections(documents["sources"], "connector_bindings", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.sources, _flatten_named_sections(documents["sources"], "sources", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.schemas, _flatten_named_sections(documents["schemas"], "schemas", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.semantic_definitions, _flatten_named_sections(documents["taxonomies"], "semantic_definitions", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.recipes, _flatten_named_sections(documents["enrichments"], "recipes", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.authority_policies, _flatten_named_sections(documents["policies"], "authority_policies", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.authority_assignments, _flatten_named_sections(documents["policies"], "authority", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.context_contracts, _flatten_named_sections(documents["contracts"], "context_contracts", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.retrievals, _flatten_named_sections(documents["retrievals"], "retrievals", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.outputs, _flatten_named_sections(documents["outputs"], "outputs", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.evaluations, _flatten_named_sections(documents["evals"], "evaluations", loaded.diagnostics), loaded, dependency.name)
        _merge_registry(loaded.profiles, _load_profiles(dependency.root / "profiles", loaded.diagnostics), loaded, dependency.name)


def _merge_registry(
    destination: dict[str, RegistryValue],
    incoming: Mapping[str, RegistryValue],
    loaded: LoadedProject,
    dependency_name: str,
) -> None:
    for key, entry in incoming.items():
        if key in destination:
            loaded.diagnostics.append(
                ProjectDiagnostic(
                    severity="error",
                    code="duplicate_dependency_definition",
                    file=str(entry.path),
                    object_name=key,
                    field_path="$",
                    message=f"dependency {dependency_name} duplicates an existing definition for {key}",
                    suggestion="Rename the imported definition or remove the conflicting package.",
                )
            )
            continue
        destination[key] = entry


def _build_lock(root: Path, project: ProjectFile, packages: PackagesFile) -> LockFile:
    nodes: dict[str, LockedPackage] = {}
    graph: dict[str, tuple[str, ...]] = {}
    visiting: set[Path] = set()

    def visit(package: PackageSpec, owner_root: Path) -> LockedPackage:
        dependency_root = _dependency_root_from_path(root, owner_root, package.path)
        if dependency_root is None:
            raise ProjectError(f"dependency path escapes the project boundary: {package.path}")
        if dependency_root in visiting:
            raise ProjectError(f"dependency cycle detected at {package.name}")
        if package.name in nodes:
            return nodes[package.name]
        visiting.add(dependency_root)
        dep_project = _load_required_yaml_file(dependency_root / "igor_project.yml", ProjectFile)
        dep_packages = _load_required_yaml_file(dependency_root / "packages.yml", PackagesFile)
        dependencies: list[str] = []
        for child in dep_packages.packages:
            locked_child = visit(child, dependency_root)
            dependencies.append(locked_child.name)
        visiting.remove(dependency_root)
        record = LockedPackage(
            name=package.name,
            path=str(dependency_root.relative_to(root)),
            revision=dep_project.project.version,
            compatibility=package.compatibility,
            integrity_hash=_project_content_hash(dependency_root),
            dependency_names=tuple(sorted(dependencies)),
        )
        nodes[package.name] = record
        graph[package.name] = record.dependency_names
        return record

    for package in packages.packages:
        visit(package, root)
    payload = {
        "project_version": project.project.version,
        "packages": [nodes[name].model_dump(mode="json") for name in sorted(nodes)],
        "dependency_graph": {name: list(graph.get(name, ())) for name in sorted(nodes)},
    }
    return LockFile(
        project_version=project.project.version,
        packages=tuple(nodes[name] for name in sorted(nodes)),
        dependency_graph={name: tuple(graph.get(name, ())) for name in sorted(nodes)},
        lock_identity=stable_identity(payload),
    )


def _project_content_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _tracked_project_files(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _tracked_project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if relative.parts[0] == ".igor":
            continue
        if candidate.name == ".env":
            continue
        if candidate.name == "igor.lock":
            continue
        files.append(candidate)
    return sorted(files)


def _scan_project_secrets(loaded: LoadedProject) -> None:
    def secret_key(normalized: str) -> bool:
        if normalized in NON_SECRET_KEY_ALLOWLIST:
            return False
        if normalized.endswith("_ref"):
            return False
        return any(part in normalized for part in SECRET_KEY_PARTS)

    def scan(path: Path, value: Any, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_text = str(key)
                normalized = key_text.lower().replace("-", "_")
                if secret_key(normalized):
                    if nested not in (None, "", []):
                        loaded.diagnostics.append(
                            ProjectDiagnostic(
                                severity="error",
                                code="credential_value_detected",
                                file=str(path),
                                object_name=trail[0] if trail else path.name,
                                field_path=".".join((*trail, key_text)) or "$",
                                message="tracked configuration appears to contain a secret-bearing value",
                                suggestion="Move the secret to runtime environment configuration and keep only non-secret references in versioned files.",
                            )
                        )
                scan(path, nested, (*trail, key_text))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                scan(path, nested, (*trail, str(index)))

    for file_path in _tracked_project_files(loaded.root):
        if file_path.name == ".env.example":
            for number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if "=" in line and line.split("=", 1)[1].strip():
                    loaded.diagnostics.append(
                        ProjectDiagnostic(
                            severity="error",
                            code="credential_value_detected",
                            file=str(file_path),
                            object_name=".env.example",
                            field_path=f"line.{number}",
                            message="example environment file must not contain secret values",
                            suggestion="Leave the value blank and document the variable name only.",
                        )
                    )
            continue
        if file_path.suffix in {".yml", ".yaml", ".json"}:
            payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            scan(file_path, payload)


def _build_graph(manifest: ResolvedContextModelManifest) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for source in manifest.sources:
        nodes.append({"node_id": source.role, "node_type": "source", "identity": source.identity, "label": source.role})
    objects = {item.role: item for item in manifest.objects}
    for item in manifest.objects:
        nodes.append({"node_id": item.role, "node_type": item.kind, "identity": item.identity, "label": item.display_name or item.role})
        if item.source_role:
            edges.append({"source": item.source_role, "target": item.role, "edge_type": "declares"})
        for dependency in item.derived_from:
            edges.append({"source": dependency, "target": item.role, "edge_type": "derived_from"})
    for item in manifest.authority:
        nodes.append({"node_id": item.role, "node_type": "authority", "identity": item.identity, "label": item.role})
        edges.append({"source": item.target_role, "target": item.role, "edge_type": "governed_by"})
    for item in manifest.retrievals:
        nodes.append({"node_id": item.role, "node_type": "retrieval", "identity": item.identity, "label": item.display_name or item.role})
        if item.resolution_policy_role:
            edges.append({"source": item.resolution_policy_role, "target": item.role, "edge_type": "resolved_by"})
        for target_role in item.target_roles:
            edges.append({"source": target_role, "target": item.role, "edge_type": "indexes"})
    for item in manifest.context_contracts:
        nodes.append({"node_id": item.ref, "node_type": "contract", "identity": item.identity, "label": item.title or item.ref})
    for item in manifest.outputs:
        nodes.append({"node_id": item.role, "node_type": "output", "identity": item.identity, "label": item.display_name or item.role})
        if item.retrieval_role:
            edges.append({"source": item.retrieval_role, "target": item.role, "edge_type": "packages"})
        if item.contract_ref:
            edges.append({"source": item.contract_ref, "target": item.role, "edge_type": "constrained_by"})
        for target_role in item.target_roles:
            edges.append({"source": target_role, "target": item.role, "edge_type": "includes"})
    for item in manifest.evaluations:
        nodes.append({"node_id": item.evaluation_id, "node_type": "evaluation", "identity": item.identity, "label": item.evaluation_id})
        for role in item.target_roles:
            edges.append({"source": role, "target": item.evaluation_id, "edge_type": "evaluates"})
        for role in item.target_outputs:
            edges.append({"source": role, "target": item.evaluation_id, "edge_type": "evaluates"})
        for ref in item.target_contracts:
            edges.append({"source": ref, "target": item.evaluation_id, "edge_type": "evaluates"})
        for ref in item.target_retrievals:
            edges.append({"source": ref, "target": item.evaluation_id, "edge_type": "evaluates"})
    return {
        "artifact_type": "igor-context-graph",
        "schema_version": PROJECT_VERSION,
        "context_model_id": manifest.context_model_id,
        "manifest_identity": manifest.identity,
        "nodes": sorted(nodes, key=lambda item: (item["node_type"], item["node_id"])),
        "edges": sorted(edges, key=lambda item: (item["source"], item["target"], item["edge_type"])),
    }


def _build_catalog(loaded: LoadedProject, manifest: ResolvedContextModelManifest) -> dict[str, Any]:
    return {
        "artifact_type": "igor-context-catalog",
        "schema_version": PROJECT_VERSION,
        "project_name": loaded.project.project.name,
        "context_model_id": manifest.context_model_id,
        "manifest_identity": manifest.identity,
        "sources": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.sources],
        "objects": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.objects],
        "authority": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.authority],
        "retrievals": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.retrievals],
        "contracts": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.context_contracts],
        "outputs": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.outputs],
        "evaluations": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.evaluations],
        "profiles": [
            entry.value.model_dump(mode="json") | {"identity": entry.value.identity}
            for _, entry in sorted(loaded.profiles.items())
        ],
    }


def _build_evaluation_plan(loaded: LoadedProject, manifest: ResolvedContextModelManifest) -> dict[str, Any]:
    categories = defaultdict(list)
    for item in manifest.evaluations:
        categories[item.category].append(item.evaluation_id)
    production_gate = {
        "mode": loaded.project.project.qualification_mode,
        "required_categories": sorted(REQUIRED_EVALUATION_CATEGORIES),
        "missing_categories": sorted(REQUIRED_EVALUATION_CATEGORIES - set(categories)),
        "fail_closed": loaded.project.project.qualification_mode == "production",
    }
    return {
        "artifact_type": "igor-evaluation-plan",
        "schema_version": PROJECT_VERSION,
        "context_model_id": manifest.context_model_id,
        "manifest_identity": manifest.identity,
        "evaluations": [item.model_dump(mode="json") | {"identity": item.identity} for item in manifest.evaluations],
        "categories": {key: sorted(value) for key, value in sorted(categories.items())},
        "fixture_inventory": {name: sorted(values) for name, values in sorted((key, list(items)) for key, items in ((name, cases.keys()) for name, cases in loaded.fixture_cases.items()))},
        "production_gate": production_gate,
    }


def _build_compile_report(
    loaded: LoadedProject,
    manifest: ResolvedContextModelManifest,
    graph: Mapping[str, Any],
    evaluation_plan: Mapping[str, Any],
) -> str:
    def row(label: str, value: Any) -> str:
        return f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>"

    sections = [
        "<html><head><meta charset='utf-8'><title>IGOR Compile Report</title></head><body>",
        f"<h1>{html.escape(loaded.project.project.name)}</h1>",
        "<table>",
        row("Context model", manifest.context_model_id),
        row("Revision", manifest.revision),
        row("Manifest identity", manifest.identity),
        row("Sources", len(manifest.sources)),
        row("Objects", len(manifest.objects)),
        row("Authority policies", len(manifest.authority)),
        row("Retrievals", len(manifest.retrievals)),
        row("Contracts", len(manifest.context_contracts)),
        row("Outputs", len(manifest.outputs)),
        row("Evaluations", len(manifest.evaluations)),
        "</table>",
        "<h2>Graph</h2>",
        f"<p>{len(graph['nodes'])} nodes, {len(graph['edges'])} edges.</p>",
        "<h2>Evaluation plan</h2>",
        "<ul>",
    ]
    for category, values in sorted(evaluation_plan["categories"].items()):
        sections.append(f"<li>{html.escape(category)}: {html.escape(', '.join(values))}</li>")
    sections.extend(["</ul>", "</body></html>"])
    return "".join(sections)


def _previous_manifest(root: Path, target: str | None) -> ResolvedContextModelManifest | None:
    target_root = _target_root(root, target)
    path = target_root / "manifest.previous.json"
    if not path.is_file():
        path = target_root / "manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ResolvedContextModelManifest.model_validate(payload["manifest"] if "manifest" in payload else payload)


def _plan_diff(
    previous: ResolvedContextModelManifest | None,
    current: ResolvedContextModelManifest,
    graph: Mapping[str, Any],
) -> dict[str, list[str]]:
    current_nodes = {node["node_id"]: node["identity"] for node in graph["nodes"]}
    if previous is None:
        return {
            "affected_nodes": sorted(current_nodes),
            "unchanged_nodes": [],
            "affected_outputs": [item.role for item in current.outputs],
            "unchanged_outputs": [],
            "affected_retrievals": [item.role for item in current.retrievals],
        }
    previous_graph = _build_graph(previous)
    previous_nodes = {node["node_id"]: node["identity"] for node in previous_graph["nodes"]}
    changed = {node for node, identity in current_nodes.items() if previous_nodes.get(node) != identity}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph["edges"]:
        adjacency[str(edge["source"])].add(str(edge["target"]))
    affected = set(changed)
    queue = deque(sorted(changed))
    while queue:
        node = queue.popleft()
        for target in sorted(adjacency.get(node, ())):
            if target not in affected:
                affected.add(target)
                queue.append(target)
    unchanged_outputs = [item.role for item in current.outputs if item.role not in affected]
    return {
        "affected_nodes": sorted(affected),
        "unchanged_nodes": sorted(set(current_nodes) - affected),
        "affected_outputs": [item.role for item in current.outputs if item.role in affected],
        "unchanged_outputs": unchanged_outputs,
        "affected_retrievals": [item.role for item in current.retrievals if item.role in affected],
    }


def _estimate_items(manifest: ResolvedContextModelManifest) -> int:
    return sum(contract.budgets.get("max_package_items", 0) for contract in manifest.context_contracts)


def _estimate_tokens(manifest: ResolvedContextModelManifest) -> int:
    return sum(contract.budgets.get("max_tokens", 0) for contract in manifest.context_contracts)


def _estimate_requests(manifest: ResolvedContextModelManifest) -> int:
    return len([item for item in manifest.objects if item.kind in {"semantic_derivation", "retrieval_projection"}])


def _estimate_concurrency(loaded: LoadedProject) -> int:
    values = []
    for entry in loaded.profiles.values():
        parameters = entry.value.parameters
        if isinstance(parameters, dict) and "max_concurrency" in parameters:
            values.append(int(parameters["max_concurrency"]))
    return max(values) if values else 1


def _live_configuration_status() -> dict[str, bool]:
    return {
        "IGOR_EMBEDDING_API_KEY": bool(os.environ.get("IGOR_EMBEDDING_API_KEY")),
        "IGOR_COMPLETION_API_KEY": bool(os.environ.get("IGOR_COMPLETION_API_KEY")),
    }


def _execute_project_runtime_command(project_root: Path, target_root: Path, command_name: str) -> dict[str, Any]:
    runtime = _load_project_runtime(project_root)
    declared = runtime.commands.get(command_name)
    if declared is None:
        raise ProjectError(
            f"runtime.yml does not declare a live command for `{command_name}`; add commands.{command_name} or run without --live"
        )
    if isinstance(declared, tuple):
        steps = [
            _execute_project_runtime_step(project_root, target_root, command_name, step, step_index=index)
            for index, step in enumerate(declared)
        ]
        return {"command": command_name, "steps": steps, "valid": all(step["valid"] for step in steps)}
    return _execute_project_runtime_step(project_root, target_root, command_name, declared, step_index=None)


def _execute_project_runtime_step(
    project_root: Path,
    target_root: Path,
    command_name: str,
    command: ProjectRuntimeCommand,
    *,
    step_index: int | None,
) -> dict[str, Any]:
    _validate_runtime_executable(command.command)
    placeholders = _runtime_placeholders(project_root, target_root)
    args = [_expand_runtime_arg(item, placeholders, project_root, target_root) for item in command.args]
    if command.produces is not None:
        produced = Path(_expand_runtime_arg(command.produces, placeholders, project_root, target_root))
        _assert_within(project_root, produced, "runtime produced artifact")
        parent = produced if produced.suffix == "" else produced.parent
        parent.mkdir(parents=True, exist_ok=True)
    expanded_command = [command.command, *args]
    try:
        completed = subprocess.run(
            expanded_command,
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise ProjectError(f"live `{command_name}` command could not start `{command.command}`: {error}") from error
    artifact: dict[str, Any] = {
        "command": command_name,
        "executable": command.command,
        "args": args,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4096:],
        "stderr": completed.stderr[-4096:],
        "valid": completed.returncode == 0,
    }
    if command.produces is not None:
        artifact["produces"] = str(produced)
        if produced.is_file():
            artifact["produces_sha256"] = _file_sha256(produced)
        elif produced.is_dir():
            artifact["produces_files"] = sorted(str(path.relative_to(produced)) for path in produced.rglob("*") if path.is_file())
        else:
            artifact["valid"] = False
            artifact["missing_artifact"] = str(produced)
    if completed.returncode != 0:
        raise ProjectError(f"live `{command_name}` command failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")
    return artifact


def _query_project_standard_relations(project_root: Path, sql: str, *, store: str | Path | None = None) -> dict[str, Any]:
    if not sql.strip():
        raise ProjectError("query SQL cannot be empty")
    store_path = _resolve_project_relation_store(project_root, store)
    try:
        completed = subprocess.run(
            ["igor-runner", "query-standard-relations", str(store_path), sql],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise ProjectError(f"query command could not start igor-runner: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown failure"
        raise ProjectError(detail)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ProjectError("query command returned invalid JSON") from error


def _resolve_project_relation_store(project_root: Path, store: str | Path | None) -> Path:
    if store is None:
        candidate = project_root / ".igor" / "runtime" / "qualification" / "allowed-relations"
    else:
        store_path = Path(store)
        candidate = store_path if store_path.is_absolute() else project_root / store_path
    candidate = candidate.resolve()
    _assert_within(project_root, candidate, "query relation store")
    if not candidate.is_dir():
        raise ProjectError(f"query relation store does not exist: {candidate}")
    return candidate


def _load_project_runtime(project_root: Path) -> ProjectRuntimeFile:
    path = project_root / "runtime.yml"
    if not path.is_file():
        raise ProjectError("runtime.yml is required for --live project commands")
    return _load_required_yaml_file(path, ProjectRuntimeFile)


def _validate_runtime_executable(command: str) -> None:
    if "/" in command or "\\" in command or command in {".", ".."}:
        raise ProjectError(f"runtime command must name an executable, not a path: {command}")
    if not command.startswith("igor-"):
        raise ProjectError(f"runtime command must use an IGOR-owned executable: {command}")


def _runtime_placeholders(project_root: Path, target_root: Path) -> dict[str, str]:
    return {
        "project_root": str(project_root),
        "target_root": str(target_root),
        "source_output": str(target_root / "sources" / "acquisition.json"),
        "source_dir": str(target_root / "sources"),
        "image_selection": str(project_root / "image-selection.json"),
        "image_source_output": str(target_root / "images" / "acquisition.json"),
        "image_source_dir": str(target_root / "images"),
        "qualification_output": str(project_root / ".igor" / "runtime" / "qualification"),
        "qualification_json": str(project_root / ".igor" / "runtime" / "qualification" / "qualification.json"),
        "evaluation_output": str(project_root / ".igor" / "runtime" / "evaluation" / "evaluation.json"),
        "lance_root": str(project_root / ".igor" / "runtime" / "lance"),
        "selection": str(project_root / "selection.json"),
        "reference_profile": str(project_root / "profiles" / "compiler" / "reference.yml"),
        "live_profile": str(project_root / "profiles" / "compiler" / "live.yml"),
    }


def _expand_runtime_arg(value: str, placeholders: Mapping[str, str], project_root: Path, target_root: Path) -> str:
    try:
        expanded = value.format(**placeholders)
    except KeyError as error:
        raise ProjectError(f"unknown runtime command placeholder: {error.args[0]}") from error
    if expanded.startswith(str(project_root)) or expanded.startswith(str(target_root)):
        path = Path(expanded).resolve()
        _assert_within(project_root, path, "runtime command path")
        return str(path)
    if "/" in expanded or "\\" in expanded or expanded.startswith("."):
        raise ProjectError(f"runtime command path must use a project-bounded placeholder: {value}")
    return expanded


def _assert_within(project_root: Path, path: Path, label: str) -> None:
    allowed_roots = (project_root.resolve(), (project_root / ".igor").resolve())
    resolved = path.resolve()
    if any(_is_relative_to(resolved, root) for root in allowed_roots):
        return
    raise ProjectError(f"{label} must stay within the project boundary: {path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _write_compile_artifacts(target_root: Path, artifact: ProjectCompileOutput) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = target_root / "manifest.json"
    if manifest_path.is_file():
        previous_payload = manifest_path.read_text(encoding="utf-8")
        current_payload = canonical_json(artifact.manifest) + "\n"
        if previous_payload != current_payload:
            (target_root / "manifest.previous.json").write_text(previous_payload, encoding="utf-8")
    _write_json(manifest_path, artifact.manifest)
    _write_json(target_root / "graph.json", artifact.graph)
    _write_json(target_root / "catalog.json", artifact.catalog)
    _write_json(target_root / "evaluation-plan.json", artifact.evaluation_plan)
    (target_root / "compile-report.html").write_text(artifact.compile_report_html, encoding="utf-8")


def _target_root(root: Path, target: str | None) -> Path:
    base = (root / ".igor" / TARGET_DIRNAME).resolve()
    if target is None or target == "default":
        return base
    candidate = (base / target).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ProjectError(f"target must stay within {base}: {target}") from error
    return candidate


def _package_artifact_name(path: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip("-")
    return slug or "package"


def _dependency_root_from_path(project_root: Path, owner_root: Path, relative_path: str) -> Path | None:
    dependency_root = (owner_root / relative_path).resolve()
    try:
        dependency_root.relative_to(project_root)
    except ValueError:
        return None
    return dependency_root


def _diagnostics_from_exception(path: Path, error: Exception) -> list[ProjectDiagnostic]:
    if isinstance(error, ValidationError):
        result = []
        for item in error.errors():
            result.append(
                ProjectDiagnostic(
                    severity="error",
                    code="invalid_field",
                    file=str(path),
                    object_name=path.stem,
                    field_path=".".join(str(part) for part in item["loc"]) or "$",
                    message=str(item["msg"]),
                    suggestion="Correct the field shape or remove the unsupported field.",
                )
            )
        return result
    if isinstance(error, yaml.YAMLError):
        return [
            ProjectDiagnostic(
                severity="error",
                code="invalid_yaml",
                file=str(path),
                object_name=path.stem,
                field_path="$",
                message=str(error),
                suggestion="Fix the YAML syntax and try again.",
            )
        ]
    return [
        ProjectDiagnostic(
            severity="error",
            code="invalid_file",
            file=str(path),
            object_name=path.stem,
            field_path="$",
            message=str(error),
            suggestion="Correct the file content to match the project contract.",
        )
    ]


def _missing_ref(path: Path, object_name: str, field_path: str, ref: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        severity="error",
        code="missing_reference",
        file=str(path),
        object_name=object_name,
        field_path=field_path,
        message=f"missing reference: {ref}",
        suggestion="Declare the referenced definition or update the reference value.",
    )


def _yaml_text(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def _project_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ProjectError("project name must contain at least one letter or number")
    return slug


def _init_project_from_template(root: Path, template: Path) -> Path:
    template_root = template.resolve()
    if not (template_root / "igor_project.yml").is_file():
        raise ProjectError(f"template is not an IGOR project: {template_root}")
    if root.exists():
        if not root.is_dir():
            raise ProjectError(f"target exists and is not a directory: {root}")
        if any(root.iterdir()):
            raise ProjectError(f"target directory contains conflicting files: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for path in sorted(template_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(template_root)
        if any(part == ".igor" for part in relative.parts):
            continue
        if not _is_allowed_template_file(relative):
            raise ProjectError(f"template contains unsupported project file: {relative}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return root


def _is_allowed_template_file(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return relative.name in ALLOWED_TOP_LEVEL_FILES or relative.suffix in ALLOWED_ASSET_SUFFIXES
    if relative.parts[0] not in (*PROJECT_DIRECTORIES, "profiles"):
        return False
    return relative.suffix in ALLOWED_ASSET_SUFFIXES or relative.name == ".gitkeep"


def _existing_layout_state(root: Path, files: Mapping[Path, str]) -> Literal["same", "conflict"]:
    for relative, content in files.items():
        path = root / relative
        if not path.is_file():
            return "conflict"
        if path.read_text(encoding="utf-8") != content:
            return "conflict"
    return "same"


def _starter_project_files(project_name: str) -> dict[Path, str]:
    slug = _project_slug(project_name)
    model_id = f"{slug}.context"
    files: dict[Path, str] = {}
    files[Path("igor_project.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "project": {
                "name": slug,
                "version": "1",
                "project_version": PROJECT_VERSION,
                "qualification_mode": "development",
                "default_context_model": model_id,
                "compatibility": ["0.1"],
            },
        }
    )
    files[Path("packages.yml")] = _yaml_text({"packages": []})
    files[Path(".env.example")] = "IGOR_EMBEDDING_API_KEY=\nIGOR_COMPLETION_API_KEY=\n"
    files[Path(".gitignore")] = ".env\n.igor/\n"
    files[Path("package_request.example.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "publication": {
                "output_role": "support_delivery",
                "consumer": "support-assistant",
                "task": "support.answer",
                "purpose": "customer-support",
                "as_of": "2026-08-14T00:00:00Z",
                "evaluated_at": "2026-08-14T00:00:00Z",
                "authority_level": "verified",
                "budget_tokens": 10,
                "budget_bytes": 256,
            },
            "runtime_outputs": [
                {
                    "identity": "support-message-assessment:valid-case",
                    "authority_level": "verified",
                    "status": "active",
                    "byte_length": 256,
                    "valid_until": "2026-08-15T00:00:00Z",
                }
            ],
            "packages": [
                {
                    "task_id": "support.answer",
                    "budget_tokens": 10,
                    "metadata": {"fixture_case": "valid-case"},
                    "items": [
                        {
                            "representation_identity": "support-message-assessment:valid-case",
                            "evidence_identities": ["support-ticket:T-100"],
                            "role": "resolved-context",
                            "rank": 0,
                            "token_estimate": 10,
                            "byte_estimate": 256,
                            "authority_level": "verified",
                            "valid_until": "2026-08-15T00:00:00Z",
                            "status": "active",
                        }
                    ],
                }
            ],
        }
    )
    files[Path("README.md")] = (
        f"# {slug}\n\n"
        "This is an IGOR Context Project.\n\n"
        "Commands:\n\n"
        "```text\n"
        "igor deps\n"
        "igor validate\n"
        "igor compile\n"
        "igor plan\n"
        "igor package --request package_request.example.yml\n"
        "igor status\n"
        "igor observe\n"
        "igor sync\n"
        "igor build\n"
        "igor diff\n"
        "igor resolve\n"
        "igor query\n"
        "igor explain\n"
        "igor test\n"
        "igor evaluate\n"
        "igor qualify\n"
        "igor tune\n"
        "igor serve\n"
        "```\n"
    )
    files[Path("models/example_context.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "context_model": {"id": model_id, "revision": "1", "title": "Example support context"},
            "objects": {
                "support_ticket": {"kind": "business_object", "source": "support_messages", "display_name": "Support ticket {{ ticket_id }}"},
                "support_ticket_snapshot": {"kind": "source_snapshot", "source": "support_messages", "display_name": "{{ ticket_id }} observed at {{ observed_at }}"},
                "support_message": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_ticket_snapshot"],
                    "operation": "representation.v1",
                    "schema": "support.message.v1",
                    "display_name": "Normalized support message",
                },
                "support_message_embedding": {
                    "kind": "retrieval_projection",
                    "derived_from": ["support_message"],
                    "operation": "embedding.v1",
                    "schema": "support.embedding.v1",
                    "model_profile": "example.embedding.profile",
                    "display_name": "Support retrieval embedding",
                },
                "support_message_assessment": {
                    "kind": "semantic_derivation",
                    "derived_from": ["support_message"],
                    "operation": "enrichment.v1",
                    "schema": "support.assessment.v1",
                    "semantic_definition": "support.assessment-taxonomy.v1",
                    "recipe": "support.assessment-recipe.v1",
                    "model_profile": "example.completion.profile",
                    "display_name": "Support assessment",
                },
            },
            "presentation": {
                "support_message": {
                    "preview_fields": ["ticket_id", "customer_message", "status"],
                    "display_name": "Support message preview",
                },
                "support_message_assessment": {
                    "preview_fields": ["intent", "urgency", "recommended_action"],
                    "display_name": "Assessment preview",
                },
            },
            "tests": [
                {"identity_unique": "support_ticket"},
                {"snapshot_integrity": "support_ticket_snapshot"},
                {"evidence_required": "support_message_assessment"},
                {"lineage_complete": "support_message_assessment"},
                {"no_orphan_outputs": {}},
            ],
            "uses": {
                "sources": ["support_messages"],
                "authority": ["support_authority"],
                "retrievals": ["support_search"],
                "outputs": ["support_delivery"],
                "contracts": ["example.agent.v1"],
                "evaluations": [
                    "retrieval.relevance",
                    "authority.selection",
                    "temporal.state",
                    "contract.behavior",
                    "package.quality",
                    "lineage.completeness",
                    "quality.coverage",
                    "operational.accounting",
                ],
            },
        }
    )
    files[Path("sources/example_source.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "source_contracts": {
                "support.messages.v1": {
                    "contract_id": "support.messages",
                    "revision": "1",
                    "domain": "support",
                    "resources": [
                        {
                            "resource_id": "messages",
                            "resource_kind": "ticket_stream",
                            "identity_concepts": ["ticket_id"],
                            "fields": [
                                {"concept_id": "ticket_id", "logical_type": "string", "required": True},
                                {"concept_id": "customer_message", "logical_type": "text", "required": True},
                                {"concept_id": "status", "logical_type": "string", "required": True},
                                {"concept_id": "updated_at", "logical_type": "timestamp", "required": True},
                            ],
                            "accepted_media_types": ["text/plain"],
                            "selection": {"status": "open"},
                            "change_mode": "snapshot",
                            "deletion_semantics": "tombstone",
                        }
                    ],
                }
            },
            "connector_bindings": {
                "fixture.support.messages.v1": {
                    "binding_id": "fixture.support.messages",
                    "revision": "1",
                    "source_contract_ref": "support.messages.v1",
                    "connector": "fixture",
                    "deployment_ref": "local-fixtures",
                    "resources": [
                        {
                            "resource_id": "messages",
                            "source_resource": "support_messages",
                            "fields": [
                                {"concept_id": "ticket_id", "source_field": "ticket_id"},
                                {"concept_id": "customer_message", "source_field": "customer_message"},
                                {"concept_id": "status", "source_field": "status"},
                                {"concept_id": "updated_at", "source_field": "updated_at"},
                            ],
                            "filters": {"status": "open"},
                        }
                    ],
                }
            },
            "sources": {
                "support_messages": {
                    "source_contract": "support.messages.v1",
                    "connector_binding": "fixture.support.messages.v1",
                    "identity_fields": ["ticket_id"],
                }
            },
        }
    )
    files[Path("schemas/example_enrichment.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "schemas": {
                "support.message.v1": {
                    "schema_version": "0.1",
                    "schema_id": "support.message",
                    "revision": "1",
                    "domain": "support",
                    "fields": [
                        {"field_id": "ticket_id", "name": "ticket_id", "logical_type": "string", "required": True},
                        {"field_id": "customer_message", "name": "customer_message", "logical_type": "text", "required": True},
                        {"field_id": "status", "name": "status", "logical_type": "string", "required": True},
                    ],
                    "json_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ticket_id", "customer_message", "status"],
                        "properties": {
                            "ticket_id": {"type": "string"},
                            "customer_message": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                },
                "support.embedding.v1": {
                    "schema_version": "0.1",
                    "schema_id": "support.embedding",
                    "revision": "1",
                    "domain": "support",
                    "fields": [
                        {"field_id": "vector", "name": "vector", "logical_type": "embedding", "required": True}
                    ],
                    "json_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["vector"],
                        "properties": {"vector": {"type": "array", "items": {"type": "number"}}},
                    },
                },
                "support.assessment.v1": {
                    "schema_version": "0.1",
                    "schema_id": "support.assessment",
                    "revision": "1",
                    "domain": "support",
                    "fields": [
                        {"field_id": "intent", "name": "intent", "logical_type": "string", "required": True},
                        {"field_id": "urgency", "name": "urgency", "logical_type": "string", "required": True},
                        {"field_id": "recommended_action", "name": "recommended_action", "logical_type": "string", "required": True},
                    ],
                    "json_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["intent", "urgency", "recommended_action"],
                        "properties": {
                            "intent": {"type": "string"},
                            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
                            "recommended_action": {"type": "string"},
                        },
                    },
                },
            },
        }
    )
    files[Path("taxonomies/example_taxonomy.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "semantic_definitions": {
                "support.assessment-taxonomy.v1": {
                    "definition_id": "support.assessment-taxonomy",
                    "revision": "1",
                    "scope": "support triage",
                    "description": "Intent and urgency guidance for support messages.",
                    "concepts": [
                        {"id": "billing", "description": "Questions about invoices, refunds, or charges."},
                        {"id": "technical", "description": "Product or access issues requiring investigation."},
                        {"id": "information", "description": "General information requests."},
                    ],
                    "rules": {
                        "abstain_when": ["missing_evidence", "conflicting_evidence"],
                        "precedence": ["verified_history", "fresh_customer_message"],
                    },
                    "coverage": {"known_limitations": ["language outside the starter fixture set"]},
                    "display_name": "Support assessment taxonomy",
                }
            },
        }
    )
    files[Path("enrichments/example_recipe.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "recipes": {
                "support.assessment-recipe.v1": {
                    "recipe_id": "support.assessment-recipe",
                    "revision": "1",
                    "accepted_representation_types": ["text"],
                    "accepted_media_types": ["text/plain"],
                    "output_schema": "support.assessment.v1",
                    "prompt_version": "1",
                    "taxonomy_version": "1",
                    "evidence_required": True,
                    "abstention_allowed": True,
                }
            },
        }
    )
    files[Path("policies/example_authority.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "authority_policies": {
                "support.authority-policy.v1": {
                    "policy_id": "support.authority-policy",
                    "revision": "1",
                    "required_authority_level": "verified",
                    "historical_evidence_allowed": False,
                    "freshness_hours": 24,
                    "allowed_as_of_max_age_days": 30,
                    "conflict_behavior": "abstain",
                    "description": "Prefer fresh verified support evidence and fail closed on unresolved conflicts.",
                }
            },
            "authority": {
                "support_authority": {
                    "target": "support_message",
                    "policy": "support.authority-policy.v1",
                    "policy_id": "support.authority-policy",
                    "policy_revision": "1",
                    "accepted_outcomes": ["selected"],
                    "active_only": True,
                    "valid_at_task_time": True,
                }
            },
        }
    )
    files[Path("contracts/example_agent.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "context_contracts": {
                "example.agent.v1": {
                    "contract_id": "example.agent",
                    "version": "1",
                    "permitted_consumers": ["support-assistant"],
                    "permitted_tasks": ["support.answer"],
                    "permitted_purposes": ["customer-support"],
                    "allowed_modalities": ["text"],
                    "required_authority_level": "verified",
                    "historical_evidence_allowed": False,
                    "allowed_as_of_max_age_days": 30,
                    "budgets": {"max_package_items": 8, "max_tokens": 2000, "max_bytes": 65536},
                    "freshness_hours": 24,
                    "citations_required": True,
                    "abstain_conditions": ["missing_evidence", "conflicting_evidence"],
                    "prohibited_uses": ["training", "marketing"],
                    "title": "Starter agent contract",
                    "description": "Vendor-neutral contract for one support-answer package.",
                }
            },
        }
    )
    files[Path("retrievals/example_search.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "retrievals": {
                "support_search": {
                    "search": "hybrid",
                    "candidate_limit": 12,
                    "target_roles": ["support_message", "support_message_assessment"],
                    "eligibility": {"active_only": True, "valid_at_task_time": True},
                    "resolution": {"policy": "support_authority", "accepted_outcomes": ["selected"]},
                    "display_name": "Support search",
                }
            },
        }
    )
    files[Path("outputs/example_delivery.yml")] = _yaml_text(
        {
            "schema_version": PROJECT_VERSION,
            "outputs": {
                "support_delivery": {
                    "output_type": "context_package",
                    "retrieval": "support_search",
                    "contract": "example.agent.v1",
                    "target_roles": ["support_message", "support_message_assessment"],
                    "format": "markdown",
                    "destination": "agent",
                    "display_name": "Starter support delivery",
                    "settings": {"citation_style": "inline"},
                }
            },
        }
    )
    files[Path("profiles/embeddings.example.yml")] = _yaml_text(
        {
            "schema_version": "0.1",
            "profile_id": "example.embedding.profile",
            "capability": "embedding",
            "provider": "example",
            "model": "embedding-model",
            "revision": "1",
            "parameters": {"dimensions": 16, "max_concurrency": 4},
        }
    )
    files[Path("profiles/completions.example.yml")] = _yaml_text(
        {
            "schema_version": "0.1",
            "profile_id": "example.completion.profile",
            "capability": "completion",
            "provider": "example",
            "model": "completion-model",
            "revision": "1",
            "parameters": {"supported_content_kinds": ["text"], "max_concurrency": 4},
        }
    )
    evaluation_payloads = {
        "evals/retrieval.yml": {
            "evaluations": {
                "retrieval.relevance": {
                    "category": "retrieval",
                    "description": "Measure semantic retrieval relevance.",
                    "target_roles": ["support_message"],
                    "target_retrievals": ["support_search"],
                    "fixtures": {"source_cases": ["valid-case", "conflicting-case"], "expected_cases": ["valid-case", "conflicting-case"]},
                    "checks": ["semantic_retrieval_relevance", "evidence_coverage"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/authority.yml": {
            "evaluations": {
                "authority.selection": {
                    "category": "authority",
                    "description": "Measure authority selection and conflicting evidence handling.",
                    "target_roles": ["support_message"],
                    "target_retrievals": ["support_search"],
                    "fixtures": {"source_cases": ["valid-case", "conflicting-case"], "expected_cases": ["valid-case", "conflicting-case"]},
                    "checks": ["authority_selection", "conflicting_evidence"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/temporal.yml": {
            "evaluations": {
                "temporal.state": {
                    "category": "temporal",
                    "description": "Measure current, historical, expired, and unknown temporal states.",
                    "target_roles": ["support_message"],
                    "fixtures": {"source_cases": ["valid-case", "expired-case"], "expected_cases": ["valid-case", "expired-case"]},
                    "checks": ["current_state", "expired_state", "unknown_time_state"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/contract.yml": {
            "evaluations": {
                "contract.behavior": {
                    "category": "contract",
                    "description": "Measure allow, deny, and abstain Context Contract behavior.",
                    "target_contracts": ["example.agent.v1"],
                    "target_outputs": ["support_delivery"],
                    "fixtures": {"source_cases": ["valid-case", "contract-denied-case"], "expected_cases": ["valid-case", "contract-denied-case"]},
                    "checks": ["allow_behavior", "deny_behavior", "abstain_behavior"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/package.yml": {
            "evaluations": {
                "package.quality": {
                    "category": "package",
                    "description": "Measure package contents, citations, and budgets.",
                    "target_outputs": ["support_delivery"],
                    "fixtures": {"source_cases": ["valid-case", "missing-evidence-case"], "expected_cases": ["valid-case", "missing-evidence-case"]},
                    "checks": ["package_contents", "citation_requirement", "budget_limits"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/lineage.yml": {
            "evaluations": {
                "lineage.completeness": {
                    "category": "lineage",
                    "description": "Measure lineage completeness, readable names, and mutation invalidation.",
                    "target_roles": ["support_message_assessment"],
                    "fixtures": {"source_cases": ["valid-case"], "mutation_cases": ["mutation-and-reuse-case"], "expected_cases": ["valid-case", "mutation-and-reuse-case"]},
                    "checks": ["lineage_completeness", "readable_names", "mutation_invalidation"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/quality.yml": {
            "evaluations": {
                "quality.coverage": {
                    "category": "quality",
                    "description": "Measure evidence coverage and unsupported assertions.",
                    "target_roles": ["support_message_assessment"],
                    "fixtures": {"source_cases": ["valid-case", "missing-evidence-case"], "expected_cases": ["valid-case", "missing-evidence-case"]},
                    "checks": ["evidence_coverage", "unsupported_assertions"],
                    "required": True,
                    "hidden_expectations": True,
                }
            }
        },
        "evals/operational.yml": {
            "evaluations": {
                "operational.accounting": {
                    "category": "operational",
                    "description": "Measure request, token, byte, cost, retry, latency, CPU, and memory accounting.",
                    "target_outputs": ["support_delivery"],
                    "fixtures": {"source_cases": ["valid-case", "mutation-and-reuse-case"], "expected_cases": ["valid-case", "mutation-and-reuse-case"]},
                    "checks": ["request_accounting", "token_accounting", "byte_accounting", "cost_accounting", "retry_accounting", "latency_accounting", "cpu_accounting", "memory_accounting", "cache_isolation"],
                    "required": True,
                    "hidden_expectations": False,
                }
            }
        },
    }
    for relative, payload in evaluation_payloads.items():
        files[Path(relative)] = _yaml_text({"schema_version": PROJECT_VERSION, **payload})
    fixture_sources = {
        "fixtures/sources/valid-case.yml": {
            "case_id": "valid-case",
            "ticket_id": "T-100",
            "customer_message": "I was charged twice for the same invoice.",
            "status": "open",
            "updated_at": "2026-08-14T09:00:00Z",
        },
        "fixtures/sources/conflicting-case.yml": {
            "case_id": "conflicting-case",
            "ticket_id": "T-200",
            "customer_message": "One system says the refund is complete, another says it is pending.",
            "status": "open",
            "updated_at": "2026-08-14T09:05:00Z",
        },
        "fixtures/sources/expired-case.yml": {
            "case_id": "expired-case",
            "ticket_id": "T-300",
            "customer_message": "This escalation depends on an expired entitlement.",
            "status": "open",
            "updated_at": "2026-06-01T10:00:00Z",
        },
        "fixtures/sources/missing-evidence-case.yml": {
            "case_id": "missing-evidence-case",
            "ticket_id": "T-400",
            "customer_message": "The message references an attachment that is not present.",
            "status": "open",
            "updated_at": "2026-08-14T09:10:00Z",
        },
        "fixtures/sources/contract-denied-case.yml": {
            "case_id": "contract-denied-case",
            "ticket_id": "T-500",
            "customer_message": "The request asks for a use that is outside the permitted purpose.",
            "status": "open",
            "updated_at": "2026-08-14T09:15:00Z",
        },
        "fixtures/sources/mutation-and-reuse-case.yml": {
            "case_id": "mutation-and-reuse-case",
            "ticket_id": "T-600",
            "customer_message": "Only the display text changed; the meaning stayed the same.",
            "status": "open",
            "updated_at": "2026-08-14T09:20:00Z",
        },
    }
    for relative, payload in fixture_sources.items():
        files[Path(relative)] = _yaml_text(payload)
    expected = {
        "fixtures/expected/valid-case.yml": {"case_id": "valid-case", "selected": True, "intent": "billing", "urgency": "medium"},
        "fixtures/expected/conflicting-case.yml": {"case_id": "conflicting-case", "selected": False, "reason": "conflicting_evidence"},
        "fixtures/expected/expired-case.yml": {"case_id": "expired-case", "selected": False, "reason": "expired"},
        "fixtures/expected/missing-evidence-case.yml": {"case_id": "missing-evidence-case", "selected": False, "reason": "missing_evidence"},
        "fixtures/expected/contract-denied-case.yml": {"case_id": "contract-denied-case", "package_outcome": "deny"},
        "fixtures/expected/mutation-and-reuse-case.yml": {"case_id": "mutation-and-reuse-case", "expected_reuse": True},
    }
    for relative, payload in expected.items():
        files[Path(relative)] = _yaml_text(payload)
    files[Path("fixtures/mutations/mutation-and-reuse-case.yml")] = _yaml_text(
        {"case_id": "mutation-and-reuse-case", "mutation": "presentation_only", "field": "display_name"}
    )
    return files


def _render_diagnostics(items: Sequence[ProjectDiagnostic]) -> str:
    return render_diagnostics(items)
