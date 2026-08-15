"""Runtime Context Contract binding and package publication for IGOR projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from igor_lancedb import LanceStore
from igor_runner.context_model_lifecycle import materialize_context_model_lifecycle


class PublicationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_role: str
    consumer: str
    task: str
    purpose: str
    as_of: str
    evaluated_at: str | None = None
    authority_level: str | None = None
    budget_tokens: int | None = Field(default=None, ge=0)
    budget_bytes: int | None = Field(default=None, ge=0)


class PublicationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: str
    authority_level: str | None = None
    status: str | None = None
    byte_length: int | None = Field(default=None, ge=0)
    valid_until: str | None = None
    output_identity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PublicationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_identity: str
    evidence_identities: tuple[str, ...] = ()
    role: str
    rank: int = Field(ge=0)
    token_estimate: int = Field(ge=0)
    byte_estimate: int | None = Field(default=None, ge=0)
    authority_level: str | None = None
    valid_until: str | None = None
    status: str | None = None


class PublicationPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str | None = None
    budget_tokens: int | None = Field(default=None, ge=0)
    authority_level: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    items: tuple[PublicationItem, ...]


class PackagePublicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1"
    publication: PublicationSelection
    runtime_outputs: tuple[PublicationOutput, ...]
    packages: tuple[PublicationPackage, ...]


@dataclass(frozen=True)
class PackagePublicationResult:
    request: PackagePublicationRequest
    summary: dict[str, Any]
    published: bool


def publish_project_package(
    *,
    project_root: Path,
    target_root: Path,
    request_path: Path,
    manifest: Any,
    contract: Mapping[str, Any],
    request: PackagePublicationRequest,
) -> PackagePublicationResult:
    request_stem = request_path.stem.replace(" ", "-")
    store = LanceStore(target_root / f"{request_stem}-package-relations")

    outputs = tuple(
        {
            "identity": item.identity,
            "output_identity": item.output_identity or item.identity,
            "authority_level": item.authority_level or request.publication.authority_level or "",
            "status": item.status or "active",
            "byte_length": item.byte_length or 0,
            "valid_until": item.valid_until,
            "metadata": dict(item.metadata),
        }
        for item in request.runtime_outputs
    )
    packages = []
    for package in request.packages:
        packages.append(
            {
                "task_id": package.task_id or request.publication.task,
                "package_kind": "task_context",
                "budget_tokens": (
                    package.budget_tokens
                    or request.publication.budget_tokens
                    or sum(item.token_estimate for item in package.items)
                ),
                "authority_level": package.authority_level or request.publication.authority_level or "",
                "metadata": {
                    "output_role": request.publication.output_role,
                    "consumer": request.publication.consumer,
                    "purpose": request.publication.purpose,
                    **dict(package.metadata),
                },
                "items": [item.model_dump(mode="json") for item in package.items],
            }
        )

    lifecycle = materialize_context_model_lifecycle(
        store=store,
        manifest=manifest,
        outputs=outputs,
        packages=packages,
        contract_context={
            "consumer": request.publication.consumer,
            "task_id": request.publication.task,
            "purpose": request.publication.purpose,
            "as_of": request.publication.as_of,
            "evaluated_at": request.publication.evaluated_at or request.publication.as_of,
            "authority_level": request.publication.authority_level or "",
            "budget_tokens": request.publication.budget_tokens,
            "budget_bytes": request.publication.budget_bytes,
            "evaluator_identity": "igor.project-cli.package.v0.1",
        },
        context_contracts=(dict(contract),),
    )

    evaluations = tuple(dict(row) for row in lifecycle.contract_evaluations)
    published_packages = list(lifecycle.materialization.relations.get("context_packages", ()))
    published_items = list(lifecycle.materialization.relations.get("package_items", ()))
    bindings = list(lifecycle.materialization.relations.get("context_package_contracts", ()))
    decisions = sorted({row["decision"] for row in evaluations})
    summary = {
        "artifact_type": "igor-context-package-publication",
        "schema_version": "0.1",
        "project_root": str(project_root),
        "request_file": str(request_path),
        "output_role": request.publication.output_role,
        "manifest_identity": getattr(manifest, "identity", ""),
        "contract_identity": contract["identity"],
        "contract_ref": contract["ref"],
        "contract_id": contract["contract_id"],
        "contract_version": contract["version"],
        "published": bool(published_packages) and decisions == ["allow"],
        "decisions": decisions,
        "contract_evaluations": evaluations,
        "published_packages": published_packages,
        "published_package_items": published_items,
        "context_package_contracts": bindings,
        "relation_counts": dict(lifecycle.relation_counts),
        "test_execution": None
        if lifecycle.test_execution is None
        else {
            "passed": lifecycle.test_execution.passed,
            "failures": list(lifecycle.test_execution.failures),
        },
        "store_tables": store.names(),
        "request": request.model_dump(mode="json"),
    }
    return PackagePublicationResult(
        request=request,
        summary=summary,
        published=bool(summary["published"]),
    )
