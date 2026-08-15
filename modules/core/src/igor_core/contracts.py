"""Tool-neutral run-artifact contracts and deterministic validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from igor_core.canonical import stable_identity

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from igor_core.lineage import LineageGraph

ContractVersion = Literal["0.1"]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class ProducerIdentity(BaseModel):
    """Stable identity of the implementation that produced an artifact or stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyString
    revision: NonEmptyString


class RunIdentityLinks(BaseModel):
    """The minimum version links required to attribute one benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_contract: NonEmptyString
    scenario_pack: NonEmptyString
    scorecard: NonEmptyString
    implementation: NonEmptyString
    source_fixture: NonEmptyString
    transformation_config: NonEmptyString
    evaluator: NonEmptyString


class ArtifactReference(BaseModel):
    """A produced artifact described without exposing its storage implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    path: NonEmptyString
    producer: ProducerIdentity
    media_type: NonEmptyString
    sha256: Sha256
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or normalized == ".." or "/../" in f"/{normalized}/":
            raise ValueError("artifact paths must stay within the run directory")
        return normalized

    @field_validator("media_type")
    @classmethod
    def media_type_must_be_valid(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*(?:;\s*.+)?", value.lower()):
            raise ValueError("media_type must be a valid MIME-like value")
        return value

    @property
    def identity(self) -> str:
        return stable_identity(self)


class StageResult(BaseModel):
    """Observable result of one benchmark stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    stage_id: NonEmptyString
    status: Literal["succeeded", "failed"]
    producer: ProducerIdentity
    input_artifact_identities: list[Sha256] = Field(default_factory=list)
    output_artifact_identities: list[Sha256] = Field(default_factory=list)
    error: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "StageResult":
        if self.status == "failed" and self.error is None:
            raise ValueError("failed stages must declare an error")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("succeeded stages cannot declare an error")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class RunManifest(BaseModel):
    """Stable run-level declarations and references to its produced records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    identity_links: RunIdentityLinks
    artifact_index_path: NonEmptyString = "artifact-index.json"
    stage_result_paths: list[NonEmptyString] = Field(default_factory=list)
    lineage_path: NonEmptyString | None = None

    @field_validator("artifact_index_path", "stage_result_paths", "lineage_path")
    @classmethod
    def paths_must_be_relative(cls, value):
        values = value if isinstance(value, list) else [value]
        for path in values:
            if path is None:
                continue
            normalized = path.replace("\\", "/")
            if normalized.startswith("/") or normalized == ".." or "/../" in f"/{normalized}/":
                raise ValueError("manifest paths must stay within the run directory")
        return value

    @property
    def identity(self) -> str:
        return stable_identity(self.identity_links)


class ArtifactIndex(BaseModel):
    """Index of every artifact required to interpret a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: ContractVersion
    run_identity: Sha256
    artifacts: list[ArtifactReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_artifacts(self) -> "ArtifactIndex":
        identities = [artifact.identity for artifact in self.artifacts]
        if len(identities) != len(set(identities)):
            raise ValueError("artifact index contains duplicate artifact identities")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact index contains duplicate artifact paths")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class RunPackage(BaseModel):
    """Validated in-memory view of a complete run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: RunManifest
    artifact_index: ArtifactIndex
    stages: list[StageResult]
    lineage: "LineageGraph | None" = None

    @model_validator(mode="after")
    def validate_references(self) -> "RunPackage":
        if self.artifact_index.run_identity != self.manifest.identity:
            raise ValueError("artifact index run_identity does not match manifest identity")

        artifact_identities = {artifact.identity for artifact in self.artifact_index.artifacts}
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("run package contains duplicate stage IDs")
        for stage in self.stages:
            missing = (
                set(stage.input_artifact_identities) | set(stage.output_artifact_identities)
            ) - artifact_identities
            if missing:
                raise ValueError(f"stage {stage.stage_id} references undeclared artifacts: {sorted(missing)}")
        return self


from igor_core.lineage import LineageGraph

RunPackage.model_rebuild(_types_namespace={"LineageGraph": LineageGraph})


def validate_run_directory(path: str | Path) -> RunPackage:
    """Load and validate the manifest, artifact index, and stage results in a run directory."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")

    def load_json(relative_path: str) -> dict:
        candidate = root / relative_path
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"missing or unsafe run file: {relative_path}")
        return json.loads(candidate.read_text(encoding="utf-8"))

    manifest = RunManifest.model_validate(load_json("manifest.json"))
    artifact_index = ArtifactIndex.model_validate(load_json(manifest.artifact_index_path))
    stages = [StageResult.model_validate(load_json(stage_path)) for stage_path in manifest.stage_result_paths]
    root_resolved = root.resolve()
    for artifact in artifact_index.artifacts:
        candidate = root / artifact.path
        if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(root_resolved):
            raise ValueError(f"missing or unsafe artifact file: {artifact.path}")
        content = candidate.read_bytes()
        actual_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if actual_hash != artifact.sha256:
            raise ValueError(f"artifact integrity mismatch for {artifact.path}")
        if len(content) != artifact.size_bytes:
            raise ValueError(f"artifact size mismatch for {artifact.path}")
    lineage = None
    if manifest.lineage_path is not None:
        lineage_file = root / manifest.lineage_path
        if not lineage_file.is_file():
            raise ValueError(f"missing lineage artifact: {manifest.lineage_path}")
        lineage = LineageGraph.model_validate(json.loads(lineage_file.read_text(encoding="utf-8")))
        if lineage.run_identity != manifest.identity:
            raise ValueError("lineage run_identity does not match manifest identity")
    return RunPackage(manifest=manifest, artifact_index=artifact_index, stages=stages, lineage=lineage)
