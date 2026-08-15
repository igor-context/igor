"""Validated, non-secret model profiles and compiler composition contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator

from igor_core.canonical import stable_identity

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        strip_whitespace=True,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
NonEmptyString = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]

_SECRET_KEY_PARTS = ("api_key", "credential", "password", "secret", "token")


def _reject_secret_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SECRET_KEY_PARTS):
                location = ".".join((*path, str(key)))
                raise ValueError(f"reference profiles cannot contain secret field {location}")
            _reject_secret_keys(nested, (*path, str(key)))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_keys(nested, (*path, str(index)))


class ModelProfile(BaseModel):
    """One independently reusable, reproducibility-relevant model profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"]
    profile_id: Identifier
    capability: Literal["embedding", "completion"]
    provider: Identifier
    model: NonEmptyString
    revision: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def parameters_are_non_secret(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _reject_secret_keys(value)
        return value

    @property
    def identity(self) -> str:
        return stable_identity(self)


class CompilerProfile(BaseModel):
    """Authoring contract that composes independent capability profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"]
    profile_id: Identifier
    prompt_version: NonEmptyString
    taxonomy_version: NonEmptyString
    embedding_profile: NonEmptyString
    completion_profile: NonEmptyString


class ReferenceProfile(BaseModel):
    """Fully resolved non-secret compiler configuration recorded in run identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"]
    profile_id: Identifier
    prompt_version: NonEmptyString
    taxonomy_version: NonEmptyString
    embedding: ModelProfile
    completion: ModelProfile

    @property
    def identity(self) -> str:
        return stable_identity(self)


def _load_yaml_mapping(path: Path, kind: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{kind} must contain a YAML mapping: {path}")
    return data


def load_model_profile(
    path: str | Path,
    *,
    expected_capability: Literal["embedding", "completion"] | None = None,
) -> ModelProfile:
    """Load one independent model profile and optionally enforce its capability."""
    profile_path = Path(path)
    profile = ModelProfile.model_validate(_load_yaml_mapping(profile_path, "model profile"))
    if expected_capability is not None and profile.capability != expected_capability:
        raise ValueError(
            f"model profile {profile.profile_id} has capability {profile.capability}; "
            f"expected {expected_capability}"
        )
    return profile


def load_reference_profile(path: str | Path) -> ReferenceProfile:
    """Resolve one compiler composition into a validated run-identity profile."""
    profile_path = Path(path)
    composition = CompilerProfile.model_validate(
        _load_yaml_mapping(profile_path, "compiler profile")
    )
    embedding = load_model_profile(
        profile_path.parent / composition.embedding_profile,
        expected_capability="embedding",
    )
    completion = load_model_profile(
        profile_path.parent / composition.completion_profile,
        expected_capability="completion",
    )
    return ReferenceProfile(
        schema_version=composition.schema_version,
        profile_id=composition.profile_id,
        prompt_version=composition.prompt_version,
        taxonomy_version=composition.taxonomy_version,
        embedding=embedding,
        completion=completion,
    )
