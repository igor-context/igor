"""Tool-neutral Context IR and semantic-layer contracts.

The models in this module describe immutable records and references. They do not
perform derivation, model inference, retrieval, storage, or context compilation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from igor_core.canonical import stable_identity

SemanticVersion = Literal["0.1"]
NonEmpty = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Identity = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class SchemaField(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: NonEmpty
    name: NonEmpty
    logical_type: NonEmpty
    required: bool = False


class SchemaDescriptor(BaseModel):
    """Versioned meaning of a payload; shape alone is not its identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SemanticVersion
    schema_id: NonEmpty
    revision: NonEmpty
    fields: tuple[SchemaField, ...] = ()
    domain: NonEmpty | None = None
    semantic_annotations: dict[str, str] = Field(default_factory=dict)
    json_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def json_schema_is_valid(self) -> "SchemaDescriptor":
        if self.json_schema is not None:
            try:
                Draft202012Validator.check_schema(self.json_schema)
            except SchemaError as error:
                raise ValueError(f"invalid JSON Schema: {error.message}") from error
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class SourceFieldRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: NonEmpty
    logical_type: NonEmpty
    required: bool = False


class SourceResourceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: NonEmpty
    resource_kind: NonEmpty
    identity_concepts: tuple[NonEmpty, ...] = Field(min_length=1)
    fields: tuple[SourceFieldRequirement, ...] = Field(min_length=1)
    accepted_media_types: tuple[NonEmpty, ...] = Field(min_length=1)
    selection: dict[str, str] = Field(default_factory=dict)
    change_mode: Literal["snapshot", "incremental", "cdc"]
    cursor_concept: NonEmpty | None = None
    deletion_semantics: Literal["ignore", "tombstone", "hard-delete-event"]

    @model_validator(mode="after")
    def concepts_are_closed(self) -> "SourceResourceContract":
        concepts = [field.concept_id for field in self.fields]
        if len(concepts) != len(set(concepts)):
            raise ValueError("source field concepts must be unique")
        missing = set(self.identity_concepts) - set(concepts)
        if missing:
            raise ValueError(f"identity concepts missing from fields: {sorted(missing)}")
        if self.change_mode in {"incremental", "cdc"} and self.cursor_concept is None:
            raise ValueError("incremental or cdc resources require cursor_concept")
        if self.cursor_concept is not None and self.cursor_concept not in concepts:
            raise ValueError("cursor_concept must reference a declared field concept")
        return self


class ContextSourceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: NonEmpty
    revision: NonEmpty
    domain: NonEmpty
    resources: tuple[SourceResourceContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def resource_ids_are_unique(self) -> "ContextSourceContract":
        values = [resource.resource_id for resource in self.resources]
        if len(values) != len(set(values)):
            raise ValueError("source resource IDs must be unique")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class ConnectorFieldBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: NonEmpty
    source_field: NonEmpty


class ConnectorResourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_id: NonEmpty
    source_resource: NonEmpty
    fields: tuple[ConnectorFieldBinding, ...] = Field(min_length=1)
    filters: dict[str, str] = Field(default_factory=dict)
    cursor_field: NonEmpty | None = None


class ConnectorBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_id: NonEmpty
    revision: NonEmpty
    source_contract_identity: Identity
    connector: NonEmpty
    deployment_ref: NonEmpty
    credential_ref: NonEmpty | None = None
    resources: tuple[ConnectorResourceBinding, ...] = Field(min_length=1)

    @property
    def identity(self) -> str:
        return stable_identity(self)

    def validate_against(self, contract: ContextSourceContract) -> None:
        if self.source_contract_identity != contract.identity:
            raise ValueError("connector binding source-contract identity mismatch")
        contracts = {resource.resource_id: resource for resource in contract.resources}
        bindings = {resource.resource_id: resource for resource in self.resources}
        if len(bindings) != len(self.resources):
            raise ValueError("connector resource bindings must be unique")
        if set(bindings) != set(contracts):
            raise ValueError("connector binding must cover every source-contract resource")
        for resource_id, requirement in contracts.items():
            binding = bindings[resource_id]
            mapped = [field.concept_id for field in binding.fields]
            if len(mapped) != len(set(mapped)):
                raise ValueError(f"duplicate concept mapping for resource {resource_id}")
            required = {field.concept_id for field in requirement.fields if field.required}
            required.update(requirement.identity_concepts)
            missing = required - set(mapped)
            if missing:
                raise ValueError(f"required concepts not bound for {resource_id}: {sorted(missing)}")
            if requirement.cursor_concept is not None:
                cursor_sources = {field.source_field for field in binding.fields if field.concept_id == requirement.cursor_concept}
                if binding.cursor_field not in cursor_sources:
                    raise ValueError(f"cursor binding mismatch for resource {resource_id}")


class ContentPart(BaseModel):
    """Identity-bearing runtime content; non-text bytes remain behind a reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text", "image", "document", "audio", "video"]
    media_type: NonEmpty
    content_sha256: Identity
    text: NonEmpty | None = None
    content_ref: NonEmpty | None = None
    locator: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def content_matches_kind(self) -> "ContentPart":
        if self.kind == "text":
            if self.text is None or self.content_ref is not None:
                raise ValueError("text content requires text and forbids content_ref")
            expected = stable_identity(self.text)
            if self.content_sha256 != expected:
                raise ValueError("text content hash mismatch")
        elif self.content_ref is None or self.text is not None:
            raise ValueError("non-text content requires content_ref and forbids text")
        if self.kind == "image" and not self.media_type.startswith("image/"):
            raise ValueError("image content requires an image media type")
        if self.kind == "audio" and not self.media_type.startswith("audio/"):
            raise ValueError("audio content requires an audio media type")
        if self.kind == "video" and not self.media_type.startswith("video/"):
            raise ValueError("video content requires a video media type")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class EnrichmentRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe_id: NonEmpty
    revision: NonEmpty
    accepted_representation_types: tuple[NonEmpty, ...] = Field(min_length=1)
    accepted_media_types: tuple[NonEmpty, ...] = Field(min_length=1)
    output_schema_identity: Identity
    prompt_version: NonEmpty
    taxonomy_version: NonEmpty
    evidence_required: bool = True
    abstention_allowed: bool = True

    @property
    def identity(self) -> str:
        return stable_identity(self)


def validate_schema_payload(schema: SchemaDescriptor, payload: Any) -> None:
    """Validate untrusted output through the registered Draft 2020-12 schema."""
    if schema.json_schema is None:
        raise ValueError("schema descriptor has no registered JSON Schema")
    try:
        Draft202012Validator(schema.json_schema).validate(payload)
    except ValidationError as error:
        location = ".".join(str(item) for item in error.absolute_path) or "$"
        raise ValueError(f"payload violates schema at {location}: {error.message}") from error


class SourceSnapshot(BaseModel):
    """An immutable observation of source content; bytes live behind a reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    source_system: NonEmpty
    source_key: NonEmpty
    observed_at: datetime
    content_ref: NonEmpty
    content_sha256: NonEmpty
    media_type: NonEmpty
    schema_ref: SchemaDescriptor | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def identity(self) -> str:
        return stable_identity(self)


class Representation(BaseModel):
    """A versioned interpretation of one or more source snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    representation_type: Literal["text", "clause", "entity", "event", "table", "image", "video", "code", "custom"]
    schema_ref: SchemaDescriptor
    source_snapshot_ids: tuple[Identity, ...] = Field(min_length=1)
    payload: str | int | float | bool | list | dict | None
    derivation_id: Identity | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def identity(self) -> str:
        return stable_identity(self)


class Evidence(BaseModel):
    """A bounded, addressable explanation for a representation or enrichment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    subject_identity: Identity
    source_snapshot_id: Identity
    locator: NonEmpty
    excerpt: NonEmpty | None = None
    observed_at: datetime | None = None

    @property
    def identity(self) -> str:
        return stable_identity(self)


class Derivation(BaseModel):
    """A reproducible dependency declaration, not an execution request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    operation: NonEmpty
    input_identities: tuple[Identity, ...] = Field(min_length=1)
    output_identity: Identity
    code_revision: NonEmpty
    configuration_identity: NonEmpty
    model_revision: NonEmpty | None = None

    @property
    def identity(self) -> str:
        return stable_identity(self)


class EmbeddingSpace(BaseModel):
    """Immutable vector-space meaning; vectors are never identified by values alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    provider: NonEmpty
    model: NonEmpty
    model_revision: NonEmpty
    dimension: int = Field(gt=0)
    dtype: Literal["float16", "float32", "float64"]
    metric: Literal["cosine", "dot", "l2"]
    normalized: bool
    input_schema_identity: NonEmpty

    @property
    def identity(self) -> str:
        return stable_identity(self)


class Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    representation_identity: Identity
    space_identity: NonEmpty
    vector: list[float]

    @field_validator("vector")
    @classmethod
    def vector_values_are_finite(cls, value: list[float]) -> list[float]:
        if any(value != value or value in (float("inf"), float("-inf")) for value in value):
            raise ValueError("embedding vectors must contain finite values")
        return value

    @property
    def identity(self) -> str:
        return stable_identity(self)


class EnrichmentOutput(BaseModel):
    """Structured interpretation with explicit abstention and evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    representation_identity: Identity
    schema_ref: SchemaDescriptor
    payload: dict
    evidence_identities: tuple[Identity, ...] = ()
    derivation_identity: Identity
    status: Literal["accepted", "abstained", "failed"]
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def confidence_matches_status(self) -> "EnrichmentOutput":
        if self.status == "accepted" and self.confidence is None:
            raise ValueError("accepted enrichment must declare confidence")
        if self.status != "accepted" and self.confidence is not None:
            raise ValueError("abstained or failed enrichment cannot declare confidence")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


class ContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    representation_identity: Identity
    evidence_identities: tuple[Identity, ...] = ()
    role: NonEmpty
    rank: int = Field(ge=0)
    token_estimate: int = Field(ge=0)


class ContextPackage(BaseModel):
    """Task-scoped, evidence-linked delivery payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ir_version: SemanticVersion
    task_id: NonEmpty
    schema_revision: NonEmpty
    items: tuple[ContextItem, ...]
    budget_tokens: int = Field(gt=0)
    created_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_budget_and_order(self) -> "ContextPackage":
        if sum(item.token_estimate for item in self.items) > self.budget_tokens:
            raise ValueError("context package exceeds token budget")
        ranks = [item.rank for item in self.items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("context package ranks must be unique")
        return self

    @property
    def identity(self) -> str:
        return stable_identity(self)


def affected_outputs(changed_identities: set[str], derivations: list[Derivation]) -> set[str]:
    """Return the transitive output closure for selective invalidation."""

    affected = set(changed_identities)
    changed = True
    while changed:
        changed = False
        for derivation in derivations:
            if derivation.output_identity not in affected and any(
                identity in affected for identity in derivation.input_identities
            ):
                affected.add(derivation.output_identity)
                changed = True
    return affected


def validate_context_artifact(path: str) -> ContextPackage:
    """Validate one canonical ContextPackage JSON artifact."""

    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ContextPackage.model_validate(payload)
