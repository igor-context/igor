"""Declarative Context Model manifest compilation.

This module implements executable slices of the Context Model contract: a human-authored
declaration is validated and resolved into an identity-bearing manifest, and declared
runtime derivations can be compiled into typed compiler work. It performs no source
acquisition, storage, retrieval, provider execution, or package compilation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from igor_core import Evidence, stable_identity

from .compiler import (
    AuthorityAssertion,
    CompilerFailure,
    DerivationSpec,
    ResolutionCandidate,
    ResolutionRequest,
    RetrievalQuery,
    TemporalAssertion,
)

Identity = str

CONTEXT_OBJECT_KINDS = {
    "business_object",
    "source_snapshot",
    "semantic_derivation",
    "retrieval_projection",
    "authority_assertion",
}
ContextObjectKind = Literal[
    "business_object",
    "source_snapshot",
    "semantic_derivation",
    "retrieval_projection",
    "authority_assertion",
]


class ContextModelReference(BaseModel):
    """Identity-bearing artifact available to the manifest compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    ref: str
    identity: Identity
    revision: str | None = None
    material: dict[str, Any] = Field(default_factory=dict)


class ContextModelCompileRequest(BaseModel):
    """Authoring declaration plus the registry used to resolve references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    declaration: dict[str, Any]
    references: tuple[ContextModelReference, ...] = ()
    compiler_version: str = "context-model-manifest.v0.1"

    @model_validator(mode="after")
    def registry_keys_are_unique(self) -> "ContextModelCompileRequest":
        keys = [(item.kind, item.ref) for item in self.references]
        if len(keys) != len(set(keys)):
            raise CompilerFailure("contract_invalid", "duplicate context model registry reference")
        return self


class ResolvedSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    source_contract_ref: str
    source_contract_identity: Identity
    connector_binding_ref: str
    connector_binding_identity: Identity
    identity_fields: tuple[str, ...] = ()

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "role": self.role,
            "source_contract_identity": self.source_contract_identity,
            "connector_binding_identity": self.connector_binding_identity,
            "identity_fields": self.identity_fields,
        })


class ResolvedContextObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    kind: ContextObjectKind
    scope: Literal["instance", "context_model"] = "instance"
    source_role: str | None = None
    derived_from: tuple[str, ...] = ()
    operation_ref: str | None = None
    operation_identity: Identity | None = None
    schema_ref: str | None = None
    schema_identity: Identity | None = None
    semantic_definition_ref: str | None = None
    semantic_definition_identity: Identity | None = None
    recipe_ref: str | None = None
    recipe_identity: Identity | None = None
    model_profile_ref: str | None = None
    model_profile_identity: Identity | None = None
    display_name: str | None = None

    def semantic_material(self) -> dict[str, object]:
        return {
            "role": self.role,
            "kind": self.kind,
            "scope": self.scope,
            "source_role": self.source_role,
            "derived_from": self.derived_from,
            "operation_identity": self.operation_identity,
            "schema_identity": self.schema_identity,
            "semantic_definition_identity": self.semantic_definition_identity,
            "recipe_identity": self.recipe_identity,
            "model_profile_identity": self.model_profile_identity,
        }

    @property
    def identity(self) -> Identity:
        return stable_identity(self.semantic_material())


class ResolvedAuthorityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    target_role: str
    policy_ref: str
    policy_identity: Identity
    settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "role": self.role,
            "target_role": self.target_role,
            "policy_identity": self.policy_identity,
            "settings": _semantic_settings(self.settings),
        })


class ResolvedRetrievalDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    search: Literal["vector", "full_text", "hybrid"]
    candidate_limit: int = Field(default=10, gt=0)
    target_roles: tuple[str, ...] = ()
    eligibility: dict[str, Any] = Field(default_factory=dict)
    resolution_policy_role: str | None = None
    accepted_outcomes: tuple[str, ...] = ()
    display_name: str | None = None

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "role": self.role,
            "search": self.search,
            "candidate_limit": self.candidate_limit,
            "target_roles": self.target_roles,
            "eligibility": _semantic_settings(self.eligibility),
            "resolution_policy_role": self.resolution_policy_role,
            "accepted_outcomes": self.accepted_outcomes,
        })


class ResolvedPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    target_role: str
    preview_fields: tuple[str, ...] = ()
    display_name: str | None = None

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "role": self.role,
            "target_role": self.target_role,
            "preview_fields": self.preview_fields,
        })


class ResolvedContextModelTest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test_type: str
    target_role: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "test_type": self.test_type,
            "target_role": self.target_role,
            "parameters": _semantic_settings(self.parameters),
        })


class ResolvedContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    contract_id: str
    version: str
    title: str | None = None
    rules: dict[str, Any] = Field(default_factory=dict)

    @property
    def semantic_identity(self) -> Identity:
        return stable_identity({
            "contract_id": self.contract_id,
            "version": self.version,
            "rules": _semantic_settings(self.rules),
        })

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "ref": self.ref,
            "semantic_identity": self.semantic_identity,
        })

    @property
    def budgets(self) -> dict[str, Any]:
        value = self.rules.get("budgets", {})
        return dict(value) if isinstance(value, Mapping) else {}


class ResolvedContextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    output_type: str
    retrieval_role: str | None = None
    retrieval_identity: Identity | None = None
    contract_ref: str | None = None
    contract_identity: Identity | None = None
    target_roles: tuple[str, ...] = ()
    format: str | None = None
    destination: str | None = None
    display_name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "role": self.role,
            "output_type": self.output_type,
            "retrieval_identity": self.retrieval_identity,
            "contract_identity": self.contract_identity,
            "target_roles": self.target_roles,
            "format": self.format,
            "destination": self.destination,
            "settings": _semantic_settings(self.settings),
        })


class ResolvedContextEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str
    category: str
    description: str
    target_roles: tuple[str, ...] = ()
    target_outputs: tuple[str, ...] = ()
    target_contracts: tuple[str, ...] = ()
    target_retrievals: tuple[str, ...] = ()
    fixtures: dict[str, Any] = Field(default_factory=dict)
    checks: tuple[str, ...] = ()
    required: bool = True
    hidden_expectations: bool = True

    @property
    def identity(self) -> Identity:
        return stable_identity({
            "evaluation_id": self.evaluation_id,
            "category": self.category,
            "description": self.description,
            "target_roles": self.target_roles,
            "target_outputs": self.target_outputs,
            "target_contracts": self.target_contracts,
            "target_retrievals": self.target_retrievals,
            "fixtures": _semantic_settings(self.fixtures),
            "checks": self.checks,
            "required": self.required,
            "hidden_expectations": self.hidden_expectations,
        })


class ResolvedContextModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_model_id: str
    revision: str
    title: str | None = None
    compiler_version: str
    sources: tuple[ResolvedSourceBinding, ...] = ()
    objects: tuple[ResolvedContextObject, ...] = ()
    authority: tuple[ResolvedAuthorityPolicy, ...] = ()
    retrievals: tuple[ResolvedRetrievalDefinition, ...] = ()
    presentation: tuple[ResolvedPresentation, ...] = ()
    tests: tuple[ResolvedContextModelTest, ...] = ()
    context_contracts: tuple[ResolvedContextContract, ...] = ()
    outputs: tuple[ResolvedContextOutput, ...] = ()
    evaluations: tuple[ResolvedContextEvaluation, ...] = ()

    def semantic_material(self) -> dict[str, object]:
        return {
            "context_model_id": self.context_model_id,
            "revision": self.revision,
            "compiler_version": self.compiler_version,
            "sources": [
                {
                    "role": item.role,
                    "identity": item.identity,
                    "source_contract_identity": item.source_contract_identity,
                    "connector_binding_identity": item.connector_binding_identity,
                    "identity_fields": item.identity_fields,
                }
                for item in self.sources
            ],
            "objects": [{"identity": item.identity, **item.semantic_material()} for item in self.objects],
            "authority": [
                {
                    "role": item.role,
                    "identity": item.identity,
                    "target_role": item.target_role,
                    "policy_identity": item.policy_identity,
                    "settings": _semantic_settings(item.settings),
                }
                for item in self.authority
            ],
            "retrievals": [item.model_dump(mode="json") | {"identity": item.identity} for item in self.retrievals],
            "presentation": [
                item.model_dump(mode="json", exclude={"display_name"}) | {"identity": item.identity}
                for item in self.presentation
            ],
            "tests": [item.model_dump(mode="json") | {"identity": item.identity} for item in self.tests],
            "context_contracts": [
                item.model_dump(mode="json", exclude={"title"}) | {"identity": item.identity, "semantic_identity": item.semantic_identity}
                for item in self.context_contracts
            ],
            "outputs": [
                item.model_dump(mode="json", exclude={"display_name"}) | {"identity": item.identity}
                for item in self.outputs
            ],
            "evaluations": [item.model_dump(mode="json") | {"identity": item.identity} for item in self.evaluations],
        }

    @property
    def semantic_identity(self) -> Identity:
        return stable_identity(self.semantic_material())

    @property
    def identity(self) -> Identity:
        return self.semantic_identity

    def artifact(self) -> dict[str, object]:
        return {
            "artifact_type": "resolved-context-model-manifest",
            "schema_version": "0.1",
            "context_model_id": self.context_model_id,
            "context_model_revision": self.revision,
            "manifest_identity": self.identity,
            "semantic_identity": self.semantic_identity,
            "manifest": self.model_dump(mode="json"),
        }


def compile_context_model_derivations(
    manifest: "ResolvedContextModelManifest | Mapping[str, Any]",
    role_identities: Mapping[str, str | Sequence[str]],
    *,
    code_revision: str,
    output_identities: Mapping[str, str] | None = None,
) -> tuple[DerivationSpec, ...]:
    """Compile declared Context Model derivations into typed compiler specs."""
    resolved = _manifest_model(manifest)
    bindings = {role: _identity_tuple(value, f"role_identities.{role}") for role, value in role_identities.items()}
    output_bindings = dict(output_identities or {})
    derivations: list[DerivationSpec] = []
    for item in sorted(resolved.objects, key=lambda value: value.role):
        if item.kind not in {"semantic_derivation", "retrieval_projection", "authority_assertion"}:
            continue
        if not item.derived_from:
            raise CompilerFailure("contract_invalid", f"declared derivation has no dependencies: {item.role}")
        input_identities: list[str] = []
        for dependency in item.derived_from:
            values = bindings.get(dependency)
            if not values:
                raise CompilerFailure("contract_invalid", f"missing runtime identity for Context Model role: {dependency}")
            input_identities.extend(values)
        output_identity = output_bindings.get(item.role) or stable_identity({
            "context_model_identity": resolved.identity,
            "role": item.role,
            "object_identity": item.identity,
            "operation_identity": item.operation_identity,
            "schema_identity": item.schema_identity,
            "semantic_definition_identity": item.semantic_definition_identity,
            "recipe_identity": item.recipe_identity,
            "model_profile_identity": item.model_profile_identity,
            "input_identities": tuple(sorted(input_identities)),
            "code_revision": code_revision,
        })
        derivations.append(DerivationSpec(
            operation=item.operation_ref or item.role,
            input_identities=tuple(sorted(input_identities)),
            output_identity=output_identity,
            code_revision=code_revision,
            configuration_identity=_derivation_configuration_identity(item),
            model_revision=item.model_profile_identity,
        ))
    return tuple(derivations)


def compile_context_model_derivation_instances(
    manifest: "ResolvedContextModelManifest | Mapping[str, Any]",
    instances: Sequence[Mapping[str, Any]],
    *,
    code_revision: str,
) -> tuple[DerivationSpec, ...]:
    """Compile per-instance Context Model derivation facts into typed compiler specs."""
    resolved = _manifest_model(manifest)
    objects = {item.role: item for item in resolved.objects}
    derivations: list[DerivationSpec] = []
    for index, raw in enumerate(instances):
        value = dict(raw)
        role = _required_str(value, "role", f"derivation_instances[{index}]")
        item = objects.get(role)
        if item is None:
            raise CompilerFailure("contract_invalid", f"unknown Context Model derivation role: {role}")
        if item.kind not in {"semantic_derivation", "retrieval_projection", "authority_assertion"}:
            raise CompilerFailure("contract_invalid", f"Context Model role is not derivable: {role}")
        if not item.derived_from:
            raise CompilerFailure("contract_invalid", f"declared derivation has no dependencies: {role}")
        input_identities = _identity_tuple(value.get("input_identities", ()), f"derivation_instances[{index}].input_identities")
        output_identity = str(value.get("output_identity") or stable_identity({
            "context_model_identity": resolved.identity,
            "role": item.role,
            "object_identity": item.identity,
            "operation_identity": item.operation_identity,
            "schema_identity": item.schema_identity,
            "semantic_definition_identity": item.semantic_definition_identity,
            "recipe_identity": item.recipe_identity,
            "model_profile_identity": item.model_profile_identity,
            "input_identities": input_identities,
            "code_revision": code_revision,
        }))
        model_revision = value.get("model_revision")
        derivations.append(DerivationSpec(
            operation=item.operation_ref or item.role,
            input_identities=input_identities,
            output_identity=output_identity,
            code_revision=code_revision,
            configuration_identity=str(value.get("configuration_identity") or _derivation_configuration_identity(item)),
            model_revision=str(model_revision) if model_revision is not None else item.model_profile_identity,
        ))
    if len({item.output_identity for item in derivations}) != len(derivations):
        raise CompilerFailure("contract_invalid", "duplicate Context Model derivation output identity")
    return tuple(derivations)


def compile_context_model_retrieval_queries(
    manifest: "ResolvedContextModelManifest | Mapping[str, Any]",
    retrieval_inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[RetrievalQuery, ...]:
    """Compile declared retrieval roles into typed runtime retrieval queries."""
    resolved = _manifest_model(manifest)
    declarations = {item.role: item for item in resolved.retrievals}
    unknown = sorted(set(retrieval_inputs) - set(declarations))
    if unknown:
        raise CompilerFailure("contract_invalid", "unknown Context Model retrieval role: " + ", ".join(unknown))
    queries: list[RetrievalQuery] = []
    for role in sorted(retrieval_inputs):
        declaration = declarations[role]
        value = dict(retrieval_inputs[role])
        query_id = str(value.get("query_id") or stable_identity({
            "context_model_identity": resolved.identity,
            "retrieval_identity": declaration.identity,
            "text": value.get("text", ""),
            "vector": tuple(value.get("vector", ())),
            "space_identity": value.get("space_identity"),
        }))
        queries.append(RetrievalQuery(
            query_id=query_id,
            query_type=declaration.search,
            text=str(value.get("text", "")),
            vector=tuple(float(item) for item in value.get("vector", ())),
            limit=int(value.get("limit", declaration.candidate_limit)),
            space_identity=None if value.get("space_identity") is None else str(value["space_identity"]),
        ))
    return tuple(queries)


def compile_context_model_resolution_request(
    manifest: "ResolvedContextModelManifest | Mapping[str, Any]",
    *,
    retrieval_role: str,
    task_id: str,
    task_schema_revision: str,
    as_of: Any,
    candidates: Sequence[ResolutionCandidate | Mapping[str, Any]] = (),
    evidence: Sequence[Evidence | Mapping[str, Any]] = (),
    authority_assertions: Sequence[AuthorityAssertion | Mapping[str, Any]] = (),
    temporal_assertions: Sequence[TemporalAssertion | Mapping[str, Any]] = (),
    request_identity: str | None = None,
    policy_role: str | None = None,
) -> ResolutionRequest:
    """Compile a manifest-declared retrieval policy into a typed resolution request."""
    resolved = _manifest_model(manifest)
    retrieval = _retrieval_definition(resolved, retrieval_role)
    resolved_policy_role = policy_role or retrieval.resolution_policy_role
    if resolved_policy_role is None:
        raise CompilerFailure("contract_invalid", f"retrieval has no resolution policy: {retrieval_role}")
    policy = _authority_policy(resolved, resolved_policy_role)
    candidate_values = tuple(
        item if isinstance(item, ResolutionCandidate) else ResolutionCandidate.model_validate(dict(item))
        for item in candidates
    )
    evidence_values = tuple(item if isinstance(item, Evidence) else Evidence.model_validate(dict(item)) for item in evidence)
    authority_values = tuple(
        item if isinstance(item, AuthorityAssertion) else AuthorityAssertion.model_validate(dict(item))
        for item in authority_assertions
    )
    temporal_values = tuple(
        item if isinstance(item, TemporalAssertion) else TemporalAssertion.model_validate(dict(item))
        for item in temporal_assertions
    )
    policy_revision = _policy_revision(policy)
    identity = request_identity or stable_identity({
        "context_model_identity": resolved.identity,
        "retrieval_role": retrieval.role,
        "retrieval_identity": retrieval.identity,
        "policy_role": policy.role,
        "policy_identity": policy.policy_identity,
        "policy_revision": policy_revision,
        "task_id": task_id,
        "task_schema_revision": task_schema_revision,
        "as_of": str(as_of),
        "candidate_identities": tuple(sorted(item.candidate_identity for item in candidate_values)),
        "evidence_identities": tuple(sorted(item.identity for item in evidence_values)),
        "authority_assertion_ids": tuple(sorted(item.assertion_identity for item in authority_values)),
        "temporal_assertion_ids": tuple(sorted(item.assertion_identity for item in temporal_values)),
    })
    return ResolutionRequest(
        request_identity=identity,
        task_id=task_id,
        task_schema_revision=task_schema_revision,
        as_of=as_of,
        candidates=tuple(sorted(candidate_values, key=lambda item: item.candidate_identity)),
        evidence=tuple(sorted(evidence_values, key=lambda item: item.identity)),
        authority_assertions=tuple(sorted(authority_values, key=lambda item: item.assertion_identity)),
        temporal_assertions=tuple(sorted(temporal_values, key=lambda item: item.assertion_identity)),
        policy_id=_policy_id(policy),
        policy_revision=policy_revision,
    )


def compile_context_model(request: ContextModelCompileRequest) -> ResolvedContextModelManifest:
    """Resolve a Context Model declaration into an execution manifest."""

    declaration = request.declaration
    _reject_unknown_sections(declaration)
    model = _mapping(declaration.get("context_model"), "context_model")
    context_model_id = _required_str(model, "id", "context_model")
    revision = _required_str(model, "revision", "context_model")
    registry = {(item.kind, item.ref): item for item in request.references}
    sources = _resolve_sources(declaration.get("sources", {}), registry)
    objects = _resolve_objects(declaration.get("objects", {}), sources, registry)
    authority = _resolve_authority(declaration.get("authority", {}), objects, registry)
    retrievals = _resolve_retrievals(declaration.get("retrievals", {}), objects, authority)
    presentation = _resolve_presentation(declaration.get("presentation", {}), objects)
    tests = _resolve_tests(declaration.get("tests", ()), objects)
    context_contracts = _resolve_context_contracts(
        declaration.get("context_contracts", declaration.get("contracts", {}))
    )
    outputs = _resolve_outputs(declaration.get("outputs", {}), retrievals, context_contracts, objects)
    evaluations = _resolve_evaluations(
        declaration.get("evaluations", {}),
        objects,
        outputs,
        context_contracts,
        retrievals,
    )
    return ResolvedContextModelManifest(
        context_model_id=context_model_id,
        revision=revision,
        title=model.get("title"),
        compiler_version=request.compiler_version,
        sources=sources,
        objects=objects,
        authority=authority,
        retrievals=retrievals,
        presentation=presentation,
        tests=tests,
        context_contracts=context_contracts,
        outputs=outputs,
        evaluations=evaluations,
    )


def _manifest_model(value: ResolvedContextModelManifest | Mapping[str, Any]) -> ResolvedContextModelManifest:
    if isinstance(value, ResolvedContextModelManifest):
        return value
    material = dict(value)
    if "manifest" in material:
        material = dict(_mapping(material["manifest"], "manifest"))
    if "context_model_revision" in material and "revision" not in material:
        material["revision"] = material["context_model_revision"]
    material.pop("context_model_revision", None)
    for runtime_key in ("context_model_identity", "manifest_identity", "semantic_identity"):
        material.pop(runtime_key, None)
    material.setdefault("compiler_version", "context-model-manifest.v0.1")
    return ResolvedContextModelManifest.model_validate(material)


def _identity_tuple(value: str | Sequence[str], path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_string(value, path),)
    if not isinstance(value, Sequence):
        raise CompilerFailure("contract_invalid", f"{path} must be an identity or sequence of identities")
    result = tuple(_string(item, f"{path}[]") for item in value)
    if not result:
        raise CompilerFailure("contract_invalid", f"{path} cannot be empty")
    return tuple(sorted(result))


def _derivation_configuration_identity(item: ResolvedContextObject) -> str:
    return (
        item.schema_identity
        or item.semantic_definition_identity
        or item.recipe_identity
        or item.operation_identity
        or item.identity
    )


def _retrieval_definition(manifest: ResolvedContextModelManifest, role: str) -> ResolvedRetrievalDefinition:
    for item in manifest.retrievals:
        if item.role == role:
            return item
    raise CompilerFailure("contract_invalid", f"unknown Context Model retrieval role: {role}")


def _authority_policy(manifest: ResolvedContextModelManifest, role: str) -> ResolvedAuthorityPolicy:
    for item in manifest.authority:
        if item.role == role:
            return item
    raise CompilerFailure("contract_invalid", f"unknown Context Model authority policy role: {role}")


def _policy_revision(policy: ResolvedAuthorityPolicy) -> str:
    raw = policy.settings.get("policy_revision", policy.settings.get("revision", "1"))
    return _string(str(raw), f"authority.{policy.role}.policy_revision")


def _policy_id(policy: ResolvedAuthorityPolicy) -> str:
    raw = policy.settings.get("policy_id", policy.policy_ref)
    return _string(str(raw), f"authority.{policy.role}.policy_id")


def _resolve_sources(section: object, registry: Mapping[tuple[str, str], ContextModelReference]) -> tuple[ResolvedSourceBinding, ...]:
    result: list[ResolvedSourceBinding] = []
    for role, raw in _entries(section, "sources"):
        value = _mapping(raw, f"sources.{role}")
        contract_ref = _required_str(value, "source_contract", f"sources.{role}")
        binding_ref = _required_str(value, "connector_binding", f"sources.{role}")
        contract = _reference(registry, "source_contract", contract_ref)
        binding = _reference(registry, "connector_binding", binding_ref)
        result.append(ResolvedSourceBinding(
            role=role,
            source_contract_ref=contract_ref,
            source_contract_identity=contract.identity,
            connector_binding_ref=binding_ref,
            connector_binding_identity=binding.identity,
            identity_fields=tuple(_strings(value.get("identity_fields", ()), f"sources.{role}.identity_fields")),
        ))
    return tuple(sorted(result, key=lambda item: item.role))


def _resolve_objects(section: object, sources: Sequence[ResolvedSourceBinding],
                     registry: Mapping[tuple[str, str], ContextModelReference]) -> tuple[ResolvedContextObject, ...]:
    source_roles = {item.role for item in sources}
    raw_entries = _entries(section, "objects")
    raw_roles = {role for role, _ in raw_entries}
    result: list[ResolvedContextObject] = []
    for role, raw in raw_entries:
        value = _mapping(raw, f"objects.{role}")
        kind = _required_str(value, "kind", f"objects.{role}")
        if kind not in CONTEXT_OBJECT_KINDS:
            raise CompilerFailure("contract_invalid", f"unknown Context Model object kind: {role}")
        source_role = value.get("source")
        if source_role is not None:
            source_role = _string(source_role, f"objects.{role}.source")
            if source_role not in source_roles:
                raise CompilerFailure("contract_invalid", f"object references missing source: {role}")
        derived_from = tuple(_strings(value.get("derived_from", ()), f"objects.{role}.derived_from"))
        missing_dependencies = set(derived_from) - raw_roles
        if missing_dependencies:
            raise CompilerFailure("contract_invalid", f"object references missing dependency: {role}")
        result.append(ResolvedContextObject(
            role=role,
            kind=kind,  # type: ignore[arg-type]
            scope=value.get("scope", "instance"),
            source_role=source_role,
            derived_from=tuple(sorted(derived_from)),
            operation_ref=value.get("operation"),
            operation_identity=_optional_reference_identity(registry, "operation", value.get("operation")),
            schema_ref=value.get("schema"),
            schema_identity=_optional_reference_identity(registry, "schema", value.get("schema")),
            semantic_definition_ref=value.get("semantic_definition"),
            semantic_definition_identity=_optional_reference_identity(registry, "semantic_definition", value.get("semantic_definition")),
            recipe_ref=value.get("recipe"),
            recipe_identity=_optional_reference_identity(registry, "recipe", value.get("recipe")),
            model_profile_ref=value.get("model_profile"),
            model_profile_identity=_optional_reference_identity(registry, "model_profile", value.get("model_profile")),
            display_name=value.get("display_name"),
        ))
    ordered = tuple(sorted(result, key=lambda item: item.role))
    _reject_cycles(ordered)
    return ordered


def _resolve_authority(section: object, objects: Sequence[ResolvedContextObject],
                       registry: Mapping[tuple[str, str], ContextModelReference]) -> tuple[ResolvedAuthorityPolicy, ...]:
    object_roles = {item.role for item in objects}
    result: list[ResolvedAuthorityPolicy] = []
    for role, raw in _entries(section, "authority"):
        value = _mapping(raw, f"authority.{role}")
        target_role = _string(value.get("target", role), f"authority.{role}.target")
        if target_role not in object_roles:
            raise CompilerFailure("contract_invalid", f"authority references missing object: {role}")
        policy_ref = _required_str(value, "policy", f"authority.{role}")
        policy = _reference(registry, "authority_policy", policy_ref)
        settings = {key: val for key, val in value.items() if key not in {"target", "policy"}}
        result.append(ResolvedAuthorityPolicy(role=role, target_role=target_role, policy_ref=policy_ref,
                                              policy_identity=policy.identity, settings=settings))
    return tuple(sorted(result, key=lambda item: item.role))


def _resolve_retrievals(
    section: object,
    objects: Sequence[ResolvedContextObject],
    authority: Sequence[ResolvedAuthorityPolicy],
) -> tuple[ResolvedRetrievalDefinition, ...]:
    object_roles = {item.role for item in objects}
    authority_roles = {item.role for item in authority}
    result: list[ResolvedRetrievalDefinition] = []
    for role, raw in _entries(section, "retrievals"):
        value = _mapping(raw, f"retrievals.{role}")
        resolution = _mapping(value.get("resolution", {}), f"retrievals.{role}.resolution")
        policy_role = resolution.get("policy")
        if policy_role is not None:
            policy_role = _string(policy_role, f"retrievals.{role}.resolution.policy")
            if policy_role not in authority_roles:
                raise CompilerFailure("contract_invalid", f"retrieval references missing authority policy: {role}")
        target_roles = tuple(_strings(value.get("target_roles", ()), f"retrievals.{role}.target_roles"))
        if set(target_roles) - object_roles:
            raise CompilerFailure("contract_invalid", f"retrieval references missing object: {role}")
        result.append(ResolvedRetrievalDefinition(
            role=role,
            search=value.get("search", "hybrid"),
            candidate_limit=int(value.get("candidate_limit", value.get("limit", 10))),
            target_roles=tuple(sorted(target_roles)),
            eligibility=dict(_mapping(value.get("eligibility", {}), f"retrievals.{role}.eligibility")),
            resolution_policy_role=policy_role,
            accepted_outcomes=tuple(_strings(resolution.get("accepted_outcomes", ()), f"retrievals.{role}.accepted_outcomes")),
            display_name=value.get("display_name"),
        ))
    return tuple(sorted(result, key=lambda item: item.role))


def _resolve_presentation(section: object, objects: Sequence[ResolvedContextObject]) -> tuple[ResolvedPresentation, ...]:
    object_roles = {item.role for item in objects}
    result: list[ResolvedPresentation] = []
    for role, raw in _entries(section, "presentation"):
        value = _mapping(raw, f"presentation.{role}")
        target_role = _string(value.get("target", role), f"presentation.{role}.target")
        if target_role not in object_roles:
            raise CompilerFailure("contract_invalid", f"presentation references missing object: {role}")
        result.append(ResolvedPresentation(
            role=role,
            target_role=target_role,
            preview_fields=tuple(_strings(value.get("preview_fields", ()), f"presentation.{role}.preview_fields")),
            display_name=value.get("display_name"),
        ))
    return tuple(sorted(result, key=lambda item: item.role))


def _resolve_tests(section: object, objects: Sequence[ResolvedContextObject]) -> tuple[ResolvedContextModelTest, ...]:
    object_roles = {item.role for item in objects}
    result: list[ResolvedContextModelTest] = []
    if not section:
        return ()
    if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
        raise CompilerFailure("contract_invalid", "tests must be a sequence")
    for index, raw in enumerate(section):
        if not isinstance(raw, Mapping) or len(raw) != 1:
            raise CompilerFailure("contract_invalid", f"tests.{index} must contain one test")
        test_type, value = next(iter(raw.items()))
        parameters: dict[str, Any]
        target_role: str | None = None
        if isinstance(value, str):
            target_role = value
            parameters = {}
        elif isinstance(value, Mapping):
            parameters = dict(value)
            raw_target = parameters.get("object") or parameters.get("target")
            if raw_target is not None:
                target_role = _string(raw_target, f"tests.{index}.{test_type}.target")
        elif value is None:
            parameters = {}
        else:
            raise CompilerFailure("contract_invalid", f"tests.{index}.{test_type} has invalid parameters")
        if target_role is not None and target_role not in object_roles:
            raise CompilerFailure("contract_invalid", f"test references missing object: {test_type}")
        result.append(ResolvedContextModelTest(test_type=str(test_type), target_role=target_role, parameters=parameters))
    return tuple(sorted(result, key=lambda item: (item.test_type, item.target_role or "", item.identity)))


def _resolve_context_contracts(section: object) -> tuple[ResolvedContextContract, ...]:
    result: list[ResolvedContextContract] = []
    for ref, raw in _entries(section, "context_contracts"):
        value = _mapping(raw, f"context_contracts.{ref}")
        contract_id = _required_str(value, "contract_id", f"context_contracts.{ref}")
        version = _required_str(value, "version", f"context_contracts.{ref}")
        rules = {key: val for key, val in value.items() if key not in {"contract_id", "version", "title"}}
        result.append(
            ResolvedContextContract(
                ref=ref,
                contract_id=contract_id,
                version=version,
                title=value.get("title"),
                rules=rules,
            )
        )
    return tuple(sorted(result, key=lambda item: item.ref))


def _resolve_outputs(
    section: object,
    retrievals: Sequence[ResolvedRetrievalDefinition],
    contracts: Sequence[ResolvedContextContract],
    objects: Sequence[ResolvedContextObject],
) -> tuple[ResolvedContextOutput, ...]:
    retrieval_by_role = {item.role: item for item in retrievals}
    contract_by_ref = {item.ref: item for item in contracts}
    object_roles = {item.role for item in objects}
    result: list[ResolvedContextOutput] = []
    for role, raw in _entries(section, "outputs"):
        value = _mapping(raw, f"outputs.{role}")
        retrieval_role = value.get("retrieval")
        if retrieval_role is not None:
            retrieval_role = _string(retrieval_role, f"outputs.{role}.retrieval")
            if retrieval_role not in retrieval_by_role:
                raise CompilerFailure("contract_invalid", f"output references missing retrieval: {role}")
        contract_ref = value.get("contract")
        if contract_ref is not None:
            contract_ref = _string(contract_ref, f"outputs.{role}.contract")
            if contract_ref not in contract_by_ref:
                raise CompilerFailure("contract_invalid", f"output references missing contract: {role}")
        target_roles = tuple(_strings(value.get("target_roles", ()), f"outputs.{role}.target_roles"))
        if set(target_roles) - object_roles:
            raise CompilerFailure("contract_invalid", f"output references missing object: {role}")
        result.append(
            ResolvedContextOutput(
                role=role,
                output_type=_required_str(value, "output_type", f"outputs.{role}"),
                retrieval_role=retrieval_role,
                retrieval_identity=None if retrieval_role is None else retrieval_by_role[retrieval_role].identity,
                contract_ref=contract_ref,
                contract_identity=None if contract_ref is None else contract_by_ref[contract_ref].identity,
                target_roles=tuple(sorted(target_roles)),
                format=value.get("format"),
                destination=value.get("destination"),
                display_name=value.get("display_name"),
                settings=dict(_mapping(value.get("settings", {}), f"outputs.{role}.settings")),
            )
        )
    return tuple(sorted(result, key=lambda item: item.role))


def _resolve_evaluations(
    section: object,
    objects: Sequence[ResolvedContextObject],
    outputs: Sequence[ResolvedContextOutput],
    contracts: Sequence[ResolvedContextContract],
    retrievals: Sequence[ResolvedRetrievalDefinition],
) -> tuple[ResolvedContextEvaluation, ...]:
    object_roles = {item.role for item in objects}
    output_roles = {item.role for item in outputs}
    contract_refs = {item.ref for item in contracts}
    retrieval_roles = {item.role for item in retrievals}
    result: list[ResolvedContextEvaluation] = []
    for evaluation_id, raw in _entries(section, "evaluations"):
        value = _mapping(raw, f"evaluations.{evaluation_id}")
        target_roles = tuple(_strings(value.get("target_roles", ()), f"evaluations.{evaluation_id}.target_roles"))
        target_outputs = tuple(_strings(value.get("target_outputs", ()), f"evaluations.{evaluation_id}.target_outputs"))
        target_contracts = tuple(_strings(value.get("target_contracts", ()), f"evaluations.{evaluation_id}.target_contracts"))
        target_retrievals = tuple(_strings(value.get("target_retrievals", ()), f"evaluations.{evaluation_id}.target_retrievals"))
        if set(target_roles) - object_roles:
            raise CompilerFailure("contract_invalid", f"evaluation references missing object: {evaluation_id}")
        if set(target_outputs) - output_roles:
            raise CompilerFailure("contract_invalid", f"evaluation references missing output: {evaluation_id}")
        if set(target_contracts) - contract_refs:
            raise CompilerFailure("contract_invalid", f"evaluation references missing contract: {evaluation_id}")
        if set(target_retrievals) - retrieval_roles:
            raise CompilerFailure("contract_invalid", f"evaluation references missing retrieval: {evaluation_id}")
        result.append(
            ResolvedContextEvaluation(
                evaluation_id=evaluation_id,
                category=_required_str(value, "category", f"evaluations.{evaluation_id}"),
                description=_required_str(value, "description", f"evaluations.{evaluation_id}"),
                target_roles=tuple(sorted(target_roles)),
                target_outputs=tuple(sorted(target_outputs)),
                target_contracts=tuple(sorted(target_contracts)),
                target_retrievals=tuple(sorted(target_retrievals)),
                fixtures=dict(_mapping(value.get("fixtures", {}), f"evaluations.{evaluation_id}.fixtures")),
                checks=tuple(_strings(value.get("checks", ()), f"evaluations.{evaluation_id}.checks")),
                required=bool(value.get("required", True)),
                hidden_expectations=bool(value.get("hidden_expectations", True)),
            )
        )
    return tuple(sorted(result, key=lambda item: item.evaluation_id))


def _entries(section: object, name: str) -> tuple[tuple[str, object], ...]:
    if section is None:
        return ()
    entries: list[tuple[str, object]] = []
    if isinstance(section, Mapping):
        entries = [(str(key), value) for key, value in section.items()]
    elif isinstance(section, Sequence) and not isinstance(section, (str, bytes)):
        for index, item in enumerate(section):
            if not isinstance(item, Mapping):
                raise CompilerFailure("contract_invalid", f"{name}.{index} must be a mapping")
            if "role" in item:
                role = _string(item["role"], f"{name}.{index}.role")
                value = {key: val for key, val in item.items() if key != "role"}
            elif len(item) == 1:
                role, value = next(iter(item.items()))
                role = str(role)
            else:
                raise CompilerFailure("contract_invalid", f"{name}.{index} must declare a role")
            entries.append((role, value))
    else:
        raise CompilerFailure("contract_invalid", f"{name} must be a mapping or sequence")
    roles = [role for role, _ in entries]
    if len(roles) != len(set(roles)):
        raise CompilerFailure("contract_invalid", f"duplicate Context Model role in {name}")
    return tuple(entries)


def _reject_unknown_sections(declaration: Mapping[str, Any]) -> None:
    allowed = {
        "context_model",
        "sources",
        "objects",
        "authority",
        "retrievals",
        "presentation",
        "tests",
        "context_contracts",
        "contracts",
        "outputs",
        "evaluations",
    }
    unknown = sorted(set(declaration) - allowed)
    if unknown:
        raise CompilerFailure("contract_invalid", "unknown Context Model section: " + ", ".join(unknown))


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompilerFailure("contract_invalid", f"{path} must be a mapping")
    return value


def _required_str(value: Mapping[str, Any], key: str, path: str) -> str:
    if key not in value:
        raise CompilerFailure("contract_invalid", f"{path}.{key} is required")
    return _string(value[key], f"{path}.{key}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompilerFailure("contract_invalid", f"{path} must be a non-empty string")
    return value


def _strings(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise CompilerFailure("contract_invalid", f"{path} must be a sequence of strings")
    return tuple(_string(item, f"{path}[]") for item in value)


def _reference(registry: Mapping[tuple[str, str], ContextModelReference], kind: str, ref: str) -> ContextModelReference:
    artifact = registry.get((kind, ref))
    if artifact is None:
        raise CompilerFailure("contract_invalid", f"missing {kind} reference: {ref}")
    return artifact


def _optional_reference_identity(registry: Mapping[tuple[str, str], ContextModelReference], kind: str,
                                 ref: object) -> Identity | None:
    if ref is None:
        return None
    value = _string(ref, kind)
    if kind == "operation" and (kind, value) not in registry:
        return stable_identity({"operation": value})
    return _reference(registry, kind, value).identity


def _reject_cycles(objects: Sequence[ResolvedContextObject]) -> None:
    dependencies = {item.role: set(item.derived_from) for item in objects}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(role: str) -> None:
        if role in visited:
            return
        if role in visiting:
            raise CompilerFailure("contract_invalid", "Context Model object dependency cycle")
        visiting.add(role)
        for dependency in sorted(dependencies[role]):
            visit(dependency)
        visiting.remove(role)
        visited.add(role)

    for role in sorted(dependencies):
        visit(role)


def _semantic_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in sorted(value)}
