from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from igor_context_compiler import (
    CompilationRequest,
    CompilerFailure,
    CompilerPorts,
    DependencyInventory,
    compile_context_model_derivation_instances,
    compile_context_model_derivations,
    compile_context_model_resolution_request,
    compile_context_model_retrieval_queries,
    execute,
    plan,
)
from igor_core import Evidence, ContextItem, ContextPackage, EmbeddingSpace, ModelProfile, canonical_json, stable_identity
from igor_datafusion import AnalyticalEngine
from igor_lancedb import (
    STANDARD_CONTEXT_RELATIONS,
    ContextModelTestExecutionResult,
    LanceContextModelMaterializer,
    LanceRetrievalAdapter,
    LanceStore,
)


@dataclass(frozen=True)
class ContextModelLifecycleResult:
    """Observable result of one runner-composed Context Model lifecycle pass."""

    materialization: Any
    compilation: Any | None = None
    compilation_plan: Any | None = None
    retrieval_results: tuple[dict[str, Any], ...] = ()
    resolution_requests: tuple[Any, ...] = ()
    resolution_results: tuple[Any, ...] = ()
    packages: tuple[ContextPackage, ...] = ()
    contract_evaluations: tuple[dict[str, Any], ...] = ()
    standard_sql_relations: tuple[str, ...] = STANDARD_CONTEXT_RELATIONS
    relation_counts: dict[str, int] = field(default_factory=dict)
    test_execution: ContextModelTestExecutionResult | None = None
    store: LanceStore | None = field(default=None, repr=False, compare=False)

    def query_standard_sql(
        self,
        sql: str,
        *,
        table_name: str = "context_models",
        retrieval: Any | None = None,
        query_vectors: Mapping[str, tuple[float, ...]] | None = None,
    ):
        """Query materialized standard relations through the DataFusion owner."""
        if self.store is None:
            raise ValueError("Context Model lifecycle result does not expose a Lance store")
        if table_name not in self.standard_sql_relations:
            raise ValueError(f"unknown standard Context Model SQL relation: {table_name}")
        engine = AnalyticalEngine(
            self.store,
            retrieval if retrieval is not None else LanceRetrievalAdapter(self.store),
            query_vectors=query_vectors,
        )
        return engine.query(sql, table_name)


def materialize_context_model_lifecycle(
    *,
    store: LanceStore,
    manifest: Any,
    observations: Iterable[dict[str, Any]] = (),
    outputs: Iterable[dict[str, Any]] = (),
    assertions: Iterable[dict[str, Any]] = (),
    retrieval: Iterable[dict[str, Any]] = (),
    resolution_decisions: Iterable[dict[str, Any]] = (),
    resolution_candidates: Iterable[dict[str, Any]] = (),
    resolution_inputs: Iterable[Mapping[str, Any]] = (),
    resolution_requests: Iterable[Any] = (),
    resolution_results: Iterable[Any] = (),
    resolution_port: Any | None = None,
    retrieval_inputs: Iterable[Mapping[str, Any]] = (),
    retrieval_port: Any | None = None,
    contract_context: Mapping[str, Any] | None = None,
    context_contracts: Iterable[Mapping[str, Any]] = (),
    contract_evaluations: Iterable[Mapping[str, Any]] = (),
    lineage_edges: Iterable[dict[str, Any]] = (),
    packages: Iterable[dict[str, Any]] = (),
    package_specs: Iterable[Any] = (),
    package_inputs: Iterable[Mapping[str, Any]] = (),
    package_items: Iterable[dict[str, Any]] = (),
) -> ContextModelLifecycleResult:
    output_rows = tuple(outputs)
    assertion_rows = tuple(assertions)
    lineage_rows = tuple(lineage_edges)
    package_item_rows = tuple(package_items)
    retrieval_rows = tuple(retrieval)
    executed_retrieval_rows = _execute_context_model_retrieval_inputs(
        manifest,
        tuple(retrieval_inputs),
        retrieval_port,
    )
    package_specs_tuple = tuple(package_specs)
    package_inputs_tuple = tuple(package_inputs)
    explicit_package_rows = tuple(packages)
    explicit_resolution_requests = tuple(resolution_requests)
    resolution_input_rows = tuple(resolution_inputs)
    if not resolution_input_rows and not explicit_resolution_requests:
        resolution_input_rows = _resolution_inputs_from_retrieval_rows((*retrieval_rows, *executed_retrieval_rows))
    compiled_requests = _compile_context_model_resolution_inputs(manifest, resolution_input_rows)
    requests = (*compiled_requests, *explicit_resolution_requests)
    resolved_pairs = _resolve_context_model_requests(
        requests=requests,
        results=tuple(resolution_results),
        resolution_port=resolution_port,
    )
    default_packages = ()
    if not package_specs_tuple and not package_inputs_tuple and not explicit_package_rows:
        default_packages = _default_context_packages_from_resolution_pairs(resolved_pairs)
    compiled_packages = (
        *_compile_context_packages(package_specs_tuple),
        *_compile_context_packages_from_resolution_inputs(resolved_pairs, package_inputs_tuple),
        *default_packages,
    )
    compiled_decisions, compiled_candidates = _resolution_relation_rows(resolved_pairs)
    compiled_package_rows = _context_package_rows(compiled_packages)
    contract = _resolve_context_contract(manifest, tuple(context_contracts))
    allowed_package_rows, contract_bindings, evaluated_contracts = _evaluate_context_package_contracts(
        contract=contract,
        packages=(*explicit_package_rows, *compiled_package_rows),
        package_items=package_item_rows,
        outputs=output_rows,
        context=contract_context or {},
    )
    allowed_package_ids = {row["package_identity"] for row in allowed_package_rows}
    allowed_package_items = _package_items_for_allowed_packages(tuple(package_items), allowed_package_ids)
    compiled_lineage_edges = _standard_lifecycle_lineage_edges(
        resolution_decisions=compiled_decisions,
        resolution_candidates=compiled_candidates,
        packages=allowed_package_rows,
    )
    contract_lineage_edges = _contract_lifecycle_lineage_edges(evaluated_contracts)
    materialized_retrieval_rows = _context_retrieval_rows((*retrieval_rows, *executed_retrieval_rows))
    materializer = LanceContextModelMaterializer(store)
    materialized = materializer.replace(
        manifest=manifest,
        observations=tuple(observations),
        outputs=output_rows,
        assertions=assertion_rows,
        retrieval=materialized_retrieval_rows,
        resolution_decisions=(*tuple(resolution_decisions), *compiled_decisions),
        resolution_candidates=(*tuple(resolution_candidates), *compiled_candidates),
        context_contracts=(contract,),
        context_package_contracts=contract_bindings,
        contract_evaluations=(*tuple(dict(row) for row in contract_evaluations), *evaluated_contracts),
        lineage_edges=(*lineage_rows, *compiled_lineage_edges, *contract_lineage_edges),
        packages=allowed_package_rows,
        package_items=allowed_package_items,
    )
    test_execution = materializer.execute_tests(manifest)
    return ContextModelLifecycleResult(
        materialization=materialized,
        retrieval_results=executed_retrieval_rows,
        resolution_requests=requests,
        resolution_results=tuple(result for _, result in resolved_pairs),
        packages=tuple(
            package for package in compiled_packages
            if package.identity in allowed_package_ids
        ),
        contract_evaluations=evaluated_contracts,
        relation_counts=dict(materialized.relation_counts),
        test_execution=test_execution,
        store=store,
    )


def compile_context_model_lifecycle(
    *,
    store: LanceStore,
    manifest: Any,
    observations: Iterable[dict[str, Any]],
    ports: CompilerPorts,
    request: CompilationRequest | None = None,
    inventory: DependencyInventory | None = None,
    outputs: Iterable[dict[str, Any]] = (),
    assertions: Iterable[dict[str, Any]] = (),
    retrieval: Iterable[dict[str, Any]] = (),
    lineage_edges: Iterable[dict[str, Any]] = (),
    package_specs: Iterable[Any] = (),
    package_inputs: Iterable[Mapping[str, Any]] = (),
    runtime_role_identities: Mapping[str, str | Iterable[str]] | None = None,
    output_role_identities: Mapping[str, str] | None = None,
    run_identity: str | None = None,
    task_id: str | None = None,
    embedding_profile: ModelProfile | None = None,
    completion_profile: ModelProfile | None = None,
    embedding_space: EmbeddingSpace | None = None,
    code_revision: str | None = None,
    budget_tokens: int | None = None,
    package_items: Iterable[ContextItem | Mapping[str, Any]] = (),
    retrieval_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    resolution_inputs: Iterable[Mapping[str, Any]] = (),
    derivation_inputs: Iterable[Mapping[str, Any]] = (),
    embedding_inputs: Iterable[Any] = (),
    completion_inputs: Iterable[Any] = (),
    completion_output_schema: Any | None = None,
    enrichment_recipe: Any | None = None,
    prompt_version: str = "0.1",
    taxonomy_version: str = "0.1",
    required_context_identities: Iterable[str] = (),
    contract_context: Mapping[str, Any] | None = None,
    context_contracts: Iterable[Mapping[str, Any]] = (),
    contract_evaluations: Iterable[Mapping[str, Any]] = (),
    plan_only: bool = False,
) -> ContextModelLifecycleResult:
    runtime_outputs = tuple(outputs)
    observation_rows = tuple(observations)
    assertion_rows = tuple(assertions)
    retrieval_rows = tuple(retrieval)
    lineage_rows = tuple(lineage_edges)
    resolution_input_rows = tuple(resolution_inputs)
    if request is None:
        role_identities = _role_identities(
            observations=observation_rows,
            outputs=runtime_outputs,
            runtime_role_identities=runtime_role_identities or {},
        )
        request = _request_from_manifest_derivations(
            manifest=manifest,
            role_identities=role_identities,
            output_role_identities=output_role_identities or {},
            run_identity=run_identity,
            task_id=task_id,
            embedding_profile=embedding_profile,
            completion_profile=completion_profile,
            embedding_space=embedding_space,
            code_revision=code_revision,
            budget_tokens=budget_tokens,
            package_items=tuple(package_items),
            retrieval_inputs=retrieval_inputs or {},
            resolution_inputs=resolution_input_rows,
            derivation_inputs=tuple(derivation_inputs),
            embedding_inputs=tuple(embedding_inputs),
            completion_inputs=tuple(completion_inputs),
            completion_output_schema=completion_output_schema,
            enrichment_recipe=enrichment_recipe,
            prompt_version=prompt_version,
            taxonomy_version=taxonomy_version,
            required_context_identities=tuple(required_context_identities),
        )
        if inventory is None:
            inventory = DependencyInventory(known_output_identities=tuple(sorted(_flatten_role_identities(role_identities))))
    elif inventory is None:
        inventory = DependencyInventory()
    if request is not None and request.resolution is None:
        if not resolution_input_rows:
            resolution_input_rows = _resolution_inputs_from_retrieval_rows(retrieval_rows)
        resolution_request = _single_context_model_resolution_input(manifest, resolution_input_rows)
        if resolution_request is not None:
            request = request.model_copy(update={"resolution": resolution_request})
    preserved_outputs = _existing_context_outputs(store)
    initial_materialization = materialize_context_model_lifecycle(
        store=store,
        manifest=manifest,
        observations=observation_rows,
        outputs=_merge_outputs(preserved_outputs, runtime_outputs),
        assertions=assertion_rows,
        retrieval=retrieval_rows,
        lineage_edges=lineage_rows,
    )
    compilation_plan = plan(request, inventory)
    if plan_only:
        return ContextModelLifecycleResult(
            materialization=initial_materialization.materialization,
            compilation_plan=compilation_plan,
            relation_counts=dict(initial_materialization.relation_counts),
            test_execution=initial_materialization.test_execution,
            store=store,
        )
    compilation = execute(compilation_plan, ports, request)
    compiler_outputs = _compiler_output_rows(request, compilation, ports)
    compiled_packages = _compile_context_packages(tuple(package_specs))
    package_rows = (*_package_rows(request, compilation), *_context_package_rows(compiled_packages))
    contract = _resolve_context_contract(manifest, tuple(context_contracts))
    allowed_package_rows, contract_bindings, evaluated_contracts = _evaluate_context_package_contracts(
        contract=contract,
        packages=package_rows,
        package_items=(),
        outputs=_merge_outputs(preserved_outputs, runtime_outputs, compiler_outputs),
        context={
            "task_id": request.task_id,
            "budget_tokens": request.budget_tokens,
            **dict(contract_context or {}),
        },
    )
    allowed_package_ids = {row["package_identity"] for row in allowed_package_rows}
    resolution_decision_rows = _compilation_resolution_decision_rows(compilation)
    resolution_candidate_rows = (*_resolution_candidates(compilation), *_retrieval_candidate_rows(compilation))
    compiled_lineage_edges = _standard_lifecycle_lineage_edges(
        resolution_decisions=resolution_decision_rows,
        resolution_candidates=resolution_candidate_rows,
        packages=allowed_package_rows,
    )
    contract_lineage_edges = _contract_lifecycle_lineage_edges(evaluated_contracts)
    materializer = LanceContextModelMaterializer(store)
    materialized = materializer.replace(
        manifest=manifest,
        observations=observation_rows,
        outputs=_merge_outputs(preserved_outputs, runtime_outputs, compiler_outputs),
        assertions=assertion_rows,
        retrieval=retrieval_rows,
        resolution_decisions=resolution_decision_rows,
        resolution_candidates=resolution_candidate_rows,
        context_contracts=(contract,),
        context_package_contracts=contract_bindings,
        contract_evaluations=(*tuple(dict(row) for row in contract_evaluations), *evaluated_contracts),
        lineage_edges=(*lineage_rows, *compiled_lineage_edges, *contract_lineage_edges),
        packages=allowed_package_rows,
    )
    test_execution = materializer.execute_tests(manifest)
    return ContextModelLifecycleResult(
        materialization=materialized,
        compilation=compilation,
        compilation_plan=compilation_plan,
        packages=tuple(
            package for package in compiled_packages
            if package.identity in allowed_package_ids
        ),
        contract_evaluations=evaluated_contracts,
        relation_counts=dict(materialized.relation_counts),
        test_execution=test_execution,
        store=store,
    )


def _request_from_manifest_derivations(
    *,
    manifest: Any,
    role_identities: Mapping[str, tuple[str, ...]],
    output_role_identities: Mapping[str, str],
    run_identity: str | None,
    task_id: str | None,
    embedding_profile: ModelProfile | None,
    completion_profile: ModelProfile | None,
    embedding_space: EmbeddingSpace | None,
    code_revision: str | None,
    budget_tokens: int | None,
    package_items: tuple[ContextItem | Mapping[str, Any], ...],
    retrieval_inputs: Mapping[str, Mapping[str, Any]],
    resolution_inputs: tuple[Mapping[str, Any], ...],
    derivation_inputs: tuple[Mapping[str, Any], ...],
    embedding_inputs: tuple[Any, ...],
    completion_inputs: tuple[Any, ...],
    completion_output_schema: Any | None,
    enrichment_recipe: Any | None,
    prompt_version: str,
    taxonomy_version: str,
    required_context_identities: tuple[str, ...],
) -> CompilationRequest:
    missing = [
        name for name, value in (
            ("embedding_profile", embedding_profile),
            ("completion_profile", completion_profile),
            ("embedding_space", embedding_space),
            ("code_revision", code_revision),
        )
        if value is None
    ]
    if missing:
        raise CompilerFailure("contract_invalid", "manifest-derived compilation request missing: " + ", ".join(missing))
    if derivation_inputs:
        derivations = compile_context_model_derivation_instances(
            manifest,
            derivation_inputs,
            code_revision=str(code_revision),
        )
    else:
        derivations = compile_context_model_derivations(
            manifest,
            role_identities,
            code_revision=str(code_revision),
            output_identities=output_role_identities,
        )
    if not derivations:
        raise CompilerFailure("contract_invalid", "resolved Context Model manifest declares no compiler derivations")
    request_items = _context_items(package_items, derivations, required_context_identities)
    resolution_request = _single_context_model_resolution_input(manifest, resolution_inputs)
    return CompilationRequest(
        run_identity=run_identity or stable_identity({
            "context_model": _manifest_field(manifest, "context_model_id", "context-model"),
            "revision": _manifest_field(manifest, "revision", _manifest_field(manifest, "context_model_revision", "1")),
            "role_identities": role_identities,
            "code_revision": code_revision,
        }),
        task_id=task_id or str(_manifest_field(manifest, "context_model_id", "context-model")),
        required_output_identities=tuple(item.output_identity for item in derivations),
        derivations=derivations,
        embedding_profile=embedding_profile,  # type: ignore[arg-type]
        completion_profile=completion_profile,  # type: ignore[arg-type]
        embedding_space=embedding_space,  # type: ignore[arg-type]
        embedding_inputs=embedding_inputs,
        completion_inputs=completion_inputs,
        completion_output_schema=completion_output_schema,
        enrichment_recipe=enrichment_recipe,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        code_revision=str(code_revision),
        budget_tokens=budget_tokens,
        package_items=request_items,
        retrieval_queries=compile_context_model_retrieval_queries(manifest, retrieval_inputs),
        required_context_identities=required_context_identities or tuple(item.output_identity for item in derivations),
        resolution=resolution_request,
    )


def _context_items(
    values: tuple[ContextItem | Mapping[str, Any], ...],
    derivations: tuple[Any, ...],
    required_context_identities: tuple[str, ...],
) -> tuple[ContextItem, ...]:
    if values:
        return tuple(item if isinstance(item, ContextItem) else ContextItem.model_validate(dict(item)) for item in values)
    if required_context_identities:
        by_output = {str(derivation.output_identity): derivation for derivation in derivations}
        return tuple(
            ContextItem(
                representation_identity=identity,
                role=str(getattr(by_output.get(identity), "operation", "required-context")),
                rank=index,
                token_estimate=0,
            )
            for index, identity in enumerate(required_context_identities)
        )
    else:
        selected = derivations
    return tuple(
        ContextItem(
            representation_identity=derivation.output_identity,
            role=derivation.operation,
            rank=index,
            token_estimate=0,
        )
        for index, derivation in enumerate(selected)
    )


def _execute_context_model_retrieval_inputs(
    manifest: Any,
    values: tuple[Mapping[str, Any], ...],
    retrieval_port: Any | None,
) -> tuple[dict[str, Any], ...]:
    if not values:
        return ()
    if retrieval_port is None:
        raise CompilerFailure("port_unavailable", "retrieval inputs require a retrieval port")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        payload = dict(raw)
        retrieval_role = str(payload.pop("retrieval_role", payload.pop("role", ""))).strip()
        if not retrieval_role:
            raise CompilerFailure("contract_invalid", f"retrieval input {index} requires retrieval_role")
        query = compile_context_model_retrieval_queries(manifest, {retrieval_role: payload})[0]
        facts = tuple(retrieval_port.retrieve(query))
        for rank, fact in enumerate(facts):
            fact_value = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else dict(fact)
            context_identity = str(fact_value.get("context_identity", "")).strip()
            if not context_identity:
                raise CompilerFailure("contract_invalid", "retrieval fact requires context_identity")
            result_identity = str(fact_value.get("result_identity", fact_value.get("retrieval_identity", ""))).strip()
            if not result_identity:
                result_identity = stable_identity({
                    "retrieval_role": retrieval_role,
                    "query_id": query.query_id,
                    "context_identity": context_identity,
                    "rank": int(fact_value.get("rank", rank) or 0),
                })
            row = {
                **payload,
                "retrieval_role": retrieval_role,
                "query_id": query.query_id,
                "result_identity": result_identity,
                "retrieval_identity": result_identity,
                "context_identity": context_identity,
                "rank": int(fact_value.get("rank", rank) or 0),
                "score": float(fact_value.get("score", 0.0) or 0.0),
                "facts": dict(fact_value.get("facts", {})),
            }
            if len(facts) != 1 or not row.get("candidate_identity"):
                row["candidate_identity"] = stable_identity({
                    "request_identity": row.get("request_identity", query.query_id),
                    "context_identity": context_identity,
                    "rank": row["rank"],
                })
            row.setdefault("source_snapshot_id", context_identity)
            row.setdefault("evidence_locator", f"context:{context_identity}")
            rows.append(row)
    return tuple(rows)


def _context_retrieval_rows(rows: tuple[Mapping[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        context_identity = str(row.get("context_identity", row.get("identity", ""))).strip()
        if not context_identity:
            continue
        if row.get("vector") is None and not row.get("materialize_retrieval", False):
            continue
        retrieval_identity = str(
            row.get("retrieval_identity")
            or row.get("result_identity")
            or stable_identity({"retrieval": context_identity})
        )
        vector = row.get("vector")
        if vector is not None:
            vector = [float(item) for item in vector]
        values.append({
            "identity": str(row.get("identity", context_identity)),
            "context_identity": context_identity,
            "retrieval_identity": retrieval_identity,
            "vector": vector,
            "text": str(row.get("text", row.get("query", row.get("preview", "")))),
            "context_model_id": str(row.get("context_model_id", "")),
            "status": str(row.get("status", "active")),
        })
    return tuple(values)


def _resolution_inputs_from_retrieval_rows(rows: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        retrieval_role = str(row.get("retrieval_role", row.get("role", ""))).strip()
        if not retrieval_role:
            continue
        task_id = str(row.get("task_id", "")).strip()
        as_of = row.get("as_of")
        context_identity = str(row.get("context_identity", row.get("identity", ""))).strip()
        if not task_id or as_of is None or not context_identity:
            raise CompilerFailure(
                "contract_invalid",
                "resolution retrieval row requires retrieval_role, task_id, as_of, and context_identity",
            )
        task_schema_revision = str(row.get("task_schema_revision", row.get("schema_revision", "1")))
        request_identity = str(row.get("request_identity") or stable_identity({
            "retrieval_role": retrieval_role,
            "task_id": task_id,
            "task_schema_revision": task_schema_revision,
            "as_of": str(as_of),
            "query_identity": row.get("query_id", row.get("retrieval_identity", row.get("result_identity", ""))),
        }))
        group = grouped.setdefault(request_identity, {
            "request_identity": request_identity,
            "retrieval_role": retrieval_role,
            "task_id": task_id,
            "task_schema_revision": task_schema_revision,
            "as_of": as_of,
            "candidates": [],
            "evidence": [],
            "authority_assertions": [],
            "temporal_assertions": [],
        })
        for field, expected in (
            ("retrieval_role", retrieval_role),
            ("task_id", task_id),
            ("task_schema_revision", task_schema_revision),
            ("as_of", as_of),
        ):
            if group[field] != expected:
                raise CompilerFailure("contract_invalid", "resolution retrieval rows mix request metadata")
        evidence_items = _evidence_from_retrieval_row(row, context_identity=context_identity, as_of=as_of)
        group["evidence"].extend(evidence_items)
        group["authority_assertions"].extend(_sequence_field(row.get("authority_assertions")))
        group["temporal_assertions"].extend(_sequence_field(row.get("temporal_assertions")))
        evidence_identities = tuple(str(value) for value in _sequence_field(row.get("evidence_identities")))
        if not evidence_identities:
            evidence_identities = tuple(item.identity for item in evidence_items)
        candidate = {
            "candidate_identity": str(row.get("candidate_identity") or stable_identity({
                "request_identity": request_identity,
                "context_identity": context_identity,
                "rank": int(row.get("rank", 0) or 0),
            })),
            "context_identity": context_identity,
            "evidence_identities": evidence_identities,
            "authority_assertion_ids": tuple(str(value) for value in _sequence_field(row.get("authority_assertion_ids"))),
            "temporal_assertion_ids": tuple(str(value) for value in _sequence_field(row.get("temporal_assertion_ids"))),
        }
        retrieval = _retrieval_fact_from_row(row, context_identity)
        if retrieval is not None:
            candidate["retrieval"] = retrieval
        group["candidates"].append(candidate)
        if row.get("policy_role"):
            group["policy_role"] = str(row["policy_role"])
    return tuple(
        {
            **{key: value for key, value in group.items() if key not in {"candidates", "evidence", "authority_assertions", "temporal_assertions"}},
            "candidates": tuple(group["candidates"]),
            "evidence": tuple(group["evidence"]),
            "authority_assertions": tuple(group["authority_assertions"]),
            "temporal_assertions": tuple(group["temporal_assertions"]),
        }
        for _, group in sorted(grouped.items())
    )


def _sequence_field(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)):
        return (value,)
    return tuple(value)


def _evidence_from_retrieval_row(row: Mapping[str, Any], *, context_identity: str, as_of: Any) -> tuple[Evidence, ...]:
    explicit = _sequence_field(row.get("evidence"))
    if explicit:
        return tuple(item if isinstance(item, Evidence) else Evidence.model_validate(dict(item)) for item in explicit)
    if row.get("evidence_identities") and not (row.get("source_snapshot_id") or row.get("snapshot_identity")):
        return ()
    locator = str(row.get("locator", row.get("evidence_locator", row.get("content_ref", f"context:{context_identity}"))))
    evidence = Evidence(
        ir_version="0.1",
        subject_identity=context_identity,
        source_snapshot_id=str(row.get("source_snapshot_id", row.get("snapshot_identity", context_identity))),
        locator=locator,
        excerpt=row.get("excerpt", row.get("evidence_excerpt", row.get("preview"))),
        observed_at=row.get("observed_at", as_of),
    )
    return (evidence,)


def _retrieval_fact_from_row(row: Mapping[str, Any], context_identity: str) -> dict[str, Any] | None:
    if not any(key in row for key in ("result_identity", "retrieval_identity", "score", "semantic_score", "rank", "facts")):
        return None
    facts = row.get("facts", {})
    if not isinstance(facts, Mapping):
        facts = {}
    result_identity = str(row.get("result_identity", row.get("retrieval_identity", stable_identity({
        "retrieval": context_identity,
        "rank": int(row.get("rank", 0) or 0),
    }))))
    score = row.get("score", row.get("semantic_score", 0.0))
    return {
        "result_identity": result_identity,
        "context_identity": context_identity,
        "score": float(score if score is not None else 0.0),
        "rank": int(row.get("rank", 0) or 0),
        "facts": {str(key): str(value) for key, value in facts.items()},
    }


def _compile_context_model_resolution_inputs(
    manifest: Any,
    values: tuple[Mapping[str, Any], ...],
) -> tuple[Any, ...]:
    return tuple(_compile_context_model_resolution_input(manifest, value) for value in values)


def _single_context_model_resolution_input(
    manifest: Any,
    values: tuple[Mapping[str, Any], ...],
) -> Any | None:
    if not values:
        return None
    if len(values) != 1:
        raise CompilerFailure(
            "contract_invalid",
            "manifest-derived compiler execution supports exactly one resolution input",
        )
    return _compile_context_model_resolution_input(manifest, values[0])


def _compile_context_model_resolution_input(manifest: Any, value: Mapping[str, Any]) -> Any:
    payload = dict(value)
    retrieval_role = str(payload.pop("retrieval_role", payload.pop("role", "")))
    if not retrieval_role:
        raise CompilerFailure("contract_invalid", "resolution input requires retrieval_role")
    try:
        return compile_context_model_resolution_request(manifest, retrieval_role=retrieval_role, **payload)
    except TypeError as error:
        raise CompilerFailure("contract_invalid", f"invalid resolution input: {error}") from error


def _role_identities(
    *,
    observations: tuple[dict[str, Any], ...],
    outputs: tuple[dict[str, Any], ...],
    runtime_role_identities: Mapping[str, str | Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {}
    for row in observations:
        snapshot_identity = row.get("snapshot_identity") or row.get("canonical_identity")
        if row.get("snapshot_role") and snapshot_identity:
            _add_role_identity(values, str(row["snapshot_role"]), str(snapshot_identity))
        object_identity = row.get("object_identity")
        if row.get("object_role") and object_identity:
            _add_role_identity(values, str(row["object_role"]), str(object_identity))
    for row in outputs:
        identity = row.get("identity") or row.get("output_identity")
        role = row.get("role") or row.get("output_role")
        if role and identity:
            _add_role_identity(values, str(role), str(identity))
    for role, identities in runtime_role_identities.items():
        if isinstance(identities, str):
            _add_role_identity(values, role, identities)
        else:
            for identity in identities:
                _add_role_identity(values, role, str(identity))
    return {role: tuple(sorted(identities)) for role, identities in sorted(values.items())}


def _add_role_identity(values: dict[str, set[str]], role: str, identity: str) -> None:
    if not role.strip() or not identity.strip():
        return
    values.setdefault(role, set()).add(identity)


def _flatten_role_identities(values: Mapping[str, tuple[str, ...]]) -> set[str]:
    return {identity for identities in values.values() for identity in identities}


def _manifest_field(manifest: Any, key: str, default: Any = None) -> Any:
    if hasattr(manifest, key):
        return getattr(manifest, key)
    if isinstance(manifest, Mapping):
        value = dict(manifest)
        if key in value:
            return value[key]
        inner = value.get("manifest")
        if isinstance(inner, Mapping) and key in inner:
            return inner[key]
    return default


def _resolve_context_model_requests(
    *,
    requests: tuple[Any, ...],
    results: tuple[Any, ...],
    resolution_port: Any | None,
) -> tuple[tuple[Any, Any], ...]:
    if requests and resolution_port is not None:
        resolved = tuple(resolution_port.resolve(request) for request in requests)
    elif requests:
        if len(results) != len(requests):
            raise CompilerFailure("contract_invalid", "resolution results must match resolution requests")
        resolved = results
    else:
        if results:
            return tuple((None, result) for result in results)
        return ()
    pairs = tuple(zip(requests, resolved, strict=True))
    for request, result in pairs:
        _validate_context_model_resolution(request, result)
    return pairs


def _validate_context_model_resolution(request: Any, result: Any) -> None:
    if result.request_identity != request.request_identity or result.resolution_identity != request.identity:
        raise CompilerFailure("contract_invalid", "resolution result identity does not match request")
    expected = {item.candidate_identity for item in request.candidates}
    actual = {item.candidate_identity for item in result.decisions}
    if actual != expected or len(actual) != len(result.decisions):
        raise CompilerFailure("contract_invalid", "resolution must contain exactly one decision per candidate")
    ordered = tuple(sorted(result.decisions, key=lambda item: item.candidate_identity))
    if ordered != tuple(result.decisions):
        raise CompilerFailure("contract_invalid", "resolution decisions are not stably ordered")
    for decision in result.decisions:
        if (decision.as_of != request.as_of or decision.policy_id != request.policy_id or
                decision.policy_revision != request.policy_revision):
            raise CompilerFailure("contract_invalid", "decision basis does not match resolution request")
        if decision.outcome == "conflict" and not decision.conflict_set_id:
            raise CompilerFailure("contract_invalid", "conflict decision requires conflict_set_id")


def _resolution_relation_rows(pairs: Iterable[tuple[Any, Any]]) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    decisions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for request, result in pairs:
        candidate_by_id = {item.candidate_identity: item for item in request.candidates} if request is not None else {}
        if request is not None:
            for candidate in request.candidates:
                row = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else dict(candidate)
                retrieval = row.pop("retrieval", None)
                if isinstance(retrieval, Mapping):
                    row["retrieval_identity"] = str(retrieval.get("result_identity", ""))
                    row["semantic_score"] = float(retrieval.get("score", 0.0) or 0.0)
                    row["retrieval_rank"] = int(retrieval.get("rank", 0) or 0)
                row.setdefault("candidate_identity", candidate.candidate_identity)
                row.setdefault("context_identity", candidate.context_identity)
                row["request_identity"] = request.request_identity
                row["resolution_identity"] = result.resolution_identity
                row["task_id"] = request.task_id
                candidates.append(row)
        for decision in result.decisions:
            row = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else dict(decision)
            candidate_identity = str(row["candidate_identity"])
            row.setdefault("decision_identity", _decision_identity(
                request_identity=str(getattr(result, "request_identity", "")),
                candidate_identity=candidate_identity,
                decision=row,
            ))
            row["request_identity"] = result.request_identity
            row["resolution_identity"] = result.resolution_identity
            if candidate_identity in candidate_by_id:
                row["context_identity"] = candidate_by_id[candidate_identity].context_identity
            if row.get("outcome") == "selected":
                row["selected_identity"] = candidate_identity
            decisions.append(row)
    return tuple(decisions), tuple(candidates)


def _decision_identity(*, request_identity: str, candidate_identity: str, decision: Mapping[str, Any]) -> str:
    basis = dict(decision)
    basis.pop("decision_identity", None)
    return stable_identity({
        "request": request_identity,
        "candidate": candidate_identity,
        "decision": basis,
    })


def _compile_context_packages(specs: Iterable[Any]) -> tuple[ContextPackage, ...]:
    packages: list[ContextPackage] = []
    for spec in specs:
        if isinstance(spec, ContextPackage):
            packages.append(spec)
            continue
        payload = dict(spec)
        payload.setdefault("ir_version", "0.1")
        payload.setdefault("schema_revision", "1")
        payload["items"] = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in payload.get("items", ())
        ]
        packages.append(ContextPackage.model_validate(payload))
    return tuple(packages)


def _compile_context_packages_from_resolution_inputs(
    pairs: Iterable[tuple[Any, Any]],
    inputs: tuple[Mapping[str, Any], ...],
) -> tuple[ContextPackage, ...]:
    if not inputs:
        return ()
    pair_values = tuple(pairs)
    packages: list[ContextPackage] = []
    for raw in inputs:
        payload = dict(raw)
        request_identity = payload.get("request_identity")
        selected_pairs = tuple(
            (request, result) for request, result in pair_values
            if request is not None and (request_identity is None or request.request_identity == request_identity)
        )
        if not selected_pairs:
            raise CompilerFailure("contract_invalid", "package input references no resolved request")
        for request, result in selected_pairs:
            packages.append(_context_package_from_resolution(request, result, payload))
    return tuple(packages)


def _default_context_packages_from_resolution_pairs(pairs: Iterable[tuple[Any, Any]]) -> tuple[ContextPackage, ...]:
    packages: list[ContextPackage] = []
    for request, result in pairs:
        if request is None:
            continue
        if not any(decision.outcome == "selected" for decision in result.decisions):
            continue
        packages.append(_context_package_from_resolution(request, result, {
            "request_identity": request.request_identity,
            "created_at": request.as_of,
            "token_estimate": 1,
            "metadata": {"package_source": "context-model-lifecycle"},
        }))
    return tuple(packages)


def _context_package_from_resolution(request: Any, result: Any, payload: Mapping[str, Any]) -> ContextPackage:
    accepted = set(payload.get("accepted_outcomes", ("selected",)))
    candidate_by_id = {candidate.candidate_identity: candidate for candidate in request.candidates}
    decisions = tuple(decision for decision in result.decisions if decision.outcome in accepted)
    if not decisions and not bool(payload.get("allow_empty", False)):
        raise CompilerFailure("contract_invalid", "package input selected no resolution decisions")
    default_token_estimate = int(payload.get("token_estimate", payload.get("default_token_estimate", 0)))
    item_role = str(payload.get("item_role", "resolved-context"))
    items = tuple(
        ContextItem(
            representation_identity=candidate_by_id[decision.candidate_identity].context_identity,
            evidence_identities=candidate_by_id[decision.candidate_identity].evidence_identities,
            role=item_role,
            rank=index,
            token_estimate=default_token_estimate,
        )
        for index, decision in enumerate(decisions)
    )
    metadata = {
        "resolution_identity": str(result.resolution_identity),
        "resolution_request_identity": str(request.request_identity),
        **{str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
    }
    return ContextPackage(
        ir_version=str(payload.get("ir_version", "0.1")),
        task_id=str(payload.get("task_id", request.task_id)),
        schema_revision=str(payload.get("schema_revision", request.task_schema_revision)),
        items=items,
        budget_tokens=int(payload.get("budget_tokens", max(1, sum(item.token_estimate for item in items)))),
        created_at=payload.get("created_at", request.as_of),
        metadata=metadata,
    )


def _context_package_rows(packages: Iterable[ContextPackage]) -> tuple[dict[str, Any], ...]:
    return tuple({
        "package_identity": package.identity,
        "task_id": package.task_id,
        "package_kind": "task_context",
        "budget_tokens": package.budget_tokens,
        "metadata": package.metadata,
        "items": [item.model_dump(mode="json") for item in package.items],
    } for package in packages)


def _resolve_context_contract(manifest: Any, supplied: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    contracts = supplied or _manifest_contracts(manifest)
    if len(contracts) > 1:
        raise CompilerFailure("contract_invalid", "Context Package publication requires exactly one contract")
    raw = dict(contracts[0]) if contracts else _default_context_contract(manifest)
    rules = _contract_rules(raw)
    semantic_basis = {
        key: value for key, value in rules.items()
        if key not in {"display_name", "title", "description"}
    }
    contract_id = str(raw.get("contract_id", raw.get("id", "context-package-contract")))
    version = str(raw.get("contract_version", raw.get("version", raw.get("revision", stable_identity(semantic_basis)))))
    identity = str(raw.get("contract_identity") or raw.get("identity") or stable_identity({
        "contract_id": contract_id,
        "contract_version": version,
        "rules": semantic_basis,
    }))
    return {
        "contract_identity": identity,
        "contract_id": contract_id,
        "contract_version": version,
        "semantic_identity": stable_identity({"contract": contract_id, "version": version, "rules": semantic_basis}),
        "rules": rules,
        "rules_json": canonical_json(rules),
        "status": "active",
    }


def _manifest_contracts(manifest: Any) -> tuple[Mapping[str, Any], ...]:
    value = _manifest_field(manifest, "context_contracts")
    if value is None:
        value = _manifest_field(manifest, "contracts")
    if value is None:
        single = _manifest_field(manifest, "context_contract")
        value = () if single is None else (single,)
    if isinstance(value, Mapping):
        return (value,)
    return tuple(dict(item) for item in value or ())


def _default_context_contract(manifest: Any) -> dict[str, Any]:
    model_id = str(_manifest_field(manifest, "context_model_id", "context-model"))
    return {
        "contract_id": f"{model_id}.default-package-contract",
        "permitted_consumers": ["*"],
        "permitted_tasks": ["*"],
        "permitted_purposes": ["*"],
        "allowed_modalities": ["*"],
        "allowed_evidence": ["*"],
        "required_authority_level": "none",
        "valid_time_range": {},
        "historical_evidence_allowed": True,
        "max_items": 1000000,
        "max_tokens": 1000000000,
        "max_bytes": 1000000000,
        "citation_required": False,
        "freshness": {},
        "abstention_conditions": ["missing_authoritative_evidence"],
        "prohibited_uses": [],
    }


def _contract_rules(raw: Mapping[str, Any]) -> dict[str, Any]:
    rules = dict(raw.get("rules", raw))
    return {
        "permitted_consumers": _list_rule(rules.get("permitted_consumers", rules.get("permitted_agent_roles", rules.get("consumers", ("*",))))),
        "permitted_tasks": _list_rule(rules.get("permitted_tasks", rules.get("tasks", ("*",)))),
        "permitted_purposes": _list_rule(rules.get("permitted_purposes", rules.get("purposes", ("*",)))),
        "allowed_modalities": _list_rule(rules.get("allowed_modalities", ("*",))),
        "allowed_evidence": _list_rule(rules.get("allowed_evidence", ("*",))),
        "required_authority_level": str(rules.get("required_authority_level", "none")),
        "valid_time_range": dict(rules.get("valid_time_range", {})),
        "valid_as_of": rules.get("valid_as_of"),
        "historical_evidence_allowed": bool(rules.get("historical_evidence_allowed", rules.get("allow_historical", False))),
        "max_items": int(rules.get("max_items", 1000000)),
        "max_tokens": int(rules.get("max_tokens", rules.get("budget_tokens", 1000000000))),
        "max_bytes": int(rules.get("max_bytes", 1000000000)),
        "citation_required": bool(rules.get("citation_required", rules.get("citations_required", False))),
        "freshness": dict(rules.get("freshness", {})),
        "abstention_conditions": _list_rule(rules.get("abstention_conditions", ("missing_authoritative_evidence",))),
        "prohibited_uses": _list_rule(rules.get("prohibited_uses", ())),
    }


def _evaluate_context_package_contracts(
    *,
    contract: Mapping[str, Any],
    packages: tuple[Mapping[str, Any], ...],
    package_items: tuple[Mapping[str, Any], ...],
    outputs: tuple[Mapping[str, Any], ...],
    context: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    allowed: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    items_by_package: dict[str, list[dict[str, Any]]] = {}
    for item in package_items:
        value = dict(item)
        items_by_package.setdefault(str(value.get("package_identity", "")), []).append(value)
    output_by_identity = {
        str(row.get("identity", row.get("output_identity", ""))): dict(row)
        for row in outputs
        if row.get("identity") or row.get("output_identity")
    }
    for raw in packages:
        package = dict(raw)
        package_identity = str(package.get("package_identity", package.get("identity", stable_identity(package))))
        package["package_identity"] = package_identity
        items = tuple(
            dict(item)
            for item in package.get("items", items_by_package.get(package_identity, ()))
        )
        evaluation = _evaluate_context_package_contract(
            contract=contract,
            package=package,
            items=items,
            output_by_identity=output_by_identity,
            context=context,
        )
        evaluations.append(evaluation)
        if evaluation["decision"] == "allow":
            allowed.append(package)
            bindings.append({
                "binding_identity": stable_identity({
                    "package_identity": package_identity,
                    "contract_identity": evaluation["contract_identity"],
                    "contract_version": evaluation["contract_version"],
                    "evaluation_identity": evaluation["evaluation_identity"],
                }),
                "package_identity": package_identity,
                "contract_identity": evaluation["contract_identity"],
                "contract_version": evaluation["contract_version"],
                "evaluation_identity": evaluation["evaluation_identity"],
                "decision": "allow",
            })
    return tuple(allowed), tuple(bindings), tuple(evaluations)


def _evaluate_context_package_contract(
    *,
    contract: Mapping[str, Any],
    package: Mapping[str, Any],
    items: tuple[Mapping[str, Any], ...],
    output_by_identity: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    rules = dict(contract["rules"])
    metadata = dict(package.get("metadata", {})) if isinstance(package.get("metadata", {}), Mapping) else {}
    consumer = str(context.get("consumer", context.get("consumer_role", metadata.get("consumer", ""))))
    task = str(context.get("task", context.get("task_id", package.get("task_id", ""))))
    purpose = str(context.get("purpose", metadata.get("purpose", task)))
    violated: list[str] = []
    abstain: list[str] = []
    if not _allowed_value(consumer, rules["permitted_consumers"]):
        violated.append("unauthorized_consumer")
    if not _allowed_value(task, rules["permitted_tasks"]):
        violated.append("unsupported_task")
    if not _allowed_value(purpose, rules["permitted_purposes"]):
        violated.append("unsupported_purpose")
    if task in set(rules["prohibited_uses"]) or purpose in set(rules["prohibited_uses"]):
        violated.append("prohibited_use")
    if len(items) > int(rules["max_items"]):
        violated.append("max_items_exceeded")
    if _package_tokens(package, items, context) > int(rules["max_tokens"]):
        violated.append("max_tokens_exceeded")
    if _package_bytes(package, items, output_by_identity, context) > int(rules["max_bytes"]):
        violated.append("max_bytes_exceeded")
    if bool(rules["citation_required"]) and any(not _item_evidence_identities(item) for item in items):
        violated.append("missing_required_citations")
    if not bool(rules["historical_evidence_allowed"]) and any(_item_is_superseded(item, output_by_identity) for item in items):
        violated.append("superseded_evidence_rejected")
    if _freshness_expired(rules, package, items, output_by_identity, context):
        violated.append("freshness_expired")
    if not _package_has_required_authority(rules, package, items, output_by_identity, context):
        abstain.append("missing_authoritative_evidence")
    decision = "deny" if violated else "abstain" if abstain else "allow"
    violated_rules = violated if violated else abstain
    package_identity = str(package["package_identity"])
    evaluated_at = str(context.get("evaluated_at") or datetime.now(timezone.utc).isoformat())
    evaluator_identity = str(context.get("evaluator_identity", "igor.context-contract.v0.1"))
    evaluation_identity = stable_identity({
        "package_identity": package_identity,
        "contract_identity": contract["contract_identity"],
        "contract_version": contract["contract_version"],
        "decision": decision,
        "violated_rules": violated_rules,
        "evaluator_identity": evaluator_identity,
    })
    return {
        "evaluation_identity": evaluation_identity,
        "package_identity": package_identity,
        "contract_identity": str(contract["contract_identity"]),
        "contract_version": str(contract["contract_version"]),
        "decision": decision,
        "violated_rules": violated_rules,
        "explanation": "allowed" if decision == "allow" else ", ".join(violated_rules),
        "evaluated_at": evaluated_at,
        "evaluator_identity": evaluator_identity,
    }


def _list_rule(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _allowed_value(value: str, allowed: tuple[str, ...]) -> bool:
    return "*" in allowed or value in allowed


def _item_evidence_identities(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in item.get("evidence_identities", ()) if str(value))


def _package_tokens(package: Mapping[str, Any], items: tuple[Mapping[str, Any], ...],
                    context: Mapping[str, Any]) -> int:
    value = context.get("budget_tokens", package.get("budget_tokens"))
    if value is not None:
        return int(value)
    return sum(int(item.get("token_estimate", 0) or 0) for item in items)


def _package_bytes(package: Mapping[str, Any], items: tuple[Mapping[str, Any], ...],
                   output_by_identity: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any]) -> int:
    value = context.get("budget_bytes", package.get("byte_count", package.get("bytes")))
    if value is not None:
        return int(value)
    total = 0
    for item in items:
        if item.get("byte_estimate") is not None:
            total += int(item["byte_estimate"])
            continue
        output = output_by_identity.get(str(item.get("representation_identity", "")), {})
        total += int(output.get("byte_length", output.get("byte_count", 0)) or 0)
    return total


def _item_is_superseded(item: Mapping[str, Any], output_by_identity: Mapping[str, Mapping[str, Any]]) -> bool:
    if str(item.get("status", "")).lower() == "superseded":
        return True
    output = output_by_identity.get(str(item.get("representation_identity", "")), {})
    return str(output.get("status", "")).lower() == "superseded"


def _freshness_expired(
    rules: Mapping[str, Any],
    package: Mapping[str, Any],
    items: tuple[Mapping[str, Any], ...],
    output_by_identity: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> bool:
    freshness = dict(rules.get("freshness", {}))
    requested = _parse_time(context.get("as_of", context.get("requested_as_of", package.get("created_at"))))
    if requested is None:
        return False
    valid_range = dict(rules.get("valid_time_range", {}))
    starts_at = _parse_time(valid_range.get("start", valid_range.get("from")))
    ends_at = _parse_time(valid_range.get("end", valid_range.get("until", rules.get("valid_as_of"))))
    if starts_at is not None and requested < starts_at:
        return True
    if ends_at is not None and requested > ends_at:
        return True
    if not freshness.get("required", bool(freshness.get("max_age_seconds"))):
        return False
    for item in items:
        item_until = _parse_time(item.get("valid_until", item.get("expires_at")))
        output = output_by_identity.get(str(item.get("representation_identity", "")), {})
        output_until = _parse_time(output.get("valid_until", output.get("expires_at")))
        until = item_until or output_until
        if until is not None and requested > until:
            return True
    return False


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _package_has_required_authority(
    rules: Mapping[str, Any],
    package: Mapping[str, Any],
    items: tuple[Mapping[str, Any], ...],
    output_by_identity: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> bool:
    required = str(rules.get("required_authority_level", "none"))
    if required in {"", "none", "*"}:
        return True
    values = {
        str(context.get("authority_level", "")),
        str(package.get("authority_level", "")),
    }
    metadata = package.get("metadata", {})
    if isinstance(metadata, Mapping):
        values.add(str(metadata.get("authority_level", "")))
    for item in items:
        values.add(str(item.get("authority_level", "")))
        output = output_by_identity.get(str(item.get("representation_identity", "")), {})
        values.add(str(output.get("authority_level", output.get("authority_status", ""))))
    return required in values


def _package_items_for_allowed_packages(
    package_items: tuple[Mapping[str, Any], ...],
    allowed_package_ids: set[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(item) for item in package_items
        if str(item.get("package_identity", "")) in allowed_package_ids
    )


def _contract_lifecycle_lineage_edges(evaluations: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    edges: list[dict[str, Any]] = []
    for row in evaluations:
        package_identity = str(row.get("package_identity", ""))
        contract_identity = str(row.get("contract_identity", ""))
        evaluation_identity = str(row.get("evaluation_identity", ""))
        if package_identity and evaluation_identity:
            edges.append({
                "source_node_id": package_identity,
                "target_node_id": evaluation_identity,
                "edge_type": "evaluated_by_contract",
                "operation_identity": contract_identity,
            })
        if contract_identity and evaluation_identity:
            edges.append({
                "source_node_id": contract_identity,
                "target_node_id": evaluation_identity,
                "edge_type": "applied_contract",
                "operation_identity": contract_identity,
            })
    return tuple(edges)


def _standard_lifecycle_lineage_edges(
    *,
    resolution_decisions: Iterable[Mapping[str, Any]],
    resolution_candidates: Iterable[Mapping[str, Any]],
    packages: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    edges: list[dict[str, Any]] = []
    candidate_context = {
        str(candidate.get("candidate_identity", "")): str(candidate.get("context_identity", ""))
        for candidate in resolution_candidates
        if candidate.get("candidate_identity") and candidate.get("context_identity")
    }
    for decision in resolution_decisions:
        candidate_identity = str(decision.get("candidate_identity", ""))
        decision_identity = str(decision.get("decision_identity", ""))
        context_identity = str(decision.get("context_identity") or candidate_context.get(candidate_identity, ""))
        if not decision_identity or not context_identity:
            continue
        outcome = str(decision.get("outcome", decision.get("decision_status", "evaluated")))
        edges.append({
            "source_node_id": context_identity,
            "target_node_id": decision_identity,
            "edge_type": "selected_by_resolution" if outcome == "selected" else "evaluated_by_resolution",
            "operation_identity": str(decision.get("resolution_identity", decision.get("request_identity", ""))),
        })
    for package in packages:
        package_identity = str(package.get("package_identity", package.get("identity", "")))
        if not package_identity:
            continue
        metadata = package.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        operation_identity = str(metadata.get("resolution_identity", package_identity))
        for item in package.get("items", ()):
            item_value = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            representation_identity = str(item_value.get("representation_identity") or item_value.get("context_identity", ""))
            if not representation_identity:
                continue
            edges.append({
                "source_node_id": representation_identity,
                "target_node_id": package_identity,
                "edge_type": "included_in_package",
                "operation_identity": operation_identity,
            })
    return tuple(edges)


def _merge_outputs(
    *groups: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: dict[str, dict[str, Any]] = {}
    for group in groups:
        for raw in group:
            row = dict(raw)
            identity = str(row.get("identity", row.get("output_identity", "")))
            if not identity:
                continue
            current = rows.get(identity, {})
            current.update(row)
            rows[identity] = current
    return tuple(rows[key] for key in sorted(rows))


def _existing_context_outputs(store: LanceStore) -> tuple[dict[str, Any], ...]:
    if "context_outputs" not in store.names():
        return ()
    return tuple(dict(row) for row in store.read("context_outputs").to_pylist())


def _compiler_output_rows(
    request: CompilationRequest,
    compilation: Any,
    ports: CompilerPorts,
) -> tuple[dict[str, Any], ...]:
    read = getattr(ports.store, "read", None)
    rows: list[dict[str, Any]] = []
    nodes = compilation.artifact_records.get("plan", {}).get("nodes", ())
    for node in nodes:
        identity = str(node["output_identity"])
        row = {
            "identity": identity,
            "role": str(node["operation"]),
            "kind": "semantic_derivation" if str(node["operation"]).startswith(("embedding", "enrichment")) else "context_representation",
            "derived_from": list(node.get("input_identities", ())),
            "operation_name": str(node["operation"]),
            "status": compilation.outcomes.get(identity, "unknown"),
        }
        value = compilation.semantic_outputs.get(identity)
        if value is not None:
            row["value"] = _encoded_value(value)
        elif callable(read):
            try:
                row["value"] = canonical_json(read(identity))
            except KeyError:
                pass
        rows.append(row)
    if compilation.package_identity and callable(read):
        try:
            rows.append({
                "identity": compilation.package_identity,
                "role": "context_package",
                "kind": "context_package",
                "derived_from": [item.representation_identity for item in request.package_items],
                "operation_name": "compile_package",
                "status": "published",
                "value": canonical_json(read(compilation.package_identity)),
            })
        except KeyError:
            pass
    return tuple(rows)


def _encoded_value(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return canonical_json(value.model_dump(mode="json"))
    return canonical_json(value)


def _resolution_candidates(compilation: Any) -> tuple[dict[str, Any], ...]:
    request = compilation.artifact_records.get("resolution_request") or {}
    candidates = request.get("candidates") if isinstance(request, dict) else None
    if not candidates:
        return ()
    return tuple(dict(item) for item in candidates)


def _compilation_resolution_decision_rows(compilation: Any) -> tuple[dict[str, Any], ...]:
    request = compilation.artifact_records.get("resolution_request") or {}
    result = compilation.artifact_records.get("resolution_result") or {}
    candidates = request.get("candidates") if isinstance(request, Mapping) else ()
    candidate_by_id = {
        str(candidate.get("candidate_identity", "")): dict(candidate)
        for candidate in candidates or ()
        if candidate.get("candidate_identity")
    }
    raw_decisions = result.get("decisions") if isinstance(result, Mapping) else None
    if raw_decisions is None:
        raw_decisions = tuple(item.model_dump(mode="json") for item in compilation.resolution)
    request_identity = str(result.get("request_identity", request.get("request_identity", ""))) if isinstance(result, Mapping) else ""
    resolution_identity = str(result.get("resolution_identity", "")) if isinstance(result, Mapping) else ""
    rows: list[dict[str, Any]] = []
    for raw in raw_decisions:
        row = dict(raw)
        candidate_identity = str(row.get("candidate_identity", ""))
        row.setdefault("request_identity", request_identity)
        row.setdefault("resolution_identity", resolution_identity)
        row.setdefault("decision_identity", _decision_identity(
            request_identity=str(row.get("request_identity", "")),
            candidate_identity=candidate_identity,
            decision=row,
        ))
        if candidate_identity in candidate_by_id:
            row.setdefault("context_identity", candidate_by_id[candidate_identity].get("context_identity", ""))
        if row.get("outcome") == "selected":
            row.setdefault("selected_identity", candidate_identity)
        rows.append(row)
    return tuple(rows)


def _retrieval_candidate_rows(compilation: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for fact in getattr(compilation, "retrieval_facts", ()):
        row = fact.model_dump(mode="json") if hasattr(fact, "model_dump") else dict(fact)
        candidate_identity = str(row.get("result_identity") or stable_identity(row))
        rows.append({
            "candidate_identity": candidate_identity,
            "context_identity": str(row["context_identity"]),
            "retrieval_identity": candidate_identity,
            "retrieval_rank": int(row.get("rank", 0)),
            "semantic_score": float(row.get("score", 0.0)),
            "facts": dict(row.get("facts", {})),
        })
    return tuple(rows)


def _package_rows(request: CompilationRequest, compilation: Any) -> tuple[dict[str, Any], ...]:
    if not compilation.package_identity:
        return ()
    return ({
        "package_identity": compilation.package_identity,
        "task_id": request.task_id,
        "package_kind": "task_context",
        "items": [item.model_dump(mode="json") for item in request.package_items],
    },)
