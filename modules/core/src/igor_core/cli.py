"""Container-facing command interface for IGOR core contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import yaml
from pydantic import ValidationError

from igor_core.contracts import (
    ArtifactIndex,
    ArtifactReference,
    RunManifest,
    StageResult,
    validate_run_directory,
)
from igor_core.profiles import load_model_profile, load_reference_profile
from igor_core.semantic import ContextPackage, validate_context_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="igor-core")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="model and compiler profile operations")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    validate = profile_commands.add_parser("validate", help="validate one YAML profile")
    validate.add_argument("path")
    validate.add_argument("--json", action="store_true", dest="as_json")
    validate_model = profile_commands.add_parser(
        "validate-model", help="validate one independent model profile"
    )
    validate_model.add_argument("path")
    validate_model.add_argument(
        "--capability", choices=("embedding", "completion"), required=True
    )
    validate_model.add_argument("--json", action="store_true", dest="as_json")

    contract = commands.add_parser("contract", help="run-artifact contract operations")
    contract_commands = contract.add_subparsers(dest="contract_command", required=True)
    contract_validate = contract_commands.add_parser("validate", help="validate one run directory")
    contract_validate.add_argument("path")
    contract_validate.add_argument("--json", action="store_true", dest="as_json")
    contract_schema = contract_commands.add_parser("schema", help="export JSON schemas")
    contract_schema.add_argument("--output", help="write schemas to a file instead of stdout")
    context = commands.add_parser("context", help="Context IR operations")
    context_commands = context.add_subparsers(dest="context_command", required=True)
    context_validate = context_commands.add_parser("validate", help="validate one ContextPackage JSON artifact")
    context_validate.add_argument("path")
    context_validate.add_argument("--json", action="store_true", dest="as_json")
    context_schema = context_commands.add_parser("schema", help="export the Context IR JSON schema")
    context_schema.add_argument("--output", help="write the schema to a file instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "profile" and args.profile_command == "validate-model":
            model_profile = load_model_profile(
                args.path, expected_capability=args.capability
            )
            result = {
                "capability": model_profile.capability,
                "profile_id": model_profile.profile_id,
                "profile_identity": model_profile.identity,
                "schema_version": model_profile.schema_version,
                "valid": True,
            }
            if args.as_json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(
                    f"valid {model_profile.capability} profile "
                    f"{model_profile.profile_id} ({model_profile.identity})"
                )
            return 0
        if args.command == "profile" and args.profile_command == "validate":
            profile = load_reference_profile(args.path)
            result = {
                "profile_id": profile.profile_id,
                "profile_identity": profile.identity,
                "embedding_profile_id": profile.embedding.profile_id,
                "completion_profile_id": profile.completion.profile_id,
                "schema_version": profile.schema_version,
                "valid": True,
            }
            if args.as_json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(f"valid reference profile {profile.profile_id} ({profile.identity})")
            return 0
        if args.command == "contract" and args.contract_command == "validate":
            package = validate_run_directory(args.path)
            result = {
                "valid": True,
                "run_identity": package.manifest.identity,
                "artifact_count": len(package.artifact_index.artifacts),
                "stage_count": len(package.stages),
            }
            if args.as_json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(
                    f"valid run {package.manifest.identity} "
                    f"({result['artifact_count']} artifacts, {result['stage_count']} stages)"
                )
            return 0
        if args.command == "contract" and args.contract_command == "schema":
            schemas = {
                "artifact-index": ArtifactIndex.model_json_schema(),
                "artifact-reference": ArtifactReference.model_json_schema(),
                "run-manifest": RunManifest.model_json_schema(),
                "stage-result": StageResult.model_json_schema(),
            }
            payload = json.dumps(schemas, indent=2, sort_keys=True) + "\n"
            if args.output:
                with open(args.output, "w", encoding="utf-8") as stream:
                    stream.write(payload)
            else:
                print(payload, end="")
            return 0
        if args.command == "context" and args.context_command == "validate":
            package = validate_context_artifact(args.path)
            result = {
                "valid": True,
                "package_identity": package.identity,
                "task_id": package.task_id,
                "item_count": len(package.items),
            }
            if args.as_json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(f"valid context package {package.identity} ({result['item_count']} items)")
            return 0
        if args.command == "context" and args.context_command == "schema":
            payload = json.dumps(ContextPackage.model_json_schema(), indent=2, sort_keys=True) + "\n"
            if args.output:
                with open(args.output, "w", encoding="utf-8") as stream:
                    stream.write(payload)
            else:
                print(payload, end="")
            return 0
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as error:
        print(f"invalid IGOR contract: {error}", file=sys.stderr)
        return 2

    return 1
