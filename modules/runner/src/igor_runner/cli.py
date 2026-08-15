from __future__ import annotations

import argparse
import json
import sys

from igor_runner.compose import RunnerError, run_context_e2e, run_image_context, run_support_smoke, run_support_huggingface_smoke, run_support_huggingface_e2e
from igor_runner.document_qualification import qualify_document
from igor_runner.recare_qualification import qualify_recare, qualify_recare_live
from igor_runner.semantic_sql_qualification import qualify_live
from igor_runner.image_qualification import qualify_huggingface_images
from igor_runner.standard_query import query_standard_relations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igor-runner")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("support-smoke")
    smoke.add_argument("scenario")
    smoke.add_argument("profile")
    smoke.add_argument("output")
    hf = commands.add_parser("support-huggingface-smoke")
    hf.add_argument("scenario")
    hf.add_argument("profile")
    hf.add_argument("acquisition")
    hf.add_argument("output")
    hf_e2e = commands.add_parser("support-huggingface-e2e")
    hf_e2e.add_argument("scenario")
    hf_e2e.add_argument("profile")
    hf_e2e.add_argument("acquisition")
    hf_e2e.add_argument("output")
    document = commands.add_parser("document-qualification")
    document.add_argument("scenario")
    document.add_argument("output")
    document.add_argument("acquisition", nargs="?")
    live_document = commands.add_parser("document-live-qualification")
    live_document.add_argument("scenario")
    live_document.add_argument("acquisition")
    live_document.add_argument("output")
    live_document.add_argument("profile")
    e2e = commands.add_parser("context-e2e")
    e2e.add_argument("scenario")
    e2e.add_argument("profile")
    e2e.add_argument("output")
    image = commands.add_parser("image-context")
    image.add_argument("profile")
    image.add_argument("output")
    image.add_argument("--mutation", action="store_true")
    semantic_live = commands.add_parser("semantic-sql-live")
    semantic_live.add_argument("profile")
    semantic_live.add_argument("output")
    semantic_live.add_argument("lance_root")
    image_hf = commands.add_parser("image-huggingface")
    image_hf.add_argument("scenario")
    image_hf.add_argument("acquisition")
    image_hf.add_argument("profile")
    image_hf.add_argument("output")
    recare = commands.add_parser("recare-qualification")
    recare.add_argument("scenario")
    recare.add_argument("output")
    recare_live = commands.add_parser("recare-qualification-live")
    recare_live.add_argument("scenario")
    recare_live.add_argument("profile")
    recare_live.add_argument("acquisition")
    recare_live.add_argument("output")
    query = commands.add_parser("query-standard-relations")
    query.add_argument("store")
    query.add_argument("sql")
    args = parser.parse_args(argv)
    try:
        if args.command == "context-e2e":
            print(json.dumps(run_context_e2e(args.scenario, args.profile, args.output), sort_keys=True))
            return 0
        if args.command == "image-context":
            print(json.dumps(run_image_context(args.profile, args.output, mutation=args.mutation), sort_keys=True))
            return 0
        if args.command == "semantic-sql-live":
            print(json.dumps(qualify_live(args.profile, args.output, args.lance_root), sort_keys=True))
            return 0
        if args.command == "image-huggingface":
            print(json.dumps(qualify_huggingface_images(args.scenario, args.acquisition, args.profile, args.output), sort_keys=True))
            return 0
        if args.command == "support-huggingface-smoke":
            package = run_support_huggingface_smoke(args.scenario, args.profile, args.acquisition, args.output)
            print(json.dumps({"valid": True, "run_identity": package.manifest.identity}, sort_keys=True))
            return 0
        if args.command == "support-huggingface-e2e":
            print(json.dumps(run_support_huggingface_e2e(args.scenario, args.profile, args.acquisition, args.output), sort_keys=True))
            return 0
        if args.command == "recare-qualification":
            print(json.dumps(qualify_recare(args.scenario, args.output), sort_keys=True))
            return 0
        if args.command == "recare-qualification-live":
            print(json.dumps(qualify_recare_live(args.scenario, args.profile, args.acquisition, args.output), sort_keys=True))
            return 0
        if args.command == "query-standard-relations":
            print(json.dumps(query_standard_relations(args.store, args.sql), sort_keys=True))
            return 0
        if args.command == "document-qualification":
            print(json.dumps(qualify_document(args.scenario, args.output, args.acquisition), sort_keys=True))
            return 0
        if args.command == "document-live-qualification":
            from igor_runner.document_qualification import qualify_document_live
            print(json.dumps(qualify_document_live(args.scenario, args.acquisition, args.output, args.profile), sort_keys=True))
            return 0
        package = run_support_smoke(args.scenario, args.profile, args.output)
    except (OSError, RunnerError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"runner failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "run_identity": package.manifest.identity, "stages": [stage.stage_id for stage in package.stages]}, sort_keys=True))
    return 0
