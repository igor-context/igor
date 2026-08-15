from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from igor_evaluator.evaluate import evaluate_authoritative_context_qualification, evaluate_document_live_artifact, evaluate_image_artifact, evaluate_semantic_sql_artifact, evaluate_submission, write_demo_run, write_static_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igor-evaluator")
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("output")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("scenario_pack")
    evaluate.add_argument("run_directory")
    evaluate.add_argument("--json", action="store_true")
    evaluate.add_argument("--output")
    report = commands.add_parser("report")
    report.add_argument("run_directory")
    report.add_argument("evaluation_json")
    report.add_argument("output_html")
    semantic = commands.add_parser("semantic-sql")
    semantic.add_argument("artifact")
    semantic.add_argument("--output")
    document = commands.add_parser("document-live")
    document.add_argument("artifact")
    document.add_argument("judgments")
    document.add_argument("--output")
    image = commands.add_parser("image")
    image.add_argument("artifact")
    image.add_argument("--judgments")
    image.add_argument("--output")
    authoritative = commands.add_parser("authoritative-context")
    authoritative.add_argument("artifact")
    authoritative.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "fixture":
        write_demo_run(args.output)
        print(args.output)
        return 0
    if args.command == "report":
        write_static_report(args.run_directory, args.evaluation_json, args.output_html)
        return 0
    if args.command == "semantic-sql":
        result = evaluate_semantic_sql_artifact(args.artifact)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0 if result["conforming"] else 1
    if args.command == "document-live":
        result = evaluate_document_live_artifact(args.artifact, args.judgments)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0 if result["conforming"] else 1
    if args.command == "image":
        result = evaluate_image_artifact(args.artifact, args.judgments)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0 if result["conforming"] else 1
    if args.command == "authoritative-context":
        result = evaluate_authoritative_context_qualification(args.artifact)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    try:
        result = evaluate_submission(args.scenario_pack, args.run_directory)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid evaluation input: {error}", file=sys.stderr)
        return 2
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"scenario={result['scenario_id']} conforming={result['conforming']} quality_threshold_met={result['quality_threshold_met']}")
        for check in result["conformance"]:
            print(f"{check['check_id']}: {'pass' if check['passed'] else 'FAIL'} — {check['evidence']}")
    return 0 if result["conforming"] else 1
