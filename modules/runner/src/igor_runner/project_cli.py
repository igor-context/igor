"""User-facing IGOR project CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from igor_runner.project import (
    ProjectError,
    compile_project,
    deps_project,
    init_project,
    package_project,
    plan_project,
    project_command,
    render_diagnostics,
    validate_project,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="igor")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="initialize a new IGOR Context Project")
    init.add_argument("project_name")
    init.add_argument("--path")
    init.add_argument("--template", help="copy an existing IGOR project template into the new project")

    commands.add_parser("deps", help="resolve local project dependencies and write igor.lock")

    validate = commands.add_parser("validate", help="validate project definitions without runtime side effects")
    validate.add_argument("--select")

    compile_cmd = commands.add_parser("compile", help="compile one immutable resolved manifest")
    compile_cmd.add_argument("--select")
    compile_cmd.add_argument("--target")

    plan = commands.add_parser("plan", help="preview execution without calling providers or sources")
    plan.add_argument("--select")
    plan.add_argument("--target")
    plan.add_argument("--live", action="store_true")

    package = commands.add_parser("package", help="publish one contract-governed Context Package from a runtime request")
    package.add_argument("--request", required=True)
    package.add_argument("--select")
    package.add_argument("--target")

    for name, help_text in (
        ("sync", "summarize project synchronization inputs and inventories"),
        ("build", "build the resolved project bundle"),
        ("diff", "show the manifest diff against the previous target"),
        ("resolve", "resolve the manifest and referenced declarations"),
        ("query", "describe the queryable project catalog"),
        ("explain", "render the compile explanation and report"),
        ("test", "run deterministic project validation"),
        ("evaluate", "summarize the evaluation plan"),
        ("qualify", "combine validation, build, and diff summaries"),
        ("observe", "describe the current project inventory"),
        ("tune", "summarize tuning hints and live profile checks"),
        ("serve", "preview the project serving state"),
        ("status", "summarize project state"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--select")
        command.add_argument("--target")
        if name == "query":
            command.add_argument("--sql", help="execute analytical SQL against a project Lance relation store")
            command.add_argument(
                "--store",
                help="project-relative Lance relation store; defaults to .igor/runtime/qualification/allowed-relations",
            )
        if name in {"sync", "evaluate", "qualify", "tune", "serve"}:
            command.add_argument("--live", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            root = init_project(args.project_name, path=args.path, template=args.template)
            print(f"initialized IGOR project at {root}")
            return 0
        if args.command == "deps":
            lock = deps_project(".")
            print(f"wrote igor.lock ({lock.lock_identity})")
            return 0
        if args.command == "validate":
            errors, warnings = validate_project(".", select=args.select)
            if errors or warnings:
                print(render_diagnostics(errors, warnings))
            if errors:
                return 2
            print("project is valid")
            return 0
        if args.command == "compile":
            result = compile_project(".", select=args.select, target=args.target)
            print(
                json.dumps(
                    {
                        "manifest_identity": result.manifest["manifest_identity"],
                        "graph_nodes": len(result.graph["nodes"]),
                        "evaluations": len(result.evaluation_plan["evaluations"]),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "plan":
            result = plan_project(".", select=args.select, target=args.target, live=args.live)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "package":
            result = package_project(".", request=args.request, select=args.select, target=args.target)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command in {"sync", "build", "diff", "resolve", "query", "explain", "test", "evaluate", "qualify", "observe", "tune", "serve", "status"}:
            result = project_command(
                ".",
                args.command,
                select=args.select,
                target=args.target,
                live=getattr(args, "live", False),
                sql=getattr(args, "sql", None),
                store=getattr(args, "store", None),
            )
            print(json.dumps(result, sort_keys=True))
            return 0
    except ProjectError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 1
