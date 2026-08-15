from __future__ import annotations

import argparse
import json
import sys

from igor_fixtures.pack import generate_support_pack, validate_scenario_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igor-fixtures")
    commands = parser.add_subparsers(dest="command", required=True)
    scenario = commands.add_parser("scenario")
    actions = scenario.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--json", action="store_true")
    generate = actions.add_parser("generate")
    generate.add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.action == "generate":
            result = {"generated": str(generate_support_pack(args.path))}
        else:
            result = validate_scenario_pack(args.path)
        print(json.dumps(result, sort_keys=True) if getattr(args, "json", False) or args.action == "validate" else result["generated"])
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid scenario pack: {error}", file=sys.stderr)
        return 2
