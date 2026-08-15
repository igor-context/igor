"""Repository-level architecture checks for IGOR capability ownership."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

EXPECTED_MODULES = {
    "context-compiler",
    "core",
    "datafusion",
    "dbt-lance",
    "dlt",
    "evaluator",
    "fixtures",
    "huggingface",
    "lancedb",
    "relational",
    "runner",
}
EXPECTED_HARNESSES = {"architecture", "contracts", "end-to-end"}
VALID_STATUSES = {"active", "planned"}
ACTIVE_FILES = {"Dockerfile", "MODULE.md", "module.toml", "pyproject.toml", "uv.lock"}
PLANNED_RUNTIME_PATHS = {"Dockerfile", "pyproject.toml", "src", "tests", "uv.lock"}
CORE_FORBIDDEN_IMPORTS = {
    "datafusion",
    "dbt",
    "dlt",
    "duckdb",
    "lancedb",
    "polars",
}
PROJECT_CLI_FORBIDDEN_IMPORTS = {
    "igor_datafusion",
    "igor_dlt",
    "igor_lancedb",
    "igor_relational",
    "igor_runner.compose",
}
PROJECT_CLI_PATHS = {
    Path("modules/runner/src/igor_runner/project.py"),
    Path("modules/runner/src/igor_runner/project_cli.py"),
}
STANDARD_CONTEXT_RELATIONS = {
    "authority_policies",
    "context_assertions",
    "context_catalog",
    "context_contracts",
    "context_models",
    "context_objects",
    "context_outputs",
    "context_packages",
    "context_retrieval",
    "context_snapshots",
    "contract_evaluations",
    "lineage_edges",
    "lineage_nodes",
    "package_items",
    "context_package_contracts",
    "resolution_candidates",
    "resolution_decisions",
}
STANDARD_RELATION_WRITE_METHODS = {"add", "create", "replace"}
IGNORED_REPOSITORY_PREFIXES = (Path(".git"), Path(".igor"), Path(".claude/worktrees"))
STANDARD_RELATION_WRITE_ALLOWLIST = {
    Path("modules/lancedb/src/igor_lancedb/context.py"),
    Path("modules/runner/src/igor_runner/context_model_lifecycle.py"),
}
REQUIRED_COMPOSE_SERVICES = {
    "architecture-check",
    "architecture-test",
    "core-contract",
    "context-compiler-test",
    "context-compiler-contract",
    "context-compiler-lock",
    "core-lock",
    "core-test",
    "evaluator-lock",
    "evaluator-test",
    "evaluator-fixture",
    "lancedb-lock",
    "lancedb-test",
    "dlt-lock",
    "dlt-test",
    "datafusion-lock",
    "datafusion-test",
    "relational-lock",
    "relational-test",
    "relational-smoke",
    "runner-lock",
    "runner-test",
    "runner-smoke",
    "huggingface-test",
    "huggingface-qualification",
    "support-hf-smoke",
    "support-hf-evaluate",
    "support-hf-e2e",
    "support-hf-live-e2e",
    "context-e2e",
}


@dataclass(frozen=True)
class ModuleDescriptor:
    name: str
    status: str
    capability: str
    python_package: str
    allowed_module_imports: frozenset[str]
    root: Path

    @property
    def package_root(self) -> str:
        return self.python_package.split(".", 1)[0]


def load_descriptor(module_root: Path) -> ModuleDescriptor:
    descriptor_path = module_root / "module.toml"
    data = tomllib.loads(descriptor_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("module.toml schema_version must be 1")

    return ModuleDescriptor(
        name=data["name"],
        status=data["status"],
        capability=data["capability"],
        python_package=data["python_package"],
        allowed_module_imports=frozenset(data.get("allowed_module_imports", [])),
        root=module_root,
    )


def imported_roots(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def standard_relation_writes(source: Path) -> list[tuple[int, str]]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    writes: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in STANDARD_RELATION_WRITE_METHODS:
            continue
        relation: ast.expr | None = None
        if not node.args:
            for keyword in node.keywords:
                if keyword.arg == "name":
                    relation = keyword.value
                    break
        else:
            relation = node.args[0]
        if isinstance(relation, ast.Constant) and relation.value in STANDARD_CONTEXT_RELATIONS:
            writes.append((node.lineno, str(relation.value)))
    return writes


def find_nested_compose_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in ("compose*.yaml", "compose*.yml", "docker-compose*.yaml", "docker-compose*.yml"):
        for candidate in root.rglob(pattern):
            relative = candidate.relative_to(root)
            if any(relative == prefix or prefix in relative.parents for prefix in IGNORED_REPOSITORY_PREFIXES):
                continue
            if relative != Path("compose.yaml"):
                matches.append(candidate)
    return sorted(set(matches))


def top_level_yaml_block(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    key_pattern = re.compile(rf"{re.escape(key)}:(?:\s+.*)?")
    start = next((index for index, line in enumerate(lines) if key_pattern.fullmatch(line)), None)
    if start is None:
        return []

    block: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        block.append(line)
    return block


def compose_service_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in top_level_yaml_block(text, "services"):
        match = re.fullmatch(r"  ([a-z0-9][a-z0-9-]*):", line)
        if match:
            names.add(match.group(1))
    return names


def validate_architecture(root: Path) -> list[str]:
    errors: list[str] = []
    modules_root = root / "modules"
    harnesses_root = root / "harnesses"

    if not (root / "compose.yaml").is_file():
        errors.append("repository must contain one root compose.yaml")
    else:
        compose_text = (root / "compose.yaml").read_text(encoding="utf-8")
        services = compose_service_names(compose_text)
        for service in sorted(REQUIRED_COMPOSE_SERVICES - services):
            errors.append(f"compose.yaml is missing service {service}")
        core_runtime = top_level_yaml_block(compose_text, "x-core-runtime")
        if not any(line.strip() == "context: ./modules/core" for line in core_runtime):
            errors.append("core image must build from modules/core only")

    for nested in find_nested_compose_files(root):
        errors.append(f"nested Compose file is forbidden: {nested.relative_to(root)}")

    gitignore = root / ".gitignore"
    if not gitignore.is_file() or "/.igor/" not in gitignore.read_text(encoding="utf-8"):
        errors.append(".gitignore must exclude the generated /.igor/ workspace")

    instruction_files = [root / "AGENTS.md", root / "CLAUDE.md"]
    instruction_files.extend(sorted((root / ".agents" / "skills").glob("*/SKILL.md")))
    for instruction in instruction_files:
        if instruction.is_file() and re.search(
            r"\bpython(?:3(?:\.\d+)?)?\s+\.agents/",
            instruction.read_text(encoding="utf-8"),
        ):
            errors.append(
                f"agent instruction requires host Python: {instruction.relative_to(root)}"
            )

    if not modules_root.is_dir():
        return [*errors, "modules/ directory is missing"]

    if not harnesses_root.is_dir():
        errors.append("harnesses/ directory is missing")
    else:
        actual_harnesses = {path.name for path in harnesses_root.iterdir() if path.is_dir()}
        for missing in sorted(EXPECTED_HARNESSES - actual_harnesses):
            errors.append(f"required harness is missing: harnesses/{missing}")
        for name in sorted(EXPECTED_HARNESSES & actual_harnesses):
            if not (harnesses_root / name / "README.md").is_file():
                errors.append(f"harnesses/{name} is missing README.md")

    actual_modules = {path.name for path in modules_root.iterdir() if path.is_dir()}
    for missing in sorted(EXPECTED_MODULES - actual_modules):
        errors.append(f"required capability module is missing: modules/{missing}")
    for extra in sorted(actual_modules - EXPECTED_MODULES):
        errors.append(f"unregistered capability module: modules/{extra}")

    descriptors: dict[str, ModuleDescriptor] = {}
    for name in sorted(EXPECTED_MODULES & actual_modules):
        module_root = modules_root / name
        descriptor_path = module_root / "module.toml"
        module_doc = module_root / "MODULE.md"
        if not descriptor_path.is_file():
            errors.append(f"modules/{name} is missing module.toml")
            continue
        if not module_doc.is_file():
            errors.append(f"modules/{name} is missing MODULE.md")

        try:
            descriptor = load_descriptor(module_root)
        except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
            errors.append(f"modules/{name}/module.toml is invalid: {error}")
            continue

        descriptors[name] = descriptor
        if descriptor.name != name:
            errors.append(f"modules/{name}/module.toml declares name {descriptor.name}")
        if descriptor.status not in VALID_STATUSES:
            errors.append(f"modules/{name} has invalid status {descriptor.status}")
        if not descriptor.capability.strip():
            errors.append(f"modules/{name} must declare a capability")
        unknown_imports = descriptor.allowed_module_imports - EXPECTED_MODULES
        if unknown_imports:
            errors.append(f"modules/{name} allows unknown modules: {sorted(unknown_imports)}")
        if name in descriptor.allowed_module_imports:
            errors.append(f"modules/{name} cannot list itself as an allowed module import")

        if descriptor.status == "active":
            for required in sorted(ACTIVE_FILES):
                if not (module_root / required).exists():
                    errors.append(f"active modules/{name} is missing {required}")
            expected_source = module_root / "src" / Path(*descriptor.python_package.split("."))
            if not expected_source.is_dir():
                errors.append(
                    f"active modules/{name} is missing package path "
                    f"{expected_source.relative_to(root)}"
                )
            if not (module_root / "tests").is_dir():
                errors.append(f"active modules/{name} is missing tests/")
        else:
            for forbidden in sorted(PLANNED_RUNTIME_PATHS):
                runtime_path = module_root / forbidden
                exposes_runtime = runtime_path.is_file() or (
                    runtime_path.is_dir()
                    and any(candidate.is_file() for candidate in runtime_path.rglob("*"))
                )
                if exposes_runtime:
                    errors.append(
                        f"planned modules/{name} exposes runtime path {forbidden}"
                    )

    package_owners = {
        descriptor.package_root: descriptor.name
        for descriptor in descriptors.values()
    }
    for name, descriptor in sorted(descriptors.items()):
        if descriptor.status != "active":
            continue
        source_root = descriptor.root / "src"
        for source in sorted(source_root.rglob("*.py")):
            relative_source = source.relative_to(root)
            try:
                imports = imported_roots(source)
            except SyntaxError as error:
                errors.append(f"invalid Python syntax in {relative_source}: {error}")
                continue

            for imported_root in sorted(imports):
                owner = package_owners.get(imported_root)
                if owner and owner != name and owner not in descriptor.allowed_module_imports:
                    errors.append(
                        f"{relative_source} imports module {owner} without declaring it"
                    )

            if name == "core":
                forbidden = imports & CORE_FORBIDDEN_IMPORTS
                for imported_root in sorted(forbidden):
                    errors.append(
                        f"core imports forbidden reference tool {imported_root} in "
                        f"{relative_source}"
                    )

            if relative_source in PROJECT_CLI_PATHS:
                for imported_root in sorted(imports & PROJECT_CLI_FORBIDDEN_IMPORTS):
                    errors.append(
                        f"{relative_source} imports forbidden project CLI dependency "
                        f"{imported_root}"
                    )

            if relative_source not in STANDARD_RELATION_WRITE_ALLOWLIST:
                for lineno, relation in standard_relation_writes(source):
                    errors.append(
                        f"{relative_source}:{lineno} writes standard Context Model relation "
                        f"{relation} outside the lifecycle/materializer owner"
                    )

    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: validate_architecture.py ROOT", file=sys.stderr)
        return 2

    root = Path(args[0]).resolve()
    errors = validate_architecture(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("IGOR architecture validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
