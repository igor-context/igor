from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_architecture import EXPECTED_MODULES, validate_architecture


class ArchitectureHarnessTest(unittest.TestCase):
    def make_repository(self, root: Path) -> None:
        (root / ".gitignore").write_text("/.igor/\n", encoding="utf-8")
        (root / "AGENTS.md").write_text("# Agent contract\n", encoding="utf-8")
        (root / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
        for name in ("architecture", "contracts", "end-to-end"):
            harness = root / "harnesses" / name
            harness.mkdir(parents=True)
            harness.joinpath("README.md").write_text(f"# {name}\n", encoding="utf-8")
        services = "\n".join(
            f"  {name}:\n    image: fixture\n"
            for name in (
                "architecture-check",
                "architecture-test",
                "core-contract",
                "core-lock",
                "core-test",
                "context-compiler-test",
                "context-compiler-contract",
                "context-compiler-lock",
                "huggingface-test",
                "huggingface-qualification",
                "support-hf-smoke",
                "support-hf-evaluate",
                "support-hf-e2e",
                "support-hf-live-e2e",
                "evaluator-fixture",
                "evaluator-lock",
                "evaluator-test",
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
                "context-e2e",
            )
        )
        (root / "compose.yaml").write_text(
            "x-core-runtime: &core-runtime\n"
            "  build:\n"
            "    context: ./modules/core\n"
            f"services:\n{services}",
            encoding="utf-8",
        )

        for name in EXPECTED_MODULES:
            module_root = root / "modules" / name
            module_root.mkdir(parents=True)
            status = "active" if name == "core" else "planned"
            package = "igor_core" if name == "core" else f"igor_{name.replace('-', '_')}"
            module_root.joinpath("module.toml").write_text(
                "\n".join(
                    [
                        "schema_version = 1",
                        f'name = "{name}"',
                        f'status = "{status}"',
                        'capability = "fixture capability"',
                        f'python_package = "{package}"',
                        "allowed_module_imports = []",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            module_root.joinpath("MODULE.md").write_text(f"# {name}\n", encoding="utf-8")

        core = root / "modules" / "core"
        for name in ("Dockerfile", "pyproject.toml", "uv.lock"):
            core.joinpath(name).write_text("fixture\n", encoding="utf-8")
        package = core / "src" / "igor_core"
        package.mkdir(parents=True)
        package.joinpath("__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        core.joinpath("tests").mkdir()

    def activate_module(self, root: Path, name: str, package: str) -> Path:
        module_root = root / "modules" / name
        descriptor = module_root / "module.toml"
        descriptor.write_text(
            descriptor.read_text(encoding="utf-8").replace('status = "planned"', 'status = "active"'),
            encoding="utf-8",
        )
        for filename in ("Dockerfile", "pyproject.toml", "uv.lock"):
            module_root.joinpath(filename).write_text("fixture\n", encoding="utf-8")
        package_root = module_root / "src" / package
        package_root.mkdir(parents=True)
        package_root.joinpath("__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        module_root.joinpath("tests").mkdir()
        return package_root

    def test_valid_repository_passes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)

            self.assertEqual(validate_architecture(root), [])

    def test_nested_compose_file_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            nested = root / "modules" / "dlt" / "compose.yaml"
            nested.write_text("services: {}\n", encoding="utf-8")

            errors = validate_architecture(root)

            self.assertTrue(any("nested Compose file" in error for error in errors))

    def test_tool_managed_worktree_compose_is_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            nested = root / ".claude" / "worktrees" / "agent-1" / "compose.yaml"
            nested.parent.mkdir(parents=True)
            nested.write_text("services: {}\n", encoding="utf-8")

            self.assertEqual(validate_architecture(root), [])

    def test_core_tool_import_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            source = root / "modules" / "core" / "src" / "igor_core" / "__init__.py"
            source.write_text("import lancedb\n", encoding="utf-8")

            errors = validate_architecture(root)

            self.assertTrue(any("forbidden reference tool lancedb" in error for error in errors))

    def test_unregistered_module_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            (root / "modules" / "misc").mkdir()

            errors = validate_architecture(root)

            self.assertIn("unregistered capability module: modules/misc", errors)

    def test_missing_contract_harness_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            (root / "harnesses" / "contracts" / "README.md").unlink()
            (root / "harnesses" / "contracts").rmdir()

            errors = validate_architecture(root)

            self.assertIn("required harness is missing: harnesses/contracts", errors)

    def test_active_module_requires_lock(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            (root / "modules" / "core" / "uv.lock").unlink()

            errors = validate_architecture(root)

            self.assertIn("active modules/core is missing uv.lock", errors)

    def test_planned_module_cannot_expose_runtime_scaffold(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            planned = root / "modules" / "context-compiler"
            planned.joinpath("pyproject.toml").write_text("fixture\n", encoding="utf-8")

            errors = validate_architecture(root)

            self.assertIn(
                "planned modules/context-compiler exposes runtime path pyproject.toml",
                errors,
            )

    def test_host_python_instruction_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            (root / "AGENTS.md").write_text(
                "Run python3 .agents/skills/check.py\n",
                encoding="utf-8",
            )

            errors = validate_architecture(root)

            self.assertIn("agent instruction requires host Python: AGENTS.md", errors)

    def test_commented_core_context_does_not_satisfy_build_scope(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            compose = (root / "compose.yaml").read_text(encoding="utf-8")
            (root / "compose.yaml").write_text(
                compose.replace("    context: ./modules/core", "    # context: ./modules/core"),
                encoding="utf-8",
            )

            errors = validate_architecture(root)

            self.assertIn("core image must build from modules/core only", errors)

    def test_standard_context_relations_are_written_only_by_lifecycle_owner(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            runner_package = self.activate_module(root, "runner", "igor_runner")
            runner_package.joinpath("scenario.py").write_text(
                "def write(store):\n"
                "    store.replace('context_packages', [])\n",
                encoding="utf-8",
            )

            errors = validate_architecture(root)

            self.assertTrue(
                any("writes standard Context Model relation context_packages" in error for error in errors)
            )

    def test_standard_context_relation_keyword_write_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            runner_package = self.activate_module(root, "runner", "igor_runner")
            runner_package.joinpath("scenario.py").write_text(
                "def write(store):\n"
                "    store.create(name='resolution_decisions', rows=[])\n",
                encoding="utf-8",
            )

            errors = validate_architecture(root)

            self.assertTrue(
                any("writes standard Context Model relation resolution_decisions" in error for error in errors)
            )

    def test_project_cli_cannot_import_runtime_tool_modules(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repository(root)
            runner_package = self.activate_module(root, "runner", "igor_runner")
            runner_package.joinpath("project.py").write_text(
                "import igor_lancedb\n",
                encoding="utf-8",
            )
            runner_package.joinpath("project_cli.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )

            errors = validate_architecture(root)

            self.assertTrue(
                any("imports forbidden project CLI dependency igor_lancedb" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
