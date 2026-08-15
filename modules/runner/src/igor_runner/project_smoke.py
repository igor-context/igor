"""Generated-project smoke qualification for the IGOR project CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _inventory(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m igor_runner.project_smoke OUTPUT_DIR")
    base = Path(args[0]).resolve()
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        base.mkdir(parents=True)
    project_root = base / "generated-project"
    results: list[dict[str, object]] = []

    commands = [
        (["igor", "init", "generated-project", "--path", str(base)], base),
        (["igor", "deps"], project_root),
        (["igor", "validate"], project_root),
        (["igor", "compile"], project_root),
        (["igor", "plan"], project_root),
        (["igor", "package", "--request", "package_request.example.yml"], project_root),
        (["igor", "status"], project_root),
        (["igor", "observe"], project_root),
        (["igor", "sync"], project_root),
        (["igor", "build"], project_root),
        (["igor", "diff"], project_root),
        (["igor", "resolve"], project_root),
        (["igor", "query"], project_root),
        (["igor", "explain"], project_root),
        (["igor", "test"], project_root),
        (["igor", "evaluate"], project_root),
        (["igor", "qualify"], project_root),
        (["igor", "tune"], project_root),
        (["igor", "serve"], project_root),
    ]
    for command, cwd in commands:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        results.append(
            {
                "command": command,
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    smoke_dir = project_root / ".igor" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "initialized-project-inventory.json").write_text(
        json.dumps(_inventory(project_root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (smoke_dir / "cli-test-results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
