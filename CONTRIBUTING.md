# Contributing to IGOR

## Reporting issues

Open a GitHub issue with reproduction steps. For bugs, include the `docker compose
run` command that triggered it and its output.

## Running tests

Everything runs inside Docker Compose — no local Python environment needed.

```sh
docker compose run --rm runner-test
docker compose run --rm architecture-check
```

Module-specific test and lock targets are listed in each module's `MODULE.md`.

## Submitting changes

1. Fork the repo and create a branch.
2. Keep changes scoped to one capability owner (module) where possible.
3. Run `docker compose run --rm architecture-check` before opening a PR — it
   enforces module import boundaries, required module files, and Compose service
   coverage.
4. Open a pull request describing what changed and why.

## Project structure

Read a module's `module.toml` and `MODULE.md` before editing it. Core stays
tool-neutral; `runner` is the composition root. See [README.md](README.md) for the
full module ownership table.
