# Architecture harness

Active repository-level enforcement for capability registration, module activation completeness, dependency direction, one root Compose project, container-only agent commands, and clean build contexts.

Run both the harness tests and the live repository check through the root Compose project:

```sh
docker compose run --rm --build architecture-test
docker compose run --rm --build architecture-check
```
