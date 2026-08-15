FROM python:3.13.7-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

RUN groupadd --gid 10001 igor \
    && useradd --uid 10001 --gid 10001 --create-home igor

WORKDIR /opt/igor-runner

COPY modules/runner/pyproject.toml modules/runner/uv.lock ./
COPY modules/core /opt/core
COPY modules/fixtures /opt/fixtures
COPY modules/dlt /opt/dlt
COPY modules/lancedb /opt/lancedb
COPY modules/datafusion /opt/datafusion
COPY modules/relational /opt/relational
COPY modules/context-compiler /opt/context-compiler
COPY modules/evaluator /opt/evaluator
RUN uv sync --frozen --all-groups --no-install-project

COPY modules/runner/src ./src
COPY modules/runner/tests ./tests
RUN uv sync --frozen --all-groups

USER igor

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync"]
CMD ["pytest", "-p", "no:cacheprovider", "-q"]
