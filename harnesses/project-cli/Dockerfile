FROM python:3.13.7-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/opt/igor-project-cli/.venv/bin:$PATH

RUN groupadd --gid 10001 igor \
    && useradd --uid 10001 --gid 10001 --create-home igor

WORKDIR /opt/igor-project-cli
COPY modules/core /opt/core
COPY modules/fixtures /opt/fixtures
COPY modules/dlt /opt/dlt
COPY modules/lancedb /opt/lancedb
COPY modules/datafusion /opt/datafusion
COPY modules/relational /opt/relational
COPY modules/context-compiler /opt/context-compiler
COPY modules/runner /opt/runner
COPY modules/huggingface /opt/huggingface
COPY modules/evaluator /opt/evaluator

RUN uv venv /opt/igor-project-cli/.venv \
    && uv pip install --python /opt/igor-project-cli/.venv/bin/python \
      --no-cache \
      /opt/core \
      /opt/fixtures \
      /opt/dlt \
      /opt/lancedb \
      /opt/datafusion \
      /opt/relational \
      /opt/context-compiler \
      /opt/evaluator \
      /opt/huggingface \
      /opt/runner

USER igor
CMD ["igor", "--help"]
