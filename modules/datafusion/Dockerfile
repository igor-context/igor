FROM python:3.13.7-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache
RUN groupadd --gid 10001 igor && useradd --uid 10001 --gid 10001 --create-home igor
WORKDIR /opt/igor-datafusion
COPY modules/core /opt/core
COPY modules/lancedb /opt/lancedb
COPY modules/datafusion/pyproject.toml modules/datafusion/uv.lock ./
RUN uv sync --frozen --all-groups --no-install-project
COPY modules/datafusion/src ./src
COPY modules/datafusion/tests ./tests
RUN uv sync --frozen --all-groups
USER igor
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync"]
CMD ["pytest", "-p", "no:cacheprovider", "-q"]
