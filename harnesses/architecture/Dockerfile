FROM python:3.13.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 igor \
    && useradd --uid 10001 --gid 10001 --create-home igor

WORKDIR /opt/igor-architecture
COPY validate_architecture.py ./
COPY tests ./tests

USER igor
