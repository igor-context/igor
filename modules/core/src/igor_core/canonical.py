"""Deterministic serialization and identity primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with deterministic ordering."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        value = dict(value)

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_identity(value: Any) -> str:
    """Return the version-independent SHA-256 identity of a canonical value."""
    payload = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
