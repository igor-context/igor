from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import lancedb
import pyarrow as pa


@dataclass(frozen=True)
class LanceStoreConfig:
    """Provider-neutral routing for one structured Lance namespace."""

    root_uri: str
    environment: str
    domain: str
    storage_options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.root_uri.strip():
            raise ValueError("Lance root_uri cannot be empty")
        for name, value in (("environment", self.environment), ("domain", self.domain)):
            if not value.strip():
                raise ValueError(f"Lance {name} cannot be empty")
            if "/" in value or "\\" in value or value in {".", ".."}:
                raise ValueError(f"Lance {name} must be a path-safe identifier")
        parsed = urlsplit(self.root_uri)
        if parsed.scheme and not parsed.netloc:
            raise ValueError("Lance remote root_uri must include a provider and host")
        if parsed.query or parsed.fragment:
            raise ValueError("Lance root_uri cannot include a query or fragment")
        object.__setattr__(self, "storage_options", dict(self.storage_options))

    @property
    def namespace(self) -> str:
        return f"{self.environment}/{self.domain}"

    @property
    def uri(self) -> str:
        if urlsplit(self.root_uri).scheme:
            return f"{self.root_uri.rstrip('/')}/{self.namespace}"
        return str(Path(self.root_uri) / self.environment / self.domain)


class LanceStore:
    """Small storage port for namespaced local or remote Lance tables."""

    def __init__(
        self,
        path_or_config: str | Path | LanceStoreConfig,
        *,
        environment: str | None = None,
        domain: str | None = None,
        storage_options: Mapping[str, str] | None = None,
    ):
        if isinstance(path_or_config, LanceStoreConfig):
            if any(value is not None for value in (environment, domain, storage_options)):
                raise ValueError("LanceStoreConfig cannot be combined with constructor options")
            config = path_or_config
        elif environment is None and domain is None and storage_options is None:
            config = None
        else:
            if environment is None or domain is None:
                raise ValueError("Lance environment and domain are required together")
            config = LanceStoreConfig(str(path_or_config), environment, domain, storage_options or {})

        if config is None:
            self.uri = str(path_or_config)
            self.namespace = None
            self.storage_options: dict[str, str] = {}
            self.path = Path(path_or_config)
            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        else:
            self.uri = config.uri
            self.namespace = config.namespace
            self.storage_options = dict(config.storage_options)
            self.path = Path(self.uri) if not urlsplit(self.uri).scheme else None
            if self.path is not None:
                self.path.mkdir(parents=True, exist_ok=True)
            if self.storage_options:
                self._db = lancedb.connect(self.uri, storage_options=self.storage_options)
            else:
                self._db = lancedb.connect(self.uri)

    def _table(self, name: str):
        try:
            return self._db.open_table(name)
        except Exception as error:
            raise ValueError(f"unknown Lance table: {name}") from error

    def _data(self, rows: Iterable[dict[str, Any]] | pa.Table) -> Iterable[dict[str, Any]] | pa.Table:
        if isinstance(rows, pa.Table):
            return rows
        records = list(rows)
        if not records:
            raise ValueError("cannot write empty rows")
        return records

    def create(self, name: str, rows: Iterable[dict[str, Any]] | pa.Table) -> None:
        try:
            data = self._data(rows)
        except ValueError as error:
            if str(error) == "cannot write empty rows":
                raise ValueError("cannot create a table from empty rows") from error
            raise
        if data is None:
            raise ValueError("cannot create a table from empty rows")
        try:
            if isinstance(data, pa.Table) and data.num_rows == 0:
                self._db.create_table(name, data=data, schema=data.schema)
            else:
                self._db.create_table(name, data=data)
        except Exception as error:
            raise ValueError(f"could not create Lance table: {name}") from error

    def add(self, name: str, rows: Iterable[dict[str, Any]] | pa.Table) -> None:
        try:
            data = self._data(rows)
        except ValueError as error:
            raise ValueError("cannot add empty rows") from error
        try:
            self._table(name).add(data)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"could not add rows to Lance table: {name}") from error

    def replace(self, name: str, rows: Iterable[dict[str, Any]] | pa.Table) -> None:
        try:
            data = self._data(rows)
        except ValueError as error:
            raise ValueError("cannot replace with empty rows") from error
        try:
            if isinstance(data, pa.Table) and data.num_rows == 0:
                self._db.create_table(name, data=data, schema=data.schema, mode="overwrite")
            else:
                self._db.create_table(name, data=data, mode="overwrite")
        except Exception as error:
            raise ValueError(f"could not replace Lance table: {name}") from error

    def read(self, name: str) -> pa.Table:
        try:
            return self._table(name).to_arrow()
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"could not read Lance table: {name}") from error

    def metadata(self, name: str) -> dict[str, Any]:
        table = self._table(name)
        schema = table.schema
        return {"name": name, "columns": schema.names, "row_count": table.count_rows()}

    def names(self) -> list[str]:
        """Return locally available table names in stable order."""
        return sorted(self._db.list_tables().tables)
