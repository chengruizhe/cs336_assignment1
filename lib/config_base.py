from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_type_hints

TConfig = TypeVar("TConfig", bound="ConfigBase")


class ConfigBase:
    @classmethod
    def from_file(cls: type[TConfig], config_path: str | Path) -> TConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix == ".toml":
            data = tomllib.loads(path.read_text())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix}. Use .toml or .json."
            )
        if not isinstance(data, Mapping):
            raise ValueError("Config must parse to a mapping at the top level.")
        return cls.from_dict(dict(data))

    @classmethod
    def from_dict(cls: type[TConfig], data: Mapping[str, Any]) -> TConfig:
        field_map = {f.name: f for f in fields(cls)}
        type_hints = get_type_hints(cls)
        unknown = [key for key in data if key not in field_map]
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown config field(s): {keys}")

        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            incoming = data[f.name]
            field_type = type_hints.get(f.name, f.type)
            kwargs[f.name] = cls._coerce_value(field_type, incoming, path=f.name)
        return cls(**kwargs)  # type: ignore[call-arg]

    def with_updates(self: TConfig, updates: Mapping[str, Any]) -> TConfig:
        merged = asdict(self)
        self._merge_nested_dict(merged, updates, path="")
        return type(self).from_dict(merged)

    def with_flat_updates(self: TConfig, updates: Mapping[str, Any]) -> TConfig:
        return self.with_updates(self._nested_from_dotted(updates))

    @classmethod
    def _nested_from_dotted(cls, flat: Mapping[str, Any]) -> dict[str, Any]:
        nested: dict[str, Any] = {}
        for dotted_key, value in flat.items():
            if not dotted_key:
                raise ValueError("Config override keys must be non-empty.")
            parts = dotted_key.split(".")
            if any(not part for part in parts):
                raise ValueError(f"Invalid override key `{dotted_key}`.")
            cursor = nested
            for part in parts[:-1]:
                existing = cursor.get(part)
                if existing is None:
                    existing = {}
                    cursor[part] = existing
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"Cannot set nested key `{dotted_key}` because `{part}` is not a mapping."
                    )
                cursor = existing
            cursor[parts[-1]] = value
        return nested

    @classmethod
    def _coerce_value(cls, field_type: Any, incoming: Any, *, path: str) -> Any:
        if (
            isinstance(field_type, type)
            and is_dataclass(field_type)
            and issubclass(field_type, ConfigBase)
        ):
            if not isinstance(incoming, Mapping):
                raise ValueError(f"Expected mapping for nested config field `{path}`.")
            return field_type.from_dict(dict(incoming))
        return incoming

    @classmethod
    def _merge_nested_dict(
        cls, base: dict[str, Any], updates: Mapping[str, Any], *, path: str
    ) -> None:
        for key, value in updates.items():
            full_key = f"{path}.{key}" if path else key
            if key not in base:
                raise ValueError(f"Unknown config field `{full_key}`.")

            base_value = base[key]
            if isinstance(base_value, dict):
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Expected mapping for nested field `{full_key}`, got {type(value).__name__}."
                    )
                cls._merge_nested_dict(base_value, value, path=full_key)
            else:
                if isinstance(value, Mapping):
                    raise ValueError(
                        f"Expected scalar value for field `{full_key}`, got mapping."
                    )
                base[key] = value
