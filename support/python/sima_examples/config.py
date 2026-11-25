"""Read an example's config.yaml by dotted key, such as `input.rtsp.tcp`, and lists of mappings,
such as one `model` entry per stream."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(_flatten(value, f"{path}."))
        else:
            values[path] = value
    return values


def lookup(key: str, value: str, table: Mapping[str, T]) -> T:
    """`table[value]`; a value the table lacks raises with `key`, the value, and the choices.

    Use it for a value `main` derives from the config; `Config.choice` reads the key itself.
    """
    if value not in table:
        raise ValueError(f"{key}: unknown value '{value}' (expected {', '.join(sorted(table))})")
    return table[value]


class Config:
    """The values of one config file. Every key the example reads must be present.

    An entry of `mapping_list` is a Config too: its getters take a field, such as `path`, and
    its errors name the entry, such as `model[1].path`.

    Mirrors `sima_examples::Config` in support/cpp/sima_examples/config.h.
    """

    def __init__(self, path: Path, values: dict[str, Any], prefix: str = "") -> None:
        self.path = path
        self._values = values
        self._prefix = prefix

    @classmethod
    def load(cls, path: Path) -> Config:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must hold a mapping")
        return cls(path, _flatten(raw))

    def _value(self, key: str, kind: type, kind_name: str) -> Any:
        key = self._prefix + key
        if key not in self._values:
            raise ValueError(f"{self.path} is missing {key}")
        value = self._values[key]
        # YAML reads `true` as a bool, and bool is a subclass of int.
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            raise ValueError(f"{key} must be {kind_name}, got {value!r}")
        return value

    def string(self, key: str) -> str:
        return self._value(key, str, "a string")

    def integer(self, key: str) -> int:
        return self._value(key, int, "an integer")

    def number(self, key: str) -> float:
        return float(self._value(key, (int, float), "a number"))

    def boolean(self, key: str) -> bool:
        return self._value(key, bool, "true or false")

    def choice(self, key: str, table: Mapping[str, T]) -> T:
        """The `table` entry the string at `key` names, such as a codec name's Neat enum."""
        return lookup(self._prefix + key, self.string(key), table)

    def string_list(self, key: str) -> list[str]:
        """A list of strings; a single string reads as a list of one."""
        values = self._value(key, (list, str), "a list of strings")
        if isinstance(values, str):
            return [values]
        if not values or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{self._prefix + key} must be a non-empty list of strings")
        return values

    def mapping_list(self, key: str) -> list[Config]:
        """A non-empty list of mappings, one Config per entry, such as `model[1]`."""
        key = self._prefix + key
        if key not in self._values:
            if any(name.startswith(f"{key}.") for name in self._values):
                raise ValueError(f"{key} must be a list of mappings, got a mapping")
            raise ValueError(f"{self.path} is missing {key}")
        values = self._values[key]
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list of mappings, got {values!r}")
        if not values:
            raise ValueError(f"{key} must be a non-empty list of mappings")
        entries = []
        for index, value in enumerate(values):
            entry = f"{key}[{index}]."
            if not isinstance(value, dict):
                raise ValueError(f"{key}[{index}] must be a mapping, got {value!r}")
            entries.append(Config(self.path, _flatten(value, entry), entry))
        return entries
