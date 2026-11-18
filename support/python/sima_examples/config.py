"""Read an example's config.yaml by dotted key, such as `input.rtsp.tcp`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(_flatten(value, f"{path}."))
        else:
            values[path] = value
    return values


class Config:
    """The values of one config file. Every key the example reads must be present.

    Mirrors `sima_examples::ScalarConfig` in support/runtime/config_utils.h.
    """

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must hold a mapping")
        self.path = path
        self._values = _flatten(raw)
        self._unread = set(self._values)

    def _value(self, key: str, kind: type, kind_name: str) -> Any:
        if key not in self._values:
            raise ValueError(f"{self.path} is missing {key}")
        self._unread.discard(key)
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

    def string_list(self, key: str) -> list[str]:
        """A list of strings; a single string reads as a list of one."""
        values = self._value(key, (list, str), "a list of strings")
        if isinstance(values, str):
            return [values]
        if not values or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{key} must be a non-empty list of strings")
        return values

    def reject_unread_keys(self) -> None:
        """Fail on keys the example never read, such as a renamed or misspelled key."""
        if self._unread:
            raise ValueError(
                f"{self.path} has unknown keys: {', '.join(sorted(self._unread))}"
            )
