"""Helpers for the configuration-rule unit tests the examples carry.

Each application's tests keep their own case tables: the valid baseline config, the
one-key changes that must be rejected, the boundary values that must be accepted, and
the documented defaults. The two things every table needs, loading the application's
``main.py`` under a private name and writing the baseline with one change applied,
live here so the tables are all the test files hold.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

Override = tuple[tuple[str, ...], Any]


def load_example_main(example_dir: Path, module_name: str) -> types.ModuleType:
    """Import ``<example_dir>/src/python/main.py`` under ``module_name``.

    Every application has a module called ``main``, and a plain ``import main`` binds
    whichever was imported first when pytest runs several examples in one process. Each
    suite therefore binds its own under a distinct name, once; a second call returns the
    same module. The module is registered before it runs because ``main.py`` declares
    module-level dataclasses, which resolve their annotations through ``sys.modules``.
    """
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    python_dir = example_dir / "src" / "python"
    # main.py imports its own packages (utils.tracker, ...) the way a script run from its
    # directory does, so that directory has to be importable first, and a package of the
    # same name that another example already put in sys.modules must not shadow this one.
    if str(python_dir) in sys.path:
        sys.path.remove(str(python_dir))
    sys.path.insert(0, str(python_dir))
    _evict_shadowing_packages(python_dir)
    main_py = python_dir / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _evict_shadowing_packages(python_dir: Path) -> None:
    """Drop cached modules that another example's ``src/python`` provided under a name
    this example also ships, so this example's ``main.py`` imports its own copies."""
    local_names = {
        entry.stem if entry.suffix == ".py" else entry.name
        for entry in python_dir.iterdir()
        if entry.name != "main.py" and not entry.name.startswith(("_", "."))
    }
    root = python_dir.resolve()
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] not in local_names:
            continue
        origins = [getattr(module, "__file__", None)] + list(getattr(module, "__path__", []) or [])
        origins = [Path(origin).resolve() for origin in origins if origin]
        if origins and not any(origin.is_relative_to(root) for origin in origins):
            del sys.modules[name]


def write_config(
    tmp_path: Path,
    base: dict[str, Any],
    overrides: Iterable[Override] = (),
    *,
    root: Any = None,
) -> Path:
    """Write ``base`` with ``overrides`` applied as ``((section, ..., key), value)``.

    ``base`` is deep-copied, so a table can be reused across tests. ``root`` replaces it
    outright, for the malformed-file cases that need a non-mapping or an empty document.
    """
    raw = copy.deepcopy(base) if root is None else root
    for path, value in overrides:
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def config_writer(base: dict[str, Any]) -> Callable[..., Path]:
    """Bind ``write_config`` to one suite's baseline: ``write(tmp_path, overrides, root=)``."""

    def write(tmp_path: Path, overrides: Iterable[Override] | None = None, *, root: Any = None) -> Path:
        return write_config(tmp_path, base, overrides or (), root=root)

    return write
