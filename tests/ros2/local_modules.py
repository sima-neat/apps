"""Import a standalone node in tests without mixing sibling apps' helper modules."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


def load_node(entry, name, *, resolve_share=True):
    entry = Path(entry)
    local = {path.stem for path in entry.parent.glob('*.py')}
    previous = {key: sys.modules[key] for key in local if key in sys.modules}
    original_path = list(sys.path)
    for key in local:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(entry.parent))
    try:
        share = (patch('ament_index_python.packages.get_package_share_directory',
                       return_value=str(entry.parents[1])) if resolve_share else nullcontext())
        with share:
            spec = importlib.util.spec_from_file_location(name, entry)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    finally:
        sys.path[:] = original_path
        for key in local:
            sys.modules.pop(key, None)
        sys.modules.update(previous)
