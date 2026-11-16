"""Import unchanged native helpers and serialize numerical ROS results."""
from dataclasses import asdict, is_dataclass
import importlib.util
from pathlib import Path
import sys


def json_value(value):
    """Convert native result values without rounding or changing coordinates."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def load_app(root: Path, app: str):
    """Import the shipped entrypoint; its graph remains visible in main.py."""
    parts = app.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("app must be category/example")
    directory = root / "examples" / app / "src" / "python"
    entrypoint = directory / ("detector_app.py" if parts[1] == "detection-to-vlm-assistant"
                              else "main.py")
    if not entrypoint.is_file():
        raise FileNotFoundError(f"No Python pipeline: {entrypoint}")
    sys.path.insert(0, str(directory))
    name = "neat_app_" + parts[1].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, entrypoint)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their owning module here.
    spec.loader.exec_module(module)
    return module, entrypoint


def load_native(ros_entrypoint):
    """Find the native counterpart of examples/ros2/<category>/<app>/src/python/."""
    entrypoint = Path(ros_entrypoint).resolve(strict=True)
    ros_root = entrypoint.parents[4]
    if ros_root.name != "ros2" or ros_root.parent.name != "examples":
        raise ValueError("ROS entrypoint must live under examples/ros2/<category>/<app>/src/python")
    app = "/".join(entrypoint.relative_to(ros_root).parts[:2])
    return load_app(ros_root.parent.parent, app)[0]
