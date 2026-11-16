"""Select an example's explicit ros_node.py entrypoint."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="category/example")
    parser.add_argument("--apps-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    root = args.apps_root.resolve(strict=True)
    config = args.config.resolve(strict=True)
    parts = args.app.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        parser.error("app must be category/example")
    entrypoint = root / "examples/ros2" / args.app / "src/python/ros_node.py"
    if not entrypoint.is_file():
        parser.error(f"No image-topic ROS entrypoint: {entrypoint}")
    os.chdir(root)
    sys.path.insert(0, str(entrypoint.parent))
    spec = importlib.util.spec_from_file_location("neat_ros_example", entrypoint)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "main"):
        return module.main(["--config", str(config), *remaining])
    from .image_node import run
    return run(parts[1].replace("-", "_"), module.Pipeline,
               ["--config", str(config), *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
