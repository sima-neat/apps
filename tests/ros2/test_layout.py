"""ROS adapters live separately and import native helpers without changing them."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'support/ros2'))
from neat_apps.runtime import load_app, load_native


def test_each_scoped_native_example_has_a_separate_ros_directory():
    expected = {p.parent.relative_to(ROOT / 'examples')
                for p in (ROOT / 'examples').glob('*/*/README.md')
                if p.parent.name != 'pcie-high-density-multi-stream-object-detector'}
    actual = {p.parent.relative_to(ROOT / 'examples/ros2')
              for p in (ROOT / 'examples/ros2').glob('*/*/README.md')}
    assert actual == expected
    for key in expected:
        native = ROOT / 'examples' / key
        ros = ROOT / 'examples/ros2' / key
        assert not (native / 'src/python/ros_node.py').exists()
        if key.parts[0] == 'docker':
            assert (ros / 'src/common/ros-parallel-apps.sh').is_file()
        else:
            assert (ros / 'src/common/config.yaml').is_file()
            ast.parse((ros / 'src/python/ros_node.py').read_text())


def test_loader_imports_unchanged_native_helpers_from_ros_location(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    native = tmp_path / 'examples/detection/sample/src/python/main.py'
    native.parent.mkdir(parents=True)
    original = 'VALUE = 42\n'
    native.write_text(original)
    entry = tmp_path / 'examples/ros2/detection/sample/src/python/ros_node.py'
    entry.parent.mkdir(parents=True)
    entry.touch()
    module = load_native(entry)
    assert module.VALUE == 42
    assert not hasattr(module, 'result_sink')
    assert native.read_text() == original
    assert Path(module.__file__) == native


def test_loader_rejects_paths_outside_the_example_layout(tmp_path):
    with pytest.raises(ValueError, match='category/example'):
        load_app(tmp_path, '../other')
