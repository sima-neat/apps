"""Independent ROS packages with local implementations and no shared runtime."""
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_each_scoped_native_example_has_a_separate_ros_directory():
    expected = {p.parent.relative_to(ROOT / 'examples')
                for p in (ROOT / 'examples').glob('*/*/README.md')
                if p.parent.name != 'pcie-high-density-multi-stream-object-detector'}
    actual = {p.parent.relative_to(ROOT / 'examples/ros2')
              for p in (ROOT / 'examples/ros2').glob('*/*/README.md')}
    ros_only = {Path("multi-model/perception-switchboard")}
    assert actual == expected | ros_only
    for key in expected:
        native = ROOT / 'examples' / key
        ros = ROOT / 'examples/ros2' / key
        assert not (native / 'src/python/ros_node.py').exists()
        if key.parts[0] == 'docker':
            assert (ros / 'src/common/ros-parallel-apps.sh').is_file()
        else:
            assert (ros / 'package.xml').is_file()
            assert (ros / 'CMakeLists.txt').is_file()
            assert (ros / 'launch/app.launch.py').is_file()
            assert not (ros / 'src/python/ros_node.py').exists()
            nodes = list((ros / 'src/python').glob('*_node.py'))
            assert nodes
            for node in nodes:
                ast.parse(node.read_text())


@pytest.mark.parametrize('app', sorted((ROOT / 'examples/ros2').glob('*/*/package.xml')))
def test_packages_build_without_shared_ros_package(app):
    app = app.parent
    dependencies = {item.text for item in ET.parse(app / 'package.xml').getroot()
                    if item.tag.endswith('depend')}
    assert 'neat_apps' not in dependencies
    installed = (app / 'CMakeLists.txt').read_text()
    for source in (app / 'src/python').glob('*.py'):
        assert f'src/python/{source.name}' in installed or 'install(DIRECTORY src/' in installed
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('neat_apps')
            elif isinstance(node, ast.Import):
                assert all(not name.name.startswith('neat_apps') for name in node.names)
