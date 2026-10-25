"""Host checks for the isolated CI diagnostic branch, not device regressions."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from tests.utils import pytest_fixtures

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('error', [
    subprocess.TimeoutExpired(['app'], 180, output=b'partial stdout\n', stderr=b'first native reason\n'),
    subprocess.CalledProcessError(1, ['app'], output='partial stdout\n', stderr='first native reason\n'),
])
def test_subprocess_failure_retains_output_and_same_exception(tmp_path, monkeypatch, error):
    monkeypatch.setenv('SIMANEAT_APPS_TEST_OUTPUT_DIR', str(tmp_path))
    node = SimpleNamespace(name='case', fspath=ROOT / 'examples/cat/example/tests/python/test_e2e.py',
                           get_closest_marker=lambda _: object())
    request = SimpleNamespace(node=node)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, 'run', fail)
    fixture = pytest_fixtures.e2e_subprocess_artifacts.__wrapped__(request, monkeypatch)
    next(fixture)
    with pytest.raises(type(error)) as caught:
        subprocess.run(['app'], capture_output=True, timeout=180)
    assert caught.value is error
    logs = tmp_path / 'python/example/case'
    assert (logs / 'stdout.log').read_text() == 'partial stdout\n'
    assert (logs / 'stderr.log').read_text().startswith('first native reason\n')
    with pytest.raises(StopIteration):
        next(fixture)


def test_only_requested_python_cases_are_selected():
    spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'scripts/ci/run_stall_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.CASES) == {'sam-h264', 'yolov8-h265', 'yolov8-http'}
    assert all((ROOT / value.split('::')[0]).is_file() for value in module.CASES.values())
    for value in module.CASES.values():
        assert value.endswith(('rtsp-h264]', 'rtsp-h265]', 'http-mjpeg]'))


def test_diagnostic_results_cannot_be_released_and_logs_precede_cleanup():
    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    for name in ('publish-vulcan-artifacts', 'resolve-portal-publish-config',
                 'build-sysdoc-examples', 'publish-sysdoc-examples', 'publish-apps-algolia'):
        assert workflow['jobs'][name]['if'] == '${{ false }}'
    job = workflow['jobs']['test-runtime']
    assert job['strategy']['fail-fast'] is False
    steps = job['steps']
    upload = next(i for i, s in enumerate(steps) if s['name'].startswith('Upload diagnostic'))
    cleanup = next(i for i, s in enumerate(steps) if s['name'].startswith('Cleanup test'))
    assert upload < cleanup
    assert steps[upload]['if'] == '${{ always() }}'
    assert steps[upload]['with']['if-no-files-found'] == 'error'


def test_cpp_model_projection_matches_canonical_scope(tmp_path):
    spec = importlib.util.spec_from_file_location('diagnostic', ROOT / 'scripts/ci/run_stall_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scope = module.selected_model_scope(ROOT, 'object-detection/high-density-multi-stream-object-detector',
                                        'cpp', tmp_path)
    projected = subprocess.check_output([
        'python3', str(ROOT / 'tests/utils/test_scope.py'), '--scope-file', scope,
        'model-files', '--kind', 'e2e', '--language', 'cpp'], text=True)
    assert projected.strip() == 'object-detection/high-density-multi-stream-object-detector\tyolo26n-det-int8-b1.tar.gz'
    source = (ROOT / 'scripts/ci/run_stall_diagnostic.py').read_text()
    assert "env['SIMANEAT_APPS_TEST_MODEL_FILES'] = subprocess.check_output(" in source
