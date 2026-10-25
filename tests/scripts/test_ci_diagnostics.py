"""Host regression checks for the non-publishing full-suite qualification branch."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from tests.utils import ci_diagnostics as diag, pytest_fixtures

ROOT = Path(__file__).resolve().parents[2]


def test_immutable_core_and_exact_runner_are_shared():
    manifest = json.loads((ROOT / 'deps/manifest.json').read_text())
    assert manifest['neat-core'] == {'ref': 'develop', 'spec': 'd533885fe12d'}
    assert manifest['elxr_version'] == '3.0.0_master_B2372'
    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    job = workflow['jobs']['test-runtime']
    assert job['runs-on']['labels'] == '${{ fromJSON(needs.resolve-test-runner.outputs.runner_labels) }}'
    steps = job['steps']
    run = next(s for s in steps if s['name'] == 'Run all activated Apps tests')
    assert 'suite ./tests/test.sh --all --strict' in run['run']
    assert 'matrix' not in job
    assert 'SIMANEAT_APPS_TEST_SCOPE_FILE' not in run['env']
    upload = next(i for i, s in enumerate(steps) if s['name'] == 'Upload full-suite diagnostics')
    cleanup = next(i for i, s in enumerate(steps) if s['name'].startswith('Cleanup test'))
    assert upload < cleanup and steps[upload]['if'] == '${{ always() }}'
    assert steps[upload]['with']['if-no-files-found'] == 'warn'
    for name in ('publish-vulcan-artifacts', 'resolve-portal-publish-config',
                 'build-sysdoc-examples', 'publish-sysdoc-examples', 'publish-apps-algolia'):
        assert workflow['jobs'][name]['if'] == '${{ false }}'


@pytest.mark.parametrize('status', [0, 37])
def test_cpp_capture_preserves_arguments_status_and_unbounded_child_output(tmp_path, status):
    app = tmp_path / 'app with spaces'
    app.write_text(f'#!{sys.executable}\nimport os, sys\n'
                   'print(repr(sys.argv[1:]))\n'
                   'sys.stderr.write("x" * 262144)\n'
                   f'sys.exit({status})\n')
    app.chmod(0o755)
    command = diag.cpp_command('unused-test', app, tmp_path / 'capture')
    for _ in range(2):
        result = subprocess.run([command[1], 'quoted argument', "a'b"], capture_output=True, timeout=5)
        assert result.returncode == status
        assert result.stdout == result.stderr == b''
    logs = list((tmp_path / 'capture').glob('invocation.*/native.stderr.log'))
    assert len(logs) == 2 and all(p.stat().st_size == 262144 for p in logs)
    for log in logs:
        assert log.with_name('native.stdout.log').read_text().strip() == repr(['quoted argument', "a'b"])


@pytest.mark.parametrize('error', [
    subprocess.TimeoutExpired(['app'], 180, output=b'partial stdout\n', stderr=b'first native reason\n'),
    subprocess.CalledProcessError(1, ['app'], output='partial stdout\n', stderr='first native reason\n'),
])
def test_partial_failure_diagnostics_do_not_replace_the_exception(tmp_path, monkeypatch, error):
    monkeypatch.setenv('SIMANEAT_APPS_TEST_OUTPUT_DIR', str(tmp_path))
    node = SimpleNamespace(name='case', fspath=ROOT / 'examples/cat/example/tests/python/test_e2e.py',
                           get_closest_marker=lambda _: object())
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, 'run', fail)
    fixture = pytest_fixtures.e2e_subprocess_artifacts.__wrapped__(SimpleNamespace(node=node), monkeypatch)
    next(fixture)
    with pytest.raises(type(error)) as caught:
        subprocess.run(['app'], capture_output=True, timeout=180)
    assert caught.value is error
    logs = tmp_path / 'python/example/case'
    assert (logs / 'stdout.log').read_text() == 'partial stdout\n'
    assert (logs / 'stderr.log').read_text().startswith('first native reason\n')
    with pytest.raises(StopIteration):
        next(fixture)


@pytest.mark.parametrize(('example', 'case', 'enabled'), [
    (*item, True) for item in sorted(diag.PYTHON_CASES)
] + [('single-stream-instance-segmenter', 'test_yolo26_publishes_frames_video_and_metadata[rtsp-h265]', False)])
def test_python_tracing_is_scoped_to_exact_cases(tmp_path, monkeypatch, example, case, enabled):
    monkeypatch.setenv('SIMANEAT_APPS_CI_DIAGNOSTICS_DIR', str(tmp_path))
    monkeypatch.setenv('SIMANEAT_APPS_TEST_OUTPUT_DIR', str(tmp_path))
    monkeypatch.delenv('SIMA_GRAPH_PIPELINE_DIAG_MS', raising=False)
    node = SimpleNamespace(name=case, fspath=ROOT / f'examples/segmentation/{example}/tests/python/test_e2e.py',
                           get_closest_marker=lambda _: object())
    fixture = pytest_fixtures.e2e_subprocess_artifacts.__wrapped__(SimpleNamespace(node=node), monkeypatch)
    next(fixture)
    assert ('SIMA_GRAPH_PIPELINE_DIAG_MS' in os.environ) == enabled
    with pytest.raises(StopIteration):
        next(fixture)


@pytest.mark.parametrize(('expected', 'status'), [('d533885fe12d', 0), ('d533885fe12d', 37), ('wrong', 0)])
def test_suite_checks_pin_and_preserves_test_result(tmp_path, monkeypatch, expected, status):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'diagnostics'
    monkeypatch.setenv('SIMANEAT_APPS_CI_DIAGNOSTICS_DIR', str(output))
    (tmp_path / 'deps').mkdir()
    for filename, spec in [('neat-core.json', expected), ('manifest.json', 'd533885fe12d')]:
        (tmp_path / 'deps' / filename).write_text(json.dumps({'neat-core': {'ref': 'develop', 'spec': spec}}))
    monkeypatch.setattr(sys, 'argv', ['diagnostics', 'suite', sys.executable, '-c', f'import sys; sys.exit({status})'])
    monkeypatch.setattr(diag, 'snapshot', lambda _: {'meminfo': 'fixture'})
    # Do not inspect or alter host packages in this fixture.
    monkeypatch.setattr(diag.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    if expected == 'wrong':
        with pytest.raises(RuntimeError, match='Packaged Core selection differs'):
            diag.main()
        assert not (output / 'result.json').exists()
    else:
        assert diag.main() == status
        assert json.loads((output / 'result.json').read_text())['returncode'] == status


def test_fdinfo_keeps_aliases_without_opening_device_buffers(tmp_path):
    directory = tmp_path / 'fdinfo'; directory.mkdir()
    for fd in ('7', '9'):
        (directory / fd).write_text('ino:\t42\nsize:\t4096\nexp_name:\tlinux,cma\n')
    (directory / '8').write_text('ino:\t99\n')
    assert diag.dma_buf_descriptors(tmp_path) == {
        fd: {'ino': '42', 'size': '4096', 'exp_name': 'linux,cma'} for fd in ('7', '9')}
