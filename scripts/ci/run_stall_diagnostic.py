#!/usr/bin/env python3
"""Diagnostic branch only: run one unchanged failing case and retain its evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

CASES = {
    "sam-h264": "examples/segmentation/efficientsam3-promptable-segmenter/tests/python/test_e2e.py::test_publishes_insight_video_and_segments[rtsp-h264]",
    "yolov8-h265": "examples/segmentation/single-stream-instance-segmenter/tests/python/test_e2e.py::TestE2E::test_yolov8_publishes_frames_video_and_metadata[rtsp-h265]",
    "yolov8-http": "examples/segmentation/single-stream-instance-segmenter/tests/python/test_e2e.py::TestE2E::test_yolov8_publishes_frames_video_and_metadata[http-mjpeg]",
}
HD = "examples/object-detection/high-density-multi-stream-object-detector"


def read(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return f"unavailable: {error}"


def snapshot(root_pid):
    # Read-only; no ptrace, signals, device access, recovery or inferred retirement.
    table = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]), fields)
        except (OSError, ValueError, IndexError):
            continue
    owners = {root_pid}
    while True:
        children = {pid for pid, (ppid, _) in table.items() if ppid in owners}
        if children <= owners:
            break
        owners |= children
    return {
        "monotonic_ns": time.monotonic_ns(),
        "processes": {str(pid): {
            "stat": table[pid][1], "wchan": read(f'/proc/{pid}/wchan'),
            "io": read(f'/proc/{pid}/io'),
            "threads": {p.name: read(p / 'wchan') for p in Path(f'/proc/{pid}/task').glob('*')},
        } for pid in owners if pid in table},
        "meminfo": read('/proc/meminfo'),
        "memory_pressure": read('/proc/pressure/memory'),
        "io_pressure": read('/proc/pressure/io'),
        "net_snmp": read('/proc/net/snmp'),
    }


def selected_model_scope(root, example, language, output):
    # The validator requires every catalog entry. Retain them all, disabling only
    # unrelated acquisition lanes in this private, generated download scope.
    scope = yaml.safe_load(subprocess.check_output(
        [sys.executable, 'tests/utils/test_scope.py', 'generate'], cwd=root, text=True))
    selected = scope['examples'][example]['e2e'][language]
    for entry in scope['examples'].values():
        entry['e2e'] = {lang: {'enabled': False, 'models': []} for lang in ('python', 'cpp')}
    scope['examples'][example]['e2e'][language] = selected
    path = output / 'model-scope.yaml'
    path.write_text(yaml.safe_dump(scope))
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['hd-h264', *CASES], required=True)
    parser.add_argument('--mode', choices=['quiet', 'trace'], required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runtime.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'APPS_ROOT': str(root), 'PYTHONUNBUFFERED': '1',
           'SIMANEAT_APPS_TEST_REQUIRE_E2E': '1', 'SIMANEAT_APPS_TEST_KEEP_OUTPUT': '1',
           'SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS': '1',
           'SIMANEAT_APPS_TEST_MODELS_DIR': str(root / 'models'),
           'SIMANEAT_APPS_TEST_OUTPUT_DIR': str(output / 'sandbox')}
    for key in ('SIMA_GRAPH_PIPELINE_DIAG_MS', 'SIMA_GRAPH_DIAG_ON_STOP', 'SIMA_STOP_TRACE',
                'SIMA_FUSED_BRANCH_COUNTERS', 'SIMA_FUSED_BRANCH_COUNTERS_DEBUG',
                'SIMA_LATEST_MUX_STATS', 'SIMA_NEATDECODER_TIMING', 'SIMA_GRAPH_DEBUG'):
        env[key] = '0'
    env.pop('GST_DEBUG', None)
    env.pop('GST_DEBUG_FILE', None)
    if args.mode == 'trace':
        env.update({
            'SIMA_GRAPH_PIPELINE_DIAG_MS': '1000', 'SIMA_GRAPH_DIAG_ON_STOP': '1',
            'SIMA_STOP_TRACE': '1', 'SIMA_FUSED_BRANCH_COUNTERS': '1',
            'SIMA_FUSED_BRANCH_COUNTERS_DEBUG': '1', 'SIMA_FUSED_BRANCH_COUNTERS_EVERY': '20000',
            'SIMA_LATEST_MUX_STATS': '1', 'SIMA_LATEST_MUX_STATS_EVERY': '600',
            'SIMA_NEATDECODER_TIMING': '1', 'SIMA_NEATDECODER_TIMING_EVERY': '100',
            'GST_DEBUG': 'rtpjitterbuffer:3,rtph264depay:3,h264parse:3,neatdecoder:3',
            'GST_DEBUG_NO_COLOR': '1', 'GST_DEBUG_FILE': str(output / 'gstreamer.log'),
        })
    # Keep the real CI media variables, model identities, test config and deadlines.
    if args.case == 'hd-h264':
        scope = selected_model_scope(root, HD.removeprefix('examples/'), 'cpp', output)
        download = ['bash', 'scripts/download_models.sh', '--scope-file', scope, '--language', 'cpp']
    elif args.case == 'sam-h264':
        scope = selected_model_scope(root, 'segmentation/efficientsam3-promptable-segmenter', 'python', output)
        download = ['bash', 'scripts/download_models.sh', '--scope-file', scope, '--language', 'python']
    else:
        download = ['bash', 'scripts/download_models.sh', 'yolo_v8n_seg']
    subprocess.run(download, cwd=root, env=env, check=True)
    if args.case == 'hd-h264':
        env['SIMANEAT_APPS_TEST_MODEL_FILES'] = subprocess.check_output(
            [sys.executable, 'tests/utils/test_scope.py', '--scope-file', scope,
             'model-files', '--kind', 'e2e', '--language', 'cpp'],
            cwd=root, env=env, text=True).strip()
    model_hashes = {}
    for path in sorted((root / 'models').glob('*.tar.gz')):
        with path.open('rb') as handle:
            model_hashes[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (output / 'identity.json').write_text(json.dumps({
        'case': args.case, 'mode': args.mode,
        'apps_commit': os.environ.get('GITHUB_SHA'),
        'core': json.loads((root / 'deps/neat-core.json').read_text()),
        'models_sha256': model_hashes,
        'sdk': read('/etc/sdk-release'), 'os_release': read('/etc/os-release'),
        'boot_id': read('/proc/sys/kernel/random/boot_id'),
    }, indent=2))
    with (output / 'packages.txt').open('w') as log:
        subprocess.run(['dpkg-query', '-W'], stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.case == 'hd-h264':
        binary = root / HD / 'src/cpp/pre-built/high-density-multi-stream-object-detector'
        test = root / HD / 'tests/cpp/high-density-multi-stream-object-detector_e2e_test'
        subprocess.run(['file', str(binary), str(test)], check=True)
        # The C++ harness drains child pipes only at exit. An exec wrapper keeps
        # native PID/signals intact and avoids making tracing itself a pipe stall.
        wrapper = output / 'app-capture.sh'
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(binary)) + ' "$@" >' +
                           shlex.quote(str(output / 'native.stdout.log')) + ' 2>' +
                           shlex.quote(str(output / 'native.stderr.log')) + '\n')
        wrapper.chmod(0o755)
        command = [str(test), str(wrapper)]
    else:
        command = [sys.executable, '-m', 'pytest', '-c', 'tests/pytest.ini', '-vv', '-s',
                   '--tb=long', '--junitxml=' + str(output / 'junit.xml'), CASES[args.case]]
    (output / 'command.json').write_text(json.dumps(command))
    with (output / 'test.log').open('w') as log, (output / 'observations.jsonl').open('w') as observations:
        observations.write(json.dumps(snapshot(os.getpid())) + '\n')
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if args.mode == 'trace':
                observations.write(json.dumps(snapshot(process.pid)) + '\n')
                observations.flush()
            time.sleep(2)
        status = process.wait()
        observations.write(json.dumps(snapshot(os.getpid())) + '\n')
    (output / 'result.json').write_text(json.dumps({
        'returncode': status, 'device_retirement': 'not established by process exit',
        'instrumented': args.mode == 'trace',
    }, indent=2))
    print((output / 'test.log').read_text(), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
