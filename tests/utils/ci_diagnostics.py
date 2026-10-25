"""Opt-in CI evidence collection; does not change tests, deadlines or assertions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PYTHON_CASES = {
    ('efficientsam3-promptable-segmenter', 'test_publishes_insight_video_and_segments[rtsp-h264]'),
    ('single-stream-instance-segmenter', 'test_yolov8_publishes_frames_video_and_metadata[rtsp-h265]'),
    ('single-stream-instance-segmenter', 'test_yolov8_publishes_frames_video_and_metadata[http-mjpeg]'),
}


def trace_environment(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return {
        'SIMA_GRAPH_PIPELINE_DIAG_MS': '1000', 'SIMA_GRAPH_DIAG_ON_STOP': '1',
        'SIMA_STOP_TRACE': '1', 'SIMA_FUSED_BRANCH_COUNTERS': '1',
        'SIMA_FUSED_BRANCH_COUNTERS_EVERY': '20000',
        'SIMA_LATEST_MUX_STATS': '1', 'SIMA_LATEST_MUX_STATS_EVERY': '600',
        'SIMA_NEATDECODER_TIMING': '1', 'SIMA_NEATDECODER_TIMING_EVERY': '100',
        'GST_DEBUG': 'rtpjitterbuffer:3,rtph264depay:3,rtph265depay:3,h264parse:3,h265parse:3,neatdecoder:3',
        'GST_DEBUG_NO_COLOR': '1', 'GST_DEBUG_FILE': str(directory / 'gstreamer.log'),
    }


def read(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return f"unavailable: {error}"


def dma_buf_descriptors(proc):
    # Read fdinfo, never open/duplicate a DMA-BUF or change its ownership.
    result = {}
    for path in (proc / 'fdinfo').glob('*'):
        fields = dict(line.split(':', 1) for line in read(path).splitlines() if ':' in line)
        if 'exp_name' not in fields:
            continue
        result[path.name] = {key: fields[key].strip() for key in ('ino', 'size', 'exp_name')
                             if key in fields}
    return result


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
            "smaps_rollup": read(f'/proc/{pid}/smaps_rollup'),
            "dma_buf_fds": dma_buf_descriptors(Path(f'/proc/{pid}')),
            "threads": {p.name: read(p / 'wchan') for p in Path(f'/proc/{pid}/task').glob('*')},
        } for pid in owners if pid in table},
        "meminfo": read('/proc/meminfo'),
        "memory_pressure": read('/proc/pressure/memory'),
        "io_pressure": read('/proc/pressure/io'),
        "net_snmp": read('/proc/net/snmp'),
        "vmstat": read('/proc/vmstat'),
        "buddyinfo": read('/proc/buddyinfo'),
        "pagetypeinfo": read('/proc/pagetypeinfo'),
    }


def cpp_command(test, application, output):
    # Preserve the original native PID/signals. Direct files avoid filling the
    # C++ harness's pipes, which are only drained after its child exits.
    output.mkdir(parents=True, exist_ok=True)
    launcher = output / 'app-capture.sh'
    values = trace_environment(output)
    values.pop('GST_DEBUG_FILE')
    exports = ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in values.items())
    launcher.write_text('#!/bin/sh\nset -eu\n' +
                       'capture=$(mktemp -d ' + shlex.quote(str(output / 'invocation.XXXXXX')) + ')\n' +
                       exports + 'export GST_DEBUG_FILE="$capture/gstreamer.log"\n' +
                       'exec ' + shlex.quote(str(application)) + ' "$@" >"$capture/native.stdout.log"' +
                       ' 2>"$capture/native.stderr.log"\n')
    launcher.chmod(0o755)
    return [str(test), str(launcher)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['suite', 'cpp'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(os.environ['SIMANEAT_APPS_CI_DIAGNOSTICS_DIR']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'cpp':
        test, application = map(Path, args.command)
        subprocess.run(['file', str(test), str(application)], check=True)
        return subprocess.call(cpp_command(test, application, output / 'high-density'))

    root = Path.cwd()
    core = json.loads((root / 'deps/neat-core.json').read_text())['neat-core']
    expected = json.loads((root / 'deps/manifest.json').read_text())['neat-core']
    if core != expected:
        raise RuntimeError(f'Packaged Core selection differs: expected {expected}, got {core}')
    (output / 'identity.json').write_text(json.dumps({
        'core': core, 'apps_commit': os.environ.get('GITHUB_SHA'),
        'runner_name': os.environ.get('RUNNER_NAME'),
        'os_release': read('/etc/os-release'), 'sdk_release': read('/etc/sdk-release'),
        'boot_id': read('/proc/sys/kernel/random/boot_id'), 'cmdline': read('/proc/cmdline'),
    }, indent=2))
    with (output / 'packages.txt').open('w') as log:
        subprocess.run(['dpkg-query', '-W'], stdout=log, stderr=subprocess.STDOUT, check=True)
    # Inherit stdout/stderr: the original test runner remains visible in live CI.
    with (output / 'observations.jsonl').open('w') as log:
        process = subprocess.Popen(args.command)
        while True:
            log.write(json.dumps(snapshot(process.pid)) + '\n')
            log.flush()
            try:
                status = process.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                pass
        log.write(json.dumps(snapshot(os.getpid())) + '\n')
    hashes = {}
    for path in sorted((root / 'models').glob('*.tar.gz')):
        with path.open('rb') as handle:
            hashes[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (output / 'models-sha256.json').write_text(json.dumps(hashes, indent=2))
    (output / 'result.json').write_text(json.dumps({
        'returncode': status, 'instrumented': True,
        'device_retirement': 'not established by process exit',
    }, indent=2))
    return status if status >= 0 else 128 - status


if __name__ == '__main__':
    raise SystemExit(main())
