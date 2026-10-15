"""Software-only regressions for the real C++ process capture helper."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = r'''
#include "support/testing/test_process.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
using sima_examples::testing::spawn_and_wait;
constexpr int kChunks = 256;
constexpr int kChunkSize = 4096;
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "--write") {
    for (int i = 0; i < kChunks; ++i) {
      std::cout << std::string(kChunkSize, 'O') << std::flush;
      std::cerr << std::string(kChunkSize, 'E') << std::flush;
    }
    return 37;
  }
  if (mode == "--stderr-closed") {
    ::close(STDERR_FILENO);
    std::cout << std::string(kChunks * kChunkSize, 'O') << std::flush;
    return 13;
  }
  if (mode == "--sleep") {
    std::cout << "stdout before wait\n" << std::flush;
    std::cerr << "stderr before wait\n" << std::flush;
    ::sleep(20);
    return 0;
  }
  if (mode == "capture") {
    const auto result = spawn_and_wait(argv[0], {"--write"}, 3000);
    if (result.exit_code != 37 ||
        result.stdout_text != std::string(kChunks * kChunkSize, 'O') ||
        result.stderr_text != std::string(kChunks * kChunkSize, 'E')) {
      std::cerr << "capture failed: exit=" << result.exit_code
                << " stdout=" << result.stdout_text.size()
                << " stderr=" << result.stderr_text.size() << "\n";
      return 1;
    }
    return 0;
  }
  if (mode == "closed-pipe") {
    const auto result = spawn_and_wait(argv[0], {"--stderr-closed"}, 3000);
    if (result.exit_code != 13 || !result.stderr_text.empty() ||
        result.stdout_text != std::string(kChunks * kChunkSize, 'O')) {
      std::cerr << "closing stderr interrupted stdout capture\n";
      return 1;
    }
    return 0;
  }
  if (mode == "timeout") {
    const auto result = spawn_and_wait(argv[0], {"--sleep"}, 100);
    if (result.exit_code != -1 || result.stdout_text != "stdout before wait\n" ||
        result.stderr_text != "stderr before wait\n\n[test_process] killed after timeout (100ms)") {
      std::cerr << "timeout result or captured output changed\n";
      return 1;
    }
    return 0;
  }
  return 2;
}
'''


@pytest.fixture(scope="module")
def capture_executable(tmp_path_factory):
    compiler = shutil.which("c++")
    assert compiler, "A native C++ compiler is required for process capture tests"
    directory = tmp_path_factory.mktemp("process-capture")
    source = directory / "capture.cpp"
    source.write_text(HARNESS)
    executable = directory / "capture"
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
         "-I", str(ROOT), str(source), str(ROOT / "support/testing/test_process.cpp"),
         "-o", str(executable)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("mode", ["capture", "closed-pipe", "timeout"])
def test_process_capture(capture_executable, mode):
    result = subprocess.run([str(capture_executable), mode], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
