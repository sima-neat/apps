#include "support/testing/test_config.h"
#include <filesystem>
#include <iostream>
#include <unistd.h>

// Exercise the C++ binary with the same wire-level assertions as Python.
// The harness uses only the Python standard library; pytest is not required.
int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  const auto harness =
      sima_examples::testing::example_common_config_path("multi-stream-blazepose3d")
          .parent_path()
          .parent_path()
          .parent_path() /
      "tests/python/test_e2e.py";
  execl("/usr/bin/env", "env", "python3", harness.c_str(), argv[1], nullptr);
  std::cerr << "failed to launch Python E2E harness\n";
  return 127;
}
