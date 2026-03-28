// Starter e2e smoke test for multistream-object-detection-optiview.
#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  const auto r = spawn_and_wait(binary, {"--validate-config-only"}, 10000);
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] starter smoke expected exit 0, got " << r.exit_code << "\n";
    if (!r.stderr_text.empty()) {
      std::cerr << r.stderr_text << "\n";
    }
    return 1;
  }
  return 0;
}
