#include "support/testing/test_checks.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace sima_examples::testing {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& message) {
  return expect_true(haystack.find(needle) == std::string::npos, message);
}

std::filesystem::path write_scratch_config(const std::string& example_name,
                                           const std::string& test_name,
                                           const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir(example_name, test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create scratch directory for " + test_name);
  }
  const std::filesystem::path config_path = std::filesystem::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

ProcessResult validate_config_body(const std::string& example_name, const std::string& binary,
                                   const std::string& test_name, const std::string& body) {
  const std::filesystem::path config_path = write_scratch_config(example_name, test_name, body);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return result;
}

} // namespace sima_examples::testing
