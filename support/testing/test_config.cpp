#include "support/testing/test_config.h"

#include "support/runtime/config_utils.h"

#include <stdexcept>

namespace sima_examples::testing {
namespace {

const ScalarConfig& test_config() {
  static const ScalarConfig config = ScalarConfig::load("tests/e2e.yaml");
  return config;
}

std::string full_key(const std::string& example_name, const std::string& section,
                     const std::string& key) {
  return "e2e." + example_name + "." + section + "." + key;
}

void require_present(const ScalarConfig& config, const std::string& key) {
  if (!config.string_value(key).has_value()) {
    throw std::runtime_error("tests/e2e.yaml missing required key: " + key);
  }
}

} // namespace

double e2e_double(const std::string& example_name, const std::string& section,
                  const std::string& key) {
  const std::string path = full_key(example_name, section, key);
  const ScalarConfig& config = test_config();
  require_present(config, path);
  return config.double_or(path, 0.0);
}

int e2e_int(const std::string& example_name, const std::string& section,
            const std::string& key) {
  const std::string path = full_key(example_name, section, key);
  const ScalarConfig& config = test_config();
  require_present(config, path);
  return config.int_or(path, 0);
}

} // namespace sima_examples::testing
