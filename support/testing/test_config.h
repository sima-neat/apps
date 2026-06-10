#pragma once

#include "support/runtime/config_utils.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sima_examples::testing {

using ConfigScalars = std::map<std::string, std::string>;
using ConfigLists = std::map<std::string, std::vector<std::string>>;

// Locate examples/*/<example_name>/src/common/config.yaml from the apps root.
std::filesystem::path example_common_config_path(const std::string& example_name);

// Load the shipped src/common/config.yaml for an example.
const ScalarConfig& example_common_config(const std::string& example_name);

// Resolve the configured model filename under the active test model directory.
std::string configured_model_path(const std::string& example_name, const std::string& models_dir);

// Read a required scalar from an example src/common/config.yaml under:
// <section>.<key>
double e2e_double(const std::string& example_name, const std::string& section,
                  const std::string& key);

int e2e_int(const std::string& example_name, const std::string& section, const std::string& key);

bool e2e_bool(const std::string& example_name, const std::string& section, const std::string& key,
              bool default_value);

// Write an e2e runtime config by starting from src/common/config.yaml and applying
// only test-harness overrides. The generated config omits testing.* keys.
std::filesystem::path write_e2e_config(const std::string& example_name,
                                       const std::filesystem::path& config_path,
                                       const ConfigScalars& overrides,
                                       const ConfigLists& list_overrides = {});

} // namespace sima_examples::testing
