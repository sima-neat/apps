#pragma once

#include "support/runtime/config_utils.h"

#include <filesystem>
#include <string>

namespace sima_examples::testing {

// Locate examples/*/<example_name>/common/config.yaml from the apps root.
std::filesystem::path example_common_config_path(const std::string& example_name);

// Load the shipped common/config.yaml for an example.
const ScalarConfig& example_common_config(const std::string& example_name);

// Resolve the configured model filename under the active test model directory.
std::string configured_model_path(const std::string& example_name, const std::string& models_dir);

// Read a required scalar from an example common/config.yaml under:
// <section>.<key>
double e2e_double(const std::string& example_name, const std::string& section,
                  const std::string& key);

int e2e_int(const std::string& example_name, const std::string& section, const std::string& key);

bool e2e_bool(const std::string& example_name, const std::string& section, const std::string& key,
              bool default_value);

} // namespace sima_examples::testing
