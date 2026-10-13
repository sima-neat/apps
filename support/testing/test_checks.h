#pragma once

#include "support/testing/test_process.h"

#include <filesystem>
#include <string>

namespace sima_examples::testing {

// Prints "[OK] message" or "[FAIL] message" and returns `condition`, so a unit test can
// chain its checks with && and report every one that fails.
bool expect_true(bool condition, const std::string& message);
bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message);
bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& message);

// Writes `body` as config.yaml in a scratch directory for `test_name` and returns its path.
std::filesystem::path write_scratch_config(const std::string& example_name,
                                           const std::string& test_name, const std::string& body);

// Writes `body` as a scratch config, runs `binary --config <it> --validate-config-only`,
// and removes the scratch directory. What the validator said is in the result.
ProcessResult validate_config_body(const std::string& example_name, const std::string& binary,
                                   const std::string& test_name, const std::string& body);

} // namespace sima_examples::testing
