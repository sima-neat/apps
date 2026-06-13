#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace sima_examples::testing {

constexpr int kSkipCode = 77;

// Result of spawning a child process.
struct ProcessResult {
  int exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
};

// Read an environment variable; return nullptr if unset or empty.
const char* env_or_null(const char* key);

// Read an integer environment variable, returning default_value if unset.
int env_int_or_default(const char* key, int default_value);

// Read SIMANEAT_APPS_TEST_RTSP_URLS, falling back to SIMANEAT_APPS_TEST_RTSP_URL.
std::vector<std::string> rtsp_urls_from_env();

// Read a required environment variable.  If unset and
// SIMANEAT_APPS_TEST_REQUIRE_E2E=1, print an error and exit(1).
// Otherwise print a skip message and exit(77).
// Only returns if the variable is set.
const char* env_or_skip(const char* key, const char* description);

// True when SIMANEAT_APPS_TEST_REQUIRE_E2E=1.
bool require_e2e_mode();

// Return 1 in strict mode, else 77. Also prints a clear reason message.
int skip_or_fail(const std::string& reason);

// Spawn a process and capture its exit code, stdout and stderr.
// If timeout_ms > 0, SIGTERM the child after that many milliseconds.
ProcessResult spawn_and_wait(const std::string& binary, const std::vector<std::string>& args,
                             int timeout_ms = 30000);

// Spawn a process, wait until output_dir contains expected_files, then terminate it.
// Returns exit_code 0 when the output condition is reached before timeout.
ProcessResult spawn_until_output_files(const std::string& binary,
                                       const std::vector<std::string>& args,
                                       const std::string& output_dir, int expected_files,
                                       int timeout_ms = 30000);

// Create a deterministic e2e output directory under
// SIMANEAT_APPS_TEST_OUTPUT_DIR/cpp/<example>/<test>/out
// (or sandbox-test/cpp/.../out if unset).
// The example run directory is cleared before each run.
std::string create_test_output_dir(const std::string& example_name, const std::string& test_name);

// Create a deterministic scratch directory for C++ unit tests.
std::string create_test_scratch_dir(const std::string& example_name, const std::string& test_name);

// Remove a directory tree. Passing a generated out/ path removes its parent run directory.
void remove_dir(const std::string& path);

// Count regular output files in a directory tree, excluding the test config file.
int count_output_files(const std::string& dir);

// Return true if every regular output file in dir is non-empty.
bool all_output_files_nonempty(const std::string& dir);

} // namespace sima_examples::testing
