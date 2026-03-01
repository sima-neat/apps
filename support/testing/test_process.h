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
ProcessResult spawn_and_wait(const std::string& binary,
                             const std::vector<std::string>& args,
                             int timeout_ms = 30000);

// Create a temporary directory under /tmp.  Returns the path.
std::string create_temp_dir(const std::string& prefix = "sima_test_");

// Remove a directory tree.
void remove_dir(const std::string& path);

// Count regular files in a directory (non-recursive).
int count_files(const std::string& dir);

// Return true if every regular file in dir is non-empty.
bool all_files_nonempty(const std::string& dir);

} // namespace sima_examples::testing
