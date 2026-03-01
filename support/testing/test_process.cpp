#include "support/testing/test_process.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace sima_examples::testing {

const char* env_or_null(const char* key) {
  const char* value = std::getenv(key);
  return (value && *value) ? value : nullptr;
}

int env_int_or_default(const char* key, int default_value) {
  const char* raw = env_or_null(key);
  if (!raw)
    return default_value;
  return std::atoi(raw);
}

const char* env_or_skip(const char* key, const char* description) {
  const char* value = env_or_null(key);
  if (value)
    return value;

  if (require_e2e_mode()) {
    std::cerr << "[FAIL] " << key << " is required (" << description
              << ") but not set. Set it or unset SIMANEAT_APPS_TEST_REQUIRE_E2E.\n";
    std::exit(1);
  }

  std::cerr << "[SKIP] set " << key << " (" << description << ") to run this test\n";
  std::exit(kSkipCode);
}

bool require_e2e_mode() {
  const char* require = env_or_null("SIMANEAT_APPS_TEST_REQUIRE_E2E");
  return require && std::string(require) == "1";
}

int skip_or_fail(const std::string& reason) {
  if (require_e2e_mode()) {
    std::cerr << "[FAIL] " << reason << "\n";
    return 1;
  }
  std::cerr << "[SKIP] " << reason << "\n";
  return kSkipCode;
}

// ---------------------------------------------------------------------------
// Process management
// ---------------------------------------------------------------------------

namespace {

bool child_exited(pid_t pid, int& status_out) {
  const pid_t rc = ::waitpid(pid, &status_out, WNOHANG);
  return rc == pid;
}

void terminate_child(pid_t pid) {
  if (pid <= 0)
    return;
  (void)::kill(pid, SIGTERM);
  int status = 0;
  for (int i = 0; i < 20; ++i) {
    if (child_exited(pid, status))
      return;
    ::usleep(100000);
  }
  (void)::kill(pid, SIGKILL);
  (void)::waitpid(pid, &status, 0);
}

std::string read_fd(int fd) {
  std::string result;
  std::array<char, 4096> buf;
  while (true) {
    const ssize_t n = ::read(fd, buf.data(), buf.size());
    if (n <= 0)
      break;
    result.append(buf.data(), static_cast<size_t>(n));
  }
  return result;
}

} // namespace

ProcessResult spawn_and_wait(const std::string& binary,
                             const std::vector<std::string>& args,
                             int timeout_ms) {
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (::pipe(stdout_pipe) < 0 || ::pipe(stderr_pipe) < 0) {
    return {-1, "", std::string("pipe failed: ") + std::strerror(errno)};
  }

  std::vector<std::string> arg_storage;
  arg_storage.push_back(binary);
  for (auto& a : args) {
    arg_storage.push_back(a);
  }

  std::vector<char*> child_argv;
  child_argv.reserve(arg_storage.size() + 1);
  for (auto& s : arg_storage) {
    child_argv.push_back(s.data());
  }
  child_argv.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) {
    return {-1, "", std::string("fork failed: ") + std::strerror(errno)};
  }

  if (pid == 0) {
    // Child
    ::close(stdout_pipe[0]);
    ::close(stderr_pipe[0]);
    ::dup2(stdout_pipe[1], STDOUT_FILENO);
    ::dup2(stderr_pipe[1], STDERR_FILENO);
    ::close(stdout_pipe[1]);
    ::close(stderr_pipe[1]);
    ::execv(binary.c_str(), child_argv.data());
    _exit(127);
  }

  // Parent
  ::close(stdout_pipe[1]);
  ::close(stderr_pipe[1]);

  // Wait with timeout
  int elapsed_ms = 0;
  int status = 0;
  while (!child_exited(pid, status)) {
    if (timeout_ms > 0 && elapsed_ms >= timeout_ms) {
      terminate_child(pid);
      auto out = read_fd(stdout_pipe[0]);
      auto err = read_fd(stderr_pipe[0]);
      ::close(stdout_pipe[0]);
      ::close(stderr_pipe[0]);
      err += "\n[test_process] killed after timeout (" + std::to_string(timeout_ms) + "ms)";
      return {-1, std::move(out), std::move(err)};
    }
    ::usleep(50000); // 50ms
    elapsed_ms += 50;
  }

  auto out = read_fd(stdout_pipe[0]);
  auto err = read_fd(stderr_pipe[0]);
  ::close(stdout_pipe[0]);
  ::close(stderr_pipe[0]);

  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return {exit_code, std::move(out), std::move(err)};
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

std::string create_temp_dir(const std::string& prefix) {
  std::string tmpl = "/tmp/" + prefix + "XXXXXX";
  char* result = ::mkdtemp(tmpl.data());
  if (!result) {
    std::cerr << "[ERR] mkdtemp failed: " << std::strerror(errno) << "\n";
    return "";
  }
  return std::string(result);
}

void remove_dir(const std::string& path) {
  std::error_code ec;
  fs::remove_all(path, ec);
}

int count_files(const std::string& dir) {
  int count = 0;
  std::error_code ec;
  for (auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_regular_file())
      ++count;
  }
  return count;
}

bool all_files_nonempty(const std::string& dir) {
  std::error_code ec;
  for (auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_regular_file() && entry.file_size() == 0)
      return false;
  }
  return true;
}

} // namespace sima_examples::testing
