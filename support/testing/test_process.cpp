#include "support/testing/test_process.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <signal.h>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

namespace {

std::vector<std::string> split_csv(const std::string& raw) {
  std::vector<std::string> out;
  std::istringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const std::size_t start = token.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    const std::size_t end = token.find_last_not_of(" \t\r\n");
    out.push_back(token.substr(start, end - start + 1));
  }
  return out;
}

} // namespace

std::vector<std::string> rtsp_h264_urls_from_env() {
  if (const char* urls = env_or_null("SIMANEAT_TEST_RTSP_H264_URLS")) {
    return split_csv(urls);
  }
  if (const char* url = env_or_null("SIMANEAT_TEST_RTSP_H264_URL")) {
    return {url};
  }
  return {};
}

std::vector<std::string> rtsp_h265_urls_from_env() {
  if (const char* urls = env_or_null("SIMANEAT_TEST_RTSP_H265_URLS")) {
    return split_csv(urls);
  }
  if (const char* url = env_or_null("SIMANEAT_TEST_RTSP_H265_URL")) {
    return {url};
  }
  return {};
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

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::string command_line(const std::string& binary, const std::vector<std::string>& args) {
  std::string command = shell_quote(binary);
  for (const auto& arg : args) {
    command += " " + shell_quote(arg);
  }
  return command;
}

fs::path artifact_dir_from_config_arg(const std::vector<std::string>& args) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] != "--config") {
      continue;
    }
    const fs::path config_path(args[i + 1]);
    const fs::path parent = config_path.parent_path();
    if (!parent.empty() && fs::exists(parent)) {
      return parent;
    }
  }
  return {};
}

void write_process_artifacts(const fs::path& dir, const std::string& command,
                             const ProcessResult& result) {
  const char* write_logs = env_or_null("SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS");
  if (dir.empty() || (write_logs && std::string(write_logs) == "0")) {
    return;
  }
  std::ofstream(dir / "command.txt") << command << "\n";
  std::ofstream(dir / "stdout.log") << result.stdout_text;
  std::ofstream(dir / "stderr.log") << result.stderr_text;
}

bool is_output_file(const fs::directory_entry& entry) {
  return entry.is_regular_file() && entry.path().filename() != "config.yaml";
}

// What the directory-based applications accept, from their shared is_image().
bool is_image_suffix(const fs::path& path) {
  std::string suffix = path.extension().string();
  for (char& c : suffix) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png" || suffix == ".bmp";
}

std::vector<fs::path> saved_image_files(const std::string& dir) {
  std::vector<fs::path> paths;
  std::error_code ec;
  for (fs::recursive_directory_iterator entry(dir, ec), end; entry != end && !ec;
       entry.increment(ec)) {
    if (is_output_file(*entry))
      paths.push_back(entry->path());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

// The multi-stream applications name every frame after the stream that produced
// it, e.g. stream_1_frame_40.jpg, and write them all into one directory. Names
// without a stream stay together as one group, which is the whole run.
std::map<std::string, std::vector<fs::path>> group_by_stream(const std::vector<fs::path>& paths) {
  static const std::regex stream_in_name(R"(^stream[_-]?([0-9]+)[_-])");
  std::map<std::string, std::vector<fs::path>> groups;
  for (const fs::path& path : paths) {
    const std::string name = path.filename().string();
    std::smatch match;
    const std::string label =
        std::regex_search(name, match, stream_in_name) ? "stream " + match[1].str() : "the run";
    groups[label].push_back(path);
  }
  return groups;
}

using OutputSizes = std::map<fs::path, std::uintmax_t>;

OutputSizes output_sizes(const std::string& dir) {
  OutputSizes sizes;
  std::error_code ec;
  for (fs::recursive_directory_iterator entry(dir, ec), end; entry != end && !ec;
       entry.increment(ec)) {
    if (!is_output_file(*entry))
      continue;
    std::error_code size_ec;
    const std::uintmax_t size = entry->file_size(size_ec);
    // The producer may be replacing the file right now; treat it as in flight.
    sizes[entry->path()] = size_ec ? 0 : size;
  }
  return sizes;
}

// Which output files the producer has demonstrably finished writing. The
// applications write each frame straight to its final path, so a file is
// visible, and already non-empty, before its last byte lands. Counting a file
// on sight lets the poll stop the producer mid-write, and the truncated frame
// then reads as an application failure the harness caused. A file counts only
// once its size has stopped changing and it decodes as an image, which costs
// one decode per file because the answer is remembered per size.
OutputSizes confirm_finished_outputs(const OutputSizes& sizes, const OutputSizes& previous,
                                     bool have_previous, const OutputSizes& finished) {
  OutputSizes confirmed;
  for (const auto& [path, size] : finished) {
    const auto current = sizes.find(path);
    if (current != sizes.end() && current->second == size)
      confirmed[path] = size;
  }
  if (!have_previous)
    return confirmed;

  for (const auto& [path, size] : sizes) {
    if (size == 0 || confirmed.count(path))
      continue;
    const auto before = previous.find(path);
    if (before == previous.end() || before->second != size)
      continue;
    if (!cv::imread(path.string(), cv::IMREAD_COLOR).empty())
      confirmed[path] = size;
  }
  return confirmed;
}

// Drop whatever the producer had not finished when the signal reached it.
// Termination is requested as soon as enough frames are confirmed, so a later
// frame can still be mid-write when SIGTERM arrives and lose its tail. That
// truncation belongs to the harness, not to the application, and the confirmed
// frames already satisfy the count the suite asked for. Nothing is discarded
// when the producer exits on its own, so a crash mid-write is still visible.
void discard_unfinished_writes(const std::string& dir, const OutputSizes& finished) {
  for (const auto& [path, size] : output_sizes(dir)) {
    const auto confirmed = finished.find(path);
    if (confirmed == finished.end() || confirmed->second != size) {
      std::error_code ec;
      fs::remove(path, ec);
    }
  }
}

} // namespace

ProcessResult spawn_and_wait(const std::string& binary, const std::vector<std::string>& args,
                             int timeout_ms) {
  const fs::path artifact_dir = artifact_dir_from_config_arg(args);
  const std::string command = command_line(binary, args);
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (::pipe(stdout_pipe) < 0 || ::pipe(stderr_pipe) < 0) {
    ProcessResult result{-1, "", std::string("pipe failed: ") + std::strerror(errno)};
    write_process_artifacts(artifact_dir, command, result);
    return result;
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
    ProcessResult result{-1, "", std::string("fork failed: ") + std::strerror(errno)};
    write_process_artifacts(artifact_dir, command, result);
    return result;
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
      ProcessResult result{-1, std::move(out), std::move(err)};
      write_process_artifacts(artifact_dir, command, result);
      return result;
    }
    ::usleep(50000); // 50ms
    elapsed_ms += 50;
  }

  auto out = read_fd(stdout_pipe[0]);
  auto err = read_fd(stderr_pipe[0]);
  ::close(stdout_pipe[0]);
  ::close(stderr_pipe[0]);

  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  ProcessResult result{exit_code, std::move(out), std::move(err)};
  write_process_artifacts(artifact_dir, command, result);
  return result;
}

ProcessResult spawn_until_output_files(const std::string& binary,
                                       const std::vector<std::string>& args,
                                       const std::string& output_dir, int expected_files,
                                       int timeout_ms) {
  const fs::path artifact_dir = artifact_dir_from_config_arg(args);
  const std::string command = command_line(binary, args);
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (::pipe(stdout_pipe) < 0 || ::pipe(stderr_pipe) < 0) {
    ProcessResult result{-1, "", std::string("pipe failed: ") + std::strerror(errno)};
    write_process_artifacts(artifact_dir, command, result);
    return result;
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
    ProcessResult result{-1, "", std::string("fork failed: ") + std::strerror(errno)};
    write_process_artifacts(artifact_dir, command, result);
    return result;
  }

  if (pid == 0) {
    ::close(stdout_pipe[0]);
    ::close(stderr_pipe[0]);
    ::dup2(stdout_pipe[1], STDOUT_FILENO);
    ::dup2(stderr_pipe[1], STDERR_FILENO);
    ::close(stdout_pipe[1]);
    ::close(stderr_pipe[1]);
    ::execv(binary.c_str(), child_argv.data());
    _exit(127);
  }

  ::close(stdout_pipe[1]);
  ::close(stderr_pipe[1]);

  int elapsed_ms = 0;
  int status = 0;
  OutputSizes previous_sizes;
  bool have_previous = false;
  OutputSizes finished;
  while (!child_exited(pid, status)) {
    if (expected_files > 0) {
      const OutputSizes sizes = output_sizes(output_dir);
      finished = confirm_finished_outputs(sizes, previous_sizes, have_previous, finished);
      previous_sizes = sizes;
      have_previous = true;
      if (static_cast<int>(finished.size()) >= expected_files) {
        terminate_child(pid);
        discard_unfinished_writes(output_dir, finished);
        auto out = read_fd(stdout_pipe[0]);
        auto err = read_fd(stderr_pipe[0]);
        ::close(stdout_pipe[0]);
        ::close(stderr_pipe[0]);
        ProcessResult result{0, std::move(out), std::move(err)};
        write_process_artifacts(artifact_dir, command, result);
        return result;
      }
    }
    if (timeout_ms > 0 && elapsed_ms >= timeout_ms) {
      terminate_child(pid);
      auto out = read_fd(stdout_pipe[0]);
      auto err = read_fd(stderr_pipe[0]);
      ::close(stdout_pipe[0]);
      ::close(stderr_pipe[0]);
      err += "\n[test_process] killed after timeout (" + std::to_string(timeout_ms) + "ms)";
      ProcessResult result{-1, std::move(out), std::move(err)};
      write_process_artifacts(artifact_dir, command, result);
      return result;
    }
    ::usleep(100000);
    elapsed_ms += 100;
  }

  auto out = read_fd(stdout_pipe[0]);
  auto err = read_fd(stderr_pipe[0]);
  ::close(stdout_pipe[0]);
  ::close(stderr_pipe[0]);

  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  ProcessResult result{exit_code, std::move(out), std::move(err)};
  write_process_artifacts(artifact_dir, command, result);
  return result;
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

std::string create_test_output_dir(const std::string& example_name, const std::string& test_name) {
  const char* out_root_raw = env_or_null("SIMANEAT_APPS_TEST_OUTPUT_DIR");
  const char* apps_root_raw = env_or_null("APPS_ROOT");
  const fs::path base_root = out_root_raw
                                 ? fs::path(out_root_raw)
                                 : fs::path(apps_root_raw ? apps_root_raw : ".") / "sandbox-test";
  const fs::path run_dir = base_root / "cpp" / example_name / test_name;
  const fs::path out_dir = run_dir / "out";
  std::error_code ec;
  fs::remove_all(run_dir, ec);
  if (ec) {
    std::cerr << "[ERR] failed to clear output directory '" << run_dir.string()
              << "': " << ec.message() << "\n";
    return "";
  }
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "[ERR] failed to create output directory '" << out_dir.string()
              << "': " << ec.message() << "\n";
    return "";
  }
  return out_dir.string();
}

std::string create_test_scratch_dir(const std::string& example_name, const std::string& test_name) {
  const fs::path run_dir =
      fs::temp_directory_path() / "neat-apps-test-scratch" / "cpp" / example_name / test_name;
  std::error_code ec;
  fs::remove_all(run_dir, ec);
  if (ec) {
    std::cerr << "[ERR] failed to clear scratch directory '" << run_dir.string()
              << "': " << ec.message() << "\n";
    return "";
  }
  fs::create_directories(run_dir, ec);
  if (ec) {
    std::cerr << "[ERR] failed to create scratch directory '" << run_dir.string()
              << "': " << ec.message() << "\n";
    return "";
  }
  return run_dir.string();
}

void remove_dir(const std::string& path) {
  const char* keep_raw = env_or_null("SIMANEAT_APPS_TEST_KEEP_OUTPUT");
  const fs::path target(path);
  const bool is_scratch = target.string().find("neat-apps-test-scratch") != std::string::npos;
  if (keep_raw && std::string(keep_raw) == "1" && !is_scratch) {
    return;
  }
  std::error_code ec;
  fs::remove_all(target.filename() == "out" ? target.parent_path() : target, ec);
}

int count_output_files(const std::string& dir) {
  int count = 0;
  std::error_code ec;
  for (fs::recursive_directory_iterator entry(dir, ec), end; entry != end && !ec;
       entry.increment(ec)) {
    if (is_output_file(*entry))
      ++count;
  }
  return count;
}

bool all_output_files_nonempty(const std::string& dir) {
  std::error_code ec;
  for (fs::recursive_directory_iterator entry(dir, ec), end; entry != end && !ec;
       entry.increment(ec)) {
    if (is_output_file(*entry) && entry->file_size() == 0)
      return false;
  }
  return true;
}

int supported_image_count(const std::string& dir) {
  int count = 0;
  std::error_code ec;
  for (fs::directory_iterator entry(dir, ec), end; entry != end && !ec; entry.increment(ec)) {
    if (entry->is_regular_file() && is_image_suffix(entry->path()))
      ++count;
  }
  return count;
}

std::string saved_frames_problem(const std::string& dir, int minimum, int min_side) {
  const std::vector<fs::path> paths = saved_image_files(dir);
  if (static_cast<int>(paths.size()) < minimum) {
    return "expected at least " + std::to_string(minimum) + " saved frames, got " +
           std::to_string(paths.size());
  }

  for (const fs::path& path : paths) {
    const cv::Mat frame = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
      std::error_code ec;
      return path.filename().string() + " is " + std::to_string(fs::file_size(path, ec)) +
             " bytes but does not decode as an image";
    }
    if (frame.cols < min_side || frame.rows < min_side) {
      return path.filename().string() + " decoded to an implausible " +
             std::to_string(frame.cols) + "x" + std::to_string(frame.rows);
    }
  }
  return "";
}

std::string streamed_frames_problem(const std::string& dir, int minimum, int min_side) {
  const std::string problem = saved_frames_problem(dir, minimum, min_side);
  if (!problem.empty())
    return problem;

  for (const auto& [label, group] : group_by_stream(saved_image_files(dir))) {
    if (group.size() < 2)
      continue;

    // Held one at a time against the first: a run can save many large frames.
    const cv::Mat first = cv::imread(group.front().string(), cv::IMREAD_COLOR);
    bool advanced = false;
    for (std::size_t index = 1; index < group.size() && !advanced; ++index) {
      const cv::Mat frame = cv::imread(group[index].string(), cv::IMREAD_COLOR);
      advanced = frame.size() != first.size() || frame.type() != first.type() ||
                 cv::norm(frame, first, cv::NORM_L1) > 0;
    }
    if (!advanced) {
      std::string names;
      for (std::size_t index = 0; index < group.size() && index < 4; ++index) {
        names += (index ? ", " : "") + group[index].filename().string();
      }
      return "all " + std::to_string(group.size()) + " frames saved for " + label +
             " are pixel-identical (" + names +
             "...); the pipeline is not advancing through the stream";
    }
  }
  return "";
}

} // namespace sima_examples::testing
