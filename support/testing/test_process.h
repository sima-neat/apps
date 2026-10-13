#pragma once

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace sima_examples::testing {

constexpr int kSkipCode = 77;

// Result of spawning a child process.
struct ProcessResult {
  int exit_code = -1; // WEXITSTATUS, or -1 when the process died from a signal
  std::string stdout_text;
  std::string stderr_text;
  // Set by spawn_until_output_files when the harness stopped the process itself,
  // with the signal that finally ended it: SIGINT when it shut down on request,
  // SIGTERM or SIGKILL when it had to be escalated.
  bool stopped_by_harness = false;
  int stop_signal = 0;
};

// Empty when the process exited cleanly: 0 on its own, or, when the harness
// stopped it, 0 or 130 after SIGINT without escalation. Otherwise the reason,
// naming the exit code or the signal, so a suite can print it after "[FAIL]".
std::string exit_problem(const ProcessResult& result);

// Read an environment variable; return nullptr if unset or empty.
const char* env_or_null(const char* key);

// Read an integer environment variable, returning default_value if unset.
int env_int_or_default(const char* key, int default_value);

// Read SIMANEAT_TEST_RTSP_H264_URLS, falling back to SIMANEAT_TEST_RTSP_H264_URL.
std::vector<std::string> rtsp_h264_urls_from_env();

// Read SIMANEAT_TEST_RTSP_H265_URLS, falling back to SIMANEAT_TEST_RTSP_H265_URL.
std::vector<std::string> rtsp_h265_urls_from_env();

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

// Spawn a process, poll `ready` every 100 ms until it returns true, then stop the
// process: SIGINT first, which the applications handle, escalating to SIGTERM and
// SIGKILL only if it ignores that. The result carries the real exit status and how
// the process was stopped; use exit_problem() to judge it. `ready` is also where a
// caller does its own polling, such as reading metadata off a listener.
ProcessResult spawn_until(const std::string& binary, const std::vector<std::string>& args,
                          const std::function<bool()>& ready, int timeout_ms = 30000);

// spawn_until() with "output_dir holds expected_files finished files" as the condition.
// A file that was still being written when the stop landed is discarded.
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

// Count the images a directory-based application will process, so a suite can
// size its output expectation from its input instead of from a constant.
int supported_image_count(const std::string& dir);

// The output check for an application that annotates a directory of images: at
// least `minimum` files, every one of them decoding as an image no smaller than
// min_side on each side. Returns an empty string when the output is usable, and
// otherwise the reason to print. `st_size > 0` passes on a truncated JPEG; this
// does not. Nothing here asks the frames to differ, because two identical input
// images correctly produce two identical outputs.
std::string saved_frames_problem(const std::string& dir, int minimum, int min_side = 16);

// The same, plus the assertion a stream can be held to and a batch cannot: the
// frames have to move. Frames are grouped by the stream named in their filename
// (stream_<n>_frame_<m>.jpg) and each stream is required to advance against
// itself, so a stream frozen beside a working one is still caught.
//
// Without a stream count a one-frame group is skipped, because the helper
// cannot tell "this stream wrote one frame and stalled" from "this run only
// writes one frame". A suite that knows how many streams it configured passes
// that as expected_streams, and then every stream must be present with at
// least two frames — a stream that stalled after its first frame, or saved
// nothing at all, is a failure rather than a pass carried by its neighbour.
std::string streamed_frames_problem(const std::string& dir, int minimum, int expected_streams = 0,
                                    int min_side = 16);

} // namespace sima_examples::testing
