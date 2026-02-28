#include "support/testing/optiview_json_listener.h"

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kSkipCode = 77;
constexpr int kDefaultJsonPort = 9100;
constexpr int kDefaultVideoPort = 9000;
constexpr int kDefaultTimeoutMs = 10000;

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

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const char* rtsp_url = env_or_null("SIMANEAT_APPS_TEST_RTSP_URL");
  if (!rtsp_url) {
    std::cerr << "[SKIP] set SIMANEAT_APPS_TEST_RTSP_URL to run the OptiView e2e test\n";
    return kSkipCode;
  }

  const char* mpk_path = env_or_null("SIMANEAT_APPS_TEST_MPK");
  const int json_port = env_int_or_default("SIMANEAT_APPS_TEST_OPTIVIEW_JSON_PORT", kDefaultJsonPort);
  const int video_port =
      env_int_or_default("SIMANEAT_APPS_TEST_OPTIVIEW_VIDEO_PORT", kDefaultVideoPort);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", kDefaultTimeoutMs);

  sima_examples::testing::OptiViewJsonListenerOptions listener_opt;
  listener_opt.host = "127.0.0.1";
  listener_opt.base_port = json_port;
  listener_opt.num_ports = 1;
  listener_opt.timeout_ms = timeout_ms;
  listener_opt.require_all_ports = false;

  sima_examples::testing::OptiViewJsonListener listener(listener_opt);
  if (!listener.ok()) {
    std::cerr << "[ERR] listener init failed: " << listener.error() << "\n";
    return 1;
  }

  std::vector<std::string> arg_storage;
  arg_storage.push_back(argv[1]);
  arg_storage.push_back("--rtsp");
  arg_storage.push_back(rtsp_url);
  arg_storage.push_back("--optiview-host");
  arg_storage.push_back("127.0.0.1");
  arg_storage.push_back("--optiview-video-port");
  arg_storage.push_back(std::to_string(video_port));
  arg_storage.push_back("--optiview-json-port");
  arg_storage.push_back(std::to_string(json_port));
  arg_storage.push_back("--frames");
  arg_storage.push_back("300");
  if (mpk_path) {
    arg_storage.push_back("--mpk");
    arg_storage.push_back(mpk_path);
  }

  std::vector<char*> child_argv;
  child_argv.reserve(arg_storage.size() + 1);
  for (auto& arg : arg_storage) {
    child_argv.push_back(arg.data());
  }
  child_argv.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) {
    std::cerr << "[ERR] fork failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  if (pid == 0) {
    ::execv(argv[1], child_argv.data());
    std::cerr << "[ERR] execv failed: " << std::strerror(errno) << "\n";
    _exit(127);
  }

  const auto result = listener.wait_for_messages();
  terminate_child(pid);

  if (!result.success) {
    std::cerr << "[ERR] no valid OptiView JSON received: " << result.error << "\n";
    return 1;
  }

  std::cout << "[OK] received valid OptiView JSON on ports:";
  for (int port : result.ports_with_valid_json) {
    std::cout << " " << port;
  }
  std::cout << "\n";
  return 0;
}
