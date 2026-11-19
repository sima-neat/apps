#pragma once

// Stop an example on Ctrl-C, SIGTERM, or a dropped ssh session (SIGHUP). Mirrors
// sima_examples.stop_signals in support/python.

#include <atomic>
#include <csignal>

namespace sima_examples {

inline std::atomic<bool>& stop_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}

// A flag that SIGINT, SIGTERM, and SIGHUP set. Pull loops check it between pulls, and the
// application may also set it, for example when a worker thread fails.
inline std::atomic<bool>& stop_on_signals() {
  std::atomic<bool>& flag = stop_flag();
  for (const int signal_number : {SIGINT, SIGTERM, SIGHUP}) {
    std::signal(signal_number, [](int) { stop_flag() = true; });
  }
  return flag;
}

} // namespace sima_examples
