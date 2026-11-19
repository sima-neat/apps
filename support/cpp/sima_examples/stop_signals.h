#pragma once

// Stop an example on Ctrl-C, SIGTERM, or a dropped ssh session (SIGHUP). Mirrors
// sima_examples.stop_signals in support/python.

#include <atomic>
#include <csignal>

#include <signal.h>

namespace sima_examples {

inline std::atomic<bool>& stop_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}

// A flag that SIGINT, SIGTERM, and SIGHUP set. Pull loops check it between pulls. A signal the
// process started with ignored stays ignored, so a run under `nohup` keeps going when its
// terminal closes. The application may also set the flag, for example when a worker thread fails.
inline std::atomic<bool>& stop_on_signals() {
  std::atomic<bool>& flag = stop_flag();
  for (const int signal_number : {SIGINT, SIGTERM, SIGHUP}) {
    struct sigaction current{};
    sigaction(signal_number, nullptr, &current);
    if (current.sa_handler != SIG_IGN) {
      std::signal(signal_number, [](int) { stop_flag() = true; });
    }
  }
  return flag;
}

} // namespace sima_examples
