#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace high_density {

// Counts successful sends after every stream has completed warm-up.
struct MetadataMeasurement {
  std::uint64_t target;
  int warmup;
  std::vector<bool> ready;
  std::vector<std::uint64_t> frames;
  std::vector<std::uint64_t> failures;
  bool started = false;
  double start = 0.0;
  double elapsed = 0.0;
  std::uint64_t total = 0;

  MetadataMeasurement(std::size_t streams, int warmup_frames, std::uint64_t frame_target)
      : target(frame_target), warmup(warmup_frames), ready(streams, warmup_frames == 0),
        frames(streams, 0), failures(streams, 0) {}

  bool observe(std::size_t stream, int processed, bool sent, bool failed, double now) {
    if (target == 0)
      return false;
    if (!started) {
      ready.at(stream) = processed >= warmup;
      if (std::all_of(ready.begin(), ready.end(), [](bool value) { return value; })) {
        started = true;
        start = now;
      }
      return false;
    }
    frames.at(stream) += sent;
    failures.at(stream) += failed;
    total += sent;
    elapsed = now - start;
    return total >= target;
  }
};

} // namespace high_density
