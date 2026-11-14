#ifndef PCIE_HIGH_DENSITY_STALL_TRACKER_H
#define PCIE_HIGH_DENSITY_STALL_TRACKER_H

#include <chrono>
#include <cstdint>

namespace pcie_high_density {

// Use one tracker per stream. `results` counts transport arrivals, including results
// later dropped by the dispatcher. Work is live pending frames or continued input,
// not the difference between cumulative admission and completion counters.
class StallTracker {
public:
  using clock = std::chrono::steady_clock;

  // Returns true once the card has returned no result for `timeout` while work
  // is outstanding. A non-positive timeout disables detection (returns false).
  bool update(std::uint64_t results, bool work_outstanding, clock::time_point now,
              std::chrono::milliseconds timeout) {
    if (!initialized_) {
      initialized_ = true;
      last_progress_ = now;
      last_returned_ = results;
    }

    if (timeout <= std::chrono::milliseconds::zero()) {
      // Detection disabled: keep the trackers current so it can be re-enabled
      // mid-run without a spurious immediate trip.
      last_returned_ = results;
      last_progress_ = now;
      return false;
    }

    const bool made_progress = results != last_returned_;

    // Hold the clock while results are still coming back, or while nothing is
    // outstanding. In the outstanding case the previous tick left the clock at
    // ~now, so the deadline starts when work resumes -- idle time never counts.
    if (made_progress || !work_outstanding) {
      last_returned_ = results;
      last_progress_ = now;
      return false;
    }

    return (now - last_progress_) >= timeout;
  }

  // How long the currently-outstanding work has gone without a result.
  clock::duration stalled_for(clock::time_point now) const {
    return now - last_progress_;
  }

private:
  bool initialized_ = false;
  clock::time_point last_progress_{};
  std::uint64_t last_returned_ = 0;
};

} // namespace pcie_high_density

#endif // PCIE_HIGH_DENSITY_STALL_TRACKER_H
