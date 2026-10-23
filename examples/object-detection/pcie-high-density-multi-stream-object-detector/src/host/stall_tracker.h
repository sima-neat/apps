#ifndef PCIE_HIGH_DENSITY_STALL_TRACKER_H
#define PCIE_HIGH_DENSITY_STALL_TRACKER_H

#include <chrono>
#include <cstdint>

namespace pcie_high_density {

// Detects a stalled PCIe card: frames were admitted to the card but it has
// stopped returning results. Header-only and free of GStreamer/PCIe/Neat
// dependencies so the decision can be exercised by host-side unit tests.
//
// The naive "time since the last result" check got two things wrong; both are
// pinned by the regression tests:
//
//   * Idle time must NOT count toward the deadline. When every admitted frame
//     has already been returned there is nothing to wait for, so the clock is
//     held at `now`. A frame admitted after a long idle gap therefore starts a
//     fresh deadline instead of instantly tripping the timeout.
//
//   * A freeze with work already outstanding MUST trip, even when no *new*
//     frames are admitted afterwards. The deadline is driven by unfinished work
//     (admitted > returned), not by fresh admissions.
//
// `returned` and `admitted` are cumulative, monotonically non-decreasing
// counters. Call update() once per poll with the current time.
class StallTracker {
public:
  using clock = std::chrono::steady_clock;

  // Returns true once the card has returned no result for `timeout` while work
  // is outstanding. A non-positive timeout disables detection (returns false).
  bool update(std::uint64_t returned, std::uint64_t admitted, clock::time_point now,
              std::chrono::milliseconds timeout) {
    if (!initialized_) {
      initialized_ = true;
      last_progress_ = now;
      last_returned_ = returned;
    }

    if (timeout <= std::chrono::milliseconds::zero()) {
      // Detection disabled: keep the trackers current so it can be re-enabled
      // mid-run without a spurious immediate trip.
      last_returned_ = returned;
      last_progress_ = now;
      return false;
    }

    const bool made_progress = returned != last_returned_;
    const bool work_outstanding = admitted > returned;

    // Hold the clock while results are still coming back, or while nothing is
    // outstanding. In the outstanding case the previous tick left the clock at
    // ~now, so the deadline starts when work resumes -- idle time never counts.
    if (made_progress || !work_outstanding) {
      last_returned_ = returned;
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
