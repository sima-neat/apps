// Host-side unit tests for the PCIe stall detector (StallTracker).
//
// The CMake harness invokes this as "<example>.unit" and passes the built app
// binary as argv[1]; these checks are pure logic and ignore it. Exit non-zero on
// any failure. The two scenarios the code review called out are pinned here:
//   * a frame admitted after a long idle gap must NOT instantly trip the timeout;
//   * a freeze with work already outstanding MUST trip, even with no new
//     admissions afterwards.

#include "../../src/host/stall_tracker.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;
using pcie_high_density::StallTracker;

int g_failures = 0;

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    ++g_failures;
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

const Clock::time_point kBase{};
Clock::time_point at(int seconds) { return kBase + std::chrono::seconds(seconds); }
const std::chrono::milliseconds kTimeout{60000};  // 60 s, the documented default

// Healthy steady state: the card keeps returning results with a small, constant
// backlog in flight. Progress each tick must keep the detector quiet.
void test_steady_progress_never_trips() {
  StallTracker tracker;
  std::uint64_t returned = 0;
  std::uint64_t admitted = 5;  // 5 frames permanently in flight
  bool tripped = false;
  for (int s = 0; s <= 300; ++s) {
    admitted += 30;
    returned += 30;  // results keep flowing -> made_progress every tick
    tripped |= tracker.update(returned, admitted, at(s), kTimeout);
  }
  expect_true(!tripped, "steady progress with a constant backlog never trips");
}

// Regression: idle time must not count toward the deadline. After ~70 idle
// seconds with nothing outstanding, a newly admitted frame must not trip.
void test_idle_then_admit_does_not_instantly_trip() {
  StallTracker tracker;
  std::uint64_t returned = 100;
  std::uint64_t admitted = 100;  // fully caught up

  bool tripped_idle = false;
  for (int s = 0; s <= 70; ++s) {
    tripped_idle |= tracker.update(returned, admitted, at(s), kTimeout);
  }
  expect_true(!tripped_idle, "no trip during a 70 s idle gap (nothing outstanding)");

  admitted = 101;  // one frame admitted after the idle gap
  const bool tripped_now = tracker.update(returned, admitted, at(71), kTimeout);
  expect_true(!tripped_now, "a frame admitted after 70 s idle does not instantly trip");

  bool tripped_before_deadline = false;
  for (int s = 72; s <= 129; ++s) {  // < 60 s since the admission's reference tick
    tripped_before_deadline |= tracker.update(returned, admitted, at(s), kTimeout);
  }
  expect_true(!tripped_before_deadline, "does not trip before the timeout elapses");

  const bool tripped_after = tracker.update(returned, admitted, at(132), kTimeout);
  expect_true(tripped_after, "trips once the timeout elapses after the admission");
}

// Regression: a freeze with work already outstanding must trip even though no
// further frames are admitted (the old check required a later admission).
void test_freeze_with_outstanding_work_trips() {
  StallTracker tracker;
  std::uint64_t returned = 0;
  std::uint64_t admitted = 0;
  for (int s = 0; s < 10; ++s) {  // healthy ramp
    admitted += 30;
    returned += 30;
    tracker.update(returned, admitted, at(s), kTimeout);
  }

  admitted += 20;  // 20 frames outstanding, then everything freezes (no new admits, no returns)

  bool tripped_before_deadline = false;
  for (int s = 10; s <= 68; ++s) {  // < 60 s since the last result (at t=9)
    tripped_before_deadline |= tracker.update(returned, admitted, at(s), kTimeout);
  }
  expect_true(!tripped_before_deadline, "freeze does not trip before the timeout");

  const bool tripped_after = tracker.update(returned, admitted, at(72), kTimeout);
  expect_true(tripped_after,
              "freeze with outstanding work trips without any new admissions");
}

// A non-positive timeout disables detection entirely.
void test_zero_timeout_disables_detection() {
  StallTracker tracker;
  bool tripped = false;
  for (int s = 0; s < 200; ++s) {
    tripped |= tracker.update(/*returned=*/0, /*admitted=*/50, at(s), std::chrono::milliseconds{0});
  }
  expect_true(!tripped, "a non-positive timeout disables stall detection");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  test_steady_progress_never_trips();
  test_idle_then_admit_does_not_instantly_trip();
  test_freeze_with_outstanding_work_trips();
  test_zero_timeout_disables_detection();

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all stall-tracker checks passed\n";
  return 0;
}
