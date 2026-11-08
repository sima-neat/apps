// Regression checks for per-stream progress and shared PCIe configuration.
// These tests do not require a model, a PCIe device, or live services.

#include "../../src/host/stall_tracker.h"
#include "../../src/common/app_config.h"

#include <filesystem>
#include <fstream>
#include <unistd.h>

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
Clock::time_point at(int seconds) {
  return kBase + std::chrono::seconds(seconds);
}
const std::chrono::milliseconds kTimeout{60000}; // 60 s, the documented default

// Healthy steady state: the card keeps returning results with a small, constant
// backlog in flight. Progress each tick must keep the detector quiet.
void test_steady_progress_never_trips() {
  StallTracker tracker;
  std::uint64_t returned = 0;
  std::uint64_t admitted = 5; // 5 frames permanently in flight
  bool tripped = false;
  for (int s = 0; s <= 300; ++s) {
    admitted += 30;
    returned += 30; // results keep flowing -> made_progress every tick
    tripped |= tracker.update(returned, admitted > returned, at(s), kTimeout);
  }
  expect_true(!tripped, "steady progress with a constant backlog never trips");
}

// Regression: idle time must not count toward the deadline. After ~70 idle
// seconds with nothing outstanding, a newly admitted frame must not trip.
void test_idle_then_admit_does_not_instantly_trip() {
  StallTracker tracker;
  std::uint64_t returned = 100;
  std::uint64_t admitted = 100; // fully caught up

  bool tripped_idle = false;
  for (int s = 0; s <= 70; ++s) {
    tripped_idle |= tracker.update(returned, admitted > returned, at(s), kTimeout);
  }
  expect_true(!tripped_idle, "no trip during a 70 s idle gap (nothing outstanding)");

  admitted = 101; // one frame admitted after the idle gap
  const bool tripped_now = tracker.update(returned, admitted > returned, at(71), kTimeout);
  expect_true(!tripped_now, "a frame admitted after 70 s idle does not instantly trip");

  bool tripped_before_deadline = false;
  for (int s = 72; s <= 129; ++s) { // < 60 s since the admission's reference tick
    tripped_before_deadline |= tracker.update(returned, admitted > returned, at(s), kTimeout);
  }
  expect_true(!tripped_before_deadline, "does not trip before the timeout elapses");

  const bool tripped_after = tracker.update(returned, admitted > returned, at(132), kTimeout);
  expect_true(tripped_after, "trips once the timeout elapses after the admission");
}

// Regression: a freeze with work already outstanding must trip even though no
// further frames are admitted (the old check required a later admission).
void test_freeze_with_outstanding_work_trips() {
  StallTracker tracker;
  std::uint64_t returned = 0;
  std::uint64_t admitted = 0;
  for (int s = 0; s < 10; ++s) { // healthy ramp
    admitted += 30;
    returned += 30;
    tracker.update(returned, admitted > returned, at(s), kTimeout);
  }

  admitted += 20; // 20 frames outstanding, then everything freezes (no new admits, no returns)

  bool tripped_before_deadline = false;
  for (int s = 10; s <= 68; ++s) { // < 60 s since the last result (at t=9)
    tripped_before_deadline |= tracker.update(returned, admitted > returned, at(s), kTimeout);
  }
  expect_true(!tripped_before_deadline, "freeze does not trip before the timeout");

  const bool tripped_after = tracker.update(returned, admitted > returned, at(72), kTimeout);
  expect_true(tripped_after, "freeze with outstanding work trips without any new admissions");
}

// A non-positive timeout disables detection entirely.
void test_zero_timeout_disables_detection() {
  StallTracker tracker;
  bool tripped = false;
  for (int s = 0; s < 200; ++s) {
    tripped |= tracker.update(/*results=*/0, /*work_outstanding=*/true, at(s),
                              std::chrono::milliseconds{0});
  }
  expect_true(!tripped, "a non-positive timeout disables stall detection");
}

void test_each_stream_has_its_own_deadline() {
  StallTracker stalled;
  StallTracker healthy;
  bool stalled_tripped = false;
  bool healthy_tripped = false;
  for (int s = 0; s <= 300; ++s) {
    stalled_tripped |= stalled.update(0, true, at(s), kTimeout);
    healthy_tripped |= healthy.update(s, true, at(s), kTimeout);
  }
  expect_true(stalled_tripped, "one stalled stream trips despite another stream's progress");
  expect_true(!healthy_tripped, "the healthy stream does not trip");
}

void test_expired_work_does_not_poison_idle_or_restart() {
  StallTracker tracker;
  bool tripped = false;
  for (int s = 0; s <= 100; ++s) {
    tripped |= tracker.update(1, s < 30, at(s), kTimeout);
  }
  expect_true(!tripped, "expired contexts with no further input do not cause a false stall");
  expect_true(!tracker.update(1, true, at(101), kTimeout),
              "a new frame after expired work gets a fresh deadline");
  expect_true(tracker.update(1, true, at(162), kTimeout),
              "new outstanding work still trips if it freezes");
}

void test_continued_input_detects_freeze_after_contexts_expire() {
  StallTracker tracker;
  bool tripped = false;
  for (int s = 0; s <= 61; ++s) {
    tripped |= tracker.update(0, true, at(s), kTimeout);
  }
  expect_true(tripped, "continued input detects a frozen endpoint even after contexts expire");
}

void test_configuration_bounds() {
  const auto path = std::filesystem::temp_directory_path() /
                    ("pcie-config-unit-" + std::to_string(getpid()) + ".yaml");
  const auto check = [&](const std::string& settings, bool accepted, const std::string& field) {
    {
      std::ofstream out(path);
      out << "output:\n  insight:\n    host: localhost\nstreams:\n";
      for (int i = 0; i < 20; ++i) {
        out << "  - rtsp://localhost/test" << i << '\n';
      }
      out << settings;
    }
    try {
      pcie_high_density::load_config(path);
      expect_true(accepted, field + " accepted");
    } catch (const std::exception& error) {
      expect_true(!accepted && std::string(error.what()).find(field) != std::string::npos,
                  field + " rejected with its configuration key: " + error.what());
    }
  };
  check("card:\n  card_id: 7\n", true, "card.card_id");
  check("card:\n  card_id: 8\n", false, "card.card_id");
  check("pcie:\n  queue_size: 512\n", true, "pcie.queue_size");
  check("pcie:\n  queue_size: 1024\n", false, "pcie.queue_size");
  check("pcie:\n  queue_size: 512\n  max_inflight_total: 256\n", true, "pcie.max_inflight_total");
  check("pcie:\n  queue_size: 512\n  max_inflight_total: 300\n", false, "pcie.max_inflight_total");
  check("pcie:\n  buffer_size: 524287\n", false, "pcie.buffer_size");
  check("pcie:\n  buffer_size: 134217729\n", false, "pcie.buffer_size");
  check("pcie:\n  result_timeout_ms: 600001\n", false, "pcie.result_timeout_ms");
  std::filesystem::remove(path);
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  test_steady_progress_never_trips();
  test_idle_then_admit_does_not_instantly_trip();
  test_freeze_with_outstanding_work_trips();
  test_zero_timeout_disables_detection();
  test_each_stream_has_its_own_deadline();
  test_expired_work_does_not_poison_idle_or_restart();
  test_continued_input_detects_freeze_after_contexts_expire();
  test_configuration_bounds();

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all PCIe application unit checks passed\n";
  return 0;
}
