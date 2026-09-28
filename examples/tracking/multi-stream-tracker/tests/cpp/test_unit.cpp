#include "examples/tracking/multi-stream-tracker/src/cpp/utils/tracker_api.cpp"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using multi_stream_tracker::Detection;
using multi_stream_tracker::ClassEntry;
using multi_stream_tracker::MultiClassTracker;
using multi_stream_tracker::TrackedDetection;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

const std::string kTwoClasses = "tracking:\n"
                                "  classes:\n"
                                "    - class: person\n"
                                "    - class: car   # comment\n"
                                "      max_missing_frames: 45\n";

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir("multi-stream-tracker", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_accepts_four_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_four_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "input:\n"
                                            "  codec: hevc\n"
                                            "inference:\n"
                                            "  max_inflight_per_stream: 3\n"
                                            "  max_inflight_total: 12\n"
                                            + kTwoClasses +
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "classes=person,car", "validate output reports classes") &&
      expect_contains(result.stdout_text, "max_inflight_per_stream=3",
                      "validate output reports per-stream inflight limit") &&
      expect_contains(result.stdout_text, "max_inflight_total=12",
                      "validate output reports total inflight limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_too_many_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "  - rtsp://127.0.0.1:8554/src5\n"
                                            + kTwoClasses +
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to four streams",
                                  "too-many-stream error mentions four-stream phase limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_inflight_limit(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_inflight_limit",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "inference:\n"
                   "  max_inflight_total: 0\n"
                   + kTwoClasses +
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "invalid inflight limit is rejected") &&
                  expect_contains(result.stderr_text, "max_inflight_total must be -1 or > 0",
                                  "invalid inflight error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

Detection moving(int frame, int class_id = 0, float x = 10.0f, float speed = 4.0f,
                 float score = 0.9f) {
  const float x1 = x + speed * static_cast<float>(frame);
  return Detection{x1, 10.0f, x1 + 40.0f, 90.0f, score, class_id};
}

std::vector<ClassEntry> entries(std::initializer_list<ClassEntry> list) {
  return std::vector<ClassEntry>(list);
}

MultiClassTracker make_tracker(const std::vector<ClassEntry>& classes) {
  return MultiClassTracker(multi_stream_tracker::parse_class_configs(classes));
}

bool expect_rejects(const std::vector<ClassEntry>& classes, const std::string& needle) {
  try {
    multi_stream_tracker::parse_class_configs(classes);
  } catch (const std::exception& e) {
    return expect_contains(e.what(), needle, "invalid classes rejected: " + needle);
  }
  return expect_true(false, "invalid classes rejected: " + needle);
}

bool run_validate(const std::string& binary, const std::string& test_name,
                  const std::string& tracking, int expected_code, const std::string& needle) {
  const fs::path config_path = write_config(test_name, "model:\n"
                                                       "  path: models/m.tar.gz\n"
                                                       "streams:\n"
                                                       "  - rtsp://127.0.0.1:8554/src1\n" +
                                                           tracking +
                                                           "output:\n"
                                                           "  insight:\n"
                                                           "    host: 127.0.0.1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const std::string& text = expected_code == 0 ? result.stdout_text : result.stderr_text;
  const bool ok = expect_true(result.exit_code == expected_code, test_name + " exit code") &&
                  expect_contains(text, needle, test_name + " output");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_config_accepts_one_to_five_classes() {
  const std::vector<std::string> names = {"person", "car", "bicycle", "dog", "truck"};
  bool ok = true;
  for (std::size_t count = 1; count <= names.size(); ++count) {
    std::vector<ClassEntry> classes;
    for (std::size_t i = 0; i < count; ++i) {
      classes.push_back({{"class", names[i]}});
    }
    const auto configs = multi_stream_tracker::parse_class_configs(classes);
    ok &= expect_true(configs.size() == count && configs.back().label == names[count - 1],
                      "accepts " + std::to_string(count) + " classes");
  }
  const auto ids = multi_stream_tracker::parse_class_configs(
      entries({{{"class", "2"}}, {{"class", "Traffic Light"}}, {{"class", "7"}}}));
  ok &= expect_true(ids[0].class_id == 2 && ids[0].label == "car" && ids[1].class_id == 9 &&
                        ids[1].label == "traffic light" && ids[2].label == "truck",
                    "accepts numeric ids and names");
  return ok;
}

bool test_config_rejects_invalid_classes() {
  bool ok = true;
  ok &= expect_rejects({}, "1 to 5 classes");
  ok &= expect_rejects(entries({{{"class", "person"}},
                                {{"class", "car"}},
                                {{"class", "bus"}},
                                {{"class", "dog"}},
                                {{"class", "cat"}},
                                {{"class", "truck"}}}),
                       "1 to 5 classes");
  ok &= expect_rejects(entries({{{"class", "person"}}, {{"class", "person"}}}),
                       "duplicate class 'person'");
  ok &= expect_rejects(entries({{{"class", "person"}}, {{"class", "0"}}}),
                       "duplicate class 'person'");
  ok &= expect_rejects(entries({{{"class", "spaceship"}}}), "unsupported class");
  ok &= expect_rejects(entries({{{"class", "80"}}}), "unsupported class id 80");
  ok &= expect_rejects(entries({{{"class", "-1"}}}), "unsupported class");
  ok &= expect_rejects(entries({{{"class", "true"}}}), "unsupported class");
  ok &= expect_rejects(entries({{{"max_missing_frames", "3"}}}),
                       "must be a mapping with a 'class' key");
  ok &= expect_rejects(entries({{{"class", "person"}, {"speed", "1"}}}), "unknown key 'speed'");
  ok &= expect_rejects(entries({{{"class", "person"}, {"max_missing_frames", "-1"}}}),
                       "max_missing_frames must be >= 0");
  ok &= expect_rejects(entries({{{"class", "person"}, {"max_missing_frames", "2.5"}}}),
                       "must be an integer");
  ok &= expect_rejects(entries({{{"class", "person"}, {"match_iou_threshold", "1.5"}}}),
                       "between 0 and 1");
  ok &= expect_rejects(entries({{{"class", "person"}, {"match_iou_threshold", "abc"}}}),
                       "must be numeric");
  ok &= expect_rejects(entries({{{"class", "person"}, {"low_score_threshold", "0.6"}}}),
                       "low_score_threshold must be <=");
  ok &= expect_rejects(entries({{{"class", "person"}, {"new_track_threshold", "0.3"}}}),
                       "new_track_threshold must be >=");
  ok &= expect_rejects(entries({{{"class", "person"}, {"velocity_noise", "0"}}}),
                       "velocity_noise must be > 0");
  ok &= expect_rejects(entries({{{"class", "person"}, {"min_confirmed_hits", "0"}}}),
                       "min_confirmed_hits must be >= 1");
  return ok;
}

bool test_config_file_classes(const std::string& binary) {
  bool ok = true;
  ok &= run_validate(binary, "config_missing_classes", "", 1, "1 to 5 classes");
  ok &= run_validate(binary, "config_duplicate_class",
                     "tracking:\n  classes:\n    - class: car\n    - class: 2\n", 1,
                     "duplicate class 'car'");
  ok &= run_validate(binary, "config_min_score_above_low",
                     "tracking:\n  classes:\n    - class: person\n"
                     "      low_score_threshold: 0.05\n",
                     1, "min_score must be <= low_score_threshold of class 'person'");
  ok &= run_validate(binary, "config_quoted_and_blank_lines",
                     "tracking:\n  classes:\n\n    - class: \"bicycle\"\n"
                     "      max_missing_frames: 10  # short\n    - class: 'dog'\n",
                     0, "classes=bicycle,dog");
  ok &= run_validate(binary, "config_flow_list", "tracking:\n  classes: [person]\n", 1,
                     "block list");
  return ok;
}

bool test_tracker_reuses_track_id_for_moving_object() {
  auto tracker = make_tracker(entries({{{"class", "person"}}}));
  std::set<int> ids;
  int published = 0;
  for (int f = 0; f < 10; ++f) {
    for (const auto& t : tracker.update({moving(f)}, f)) {
      ids.insert(t.track_id);
      ++published;
    }
  }
  return expect_true(published == 9 && ids == std::set<int>{1},
                     "tracker keeps one id for a moving object");
}

bool test_tracker_publishes_label_and_id() {
  auto tracker = make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}}));
  std::vector<TrackedDetection> out;
  for (int f = 0; f < 3; ++f) {
    out = tracker.update({moving(f, 0), moving(f, 2, 400.0f)}, f);
  }
  return expect_true(out.size() == 2 && out[0].track_id == 1 && out[0].label == "person" &&
                         out[1].track_id == 2 && out[1].label == "car" && out[1].class_id == 2,
                     "tracker publishes class label and track id");
}

bool test_tracker_multiple_tracks_multiple_classes() {
  auto tracker =
      make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}, {{"class", "dog"}}}));
  std::vector<TrackedDetection> out;
  for (int f = 0; f < 5; ++f) {
    std::vector<Detection> dets;
    for (int cls : {0, 2, 16}) {
      for (float x : {10.0f, 300.0f, 600.0f}) {
        dets.push_back(moving(f, cls, x));
      }
    }
    out = tracker.update(dets, f);
  }
  std::set<int> ids;
  std::set<std::string> labels;
  for (const auto& t : out) {
    ids.insert(t.track_id);
    labels.insert(t.label);
  }
  return expect_true(out.size() == 9 && ids.size() == 9 && labels.size() == 3,
                     "nine tracks across three classes");
}

bool test_tracker_ignores_unconfigured_classes() {
  auto tracker = make_tracker(entries({{{"class", "person"}}}));
  std::vector<TrackedDetection> out;
  for (int f = 0; f < 4; ++f) {
    out = tracker.update({moving(f, 0), moving(f, 2, 300.0f), moving(f, 9)}, f);
  }
  return expect_true(out.size() == 1 && out[0].label == "person",
                     "tracker ignores unconfigured classes");
}

bool test_tracker_class_isolation() {
  auto tracker = make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}}));
  for (int f = 0; f < 4; ++f) {
    tracker.update({moving(f, 0)}, f);
  }
  bool car_ok = true;
  std::set<int> car_ids;
  for (int f = 4; f < 12; ++f) {
    for (const auto& t : tracker.update({moving(f, 2)}, f)) {
      car_ids.insert(t.track_id);
      car_ok &= t.class_id == 2 && t.label == "car";
    }
  }
  std::set<std::pair<int, int>> person_tracks;
  for (int f = 12; f < 16; ++f) {
    for (const auto& t : tracker.update({moving(f, 0)}, f)) {
      person_tracks.insert({t.track_id, t.class_id});
    }
  }
  return expect_true(car_ok && car_ids.count(1) == 0 &&
                         person_tracks == std::set<std::pair<int, int>>{{1, 0}},
                     "same box of another class never joins a track");
}

bool test_tracker_never_switches_class() {
  auto tracker =
      make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}, {{"class", "truck"}}}));
  std::map<int, int> seen;
  bool ok = true;
  for (int f = 0; f < 60; ++f) {
    const int cls = std::vector<int>{0, 2, 7}[static_cast<std::size_t>((f / 3) % 3)];
    for (const auto& t : tracker.update({moving(f, cls), moving(f, 0, 500.0f)}, f)) {
      ok &= seen.emplace(t.track_id, t.class_id).first->second == t.class_id;
    }
  }
  return expect_true(ok, "tracks never switch class");
}

bool test_tracker_missed_detections_keep_id() {
  auto tracker = make_tracker(entries({{{"class", "person"}, {"max_missing_frames", "5"}}}));
  std::set<int> ids;
  for (int f = 0; f < 20; ++f) {
    std::vector<Detection> dets;
    if (f < 8 || f > 10) {
      dets.push_back(moving(f, 0, 10.0f, 8.0f));
    }
    for (const auto& t : tracker.update(dets, f)) {
      ids.insert(t.track_id);
    }
  }
  return expect_true(ids == std::set<int>{1}, "short missed detections keep the id");
}

bool test_tracker_low_score_recovery() {
  auto tracker = make_tracker(entries({{{"class", "person"}}}));
  bool ok = true;
  for (int f = 0; f < 10; ++f) {
    const auto out = tracker.update({moving(f, 0, 10.0f, 4.0f, (f == 5 || f == 6) ? 0.2f : 0.9f)}, f);
    if (f > 0) {
      ok &= out.size() == 1 && out[0].track_id == 1;
    }
  }
  auto fresh = make_tracker(entries({{{"class", "person"}}}));
  bool none = true;
  for (int f = 0; f < 10; ++f) {
    none &= fresh.update({moving(f, 0, 10.0f, 4.0f, 0.3f)}, f).empty();
  }
  return expect_true(ok, "low-score detection recovers an active track") &&
         expect_true(none && fresh.active_track_count() == 0,
                     "low-score detection never starts a track");
}

bool test_tracker_expiry_uses_class_budget() {
  auto tracker = make_tracker(entries({{{"class", "person"}, {"max_missing_frames", "3"}},
                                       {{"class", "car"}, {"max_missing_frames", "12"}}}));
  for (int f = 0; f < 5; ++f) {
    tracker.update({moving(f, 0), moving(f, 2, 400.0f)}, f);
  }
  for (int f = 5; f < 10; ++f) {
    tracker.update({}, f);
  }
  const bool one_left = tracker.active_track_count() == 1;
  std::vector<TrackedDetection> out;
  for (int f = 10; f < 13; ++f) {
    out = tracker.update({moving(f, 0), moving(f, 2, 400.0f)}, f);
  }
  std::map<std::string, int> ids;
  for (const auto& t : out) {
    ids[t.label] = t.track_id;
  }
  return expect_true(one_left, "person expired, car still waiting") &&
         expect_true(ids["car"] == 2 && ids["person"] > 2, "expiry follows each class budget");
}

bool test_tracker_class_specific_settings() {
  const auto run = [](const std::string& threshold) {
    auto tracker = make_tracker(entries(
        {{{"class", "car"}, {"match_iou_threshold", threshold}, {"min_confirmed_hits", "1"}}}));
    std::set<int> ids;
    for (int f = 0; f < 3; ++f) {
      for (const auto& t : tracker.update({moving(f, 2, 10.0f, 28.0f)}, f)) {
        ids.insert(t.track_id);
      }
    }
    return ids.size();
  };
  const auto predicted_x = [](const std::string& velocity_noise) {
    auto tracker = make_tracker(entries({{{"class", "car"}, {"velocity_noise", velocity_noise}}}));
    std::vector<TrackedDetection> out;
    for (int f = 0; f < 6; ++f) {
      out = tracker.update({moving(f, 2, 10.0f, 20.0f)}, f);
    }
    return out.at(0).x1;
  };
  return expect_true(run("0.1") == 1 && run("0.3") > 1, "match threshold differs by class") &&
         expect_true(std::fabs(predicted_x("0.05") - predicted_x("0.00625")) > 1e-3f,
                     "motion noise differs by class");
}

bool test_tracker_streams_are_independent() {
  auto stream_a = make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}}));
  auto stream_b = make_tracker(entries({{{"class", "person"}}, {{"class", "car"}}}));
  std::vector<TrackedDetection> a;
  std::vector<TrackedDetection> b;
  for (int f = 0; f < 4; ++f) {
    a = stream_a.update({moving(f, 0), moving(f, 2, 300.0f)}, f);
    b = stream_b.update({moving(f, 2, 300.0f)}, f);
  }
  return expect_true(a.size() == 2 && a[0].track_id == 1 && a[1].track_id == 2 && b.size() == 1 &&
                         b[0].track_id == 1 && b[0].label == "car",
                     "streams keep independent track ids");
}

bool test_linear_assignment() {
  using Pairs = std::vector<std::pair<int, int>>;
  return expect_true(multi_stream_tracker::linear_assignment({{1, 2, 3}, {2, 4, 6}, {3, 6, 9}},
                                                             10) == Pairs{{0, 2}, {1, 1}, {2, 0}},
                     "square assignment is optimal") &&
         expect_true(multi_stream_tracker::linear_assignment({{0.1, 0.2}, {0.05, 0.9}, {0.4, 0.3}},
                                                             1) == Pairs{{0, 1}, {1, 0}},
                     "rectangular assignment is optimal") &&
         expect_true(multi_stream_tracker::linear_assignment({{0.9}}, 0.5).empty(),
                     "assignment drops pairs above max cost");
}

fs::path find_golden(const std::string& binary) {
  std::vector<fs::path> candidates;
#ifdef MULTI_STREAM_TRACKER_SOURCE_DIR
  candidates.push_back(fs::path(MULTI_STREAM_TRACKER_SOURCE_DIR) / "../../tests/common");
#endif
  std::error_code ec;
  const fs::path self = fs::read_symlink("/proc/self/exe", ec);
  if (!ec) {
    candidates.push_back(self.parent_path() / "../common");
  }
  candidates.push_back(fs::path(binary).parent_path() / "../../tests/common");
  for (const auto& dir : candidates) {
    if (fs::exists(dir / "tracker_golden.json")) {
      return dir / "tracker_golden.json";
    }
  }
  return {};
}

bool test_golden_parity(const std::string& binary) {
  const fs::path path = find_golden(binary);
  if (!expect_true(!path.empty(), "golden fixture found")) {
    return false;
  }
  std::ifstream input(path);
  const auto golden = nlohmann::json::parse(input);
  bool ok = true;
  for (const auto& test_case : golden.at("cases")) {
    std::vector<ClassEntry> classes;
    for (const auto& entry : test_case.at("classes")) {
      ClassEntry pairs;
      for (const auto& [key, value] : entry.items()) {
        pairs.emplace_back(key, value.is_string() ? value.get<std::string>() : value.dump());
      }
      classes.push_back(pairs);
    }
    auto tracker = make_tracker(classes);
    const auto& frames = test_case.at("frames");
    const auto& expected = test_case.at("expected");
    bool case_ok = true;
    for (std::size_t f = 0; f < frames.size() && case_ok; ++f) {
      std::vector<Detection> dets;
      for (const auto& d : frames[f]) {
        dets.push_back(Detection{d[0].get<float>(), d[1].get<float>(), d[2].get<float>(),
                                 d[3].get<float>(), d[4].get<float>(), d[5].get<int>()});
      }
      const auto out = tracker.update(dets, static_cast<int>(f));
      case_ok &= out.size() == expected[f].size();
      for (std::size_t i = 0; case_ok && i < out.size(); ++i) {
        const auto& e = expected[f][i];
        case_ok &= out[i].track_id == e[0].get<int>() && out[i].class_id == e[1].get<int>() &&
                   out[i].label == e[2].get<std::string>() &&
                   std::fabs(out[i].x1 - e[3].get<double>()) < 1e-2 &&
                   std::fabs(out[i].y1 - e[4].get<double>()) < 1e-2 &&
                   std::fabs(out[i].x2 - e[5].get<double>()) < 1e-2 &&
                   std::fabs(out[i].y2 - e[6].get<double>()) < 1e-2;
      }
      if (!case_ok) {
        std::cerr << "[FAIL] golden mismatch at frame " << f << "\n";
      }
    }
    ok &= expect_true(case_ok, "golden parity: " + test_case.at("name").get<std::string>());
  }
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_accepts_four_streams(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_invalid_inflight_limit(binary);
  ok &= test_config_accepts_one_to_five_classes();
  ok &= test_config_rejects_invalid_classes();
  ok &= test_config_file_classes(binary);
  ok &= test_tracker_reuses_track_id_for_moving_object();
  ok &= test_tracker_publishes_label_and_id();
  ok &= test_tracker_multiple_tracks_multiple_classes();
  ok &= test_tracker_ignores_unconfigured_classes();
  ok &= test_tracker_class_isolation();
  ok &= test_tracker_never_switches_class();
  ok &= test_tracker_missed_detections_keep_id();
  ok &= test_tracker_low_score_recovery();
  ok &= test_tracker_expiry_uses_class_budget();
  ok &= test_tracker_class_specific_settings();
  ok &= test_tracker_streams_are_independent();
  ok &= test_linear_assignment();
  ok &= test_golden_parity(binary);
  return ok ? 0 : 1;
}
