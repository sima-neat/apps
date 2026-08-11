#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;
using multi_stream_people_tracker::CameraTransform;
using multi_stream_people_tracker::Detection;
using multi_stream_people_tracker::ObjectTracker;
using multi_stream_people_tracker::TrackedDetection;
using multi_stream_people_tracker::TrackerConfig;

namespace {

struct ExpectedTrack {
  int track_id = 0;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  bool predicted = false;
  bool occluded = false;
  float association_confidence = 1.0f;
};

struct ReplayFrame {
  int frame_index = 0;
  std::vector<Detection> detections;
  CameraTransform camera_transform;
  bool has_recorded_tracks = false;
  std::vector<ExpectedTrack> recorded_tracks;
};

struct Options {
  fs::path detections_path;
  fs::path output_path;
  TrackerConfig tracker;
  bool verify_determinism = false;
  bool verify_recorded = false;
  float recorded_tolerance = 0.02f;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void print_help(const char* program) {
  std::cout << "Usage: " << program
            << " --detections <input.jsonl> --output <output.jsonl> [options]\n"
            << "\nReplays detector output through the exact production C++ tracker.\n\n"
            << "Options:\n"
            << "  --high-score-threshold <value>\n"
            << "  --new-track-threshold <value>\n"
            << "  --match-iou-threshold <value>\n"
            << "  --max-center-distance <value>\n"
            << "  --velocity-momentum <value>\n"
            << "  --box-smoothing-alpha <value>\n"
            << "  --max-missing-frames <count>\n"
            << "  --min-confirmed-hits <count>\n"
            << "  --max-prediction-frames <count>\n"
            << "  --overlap-threshold <value>\n"
            << "  --max-occlusion-frames <count>\n"
            << "  --max-active-tracks <count>\n"
            << "  --camera-motion-compensation\n"
            << "  --disable-center-distance\n"
            << "  --disable-covariance-motion\n"
            << "  --verify-determinism\n"
            << "  --verify-recorded\n"
            << "  --recorded-tolerance <pixels>\n";
}

float parse_float(std::string_view text, std::string_view option) {
  std::size_t consumed = 0;
  float value = 0.0f;
  try {
    value = std::stof(std::string(text), &consumed);
  } catch (const std::exception&) {
    fail(std::string(option) + " requires a finite number");
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    fail(std::string(option) + " requires a finite number");
  }
  return value;
}

int parse_int(std::string_view text, std::string_view option) {
  std::size_t consumed = 0;
  long value = 0;
  try {
    value = std::stol(std::string(text), &consumed);
  } catch (const std::exception&) {
    fail(std::string(option) + " requires an integer");
  }
  if (consumed != text.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    fail(std::string(option) + " requires an integer");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  const auto next_value = [&](int& index, std::string_view name) -> std::string_view {
    if (++index >= argc) {
      fail(std::string(name) + " requires a value");
    }
    return argv[index];
  };
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      print_help(argv[0]);
      std::exit(0);
    } else if (argument == "--detections") {
      options.detections_path = next_value(index, argument);
    } else if (argument == "--output") {
      options.output_path = next_value(index, argument);
    } else if (argument == "--high-score-threshold") {
      options.tracker.high_score_threshold = parse_float(next_value(index, argument), argument);
    } else if (argument == "--new-track-threshold") {
      options.tracker.new_track_threshold = parse_float(next_value(index, argument), argument);
    } else if (argument == "--match-iou-threshold") {
      options.tracker.match_iou_threshold = parse_float(next_value(index, argument), argument);
    } else if (argument == "--max-center-distance") {
      options.tracker.max_center_distance = parse_float(next_value(index, argument), argument);
    } else if (argument == "--velocity-momentum") {
      options.tracker.velocity_momentum = parse_float(next_value(index, argument), argument);
    } else if (argument == "--box-smoothing-alpha") {
      options.tracker.box_smoothing_alpha = parse_float(next_value(index, argument), argument);
    } else if (argument == "--max-missing-frames") {
      options.tracker.max_missing_frames = parse_int(next_value(index, argument), argument);
    } else if (argument == "--min-confirmed-hits") {
      options.tracker.min_confirmed_hits = parse_int(next_value(index, argument), argument);
    } else if (argument == "--max-prediction-frames") {
      options.tracker.max_prediction_frames = parse_int(next_value(index, argument), argument);
    } else if (argument == "--overlap-threshold") {
      options.tracker.overlap_threshold = parse_float(next_value(index, argument), argument);
    } else if (argument == "--max-occlusion-frames") {
      options.tracker.max_occlusion_frames = parse_int(next_value(index, argument), argument);
    } else if (argument == "--max-active-tracks") {
      options.tracker.max_active_tracks = parse_int(next_value(index, argument), argument);
    } else if (argument == "--recorded-tolerance") {
      options.recorded_tolerance = parse_float(next_value(index, argument), argument);
    } else if (argument == "--camera-motion-compensation") {
      options.tracker.camera_motion_compensation = true;
    } else if (argument == "--disable-center-distance") {
      options.tracker.center_distance_enabled = false;
    } else if (argument == "--disable-covariance-motion") {
      options.tracker.covariance_motion_enabled = false;
    } else if (argument == "--verify-determinism") {
      options.verify_determinism = true;
    } else if (argument == "--verify-recorded") {
      options.verify_recorded = true;
    } else {
      fail("unknown option: " + std::string(argument));
    }
  }
  if (options.detections_path.empty()) {
    fail("--detections is required");
  }
  if (options.output_path.empty()) {
    fail("--output is required");
  }
  if (options.recorded_tolerance < 0.0f) {
    fail("--recorded-tolerance must be non-negative");
  }
  return options;
}

float finite_number(const json& value, const std::string& location) {
  if (!value.is_number()) {
    fail(location + " must be a number");
  }
  const double number = value.get<double>();
  if (!std::isfinite(number) || number < -std::numeric_limits<float>::max() ||
      number > std::numeric_limits<float>::max()) {
    fail(location + " must be a finite float");
  }
  return static_cast<float>(number);
}

int integer(const json& value, const std::string& location, int minimum) {
  if (!value.is_number_integer()) {
    fail(location + " must be an integer");
  }
  const auto number = value.get<long long>();
  if (number < minimum || number > std::numeric_limits<int>::max()) {
    fail(location + " is outside the supported range");
  }
  return static_cast<int>(number);
}

Detection parse_detection(const json& value, const std::string& location) {
  if (!value.is_object()) {
    fail(location + " must be an object");
  }
  Detection detection;
  if (value.contains("bbox")) {
    const auto& bbox = value.at("bbox");
    if (!bbox.is_array() || bbox.size() != 4) {
      fail(location + ".bbox must contain [x, y, width, height]");
    }
    detection.x1 = finite_number(bbox.at(0), location + ".bbox[0]");
    detection.y1 = finite_number(bbox.at(1), location + ".bbox[1]");
    detection.x2 = detection.x1 + finite_number(bbox.at(2), location + ".bbox[2]");
    detection.y2 = detection.y1 + finite_number(bbox.at(3), location + ".bbox[3]");
  } else {
    detection.x1 = finite_number(value.at("x1"), location + ".x1");
    detection.y1 = finite_number(value.at("y1"), location + ".y1");
    detection.x2 = finite_number(value.at("x2"), location + ".x2");
    detection.y2 = finite_number(value.at("y2"), location + ".y2");
  }
  detection.score = finite_number(value.at("score"), location + ".score");
  detection.class_id =
      value.contains("class_id") ? integer(value.at("class_id"), location + ".class_id", 0) : 0;
  if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1) {
    fail(location + " must have positive width and height");
  }
  if (detection.score < 0.0f || detection.score > 1.0f) {
    fail(location + ".score must be in [0, 1]");
  }
  return detection;
}

ExpectedTrack parse_expected_track(const json& value, const std::string& location) {
  if (!value.is_object()) {
    fail(location + " must be an object");
  }
  ExpectedTrack track;
  if (value.at("id").is_string()) {
    track.track_id = parse_int(value.at("id").get<std::string>(), location + ".id");
  } else {
    track.track_id = integer(value.at("id"), location + ".id", 1);
  }
  const auto& bbox = value.at("bbox");
  if (!bbox.is_array() || bbox.size() != 4) {
    fail(location + ".bbox must contain [x, y, width, height]");
  }
  track.x1 = finite_number(bbox.at(0), location + ".bbox[0]");
  track.y1 = finite_number(bbox.at(1), location + ".bbox[1]");
  track.x2 = track.x1 + finite_number(bbox.at(2), location + ".bbox[2]");
  track.y2 = track.y1 + finite_number(bbox.at(3), location + ".bbox[3]");
  track.score = finite_number(value.at("confidence"), location + ".confidence");
  track.class_id = integer(value.at("class_id"), location + ".class_id", 0);
  track.predicted = value.at("predicted").get<bool>();
  track.occluded = value.at("occluded").get<bool>();
  track.association_confidence =
      finite_number(value.at("association_confidence"), location + ".association_confidence");
  return track;
}

std::vector<ReplayFrame> read_frames(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    fail("cannot open detector replay: " + path.string());
  }
  std::vector<ReplayFrame> frames;
  std::string line;
  int line_number = 0;
  int previous_frame = -1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    json document;
    try {
      document = json::parse(line);
    } catch (const std::exception& error) {
      fail(path.string() + ":" + std::to_string(line_number) + ": invalid JSON: " + error.what());
    }
    const std::string location = path.string() + ":" + std::to_string(line_number);
    if (!document.is_object()) {
      fail(location + ": frame must be an object");
    }
    ReplayFrame frame;
    frame.frame_index = integer(document.at("frame_index"), location + ".frame_index", 0);
    if (frame.frame_index <= previous_frame) {
      fail(location + ": frame_index must be strictly increasing");
    }
    previous_frame = frame.frame_index;
    const auto& detections = document.at("detections");
    if (!detections.is_array()) {
      fail(location + ".detections must be an array");
    }
    frame.detections.reserve(detections.size());
    for (std::size_t index = 0; index < detections.size(); ++index) {
      frame.detections.push_back(parse_detection(
          detections.at(index), location + ".detections[" + std::to_string(index) + "]"));
    }
    if (document.contains("camera_transform") && !document.at("camera_transform").is_null()) {
      const auto& transform = document.at("camera_transform");
      if (!transform.is_array() || transform.size() != 6) {
        fail(location + ".camera_transform must contain six affine values");
      }
      frame.camera_transform =
          CameraTransform{finite_number(transform.at(0), location + ".camera_transform[0]"),
                          finite_number(transform.at(1), location + ".camera_transform[1]"),
                          finite_number(transform.at(2), location + ".camera_transform[2]"),
                          finite_number(transform.at(3), location + ".camera_transform[3]"),
                          finite_number(transform.at(4), location + ".camera_transform[4]"),
                          finite_number(transform.at(5), location + ".camera_transform[5]"),
                          true};
      if (document.contains("camera_diagnostics")) {
        const auto& diagnostics = document.at("camera_diagnostics");
        frame.camera_transform.confidence = finite_number(
            diagnostics.at("confidence"), location + ".camera_diagnostics.confidence");
        frame.camera_transform.reprojection_error =
            finite_number(diagnostics.at("reprojection_error"),
                          location + ".camera_diagnostics.reprojection_error");
        frame.camera_transform.inliers =
            integer(diagnostics.at("inliers"), location + ".camera_diagnostics.inliers", 0);
      }
    }
    if (document.contains("tracks")) {
      const auto& tracks = document.at("tracks");
      if (!tracks.is_array()) {
        fail(location + ".tracks must be an array");
      }
      frame.has_recorded_tracks = true;
      frame.recorded_tracks.reserve(tracks.size());
      for (std::size_t index = 0; index < tracks.size(); ++index) {
        frame.recorded_tracks.push_back(parse_expected_track(
            tracks.at(index), location + ".tracks[" + std::to_string(index) + "]"));
      }
    }
    frames.push_back(std::move(frame));
  }
  if (frames.empty()) {
    fail(path.string() + ": no frames found");
  }
  return frames;
}

std::string record_json(int frame_index, const std::vector<TrackedDetection>& tracks) {
  std::ostringstream output;
  output << std::setprecision(9) << "{\"frame_index\":" << frame_index << ",\"tracks\":[";
  for (std::size_t index = 0; index < tracks.size(); ++index) {
    const auto& track = tracks[index];
    if (index != 0) {
      output << ',';
    }
    output << "{\"id\":\"" << track.track_id << "\",\"bbox\":[" << track.x1 << ',' << track.y1
           << ',' << track.x2 - track.x1 << ',' << track.y2 - track.y1
           << "],\"confidence\":" << track.score << ",\"class_id\":" << track.class_id
           << ",\"predicted\":" << (track.predicted ? "true" : "false")
           << ",\"occluded\":" << (track.occluded ? "true" : "false")
           << ",\"association_confidence\":" << track.association_confidence << '}';
  }
  output << "]}";
  return output.str();
}

void verify_tracks(const ReplayFrame& frame, const std::vector<TrackedDetection>& tracks,
                   float tolerance) {
  if (!frame.has_recorded_tracks) {
    fail("frame " + std::to_string(frame.frame_index) +
         ": tracks array is required with --verify-recorded");
  }
  if (frame.recorded_tracks.size() != tracks.size()) {
    fail("frame " + std::to_string(frame.frame_index) + ": replay produced " +
         std::to_string(tracks.size()) + " tracks but the board recorded " +
         std::to_string(frame.recorded_tracks.size()));
  }
  const auto near = [tolerance](float lhs, float rhs) { return std::abs(lhs - rhs) <= tolerance; };
  for (std::size_t index = 0; index < tracks.size(); ++index) {
    const auto& expected = frame.recorded_tracks[index];
    const auto& actual = tracks[index];
    if (expected.track_id != actual.track_id || expected.class_id != actual.class_id ||
        expected.predicted != actual.predicted || expected.occluded != actual.occluded ||
        !near(expected.x1, actual.x1) || !near(expected.y1, actual.y1) ||
        !near(expected.x2, actual.x2) || !near(expected.y2, actual.y2) ||
        !near(expected.score, actual.score) ||
        !near(expected.association_confidence, actual.association_confidence)) {
      fail("frame " + std::to_string(frame.frame_index) + ", track index " + std::to_string(index) +
           ": replay differs from the production record");
    }
  }
}

std::vector<std::string> replay(const std::vector<ReplayFrame>& frames, const Options& options) {
  ObjectTracker tracker(options.tracker);
  std::vector<std::string> records;
  records.reserve(frames.size());
  for (const auto& frame : frames) {
    auto tracks = tracker.update(frame.detections, frame.frame_index, frame.camera_transform);
    if (options.verify_recorded) {
      verify_tracks(frame, tracks, options.recorded_tolerance);
    }
    records.push_back(record_json(frame.frame_index, tracks));
  }
  return records;
}

void validate_config(const TrackerConfig& config) {
  const auto probability = [](float value) { return value >= 0.0f && value <= 1.0f; };
  if (!probability(config.high_score_threshold) || !probability(config.new_track_threshold) ||
      !probability(config.match_iou_threshold) || !probability(config.velocity_momentum) ||
      !probability(config.box_smoothing_alpha) || !probability(config.overlap_threshold)) {
    fail("probability and smoothing options must be in [0, 1]");
  }
  if (config.max_center_distance < 0.0f || config.max_missing_frames < 0 ||
      config.min_confirmed_hits <= 0 || config.max_prediction_frames < 0 ||
      config.max_occlusion_frames < 0 || config.max_occlusion_frames > config.max_missing_frames ||
      config.max_active_tracks < 1) {
    fail("invalid tracker lifecycle or distance options");
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    validate_config(options.tracker);
    const auto frames = read_frames(options.detections_path);
    const auto records = replay(frames, options);
    if (options.verify_determinism && replay(frames, options) != records) {
      fail("production C++ tracker replay is not deterministic");
    }
    if (!options.output_path.parent_path().empty()) {
      fs::create_directories(options.output_path.parent_path());
    }
    std::ofstream output(options.output_path);
    if (!output) {
      fail("cannot open output: " + options.output_path.string());
    }
    for (const auto& record : records) {
      output << record << '\n';
    }
    if (!output) {
      fail("failed while writing output: " + options.output_path.string());
    }
    std::cout << "replayed " << frames.size() << " frames with the production C++ tracker";
    if (options.verify_recorded) {
      std::cout << "; all tracks match the recorded board output";
    }
    if (options.verify_determinism) {
      std::cout << "; independent replay is byte-deterministic";
    }
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
