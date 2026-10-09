// Unit test for single-stream-instance-segmenter: CLI handling, the configuration rules
// --validate-config-only can reach without a model or a stream, and the YOLOv8 decode
// boundary, checked against the same tests/fixtures/decode_parity.json the Python unit
// test uses. Both implementations must decode the shared fixture identically.
#define main single_stream_instance_segmenter_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/runtime/pull_status.h"
#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::pull_status_has_sample;
using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_not_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::validate_config_body;

namespace {

constexpr const char* kExampleName = "single-stream-instance-segmenter";

// The smallest config the validator accepts, with the source and model lines
// supplied by the caller so each test states exactly the keys it is about.
// Everything else takes its documented default.
std::string minimal_config(const std::string& source_lines, const std::string& model_lines = "") {
  return "model:\n"
         "  path: models/model.tar.gz\n" +
         model_lines + "source:\n" + source_lines +
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
}

bool test_help_runs(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(r.exit_code == 0, "help exits with code 0") &&
         expect_contains(r.stdout_text, "Usage", "help prints usage");
}

bool test_unknown_flag_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(r.exit_code != 0, "unknown flag exits non-zero");
}

bool test_missing_config_file_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
  return expect_true(r.exit_code != 0, "missing config exits non-zero");
}

// config.yaml ships source.url present and documents source.rtsp_url as
// "used when source.url is empty", so the empty value has to fall through.
bool test_empty_url_falls_back_to_the_legacy_key(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "empty_url_falls_back",
                                      minimal_config("  url: \"\"\n"
                                                     "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "empty url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.rtsp_url",
                         "empty url selects the legacy rtsp_url");
}

bool test_absent_url_falls_back_to_the_legacy_key(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "absent_url_falls_back",
                                      minimal_config("  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "absent url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.rtsp_url",
                         "absent url selects the legacy rtsp_url");
}

bool test_present_url_wins_over_the_legacy_key(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "present_url_wins",
                                      minimal_config("  url: rtsp://127.0.0.1:8554/src1\n"
                                                     "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "url beside a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.url", "present url is selected") &&
         expect_not_contains(r.stdout_text, "source.rtsp_url", "legacy rtsp_url is not selected") &&
         expect_not_contains(r.stdout_text, "rtsp://", "validated line does not echo the URL");
}

bool test_both_urls_empty_is_rejected(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "both_urls_empty",
                                      minimal_config("  url: \"\"\n"
                                                     "  rtsp_url: \"\"\n"));
  return expect_true(r.exit_code != 0, "empty url and empty rtsp_url is rejected") &&
         expect_contains(r.stderr_text, "source.url or source.rtsp_url must be set",
                         "both-empty error names both keys");
}

// The Python loader once accepted this because Path("") is "."; the C++ side
// is held to the rule so the two implementations cannot drift apart.
bool test_empty_labels_is_rejected(const std::string& binary) {
  const auto r = validate_config_body(
      kExampleName, binary, "empty_labels",
      minimal_config("  url: rtsp://127.0.0.1:8554/src1\n", "  labels: \"\"\n"));
  return expect_true(r.exit_code != 0, "empty model.labels is rejected") &&
         expect_contains(r.stderr_text, "model.labels must be set",
                         "empty labels error names the key");
}

// ---------------------------------------------------------------------------
// Configuration rules --validate-config-only can reach without a model or a
// stream (Refs #526). Each case is the minimal valid config with exactly one
// value broken, and asserts the message that names the rule, so a failure
// says which rule stopped firing.
// ---------------------------------------------------------------------------
struct ConfigParts {
  std::string sections;      // extra top-level sections, e.g. "input:\n  latency_ms: -1\n"
  std::string output_extra;  // extra keys under output:, e.g. "  save_every: -1\n"
  std::string insight_extra; // extra keys under output.insight:, e.g. "    video_port_base: 0\n"
  std::string model_path = "models/model.tar.gz";
  std::string host = "127.0.0.1";
};

struct RejectedConfig {
  const char* name;
  ConfigParts parts;
  const char* message;
};

std::string config_body(const ConfigParts& parts) {
  return "model:\n  path: '" + parts.model_path + "'\n" +
         (parts.sections.rfind("source:", 0) == 0
              ? std::string()
              : std::string("source:\n  url: rtsp://127.0.0.1:8554/src1\n")) +
         parts.sections + "output:\n" + parts.output_extra + "  insight:\n    host: '" +
         parts.host + "'\n" + parts.insight_extra;
}

bool test_configuration_rules_are_enforced(const std::string& binary) {
  const std::vector<RejectedConfig> cases = {
      {"model-path-empty", {"", "", "", ""}, "model.path must be set"},
      {"insight-host-empty",
       {"", "", "", "models/model.tar.gz", ""},
       "output.insight.host must be set"},
      {"latency-negative",
       {"source:\n  url: rtsp://127.0.0.1:8554/src1\n  latency_ms: -1\n", "", ""},
       "source.latency_ms must be >= 0"},
      {"fps-negative",
       {"source:\n  url: rtsp://127.0.0.1:8554/src1\n  fps: -1\n", "", ""},
       "source.fps must be >= 0"},
      {"http-needs-mjpeg",
       {"source:\n  url: http://127.0.0.1:8080/stream\n  type: http\n  codec: h264\n", "", ""},
       "source.codec must be mjpeg for source.type=http"},
      {"frames-negative", {"inference:\n  frames: -1\n", "", ""}, "inference.frames must be >= 0"},
      {"min-score-above",
       {"inference:\n  min_score: 1.5\n", "", ""},
       "inference.min_score must be between 0 and 1"},
      {"nms-below",
       {"inference:\n  nms_iou: -0.5\n", "", ""},
       "inference.nms_iou must be between 0 and 1"},
      {"max-detections-zero",
       {"inference:\n  max_detections: 0\n", "", ""},
       "inference.max_detections must be > 0"},
      {"profile-interval-zero",
       {"runtime:\n  profile_interval: 0\n", "", ""},
       "runtime.profile_interval must be > 0"},
      {"video-port-zero", {"", "", "    video_port: 0\n"}, "output.insight.video_port must be > 0"},
      {"metadata-port-zero",
       {"", "", "    metadata_port: 0\n"},
       "output.insight.metadata_port must be > 0"},
      {"save-every-negative", {"", "  save_every: -1\n", ""}, "output.save_every must be >= 0"},
      {"mask-alpha-above",
       {"", "  mask_alpha: 2\n", ""},
       "output.mask_alpha must be between 0 and 1"},
      {"mask-threshold-above",
       {"", "  mask_threshold: 2\n", ""},
       "output.mask_threshold must be between 0 and 1"},
  };

  bool ok = true;
  for (const RejectedConfig& c : cases) {
    const auto result = validate_config_body(kExampleName, binary, std::string("rule_") + c.name,
                                             config_body(c.parts));
    ok &= expect_true(result.exit_code != 0, std::string(c.name) + " is rejected") &&
          expect_contains(result.stderr_text, c.message, std::string(c.name) + " names its rule");
  }

  // The control: the same minimal config with nothing broken validates, so the
  // rejections above are about the broken value and not about the baseline.
  const auto result =
      validate_config_body(kExampleName, binary, "rule_baseline", config_body(ConfigParts{}));
  ok &= expect_true(result.exit_code == 0, "minimal config validates") &&
        expect_contains(result.stdout_text, "Config validated", "validated line is printed");
  return ok;
}

bool test_closed_output_is_terminal() {
  using simaai::neat::PullStatus;
  simaai::neat::PullError pull_error;
  pull_error.message = "queue torn down";
  const auto thrown_message = [&](PullStatus status) -> std::string {
    try {
      (void)pull_status_has_sample(status, "segments", pull_error, "source reached EOS");
    } catch (const std::runtime_error& error) {
      return error.what();
    }
    return "";
  };
  return expect_true(thrown_message(PullStatus::Closed) ==
                         "segments output closed unexpectedly: source reached EOS",
                     "closed output ends the run with the runtime's reason") &&
         expect_true(thrown_message(PullStatus::Error) ==
                         "failed to pull segments: queue torn down",
                     "pull error ends the run with its message") &&
         expect_true(!pull_status_has_sample(PullStatus::Timeout, "segments", pull_error, ""),
                     "timeout is not a sample") &&
         expect_true(pull_status_has_sample(PullStatus::Ok, "segments", pull_error, ""),
                     "successful pull is a sample");
}

} // namespace

namespace {

namespace neat = simaai::neat;
using json = nlohmann::json;

std::filesystem::path fixture_path() {
  const char* apps_root = std::getenv("APPS_ROOT");
  const std::filesystem::path root = (apps_root && *apps_root) ? std::filesystem::path(apps_root)
                                                               : std::filesystem::path(".");
  return root / SIMANEAT_APPS_EXAMPLE_SOURCE_DIR / ".." / ".." / "tests" / "fixtures" /
         "decode_parity.json";
}

/// Synthetic YOLOv8 head tensors described by the shared fixture.
neat::TensorList build_yolov8_heads(const json& spec) {
  const int input_size = spec.at("input_size").get<int>();
  const int class_count = spec.at("class_count").get<int>();
  const int proto_grid = input_size / kMaskStride;

  std::vector<std::vector<float>> boxes;
  std::vector<std::vector<float>> scores;
  std::vector<std::vector<float>> coefficients;
  std::vector<int> grids;
  for (const int stride : kYolov8Strides) {
    const int grid = input_size / stride;
    grids.push_back(grid);
    const size_t cells = static_cast<size_t>(grid) * static_cast<size_t>(grid);
    boxes.emplace_back(cells * static_cast<size_t>(4 * kDflBins), 0.0f);
    scores.emplace_back(cells * static_cast<size_t>(class_count), 0.0f);
    coefficients.emplace_back(cells * static_cast<size_t>(kMaskCoefficients), 0.0f);
  }

  std::vector<float> proto(static_cast<size_t>(proto_grid) * static_cast<size_t>(proto_grid) *
                               static_cast<size_t>(kMaskCoefficients),
                           0.0f);
  const auto proto_at = [&](int y, int x, int channel) -> float& {
    return proto[(static_cast<size_t>(y) * static_cast<size_t>(proto_grid) +
                  static_cast<size_t>(x)) *
                     static_cast<size_t>(kMaskCoefficients) +
                 static_cast<size_t>(channel)];
  };
  for (const auto& channel : spec.at("prototype")) {
    const int index = channel.at("channel").get<int>();
    const float background = channel.at("background").get<float>();
    const float foreground = channel.at("foreground").get<float>();
    const auto rect = channel.at("rect").get<std::vector<int>>();
    for (int y = 0; y < proto_grid; ++y) {
      for (int x = 0; x < proto_grid; ++x) {
        const bool inside = x >= rect[0] && x < rect[2] && y >= rect[1] && y < rect[3];
        proto_at(y, x, index) = inside ? foreground : background;
      }
    }
  }

  for (const auto& cell : spec.at("cells")) {
    const int level = cell.at("level").get<int>();
    const int row = cell.at("row").get<int>();
    const int column = cell.at("column").get<int>();
    const int grid = grids[static_cast<size_t>(level)];
    const size_t cell_index =
        static_cast<size_t>(row) * static_cast<size_t>(grid) + static_cast<size_t>(column);

    float* sides = boxes[static_cast<size_t>(level)].data() +
                   cell_index * static_cast<size_t>(4 * kDflBins);
    for (int side = 0; side < 4; ++side) {
      for (int bin = 0; bin < kDflBins; ++bin) {
        sides[side * kDflBins + bin] = -8.0f;
      }
      sides[side * kDflBins + cell.at("dfl_bin").get<int>()] = 8.0f;
    }
    scores[static_cast<size_t>(level)][cell_index * static_cast<size_t>(class_count) +
                                       static_cast<size_t>(cell.at("class_id").get<int>())] =
        cell.at("score").get<float>();
    coefficients[static_cast<size_t>(level)]
                [cell_index * static_cast<size_t>(kMaskCoefficients) +
                 static_cast<size_t>(cell.at("coefficient").get<int>())] = 1.0f;
  }

  neat::TensorList tensors;
  const auto push = [&](const std::vector<float>& values, int grid, int channels) {
    tensors.push_back(neat::Tensor::from_vector(values, {1, grid, grid, channels},
                                                neat::TensorMemory::CPU));
  };
  for (size_t level = 0; level < grids.size(); ++level) {
    push(boxes[level], grids[level], 4 * kDflBins);
  }
  for (size_t level = 0; level < grids.size(); ++level) {
    push(scores[level], grids[level], class_count);
  }
  for (size_t level = 0; level < grids.size(); ++level) {
    push(coefficients[level], grids[level], kMaskCoefficients);
  }
  push(proto, proto_grid, kMaskCoefficients);
  return tensors;
}

AppConfig fixture_config(const json& spec) {
  AppConfig cfg;
  cfg.model_family = ModelFamily::YoloV8;
  cfg.input_size = spec.at("input_size").get<int>();
  cfg.min_score = spec.at("min_score").get<double>();
  cfg.nms_iou = spec.at("nms_iou").get<double>();
  cfg.max_detections = spec.at("max_detections").get<int>();
  cfg.mask_threshold = spec.at("mask_threshold").get<double>();
  return cfg;
}

int mask_above_threshold(const cv::Mat& mask) {
  return cv::countNonZero(mask > 127);
}

int check_parity(const json& fixture, int& failures) {
  const json& spec = fixture.at("input");
  const AppConfig cfg = fixture_config(spec);
  const int frame_w = spec.at("frame").at("width").get<int>();
  const int frame_h = spec.at("frame").at("height").get<int>();

  const auto detections =
      decode_yolov8_segments(build_yolov8_heads(spec), frame_w, frame_h, cfg);
  const json& expected = fixture.at("expected").at("detections");
  if (detections.size() != expected.size()) {
    std::cerr << "[FAIL] decoded " << detections.size() << " instances, fixture expects "
              << expected.size() << "\n";
    ++failures;
    return failures;
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& det = detections[i];
    const json& want = expected[i];
    const auto box = want.at("box").get<std::vector<double>>();
    const int grid = want.at("mask_grid").get<int>();
    if (det.class_id != want.at("class_id").get<int>() ||
        std::fabs(det.score - want.at("score").get<double>()) > 1e-6) {
      std::cerr << "[FAIL] instance " << i << ": class/score mismatch\n";
      ++failures;
    }
    if (std::fabs(det.x1 - box[0]) > 1e-3 || std::fabs(det.y1 - box[1]) > 1e-3 ||
        std::fabs(det.x2 - box[2]) > 1e-3 || std::fabs(det.y2 - box[3]) > 1e-3) {
      std::cerr << "[FAIL] instance " << i << ": box " << det.x1 << "," << det.y1 << ","
                << det.x2 << "," << det.y2 << " does not match the fixture\n";
      ++failures;
    }
    if (det.mask.empty() || det.mask.rows != grid || det.mask.cols != grid ||
        det.mask.type() != CV_8UC1) {
      std::cerr << "[FAIL] instance " << i << ": unexpected mask grid\n";
      ++failures;
    } else if (mask_above_threshold(det.mask) != want.at("mask_above_threshold").get<int>()) {
      std::cerr << "[FAIL] instance " << i << ": mask covers "
                << mask_above_threshold(det.mask) << " cells, fixture expects "
                << want.at("mask_above_threshold").get<int>() << "\n";
      ++failures;
    }
  }

  std::vector<std::string> labels;
  for (int index = 0; index < spec.at("class_count").get<int>(); ++index) {
    labels.push_back("class_" + std::to_string(index));
  }
  const auto segments = build_metadata_segments(detections, labels, cv::Size(frame_w, frame_h),
                                                cfg.mask_threshold);
  const json& expected_segments = fixture.at("expected").at("segments");
  if (segments.size() != expected_segments.size()) {
    std::cerr << "[FAIL] built " << segments.size() << " metadata segments, fixture expects "
              << expected_segments.size() << "\n";
    ++failures;
    return failures;
  }
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& segment = segments[i];
    const json& want = expected_segments[i];
    const auto bbox = want.at("bbox").get<std::vector<int>>();
    if (segment.id != want.at("id").get<std::string>() ||
        segment.label != want.at("label").get<std::string>() ||
        std::fabs(segment.confidence - want.at("confidence").get<double>()) > 1e-6 ||
        segment.bbox.x != bbox[0] || segment.bbox.y != bbox[1] ||
        segment.bbox.width != bbox[2] || segment.bbox.height != bbox[3] ||
        segment.polygon.size() < want.at("min_polygon_points").get<size_t>()) {
      std::cerr << "[FAIL] metadata segment " << i << " does not match the fixture\n";
      ++failures;
    }
  }
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  int failures = 0;

  // Test 1: the model family is configuration, never inferred from the package file name.
  {
    if (parse_model_family("yolo26") != ModelFamily::Yolo26 ||
        parse_model_family("YOLO26") != ModelFamily::Yolo26 ||
        parse_model_family("yolov8") != ModelFamily::YoloV8 ||
        parse_model_family("yolo_v8") != ModelFamily::YoloV8) {
      std::cerr << "[FAIL] model.family parsing does not accept the documented values\n";
      ++failures;
    }
    bool rejected = false;
    try {
      (void)parse_model_family("yolo_v8n_seg_mpk.tar.gz");
    } catch (const std::exception&) {
      rejected = true;
    }
    if (!rejected) {
      std::cerr << "[FAIL] an unknown model.family must be rejected\n";
      ++failures;
    } else {
      std::cout << "[OK] model family is selected explicitly\n";
    }
  }

  // Test 2: the YOLOv8 decode boundary matches the shared Python/C++ parity fixture.
  {
    const auto path = fixture_path();
    std::ifstream in(path);
    if (!in.good()) {
      std::cerr << "[FAIL] parity fixture not found: " << path << "\n";
      ++failures;
    } else {
      const json fixture = json::parse(in);
      const int before = failures;
      check_parity(fixture, failures);
      if (failures == before) {
        std::cout << "[OK] YOLOv8 decode matches the shared parity fixture\n";
      }
    }
  }

  // Test 3: head tensors are checked against model.input_size.
  {
    const auto path = fixture_path();
    std::ifstream in(path);
    if (in.good()) {
      const json fixture = json::parse(in);
      const auto heads = build_yolov8_heads(fixture.at("input"));
      bool rejected = false;
      try {
        (void)split_yolov8_heads(heads, 320);
      } catch (const std::exception&) {
        rejected = true;
      }
      neat::TensorList truncated(heads.begin(), heads.end() - 1);
      bool short_rejected = false;
      try {
        (void)split_yolov8_heads(truncated, fixture.at("input").at("input_size").get<int>());
      } catch (const std::exception&) {
        short_rejected = true;
      }
      if (!rejected || !short_rejected) {
        std::cerr << "[FAIL] head validation must reject a wrong input size and short output\n";
        ++failures;
      } else {
        std::cout << "[OK] head shapes are checked against model.input_size\n";
      }
    }
  }

  // Test 4: YOLOv8 pairs frames with segments here, so a saved frame is the one the
  // segments were decoded from, and an aged-out or unidentified sample pairs with nothing.
  {
    PipelineRuntime runtime;
    for (const std::int64_t frame_id : {7, 8, 9}) {
      runtime.frames.emplace_back(frame_id,
                                  cv::Mat(1, 1, CV_8UC1, cv::Scalar(static_cast<int>(frame_id))));
    }
    const auto* paired = frame_for(runtime, 8);
    if (paired == nullptr || paired->at<std::uint8_t>(0, 0) != 8) {
      std::cerr << "[FAIL] a retained frame must pair with the segments it produced\n";
      ++failures;
    } else if (frame_for(runtime, 3) != nullptr || frame_for(runtime, -1) != nullptr) {
      std::cerr << "[FAIL] aged-out and unidentified samples must not pair\n";
      ++failures;
    } else {
      std::cout << "[OK] frames pair with the segments they produced\n";
    }
  }

  // Test 5: the ring keeps host pixels, not pipeline samples. A retained sample would hold a
  // decoder-buffer loan, and the decoder stalls once its in-flight frames are all on loan.
  {
    const cv::Mat nv12(6, 4, CV_8UC1, cv::Scalar(128));
    const cv::Mat bgr = bgr_from_host_frame(nv12);
    const cv::Mat already_bgr(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
    if (bgr.rows != 4 || bgr.cols != 4 || bgr.type() != CV_8UC3) {
      std::cerr << "[FAIL] a retained NV12 frame must convert to a BGR picture of its size\n";
      ++failures;
    } else if (bgr_from_host_frame(already_bgr).data != already_bgr.data) {
      std::cerr << "[FAIL] a retained BGR frame must be used as it is\n";
      ++failures;
    } else {
      std::cout << "[OK] retained host frames convert to BGR only when saved\n";
    }
  }

  bool ok = failures == 0;
  ok &= test_help_runs(binary);
  ok &= test_unknown_flag_fails(binary);
  ok &= test_missing_config_file_fails(binary);
  ok &= test_empty_url_falls_back_to_the_legacy_key(binary);
  ok &= test_absent_url_falls_back_to_the_legacy_key(binary);
  ok &= test_present_url_wins_over_the_legacy_key(binary);
  ok &= test_both_urls_empty_is_rejected(binary);
  ok &= test_empty_labels_is_rejected(binary);
  ok &= test_configuration_rules_are_enforced(binary);
  ok &= test_closed_output_is_terminal();
  return ok ? 0 : 1;
}
