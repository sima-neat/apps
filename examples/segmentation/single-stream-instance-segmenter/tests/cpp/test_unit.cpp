// Unit test for single-stream-instance-segmenter: CLI handling plus the YOLOv8 decode
// boundary, checked against the same tests/fixtures/decode_parity.json the Python unit
// test uses. Both implementations must decode the shared fixture identically.
#define main single_stream_instance_segmenter_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::spawn_and_wait;

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
      simaai::neat::Sample frame;
      frame.frame_id = frame_id;
      runtime.frames.emplace_back(frame_id, std::move(frame));
    }
    const auto* paired = frame_for(runtime, 8);
    if (paired == nullptr || paired->frame_id != 8) {
      std::cerr << "[FAIL] a retained frame must pair with the segments it produced\n";
      ++failures;
    } else if (frame_for(runtime, 3) != nullptr || frame_for(runtime, -1) != nullptr) {
      std::cerr << "[FAIL] aged-out and unidentified samples must not pair\n";
      ++failures;
    } else {
      std::cout << "[OK] frames pair with the segments they produced\n";
    }
  }

  // Test 5: --help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] --help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] --help printed usage\n";
    }
  }

  // Test 6: unknown flag is rejected.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --bogus: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag rejected\n";
    }
  }

  // Test 7: bad config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad config: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
