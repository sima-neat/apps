// Unit test for ssd-mobilenet-object-detector: CLI arg handling and decode config validation.
#include "examples/object-detection/ssd-mobilenet-object-detector/src/cpp/aggregate_suppression.h"
#include "examples/object-detection/ssd-mobilenet-object-detector/src/cpp/output_paths.h"
#include "support/runtime/config_utils.h"
#include "support/testing/test_process.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

struct TestBox {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
};

bool has_box(const std::vector<TestBox>& boxes, float x1, int class_id) {
  return std::any_of(boxes.begin(), boxes.end(),
                     [=](const TestBox& box) { return box.x1 == x1 && box.class_id == class_id; });
}

// Write a config whose decode section carries the supplied value for `key`.
fs::path write_decode_config(const fs::path& dir, const std::string& name, const std::string& key,
                             const std::string& value) {
  const fs::path path = dir / name;
  std::ofstream out(path);
  out << "model:\n"
      << "  path: /nonexistent/ssd_model.tar.gz\n"
      << "io:\n"
      << "  input_dir: assets/datasets/coco\n"
      << "  output_dir: sandbox/ssd-mobilenet-object-detector\n"
      << "decode:\n"
      << "  " << key << ": " << value << "\n";
  return path;
}

bool expect_config_rejected(const std::string& binary, const fs::path& config,
                            const std::string& expected_message, const std::string& label) {
  ProcessResult r = spawn_and_wait(binary, {"--config", config.string()}, 20000);
  if (r.exit_code != 2) {
    std::cerr << "[FAIL] " << label << ": expected exit 2, got " << r.exit_code << "\n";
    return false;
  }
  if (r.stderr_text.find(expected_message) == std::string::npos) {
    std::cerr << "[FAIL] " << label << ": stderr does not mention '" << expected_message
              << "'\nstderr: " << r.stderr_text << "\n";
    return false;
  }
  std::cout << "[OK] " << label << " correctly rejected\n";
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  int failures = 0;

  {
    const std::vector<TestBox> boxes = {
        {0.0f, 0.0f, 600.0f, 600.0f, 0.9f, 3},
        {10.0f, 10.0f, 50.0f, 50.0f, 0.8f, 3},
        {60.0f, 60.0f, 100.0f, 100.0f, 0.8f, 3},
    };
    const auto filtered = ssd_mobilenet::suppress_aggregate_boxes(
        boxes, 640, 640, ssd_mobilenet::AggregateSuppressionOptions{});
    if (filtered.size() != boxes.size()) {
      std::cerr << "[FAIL] aggregate suppression was enabled by default\n";
      ++failures;
    } else {
      std::cout << "[OK] aggregate suppression is opt-in\n";
    }
  }

  {
    // Regression for COCO 000000210273: a road-sized class-3 crowd region enclosing individual
    // cars is hidden, while the children and an unrelated large bus remain.
    const std::vector<TestBox> boxes = {
        {43.0f, 180.0f, 617.0f, 467.0f, 0.64f, 3},  {270.0f, 330.0f, 324.0f, 374.0f, 0.76f, 3},
        {306.0f, 356.0f, 368.0f, 409.0f, 0.61f, 3}, {420.0f, 373.0f, 489.0f, 440.0f, 0.64f, 3},
        {20.0f, 80.0f, 620.0f, 500.0f, 0.90f, 6},
    };
    ssd_mobilenet::AggregateSuppressionOptions options;
    options.enabled = true;
    const auto filtered = ssd_mobilenet::suppress_aggregate_boxes(boxes, 640, 640, options);
    if (filtered.size() != boxes.size() - 1 || has_box(filtered, 43.0f, 3) ||
        !has_box(filtered, 20.0f, 6)) {
      std::cerr << "[FAIL] aggregate suppression did not isolate the same-class crowd region\n";
      ++failures;
    } else {
      std::cout << "[OK] same-class crowd region suppressed without removing large bus\n";
    }
  }

  {
    // A large object with fewer than two materially smaller same-class children is a valid
    // instance, not an aggregate.
    const std::vector<TestBox> boxes = {
        {20.0f, 20.0f, 620.0f, 620.0f, 0.95f, 3},
        {100.0f, 100.0f, 180.0f, 180.0f, 0.80f, 3},
        {300.0f, 300.0f, 380.0f, 380.0f, 0.80f, 6},
    };
    ssd_mobilenet::AggregateSuppressionOptions options;
    options.enabled = true;
    const auto filtered = ssd_mobilenet::suppress_aggregate_boxes(boxes, 640, 640, options);
    if (filtered.size() != boxes.size() || !has_box(filtered, 20.0f, 3)) {
      std::cerr << "[FAIL] valid large object was treated as an aggregate\n";
      ++failures;
    } else {
      std::cout << "[OK] valid large object preserved\n";
    }
  }

  {
    // Worst-case max_detections=100 scan: every large box is examined and none has a qualifying
    // child. Average hot-path cost must remain below the application's 1 ms budget.
    std::vector<TestBox> boxes;
    boxes.reserve(100);
    for (int i = 0; i < 100; ++i) {
      const float inset = static_cast<float>(i % 5);
      boxes.push_back({inset, inset, 500.0f + inset, 500.0f + inset, 0.8f, 3});
    }
    constexpr int kRuns = 2000;
    ssd_mobilenet::AggregateSuppressionOptions options;
    options.enabled = true;
    std::size_t observed = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int run = 0; run < kRuns; ++run) {
      observed += ssd_mobilenet::suppress_aggregate_boxes(boxes, 640, 640, options).size();
    }
    const double mean_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count() /
        static_cast<double>(kRuns);
    if (observed != boxes.size() * kRuns || mean_ms >= 1.0) {
      std::cerr << "[FAIL] aggregate suppression mean=" << mean_ms
                << " ms (budget <1 ms), observed=" << observed << "\n";
      ++failures;
    } else {
      std::cout << "[OK] aggregate suppression mean=" << mean_ms << " ms (<1 ms)\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help: stdout does not contain usage hint\n";
      ++failures;
    } else {
      std::cout << "[OK] help prints usage\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] unknown flag: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag: stderr does not mention unknown argument\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--config"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] missing config path: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path correctly rejected\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--config", "/nonexistent/ssd-config.yaml"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] bad config: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path correctly rejected\n";
    }
  }

  // The decode section drives the model-managed BoxDecode stage, so out-of-range values must be
  // rejected before the pipeline is built rather than silently forwarded to the kernel.
  const std::string scratch =
      create_test_scratch_dir("ssd-mobilenet-object-detector", "test_decode_config");
  if (scratch.empty()) {
    std::cerr << "[FAIL] could not create scratch directory for config tests\n";
    return 1;
  }
  const fs::path scratch_dir(scratch);

  const std::array<std::tuple<std::string, std::string, std::string>, 11> malformed_configs = {{
      {"root_list.yaml", "- model\n", "config root must be a mapping"},
      {"quoted_root_scalar.yaml", "\"model: []\"\n", "config root must be a mapping"},
      {"inline_model_list.yaml", "model: []\n", "model must be a mapping"},
      {"block_model_list.yaml", "model:\n  - path\n", "model must be a mapping"},
      {"scalar_io.yaml", "io: input\n", "io must be a mapping"},
      {"model_path_list.yaml", "model:\n  path:\n    - custom.tar.gz\n",
       "model.path must be a scalar value"},
      {"inline_model_values.yaml", "model: [path]\n", "model must be a mapping"},
      {"inline_model_mapping.yaml", "model: {path: custom.tar.gz}\n",
       "inline mappings are not supported for model"},
      {"inline_model_path_list.yaml", "model:\n  path: [custom.tar.gz]\n",
       "model.path must be a scalar value"},
      {"decode_max_mapping.yaml", "decode:\n  max_detections: {}\n",
       "decode.max_detections must be a scalar value"},
      {"runtime_profile_list.yaml", "runtime:\n  profile: []\n",
       "runtime.profile must be a scalar value"},
  }};
  for (const auto& [name, contents, expected] : malformed_configs) {
    const fs::path config = scratch_dir / name;
    std::ofstream(config) << contents;
    if (!expect_config_rejected(binary, config, expected, name)) {
      ++failures;
    }
  }

  {
    const fs::path config = scratch_dir / "valid_nested_sequence.yaml";
    std::ofstream(config) << "model:\n"
                             "  path: model.tar.gz\n"
                             "streams:\n"
                             "  - rtsp://example/0\n"
                             "  - rtsp://example/1\n"
                             "runtime:\n"
                             "  num_runs:\n"
                             "  profile: null\n"
                             "literal:\n"
                             "  value: \"null\"\n";
    const auto raw = sima_examples::ScalarConfig::load(config);
    if (raw.root_kind() != sima_examples::ConfigNodeKind::Mapping ||
        raw.node_kind("model") != sima_examples::ConfigNodeKind::Mapping ||
        raw.node_kind("model.path") != sima_examples::ConfigNodeKind::Scalar ||
        raw.node_kind("streams") != sima_examples::ConfigNodeKind::Sequence ||
        raw.node_kind("runtime") != sima_examples::ConfigNodeKind::Mapping ||
        raw.node_kind("runtime.num_runs") != sima_examples::ConfigNodeKind::Null ||
        raw.int_or("runtime.num_runs", 7) != 7 ||
        raw.node_kind("runtime.profile") != sima_examples::ConfigNodeKind::Null ||
        !raw.bool_or("runtime.profile", true) ||
        raw.node_kind("literal") != sima_examples::ConfigNodeKind::Mapping ||
        raw.string_or("literal.value", "missing") != "null") {
      std::cerr << "[FAIL] scalar config did not preserve mapping/sequence/null node kinds\n";
      ++failures;
    } else {
      std::cout << "[OK] scalar config preserves mapping/sequence/null node kinds\n";
    }
  }

  {
    const fs::path input = scratch_dir / "dangling-output-input" / "frame.jpg";
    const fs::path output_dir = scratch_dir / "dangling-output";
    const fs::path missing_target = scratch_dir / "outside" / "missing.png";
    const fs::path output = output_dir / "frame_jpg.png";
    fs::create_directories(input.parent_path());
    fs::create_directories(output_dir);
    std::ofstream(input).put('\n');
    fs::create_symlink(missing_target, output);
    const int removed = ssd_mobilenet::clear_output_images(output_dir, {input});
    std::error_code ec;
    const fs::file_status output_status = fs::symlink_status(output, ec);
    if (removed != 1 || output_status.type() != fs::file_type::not_found ||
        fs::exists(missing_target)) {
      std::cerr << "[FAIL] dangling output symlink was not removed by pathname\n";
      ++failures;
    } else {
      std::cout << "[OK] dangling output symlink removed by pathname\n";
    }
  }

  {
    const fs::path model = scratch_dir / "protected-model.tar.gz";
    const fs::path labels = scratch_dir / "protected-labels.txt";
    const fs::path input_dir = scratch_dir / "protected-input";
    const fs::path image = input_dir / "frame.jpg";
    const fs::path output_dir = scratch_dir / "protected-output";
    fs::create_directories(input_dir);
    std::ofstream(model).put('\n');
    std::ofstream(labels) << "background\nperson\n";
    std::ofstream(image).put('\n');
    const std::array<std::pair<std::string, fs::path>, 3> collisions = {{
        {"model", model},
        {"labels", labels},
        {"overlay", output_dir / "frame_jpg.png"},
    }};
    for (const auto& [name, collision_path] : collisions) {
      const fs::path config = scratch_dir / ("protected_" + name + ".yaml");
      std::ofstream out(config);
      out << "model:\n"
          << "  path: " << model.string() << "\n"
          << "  labels: " << labels.string() << "\n"
          << "io:\n"
          << "  input_dir: " << input_dir.string() << "\n"
          << "  output_dir: " << output_dir.string() << "\n"
          << "  detections_json: " << collision_path.string() << "\n"
          << "output:\n"
          << "  overlay: true\n";
      out.close();
      const std::string expected = name == "overlay" ? "generated overlay" : "consumed input";
      if (!expect_config_rejected(binary, config,
                                  "io.detections_json must not overwrite a " + expected,
                                  name + " detections report alias")) {
        ++failures;
      }
    }

    const fs::path config = scratch_dir / "protected_config.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "  labels: " << labels.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << output_dir.string() << "\n"
        << "  detections_json: " << config.string() << "\n"
        << "output:\n"
        << "  overlay: false\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "io.detections_json must not overwrite a consumed input",
                                "config detections report alias")) {
      ++failures;
    }

    const fs::path hardlink_report = scratch_dir / "hardlink-report.json";
    fs::create_hard_link(image, hardlink_report);
    const fs::path hardlink_config = scratch_dir / "hardlink_report.yaml";
    std::ofstream hardlink_out(hardlink_config);
    hardlink_out << "model:\n"
                 << "  path: " << model.string() << "\n"
                 << "  labels: " << labels.string() << "\n"
                 << "io:\n"
                 << "  input_dir: " << input_dir.string() << "\n"
                 << "  output_dir: " << output_dir.string() << "\n"
                 << "  detections_json: " << hardlink_report.string() << "\n"
                 << "output:\n"
                 << "  overlay: false\n";
    hardlink_out.close();
    if (!expect_config_rejected(binary, hardlink_config,
                                "io.detections_json must not overwrite an input image",
                                "hard-linked detections report alias")) {
      ++failures;
    }

    const fs::path hardlink_overlay = output_dir / "frame_jpg.png";
    fs::create_directories(output_dir);
    fs::create_hard_link(model, hardlink_overlay);
    const fs::path overlay_config = scratch_dir / "hardlink_overlay.yaml";
    std::ofstream overlay_out(overlay_config);
    overlay_out << "model:\n"
                << "  path: " << model.string() << "\n"
                << "  labels: " << labels.string() << "\n"
                << "io:\n"
                << "  input_dir: " << input_dir.string() << "\n"
                << "  output_dir: " << output_dir.string() << "\n"
                << "output:\n"
                << "  overlay: true\n";
    overlay_out.close();
    if (!expect_config_rejected(binary, overlay_config,
                                "generated overlay must not overwrite a consumed input",
                                "hard-linked overlay alias")) {
      ++failures;
    }

    const fs::path profile_config = scratch_dir / "profile_output_aliases.yaml";
    std::ofstream profile_out(profile_config);
    profile_out << "model:\n"
                << "  path: " << model.string() << "\n"
                << "  labels: " << labels.string() << "\n"
                << "io:\n"
                << "  input_dir: " << input_dir.string() << "\n"
                << "  output_dir: " << input_dir.string() << "\n"
                << "  detections_json: " << image.string() << "\n"
                << "runtime:\n"
                << "  profile: true\n"
                << "output:\n"
                << "  overlay: true\n";
    profile_out.close();
    const ProcessResult profile_result =
        spawn_and_wait(binary, {"--config", profile_config.string()}, 20000);
    const std::array<std::string, 3> unused_output_errors = {
        "io.output_dir must differ from io.input_dir",
        "io.detections_json must not overwrite",
        "generated overlay must not overwrite",
    };
    for (const std::string& message : unused_output_errors) {
      if (profile_result.stderr_text.find(message) != std::string::npos) {
        std::cerr << "[FAIL] profiling rejected an unused output path: " << message << "\n";
        ++failures;
      }
    }
  }

  if (!expect_config_rejected(
          binary, write_decode_config(scratch_dir, "score.yaml", "score_threshold", "1.5"),
          "decode.score_threshold must be in [0.0, 1.0]", "out-of-range score_threshold")) {
    ++failures;
  }
  if (!expect_config_rejected(
          binary, write_decode_config(scratch_dir, "score_nan.yaml", "score_threshold", ".nan"),
          "decode.score_threshold must be in [0.0, 1.0]", "non-finite score_threshold")) {
    ++failures;
  }
  if (!expect_config_rejected(binary,
                              write_decode_config(scratch_dir, "iou.yaml", "nms_iou", "-0.1"),
                              "decode.nms_iou must be in [0.0, 1.0]", "out-of-range nms_iou")) {
    ++failures;
  }
  if (!expect_config_rejected(binary,
                              write_decode_config(scratch_dir, "iou_nan.yaml", "nms_iou", ".nan"),
                              "decode.nms_iou must be in [0.0, 1.0]", "non-finite nms_iou")) {
    ++failures;
  }
  if (!expect_config_rejected(
          binary, write_decode_config(scratch_dir, "topk.yaml", "max_detections", "0"),
          "decode.max_detections must be >= 1", "non-positive max_detections")) {
    ++failures;
  }

  {
    const fs::path model = scratch_dir / "model.tar.gz";
    const fs::path input_dir = scratch_dir / "input";
    const fs::path image = input_dir / "frame.jpg";
    fs::create_directories(input_dir);
    std::ofstream(model).put('\n');
    std::ofstream(image).put('\n');
    const fs::path config = scratch_dir / "detections_alias.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << (scratch_dir / "output").string() << "\n"
        << "  detections_json: " << image.string() << "\n"
        << "output:\n"
        << "  overlay: false\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "io.detections_json must not overwrite an input image",
                                "input-image detections report alias")) {
      ++failures;
    }
  }

  {
    const fs::path model = scratch_dir / "image-report-model.tar.gz";
    const fs::path input_dir = scratch_dir / "image-report-input";
    const fs::path report = input_dir / "detections.png";
    fs::create_directories(input_dir);
    std::ofstream(model).put('\n');
    std::ofstream(input_dir / "frame.jpg").put('\n');
    const fs::path config = scratch_dir / "image_report.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << (scratch_dir / "image-report-output").string() << "\n"
        << "  detections_json: " << report.string() << "\n"
        << "output:\n"
        << "  overlay: false\n";
    out.close();
    if (!expect_config_rejected(
            binary, config, "io.detections_json must not use an image filename inside io.input_dir",
            "new image-named detections report inside input_dir")) {
      ++failures;
    }
    if (fs::exists(report)) {
      std::cerr << "[FAIL] rejected image-named detections report was created\n";
      ++failures;
    }
  }

  {
    const fs::path output_dir = scratch_dir / "overlay-collision-output";
    const fs::path model = output_dir / "frame_jpg.png";
    const fs::path labels = scratch_dir / "overlay-collision-labels.txt";
    const fs::path input_dir = scratch_dir / "overlay-collision-input";
    fs::create_directories(output_dir);
    fs::create_directories(input_dir);
    std::ofstream(model).put('\n');
    std::ofstream(labels) << "background\nperson\n";
    std::ofstream(input_dir / "frame.jpg").put('\n');
    const fs::path config = scratch_dir / "overlay_collision.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "  labels: " << labels.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << output_dir.string() << "\n"
        << "output:\n"
        << "  overlay: true\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "generated overlay must not overwrite a consumed input",
                                "overlay model alias")) {
      ++failures;
    }
  }

  {
    const fs::path model = scratch_dir / "model.tar.gz";
    const fs::path input_dir = scratch_dir / "input";
    const fs::path missing_labels = scratch_dir / "custom" / "coco_labels.txt";
    const fs::path config = scratch_dir / "missing_custom_labels.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "  labels: " << missing_labels.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << (scratch_dir / "output").string() << "\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "labels file does not exist: " + missing_labels.string(),
                                "missing custom coco_labels path")) {
      ++failures;
    }
  }

  remove_dir(scratch);
  return failures > 0 ? 1 : 0;
}
