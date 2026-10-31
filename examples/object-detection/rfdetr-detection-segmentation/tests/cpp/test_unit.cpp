#define main rfdetr_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_process.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

// Configuration rules load_config enforces (Refs #526), driven in-process the
// way the variant check above is: each config is the minimal valid one with
// exactly one value broken, and the exception must name the rule.
struct RuleCase {
  const char* name;
  const char* body;
  const char* message;
};

int configuration_rule_failures() {
  const std::vector<RuleCase> cases = {
      {"backbone-empty",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: ''\n      transformer: small-t.tar.gz\nsource:\n  rtsp_url: "
       "rtsp://camera/live\ninference:\n  detection: {}\noutput:\n  insight:\n    host: "
       "127.0.0.1\n",
       "backbone and transformer must be set"},
      {"labels-empty",
       "model:\n  task: detection\n  labels: ''\n  detection:\n    variant: small\n    small:\n    "
       "  backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  rtsp_url: "
       "rtsp://camera/live\ninference:\n  detection: {}\noutput:\n  insight:\n    host: "
       "127.0.0.1\n",
       "model.labels must be set"},
      {"rtsp-url-not-rtsp",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: http://127.0.0.1:8080/stream\ninference:\n  detection: {}\noutput:\n  insight:\n "
       "   host: 127.0.0.1\n",
       "source.rtsp_url must be an RTSP URL"},
      {"latency-negative",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\n  latency_ms: -1\ninference:\n  detection: {}\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "source.latency_ms and inference.frames must be >= 0"},
      {"frames-negative",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\ninference:\n  frames: -1\n  detection: {}\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "source.latency_ms and inference.frames must be >= 0"},
      {"width-negative",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\n  width: -1\ninference:\n  detection: {}\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "source.width, source.height, and source.fps must be >= 0"},
      {"min-score-above",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\ninference:\n  detection:\n    min_score: 1.5\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "min_score must be in [0, 1]"},
      {"max-detections-zero",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\ninference:\n  detection:\n    max_detections: 0\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "max_detections/max_segments must be > 0"},
      {"mask-threshold-above",
       "model:\n  task: segmentation\n  labels: labels.txt\n  segmentation:\n    backbone: "
       "seg-b.tar.gz\n    transformer: seg-t.tar.gz\nsource:\n  rtsp_url: "
       "rtsp://camera/live\ninference:\n  segmentation:\n    mask_threshold: 2.0\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "inference.segmentation.mask_threshold must be in [0, 1]"},
      {"mask-grid-below",
       "model:\n  task: segmentation\n  labels: labels.txt\n  segmentation:\n    backbone: "
       "seg-b.tar.gz\n    transformer: seg-t.tar.gz\nsource:\n  rtsp_url: "
       "rtsp://camera/live\ninference:\n  segmentation:\n    mask_grid_size: 107\noutput:\n  "
       "insight:\n    host: 127.0.0.1\n",
       "inference.segmentation.mask_grid_size must be >= 108"},
      {"insight-host-empty",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\ninference:\n  detection: {}\noutput:\n  insight:\n    host: "
       "''\n",
       "output.insight.host must be set"},
      {"video-port-zero",
       "model:\n  task: detection\n  labels: labels.txt\n  detection:\n    variant: small\n    "
       "small:\n      backbone: small-b.tar.gz\n      transformer: small-t.tar.gz\nsource:\n  "
       "rtsp_url: rtsp://camera/live\ninference:\n  detection: {}\noutput:\n  insight:\n    host: "
       "127.0.0.1\n    video_port: 0\n",
       "Insight ports must be in [1, 65535]"},
  };
  int failures = 0;
  const std::string temp_dir =
      create_test_scratch_dir("rfdetr-detection-segmentation", "configuration-rules");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] could not create config test directory\n";
    return 1;
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  for (const RuleCase& c : cases) {
    std::ofstream(config_path, std::ios::trunc) << c.body;
    try {
      (void)load_config(config_path);
      std::cerr << "[FAIL] " << c.name << ": config must be rejected\n";
      ++failures;
    } catch (const std::exception& error) {
      if (std::string(error.what()).find(c.message) == std::string::npos) {
        std::cerr << "[FAIL] " << c.name << ": error does not name the rule: " << error.what()
                  << "\n";
        ++failures;
      } else {
        std::cout << "[OK] " << c.name << " is rejected by its rule\n";
      }
    }
  }
  remove_dir(temp_dir);
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  int failures = 0;

  if (parse_source_codec("h264") != SourceCodec::H264 ||
      parse_source_codec("AVC") != SourceCodec::H264 ||
      parse_source_codec("h265") != SourceCodec::H265 ||
      parse_source_codec("HEVC") != SourceCodec::H265 ||
      parse_source_codec("mjpeg") != SourceCodec::Mjpeg ||
      parse_source_codec("JPEG") != SourceCodec::Mjpeg) {
    std::cerr << "[FAIL] source codec aliases must resolve to H.264, H.265, or MJPEG\n";
    ++failures;
  }

  const SourceGeometry probed{1280, 720, 60};
  const SourceGeometry fallback{640, 480, 30};
  const auto resolved = resolve_geometry(probed, fallback);
  const auto partial = resolve_geometry({1280, 0, 0}, fallback);
  if (resolved.width != 1280 || resolved.height != 720 || resolved.fps != 30 ||
      partial.width != 1280 || partial.height != 480 || partial.fps != 30) {
    std::cerr << "[FAIL] configured FPS must override the probe and dimensions remain fallbacks\n";
    ++failures;
  }

  auto shared_output = neat::Tensor::from_vector(
      std::vector<float>{99.0F, 1.0F, 2.0F, 3.0F, 4.0F, 88.0F}, {6}, neat::TensorMemory::CPU);
  shared_output.shape = {2};
  shared_output.byte_offset = sizeof(float);
  if (read_floats(shared_output) != std::vector<float>{1.0F, 2.0F}) {
    std::cerr << "[FAIL] tensor reads must exclude adjacent output storage\n";
    ++failures;
  }
  auto padded_output = neat::Tensor::from_vector(
      std::vector<float>{1.0F, 2.0F, 99.0F, 3.0F, 4.0F, 88.0F}, {6}, neat::TensorMemory::CPU);
  padded_output.shape = {2, 2};
  padded_output.strides_bytes = {3 * sizeof(float), sizeof(float)};
  if (read_floats(padded_output) != std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}) {
    std::cerr << "[FAIL] tensor reads must respect padded row strides\n";
    ++failures;
  }
  shared_output.shape = {8};
  try {
    (void)read_floats(shared_output);
    std::cerr << "[FAIL] tensor reads must reject insufficient storage\n";
    ++failures;
  } catch (const std::exception&) {
  }

  std::vector<float> scores(305, 0.0F);
  scores[3] = 2.0F;
  scores[4] = 2.0F;
  std::vector<float> proposals(scores.size() * 4U);
  for (std::size_t index = 0; index < scores.size(); ++index) {
    proposals[index * 4U] = static_cast<float>(index);
  }
  const auto gathered = stable_topk_gather(scores, proposals, 300);
  if (gathered.size() != 1200U || gathered[0] != 3.0F || gathered[4] != 4.0F ||
      gathered[8] != 0.0F) {
    std::cerr << "[FAIL] TopK/Gather must be stable, descending, and limited to 300 rows\n";
    ++failures;
  }

  std::vector<std::string> labels(91, "unused");
  labels[1] = "person";
  std::vector<float> boxes(1200, 0.0F);
  boxes[0] = 0.5F;
  boxes[1] = 0.5F;
  boxes[2] = 0.5F;
  boxes[3] = 0.25F;
  std::vector<float> logits(27300, -20.0F);
  logits[1] = 10.0F;
  const auto objects = postprocess(boxes, logits, 1920, 1080, labels, 0.5F, 10, 300);
  if (objects.size() != 1U || objects[0].label != "person" ||
      std::abs(objects[0].x - 480.0F) > 0.01F || std::abs(objects[0].y - 405.0F) > 0.01F ||
      std::abs(objects[0].w - 960.0F) > 0.01F || std::abs(objects[0].h - 270.0F) > 0.01F) {
    std::cerr << "[FAIL] postprocessing must preserve sparse COCO IDs and source geometry\n";
    ++failures;
  }

  Config segmentation_config;
  segmentation_config.task = Task::Segmentation;
  segmentation_config.top_k = 200;
  segmentation_config.min_score = 0.3F;
  segmentation_config.max_results = 1;
  segmentation_config.mask_threshold = 0.08F;
  std::vector<float> segmentation_boxes(200U * 4U, 0.0F);
  segmentation_boxes[0] = 0.5F;
  segmentation_boxes[1] = 0.5F;
  segmentation_boxes[2] = 0.5F;
  segmentation_boxes[3] = 0.5F;
  std::copy_n(segmentation_boxes.begin(), 4, segmentation_boxes.begin() + 4);
  std::vector<float> segmentation_logits(200U * 91U, -20.0F);
  segmentation_logits[0] = 12.0F;
  segmentation_logits[1] = 11.0F;
  segmentation_logits[91U + 1U] = 10.0F;
  std::vector<float> masks(108U * 108U * 200U, -20.0F);
  for (int y = 40; y < 68; ++y) {
    for (int x = 40; x < 68; ++x) {
      masks[static_cast<std::size_t>((y * 108 + x) * 200)] = 10.0F;
    }
  }
  TransformerOutputs segmentation_output{
      neat::Tensor::from_vector(segmentation_boxes, {1, 200, 4}, neat::TensorMemory::CPU),
      neat::Tensor::from_vector(segmentation_logits, {1, 200, 91}, neat::TensorMemory::CPU),
      neat::Tensor::from_vector(masks, {1, 108, 108, 200}, neat::TensorMemory::CPU),
  };
  for (const int grid_size : {108, 432, 640}) {
    segmentation_config.mask_grid_size = grid_size;
    const auto segments = nlohmann::json::parse(
        segmentation_metadata(segmentation_output, 1280, 720, labels, segmentation_config));
    const auto& segment_entries = segments.at("segments");
    bool valid_polygon = segment_entries.size() == 1U &&
                         segment_entries.front().at("label") == "person" &&
                         segment_entries.front().at("mask").size() >= 3U;
    if (valid_polygon) {
      for (const auto& point : segment_entries.front().at("mask")) {
        valid_polygon = valid_polygon && point.at(0).get<int>() >= 0 &&
                        point.at(0).get<int>() < 1280 && point.at(1).get<int>() >= 0 &&
                        point.at(1).get<int>() < 720;
      }
    }
    if (!valid_polygon || segments.dump().size() > kMetadataByteBudget) {
      std::cerr << "[FAIL] segmentation metadata must contain a labeled polygon\n";
      ++failures;
    }
  }

  const std::string temp_dir =
      create_test_scratch_dir("rfdetr-detection-segmentation", "model-variant");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] could not create config test directory\n";
    ++failures;
  } else {
    const fs::path config_path = fs::path(temp_dir) / "config.yaml";
    std::ofstream config(config_path);
    config << "model:\n  task: detection\n  detection:\n    variant: large\n"
              "source: {}\ninference: {}\noutput:\n  insight: {}\n";
    config.close();
    try {
      (void)load_config(config_path);
      std::cerr << "[FAIL] config must reject a variant without a model pair\n";
      ++failures;
    } catch (const std::exception& error) {
      if (std::string(error.what()).find("model.detection.large") == std::string::npos) {
        std::cerr << "[FAIL] missing model pair error must name the selected variant\n";
        ++failures;
      }
    }

    config.open(config_path, std::ios::trunc);
    config << "model:\n  task: segmentation\n  labels: labels.txt\n  segmentation:\n"
              "    backbone: segmentation-b.tar.gz\n"
              "    transformer: segmentation-t.tar.gz\n"
              "source:\n  rtsp_url: rtsp://camera/live\n"
              "inference:\n  segmentation:\n    max_segments: 24\n    mask_grid_size: 640\n"
              "output:\n  insight:\n    host: 127.0.0.1\n";
    config.close();
    try {
      const auto selected = load_config(config_path);
      if (selected.task != Task::Segmentation || selected.top_k != 200 ||
          selected.mask_grid_size != 640 || selected.backbone != "segmentation-b.tar.gz" ||
          selected.min_score != 0.3F) {
        std::cerr << "[FAIL] config must select the fixed segmentation model contract\n";
        ++failures;
      }
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] valid segmentation config was rejected: " << error.what() << "\n";
      ++failures;
    }

    config.open(config_path, std::ios::trunc);
    config << "model:\n  task: detection\n  labels: labels.txt\n  detection:\n"
              "    variant: large\n    large:\n      backbone: large-b.tar.gz\n"
              "      transformer: large-t.tar.gz\n"
              "source:\n  rtsp_url: rtsp://camera/live\n"
              "inference:\n  segmentation:\n    mask_threshold: 2.0\n"
              "output:\n  insight:\n    host: 127.0.0.1\n";
    config.close();
    try {
      const auto selected = load_config(config_path);
      if (selected.task != Task::Detection || selected.backbone != "large-b.tar.gz") {
        std::cerr << "[FAIL] inactive segmentation settings must not affect detection\n";
        ++failures;
      }
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] inactive segmentation settings rejected detection: " << error.what()
                << "\n";
      ++failures;
    }
    remove_dir(temp_dir);
  }

  const std::string binary = argv[1];
  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  if (help.exit_code != 0 || help.stdout_text.find("--config") == std::string::npos) {
    std::cerr << "[FAIL] help should describe the config-driven CLI\n";
    ++failures;
  }
  const auto no_config = spawn_and_wait(binary, {}, 20000);
  if (no_config.exit_code == 0 ||
      no_config.stderr_text.find("--config is required") == std::string::npos) {
    std::cerr << "[FAIL] the application should require an explicit config path\n";
    ++failures;
  }
  const auto missing =
      spawn_and_wait(binary, {"--config", "/nonexistent/rfdetr-config.yaml"}, 20000);
  if (missing.exit_code == 0 || missing.stderr_text.find("config") == std::string::npos) {
    std::cerr << "[FAIL] a missing config should fail clearly\n";
    ++failures;
  }
  failures += configuration_rule_failures();
  // The pull loop treats a timeout as "try again" and a closed output or pull error as the
  // end of the run, carrying the runtime's own reason.
  {
    using sima_examples::pull_status_has_sample;
    using simaai::neat::PullStatus;
    simaai::neat::PullError pull_error;
    pull_error.message = "queue torn down";
    const auto thrown_message = [&](PullStatus status) -> std::string {
      try {
        (void)pull_status_has_sample(status, "backbone", pull_error, "source reached EOS");
      } catch (const std::runtime_error& error) {
        return error.what();
      }
      return "";
    };
    const std::string closed = thrown_message(PullStatus::Closed);
    const std::string errored = thrown_message(PullStatus::Error);
    if (closed != "backbone output closed unexpectedly: source reached EOS") {
      std::cerr << "[FAIL] closed output should end the run with the reason, got: " << closed
                << "\n";
      ++failures;
    } else if (errored != "failed to pull backbone: queue torn down") {
      std::cerr << "[FAIL] pull error should end the run with its message, got: " << errored
                << "\n";
      ++failures;
    } else if (pull_status_has_sample(PullStatus::Timeout, "backbone", pull_error, "") ||
               !pull_status_has_sample(PullStatus::Ok, "backbone", pull_error, "")) {
      std::cerr << "[FAIL] a timeout is not a sample and a successful pull is\n";
      ++failures;
    } else {
      std::cout << "[OK] closed output and pull error are terminal, timeout is not\n";
    }
  }

  return failures == 0 ? 0 : 1;
}
