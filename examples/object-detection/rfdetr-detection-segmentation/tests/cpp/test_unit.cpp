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
  segmentation_logits[1] = 11.0F;
  segmentation_logits[91U + 1U] = 10.0F;
  std::vector<float> masks(108U * 108U * 200U, -20.0F);
  for (int y = 40; y < 68; ++y) {
    for (int x = 40; x < 68; ++x) {
      masks[static_cast<std::size_t>((y * 108 + x) * 200 + 1)] = 10.0F;
    }
  }
  TransformerOutputs segmentation_output{
      neat::Tensor::from_vector(segmentation_boxes, {1, 200, 4}, neat::TensorMemory::CPU),
      neat::Tensor::from_vector(segmentation_logits, {1, 200, 91}, neat::TensorMemory::CPU),
      neat::Tensor::from_vector(masks, {108, 108, 200}, neat::TensorMemory::CPU),
  };
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

  const std::string temp_dir =
      create_test_scratch_dir("rfdetr-detection-segmentation", "unknown-model-variant");
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
      std::cerr << "[FAIL] config must reject model variants other than small and medium\n";
      ++failures;
    } catch (const std::exception& error) {
      if (std::string(error.what()).find("small or medium") == std::string::npos) {
        std::cerr << "[FAIL] invalid model variant error must name the supported variants\n";
        ++failures;
      }
    }

    config.open(config_path, std::ios::trunc);
    config << "model:\n  task: segmentation\n  labels: labels.txt\n  segmentation:\n"
              "    backbone: segmentation-b.tar.gz\n"
              "    transformer: segmentation-t.tar.gz\n"
              "source:\n  rtsp_url: rtsp://camera/live\n"
              "inference:\n  segmentation:\n    max_segments: 24\n"
              "output:\n  insight:\n    host: 127.0.0.1\n";
    config.close();
    try {
      const auto selected = load_config(config_path);
      if (selected.task != Task::Segmentation || selected.input_size != 432 ||
          selected.feature_size != 36 || selected.top_k != 200 ||
          selected.backbone != "segmentation-b.tar.gz" || selected.min_score != 0.3F) {
        std::cerr << "[FAIL] config must select the fixed segmentation model contract\n";
        ++failures;
      }
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] valid segmentation config was rejected: " << error.what() << "\n";
      ++failures;
    }

    config.open(config_path, std::ios::trunc);
    config << "model:\n  task: detection\n  labels: labels.txt\n  detection:\n"
              "    variant: small\n    small:\n      backbone: small-b.tar.gz\n"
              "      transformer: small-t.tar.gz\n"
              "source:\n  rtsp_url: rtsp://camera/live\n"
              "inference:\n  segmentation:\n    mask_threshold: 2.0\n"
              "output:\n  insight:\n    host: 127.0.0.1\n";
    config.close();
    try {
      const auto selected = load_config(config_path);
      if (selected.task != Task::Detection || selected.input_size != 512) {
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
  return failures == 0 ? 0 : 1;
}
