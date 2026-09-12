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
  const std::vector<float> boxes = {480, 405, 1440, 675, 1.0F, 0, 480, 405, 1440, 675, 0.99F, 1};
  const auto objects = postprocess(boxes, 1920, 1080, labels, 0.5F, 10);
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
  const std::vector<float> segmentation_boxes = {320, 180,   960, 540, 1.0F, 0,   320, 180,   960,
                                                 540, 0.99F, 1,   320, 180,  960, 540, 0.98F, 1};
  std::vector<float> masks(3U * 108U * 108U, 0.0F);
  for (int y = 40; y < 68; ++y) {
    for (int x = 40; x < 68; ++x)
      masks[(2U * 108U + y) * 108U + x] = 1.0F;
  }
  TransformerOutputs segmentation_output{
      neat::Tensor::from_vector(segmentation_boxes, {3, 6}, neat::TensorMemory::CPU),
      neat::Tensor::from_vector(masks, {3, 108, 108}, neat::TensorMemory::CPU),
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

  // Inclusive cutoffs, clipping and repeated selected-mask associations.
  const std::vector<float> boundary_boxes = {-2, -3, 20, 15, 0.5F, 1, -2, -3, 20, 15, 0.5F, 1};
  const auto boundary_objects = postprocess(boundary_boxes, 13, 9, labels, 0.5F, 2);
  if (boundary_objects.size() != 2U || boundary_objects[0].x != 0 || boundary_objects[0].y != 0 ||
      boundary_objects[0].w != 13 || boundary_objects[0].h != 9 ||
      !postprocess({}, 13, 9, labels, 0.5F, 2).empty()) {
    std::cerr << "[FAIL] decoded boxes must preserve equality, repeats and clipping\n";
    ++failures;
  }
  segmentation_config.min_score = 0.5F;
  segmentation_config.mask_threshold = 0.5F;
  segmentation_config.mask_grid_size = 108;
  segmentation_config.max_results = 2;
  auto boundary_output = [&](int count, float probability) {
    if (count == 0) {
      const uint32_t header[] = {0x31564452, 1, 40, 0, 40, 40, 0, 108, 108, 2};
      std::vector<uint8_t> bytes(sizeof(header));
      std::memcpy(bytes.data(), header, sizeof(header));
      auto wire = neat::Tensor::from_vector(bytes, {40}, neat::TensorMemory::CPU);
      neat::tag_detection_format(wire, "RFDETR_V1");
      auto decoded = neat::decode_segmentation({wire}).front();
      return TransformerOutputs{decoded.boxes, decoded.masks};
    }
    return TransformerOutputs{
        neat::Tensor::from_vector(
            std::vector<float>(boundary_boxes.begin(), boundary_boxes.begin() + count * 6),
            {count, 6}, neat::TensorMemory::CPU),
        neat::Tensor::from_vector(std::vector<float>(count * 108U * 108U, probability),
                                  {count, 108, 108}, neat::TensorMemory::CPU)};
  };
  const auto repeated_output = boundary_output(2, 0.5F);
  const auto boundary_segments = nlohmann::json::parse(
      segmentation_metadata(repeated_output, 13, 9, labels, segmentation_config));
  nlohmann::json expected_segments = nlohmann::json::array();
  for (int i = 1; i <= 2; ++i) {
    expected_segments.push_back({{"id", "seg_" + std::to_string(i)},
                                 {"label", "person"},
                                 {"confidence", 0.5},
                                 {"bbox", {0, 0, 13, 9}},
                                 {"mask_format", "polygon"},
                                 {"mask", {{0, 0}, {0, 8}, {12, 8}, {12, 0}}}});
  }
  if (boundary_segments.at("segments") != expected_segments) {
    std::cerr << "[FAIL] native equality masks must preserve exact clipped polygons\n";
    ++failures;
  }
  for (const auto& output :
       {boundary_output(0, 0.5F), boundary_output(2, std::nextafter(0.5F, 0.0F))}) {
    const auto empty =
        nlohmann::json::parse(segmentation_metadata(output, 13, 9, labels, segmentation_config));
    if (!empty.at("segments").empty()) {
      std::cerr << "[FAIL] empty/below-threshold masks must clear segments\n";
      ++failures;
    }
  }
  auto long_labels = labels;
  long_labels[1] = std::string(kMetadataByteBudget / 2, 'x');
  const auto limited =
      segmentation_metadata(repeated_output, 13, 9, long_labels, segmentation_config);
  if (nlohmann::json::parse(limited).at("segments").size() != 1U ||
      limited.size() > kMetadataByteBudget) {
    std::cerr << "[FAIL] metadata budget must truncate only complete segments\n";
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
              "inference:\n  segmentation:\n    max_segments: 24\n    mask_grid_size: 640\n"
              "output:\n  insight:\n    host: 127.0.0.1\n";
    config.close();
    try {
      const auto selected = load_config(config_path);
      if (selected.task != Task::Segmentation || selected.input_size != 432 ||
          selected.feature_size != 36 || selected.top_k != 200 || selected.mask_grid_size != 640 ||
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
