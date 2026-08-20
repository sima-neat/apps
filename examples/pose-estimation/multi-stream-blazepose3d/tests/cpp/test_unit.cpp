#include "examples/pose-estimation/multi-stream-blazepose3d/src/cpp/pose_logic.h"
#include "support/testing/test_process.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

fs::path write_config(const std::string& test_name, const std::string& streams) {
  const fs::path directory = create_test_scratch_dir("multi-stream-blazepose3d", test_name);
  const fs::path path = directory / "config.yaml";
  std::ofstream output(path);
  output << "models:\n"
            "  detector_path: detector.tar.gz\n"
            "  pose_path: pose.tar.gz\n"
            "streams:\n"
         << streams
         << "output:\n"
            "  insight:\n"
            "    host: 127.0.0.1\n";
  return path;
}

bool test_math_contract() {
  const blazepose_app::Box box{10.0F, 20.0F, 30.0F, 60.0F, 0.9F, 0};
  const auto roi = blazepose_app::square_roi(box, 1.5);
  bool ok = expect(roi.x == -10 && roi.y == 10 && roi.width == 60 && roi.height == 60,
                   "ROI is square, centered, scaled, and may extend beyond the frame");

  const auto crop = blazepose_app::crop_plan(roi, 1280, 720);
  ok &= expect(crop.has_value() && crop->image.x == 0 && crop->image.y == 10 &&
                   crop->image.width == 50 && crop->image.height == 60,
               "crop view contains only the in-frame ROI pixels");
  ok &= expect(crop.has_value() && crop->roi.x == -10 && crop->roi.y == 0 &&
                   crop->roi.width == 60 && crop->roi.height == 60,
               "crop-relative ROI preserves out-of-frame padding");
  ok &= expect(!blazepose_app::crop_plan({1280, 10, 20, 20}, 1280, 720).has_value(),
               "ROI without frame overlap is rejected");
  const auto batch =
      blazepose_app::batch_crop_plan({roi, {1280, 10, 20, 20}, {100, 100, 20, 20}}, 1280, 720);
  ok &= expect(batch.has_value() && batch->image.x == 0 && batch->image.y == 10 &&
                   batch->image.width == 120 && batch->image.height == 110,
               "batch crop is the smallest in-frame window containing every valid ROI");
  ok &= expect(batch.has_value() && batch->indices == std::vector<std::size_t>({0, 2}) &&
                   batch->rois[0].x == -10 && batch->rois[0].y == 0 && batch->rois[1].x == 100 &&
                   batch->rois[1].y == 90,
               "batch crop keeps source indices and makes every ROI crop-relative");

  std::vector<float> raw(39 * 5, 0.0F);
  raw[0] = 4.0F;
  raw[1] = 8.0F;
  raw[3] = 2.0F;
  raw[4] = -2.0F;
  const blazepose_app::Affine affine =
      blazepose_app::offset_affine({2.0, 0.0, 10.0, 0.0, 3.0, 10.0}, 0, 10);
  const auto pose = blazepose_app::decode_pose(raw, affine, box, 2);
  ok &= expect(std::abs(pose.keypoints[0].x - 18.0F) < 0.001F &&
                   std::abs(pose.keypoints[0].y - 44.0F) < 0.001F,
               "landmarks are mapped back through ROI affine metadata");
  ok &= expect(std::abs(pose.keypoints[0].confidence - blazepose_app::sigmoid(-2.0F)) < 0.001F,
               "confidence is min(sigmoid(visibility), sigmoid(presence))");
  const auto data = blazepose_app::poses_data_json({pose});
  ok &= expect(data["poses"][0]["keypoints"].size() == 33,
               "Insight metadata contains exactly 33 body keypoints");
  ok &= expect(data["poses"][0]["keypoints"][0]["name"] == "nose",
               "Insight metadata uses BlazePose landmark names");
  ok &= expect(blazepose_app::select_frame_id(9, 8, 7, 6) == 9 &&
                   blazepose_app::select_frame_id(-1, 8, 7, 6) == 8 &&
                   blazepose_app::select_frame_id(-1, -1, 7, 6) == 7 &&
                   blazepose_app::select_frame_id(-1, -1, -1, 6) == 6,
               "frame identity falls back through source sequence fields");
  return ok;
}

bool test_cli(const std::string& binary) {
  bool ok = true;
  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  ok &= expect(help.exit_code == 0 && help.stdout_text.find("--config") != std::string::npos,
               "help documents the config option");
  const auto missing = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  ok &= expect(missing.exit_code == 2 &&
                   missing.stderr_text.find("config file not found") != std::string::npos,
               "missing configuration fails cleanly");
  return ok;
}

bool test_dynamic_stream_config(const std::string& binary) {
  const fs::path config = write_config(
      "dynamic_stream_config",
      "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n"
      "  - id: camera1\n    url: rtsp://127.0.0.1/src1\n    codec: hevc\n    insight_channel: 1\n"
      "  - id: camera2\n    url: rtsp://127.0.0.1/src2\n    codec: h264\n    insight_channel: 2\n"
      "  - id: camera3\n    url: rtsp://127.0.0.1/src3\n    codec: h264\n    insight_channel: 3\n"
      "  - id: camera4\n    url: rtsp://127.0.0.1/src4\n    codec: h264\n    insight_channel: 4\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect(result.exit_code == 0 && result.stdout_text.find("streams=5") != std::string::npos,
             "configuration accepts more than four heterogeneous streams");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_duplicate_stream_identity(const std::string& binary) {
  const fs::path config = write_config(
      "duplicate_stream_identity",
      "  - id: duplicate\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n"
      "  - id: duplicate\n    url: rtsp://127.0.0.1/src1\n    codec: h264\n    insight_channel: "
      "1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect(result.exit_code == 1 &&
                 result.stderr_text.find("stream ids must be unique") != std::string::npos,
             "duplicate stream identities are rejected");
  remove_dir(config.parent_path().string());
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  bool ok = test_math_contract();
  ok &= test_cli(argv[1]);
  ok &= test_dynamic_stream_config(argv[1]);
  ok &= test_duplicate_stream_identity(argv[1]);
  return ok ? 0 : 1;
}
