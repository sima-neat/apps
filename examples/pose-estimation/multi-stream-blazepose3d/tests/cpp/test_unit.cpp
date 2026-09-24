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
  std::vector<float> raw_world(39 * 3, 0.0F);
  raw_world[0] = 0.1F;
  raw_world[1] = -0.2F;
  raw_world[2] = 0.3F;
  const blazepose_app::Affine affine =
      blazepose_app::offset_affine({2.0, 0.0, 10.0, 0.0, 3.0, 10.0}, 0, 10);
  const auto pose = blazepose_app::decode_pose(raw, raw_world, affine, box, 2);
  ok &= expect(std::abs(pose.keypoints[0].x - 18.0F) < 0.001F &&
                   std::abs(pose.keypoints[0].y - 44.0F) < 0.001F,
               "landmarks are mapped back through ROI affine metadata");
  ok &= expect(std::abs(pose.keypoints[0].confidence - blazepose_app::sigmoid(-2.0F)) < 0.001F,
               "confidence is min(sigmoid(visibility), sigmoid(presence))");
  ok &= expect(std::abs(pose.world_keypoints[0].x - 0.1F) < 0.001F &&
                   std::abs(pose.world_keypoints[0].y + 0.2F) < 0.001F &&
                   std::abs(pose.world_keypoints[0].z - 0.3F) < 0.001F,
               "world landmarks retain the model's 3D coordinates");
  const auto data = blazepose_app::poses_data_json({pose}, "camera0");
  ok &= expect(data.at("stream_id") == "camera0",
               "2D metadata preserves the source stream identity");
  ok &= expect(data["poses"][0]["keypoints"].size() == 33,
               "Insight metadata contains exactly 33 body keypoints");
  ok &= expect(data["poses"][0]["keypoints"][0]["name"] == "nose",
               "Insight metadata uses BlazePose landmark names");
  ok &= expect(!data["poses"][0].contains("world_keypoints"),
               "normal pose-estimation metadata remains a 2D overlay contract");
  const auto auxiliary =
      blazepose_app::world_pose_auxiliary_data_json({pose}, "camera0");
  ok &= expect(auxiliary.at("stream_id") == "camera0",
               "auxiliary metadata preserves the source stream identity");
  ok &= expect(auxiliary["schema_version"] == 1 && auxiliary["id"] == "world-pose" &&
                   auxiliary["renderer"] == "blazepose-3d",
               "world landmarks use the generic auxiliary visualization envelope");
  ok &= expect(auxiliary["payload"]["poses"][0]["keypoints"].size() == 33 &&
                   auxiliary["payload"]["poses"][0]["keypoints"][0]["name"] == "nose",
               "auxiliary payload contains 33 named world keypoints");
  const nlohmann::json point_cloud = {
      {"points", {{{"x", 0.1}, {"y", 0.2}, {"z", 0.3}, {"value", 7}}}},
      {"axes", {"east", "north", "up"}},
  };
  const auto generic = blazepose_app::auxiliary_visualization_data_json(
      "depth-cloud", "point-cloud-3d", point_cloud);
  ok &= expect(generic["schema_version"] == 1 && generic["id"] == "depth-cloud" &&
                   generic["renderer"] == "point-cloud-3d" && generic["payload"] == point_cloud &&
                   !generic.contains("title"),
               "generic auxiliary envelope preserves an arbitrary renderer payload");
  ok &= expect(blazepose_app::select_frame_id(9, 8, 7, 6) == 9 &&
                   blazepose_app::select_frame_id(-1, 8, 7, 6) == 8 &&
                   blazepose_app::select_frame_id(-1, -1, 7, 6) == 7 &&
                   blazepose_app::select_frame_id(-1, -1, -1, 6) == 6,
               "frame identity falls back through source sequence fields");

  blazepose_app::Pose first_pose;
  first_pose.box = {0.0F, 0.0F, 100.0F, 100.0F, 0.9F, 0};
  first_pose.keypoints[0] = {50.0F, 50.0F, 0.2F};
  first_pose.world_keypoints[0] = {0.0F, 0.0F, 0.0F, 0.2F};
  blazepose_app::PoseSmoother smoother;
  smoother.filter({first_pose}, 1'000'000'000);
  blazepose_app::Pose second_pose = first_pose;
  second_pose.keypoints[0].x = 54.0F;
  second_pose.keypoints[0].confidence = 0.4F;
  second_pose.world_keypoints[0].x = 0.04F;
  second_pose.world_keypoints[0].confidence = 0.4F;
  const auto smoothed = smoother.filter({second_pose}, 1'040'000'000).front();
  const float image_fraction = (smoothed.keypoints[0].x - 50.0F) / 4.0F;
  const float world_fraction = smoothed.world_keypoints[0].x / 0.04F;
  ok &= expect(image_fraction > 0.45F && image_fraction < 0.90F &&
                   std::abs(image_fraction - world_fraction) < 0.001F,
               "temporal filter applies one adaptive weight to 2D and world landmarks");
  ok &= expect(std::abs(smoothed.keypoints[0].confidence - 0.24F) < 0.001F &&
                   std::abs(smoothed.world_keypoints[0].confidence - 0.24F) < 0.001F,
               "temporal filter damps confidence crossings consistently");

  blazepose_app::Pose fast_pose = second_pose;
  fast_pose.keypoints[0].x = 154.0F;
  const auto fast = smoother.filter({fast_pose}, 1'080'000'000).front();
  ok &= expect(fast.keypoints[0].x > 140.0F,
               "temporal filter follows deliberate fast motion without buffering frames");
  blazepose_app::Pose reset_pose = first_pose;
  reset_pose.keypoints[0].x = 30.0F;
  const auto reset = smoother.filter({reset_pose}, 1'400'000'000).front();
  ok &= expect(std::abs(reset.keypoints[0].x - 30.0F) < 0.001F,
               "temporal filter resets after a discontinuity instead of dragging stale state");

  blazepose_app::PoseSmoother continuity;
  continuity.filter({first_pose}, 2'000'000'000);
  const auto first_gap = continuity.filter({}, 2'040'000'000);
  const auto second_gap = continuity.filter({}, 2'080'000'000);
  const auto expired = continuity.filter({}, 2'120'000'000);
  const bool continuity_sizes = first_gap.size() == 1 && second_gap.size() == 1 && expired.empty();
  ok &= expect(continuity_sizes,
               "temporal filter bridges two missing results without buffering later frames");
  if (continuity_sizes) {
    ok &= expect(first_gap[0].keypoints[0].confidence < first_pose.keypoints[0].confidence &&
                     second_gap[0].keypoints[0].confidence < first_gap[0].keypoints[0].confidence,
                 "coasted pose confidence decays while the estimate is unavailable");
  }
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

bool test_stream_limit(const std::string& binary) {
  const fs::path config = write_config(
      "dynamic_stream_config",
      "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n"
      "  - id: camera1\n    url: rtsp://127.0.0.1/src1\n    codec: hevc\n    insight_channel: 1\n"
      "  - id: camera2\n    url: rtsp://127.0.0.1/src2\n    codec: h264\n    insight_channel: 2\n"
      "  - id: camera3\n    url: rtsp://127.0.0.1/src3\n    codec: h264\n    insight_channel: 3\n"
      "  - id: camera4\n    url: rtsp://127.0.0.1/src4\n    codec: h264\n    insight_channel: 4\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok = expect(
      result.exit_code == 1 &&
          result.stderr_text.find("streams must contain between 1 and 4 entries") !=
              std::string::npos,
      "configuration rejects more than four streams");
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
  ok &= test_stream_limit(argv[1]);
  ok &= test_duplicate_stream_identity(argv[1]);
  return ok ? 0 : 1;
}
