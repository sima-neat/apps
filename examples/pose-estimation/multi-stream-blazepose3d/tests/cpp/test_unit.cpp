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

bool test_math_contract() {
  bool ok = true;
  const blazepose_app::Box box{10.0F, 20.0F, 30.0F, 60.0F, 0.9F, 0};
  const auto roi = blazepose_app::square_roi(box, 1.5);
  ok &= expect(roi.x == -10 && roi.y == 10 && roi.width == 60 && roi.height == 60,
               "ROI is square, centered, scaled, and may extend beyond the frame");

  std::vector<float> raw(39 * 5, 0.0F);
  raw[0] = 4.0F;
  raw[1] = 8.0F;
  raw[3] = 2.0F;
  raw[4] = -2.0F;
  std::vector<float> raw_world(39 * 3, 0.0F);
  raw_world[0] = 0.1F;
  raw_world[1] = -0.2F;
  raw_world[2] = 0.3F;
  const blazepose_app::Affine affine{2.0, 0.0, 10.0, 0.0, 3.0, 20.0};
  const auto pose = blazepose_app::decode_pose(raw, raw_world, affine, box, 0.93F);
  ok &= expect(std::abs(pose.presence - 0.93F) < 0.001F,
               "global pose presence is retained for publication");
  ok &= expect(std::abs(blazepose_app::sigmoid(0.0F) - 0.5F) < 0.001F &&
                   blazepose_app::sigmoid(-0.01F) < 0.5F,
               "global pose presence logits activate before probability thresholding");
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
  ok &=
      expect(data.at("stream_id") == "camera0", "2D metadata preserves the source stream identity");
  ok &= expect(data["poses"][0]["keypoints"].size() == 33,
               "Insight metadata contains exactly 33 body keypoints");
  ok &= expect(std::abs(data["poses"][0]["presence"].get<float>() - 0.93F) < 0.001F,
               "2D metadata publishes global pose presence");
  ok &= expect(data["poses"][0]["keypoints"][0]["name"] == "nose",
               "Insight metadata uses BlazePose landmark names");
  ok &=
      expect(data["poses"][0]["world_keypoints"].size() == 33 &&
                 data["poses"][0]["world_keypoints"][0]["name"] == "nose" &&
                 std::abs(data["poses"][0]["world_keypoints"][0]["z"].get<float>() - 0.3F) < 0.001F,
             "pose-estimation metadata includes named world keypoints");
  const auto auxiliary = blazepose_app::world_pose_auxiliary_data_json(data);
  ok &= expect(auxiliary["payload"]["poses"][0]["keypoints"] == data["poses"][0]["world_keypoints"],
               "both messages reuse identical world coordinates and confidence");
  ok &= expect(auxiliary.at("stream_id") == "camera0",
               "auxiliary metadata preserves the source stream identity");
  ok &= expect(auxiliary["schema_version"] == 1 && auxiliary["id"] == "world-pose" &&
                   auxiliary["renderer"] == "blazepose-3d",
               "world landmarks use the generic auxiliary visualization envelope");
  ok &= expect(auxiliary["payload"]["poses"][0]["keypoints"].size() == 33 &&
                   auxiliary["payload"]["poses"][0]["keypoints"][0]["name"] == "nose",
               "auxiliary payload contains 33 named world keypoints");
  ok &= expect(std::abs(auxiliary["payload"]["poses"][0]["presence"].get<float>() - 0.93F) < 0.001F,
               "3D metadata publishes the same global pose presence");
  blazepose_app::Pose first_pose;
  first_pose.presence = 0.9F;
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
  ok &= expect(std::abs(image_fraction - 0.45F) < 0.001F &&
                   std::abs(image_fraction - world_fraction) < 0.001F,
               "temporal filter applies one weight to 2D and world landmarks");
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

bool test_config(const std::string& binary) {
  const fs::path dir = create_test_scratch_dir("multi-stream-blazepose3d", "config");
  const auto path = dir / "config.yaml";
  bool ok = true;
  for (const auto& [channel, expected] : std::vector<std::pair<std::string, int>>{
           {"0", 0}, {"'01'", 0}, {"-1", 1}, {"65536", 1}, {"1:20", 1}, {"1.5", 1}}) {
    std::ofstream(path) << "models: {detector_path: 'detector#v2', pose_path: pose}\n"
                           "streams:\n  - url: rtsp://host/cam#view\n    codec: h265\n"
                           "    id: camera\n    insight_channel: "
                        << channel
                        << "\n"
                           "output: {insight: {host: localhost}}\n";
    const auto result =
        spawn_and_wait(binary, {"--config", path.string(), "--validate-config-only"}, 10000);
    ok &= expect(result.exit_code == expected, "field-based YAML channel " + channel);
  }
  remove_dir(dir.string());
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  bool ok = test_math_contract();
  ok &= test_config(argv[1]);
  return ok ? 0 : 1;
}
