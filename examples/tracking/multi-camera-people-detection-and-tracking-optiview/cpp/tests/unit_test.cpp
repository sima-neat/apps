#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/config_api.cpp"
#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/image_utils_api.cpp"
#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/pipeline_api.cpp"
#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/sample_utils_api.cpp"
#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/tracker_api.cpp"
#include "support/testing/test_process.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::ProcessResult;
using sima_examples::testing::create_temp_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace multi_camera_people_tracking {
namespace {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

bool expect_near(double lhs, double rhs, double epsilon, const std::string& message) {
  return expect_true(std::fabs(lhs - rhs) <= epsilon, message);
}

bool test_help_runs(const std::string& binary) {
  const ProcessResult result = spawn_and_wait(binary, {"--help"}, 10000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 10000);
  return expect_true(result.exit_code != 0, "missing config exits nonzero") &&
         expect_contains(result.stderr_text, "config", "missing config error mentions config");
}

bool test_load_app_config_parses_dynamic_stream_list() {
  const std::string temp_dir = create_temp_dir("multi_camera_people_tracking_unit_cfg_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory for config parsing test");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model: assets/models/yolo_v8m_mpk.tar.gz\n"
         "input:\n"
         "  tcp: true\n"
         "  latency_ms: 200\n"
         "inference:\n"
         "  frames: 0\n"
         "  fps: 0\n"
         "  bitrate_kbps: 2500\n"
         "  profile: true\n"
         "  person_class_id: 0\n"
         "  detection_threshold: null\n"
         "  nms_iou_threshold: null\n"
         "  top_k: null\n"
         "tracking:\n"
         "  iou_threshold: 0.3\n"
         "  max_missing_frames: 15\n"
         "output:\n"
         "  optiview:\n"
         "    host: 192.168.0.107\n"
         "    video_port_base: 9000\n"
         "    json_port_base: 9100\n"
         "  debug_dir: null\n"
         "  save_every: 0\n"
         "streams:\n"
         "  - rtsp://192.168.0.235:8554/src1\n"
         "  - rtsp://192.168.0.235:8554/src2\n"
         "  - rtsp://192.168.0.235:8554/src3\n";
  out.close();

  bool ok = true;
  try {
    const AppConfig cfg = load_app_config(config_path);
    ok &= expect_true(cfg.model == "assets/models/yolo_v8m_mpk.tar.gz",
                      "config parser keeps model path");
    ok &= expect_true(cfg.optiview_host == "192.168.0.107",
                      "config parser keeps OptiView host");
    ok &= expect_true(cfg.optiview_video_port_base == 9000,
                      "config parser keeps video port base");
    ok &= expect_true(cfg.optiview_json_port_base == 9100,
                      "config parser keeps json port base");
    ok &= expect_true(cfg.tcp, "config parser keeps tcp=true");
    ok &= expect_true(cfg.save_every == 0, "config parser keeps save_every");
    ok &= expect_true(
        cfg.rtsp_urls ==
            std::vector<std::string>{
                "rtsp://192.168.0.235:8554/src1",
                "rtsp://192.168.0.235:8554/src2",
                "rtsp://192.168.0.235:8554/src3",
            },
        "config parser keeps all RTSP URLs");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config parser should load valid config: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_missing_streams() {
  const std::string temp_dir = create_temp_dir("multi_camera_people_tracking_unit_badcfg_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory for missing streams test");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model: assets/models/yolo_v8m_mpk.tar.gz\n"
         "input: {}\n"
         "inference: {}\n"
         "tracking: {}\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n"
         "    video_port_base: 9000\n"
         "    json_port_base: 9100\n"
         "  debug_dir: null\n"
         "  save_every: 0\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "streams", "missing streams error mentions streams");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_common_config_yaml_uses_supported_shape() {
  const fs::path config_path =
      "examples/tracking/multi-camera-people-detection-and-tracking-optiview/common/config.yaml";
  try {
    const AppConfig cfg = load_app_config(config_path);
    bool ok = true;
    ok &= expect_true(!cfg.model.empty(), "common config contains model");
    ok &= expect_true(cfg.optiview_video_port_base > 0, "common config video port base is positive");
    ok &= expect_true(cfg.optiview_json_port_base > 0, "common config json port base is positive");
    ok &= expect_true(cfg.save_every >= 0, "common config save_every is nonnegative");
    ok &= expect_true(!cfg.rtsp_urls.empty(), "common config has streams");
    return ok;
  } catch (const std::exception& ex) {
    return expect_true(false, std::string("common config should load: ") + ex.what());
  }
}

bool test_tracker_reuses_track_id_for_nearby_detection() {
  PeopleTracker tracker(0.3f, 2);
  const std::vector<Detection> first_input = {
      Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0},
  };
  const std::vector<Detection> second_input = {
      Detection{12.0f, 12.0f, 52.0f, 82.0f, 0.88f, 0},
  };

  const auto first = tracker.update(first_input, 0);
  const auto second = tracker.update(second_input, 1);

  return expect_true(first.size() == 1, "tracker returns one detection on first frame") &&
         expect_true(second.size() == 1, "tracker returns one detection on second frame") &&
         expect_true(second.front().track_id == first.front().track_id,
                     "tracker reuses track id for nearby detection");
}

bool test_tracker_drops_track_after_missing_budget() {
  PeopleTracker tracker(0.3f, 1);
  tracker.update({Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0}}, 0);
  tracker.update({}, 1);
  tracker.update({}, 2);
  return expect_true(tracker.active_track_count() == 0,
                     "tracker expires track after missing frame budget");
}

bool test_make_optiview_tracking_detection_uses_track_id_label_text() {
  const auto payload = make_optiview_tracking_detection({
      TrackedDetection{42, 1.0f, 2.0f, 11.0f, 22.0f, 0.9f, 0},
  });

  bool ok = true;
  ok &= expect_true(payload.objects.size() == 1, "OptiView payload returns one object");
  ok &= expect_true(payload.objects.front().x == 1, "OptiView payload keeps x");
  ok &= expect_true(payload.objects.front().y == 2, "OptiView payload keeps y");
  ok &= expect_true(payload.objects.front().w == 10, "OptiView payload keeps width");
  ok &= expect_true(payload.objects.front().h == 20, "OptiView payload keeps height");
  ok &= expect_near(payload.objects.front().score, 0.9, 1e-6, "OptiView payload keeps score");
  ok &= expect_true(payload.objects.front().class_id == 0, "OptiView payload uses detection index");
  ok &= expect_true(payload.labels == std::vector<std::string>{"Track ID: 42"},
                    "OptiView payload uses track-id label text");
  return ok;
}

bool test_tensor_mode_session_accepts_fp32_cv_mat_through_session_api() {
  simaai::neat::InputOptions input_options;
  input_options.media_type = "application/vnd.simaai.tensor";
  input_options.format = "FP32";
  input_options.max_width = 4;
  input_options.max_height = 4;
  input_options.max_depth = 3;

  simaai::neat::Session session;
  session.add(simaai::neat::nodes::Input(input_options));
  session.add(simaai::neat::nodes::Output());

  cv::Mat tensor_input = cv::Mat::zeros(4, 4, CV_32FC3);

  auto run = build_tensor_input_run(session, tensor_input);
  const auto output = run_tensor_input_once(run, tensor_input, 1000);

  bool ok = true;
  ok &= expect_true(output.kind == simaai::neat::SampleKind::Tensor,
                    "tensor-mode session returns tensor output for FP32 cv::Mat input");
  ok &= expect_true(output.tensor.has_value(),
                    "tensor-mode session returns tensor payload for FP32 cv::Mat input");
  return ok;
}

bool test_save_overlay_frame_converts_rgb_to_bgr_for_jpeg() {
  const std::string temp_dir = create_temp_dir("multi_camera_people_tracking_unit_overlay_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory for overlay save test");
  }

  const cv::Mat frame_rgb(32, 32, CV_8UC3, cv::Scalar(255, 0, 0));
  bool ok = save_overlay_frame(fs::path(temp_dir), 0, 0, frame_rgb, 1);
  ok &= expect_true(ok, "overlay save writes sampled frame");

  const fs::path out_path = sample_output_path(fs::path(temp_dir), 0, 0);
  const cv::Mat saved = cv::imread(out_path.string(), cv::IMREAD_COLOR);
  ok &= expect_true(!saved.empty(), "overlay save writes readable image");
  if (!saved.empty()) {
    const cv::Vec3b pixel = saved.at<cv::Vec3b>(0, 0);
    ok &= expect_true(pixel[2] > pixel[0],
                      "overlay save preserves red-dominant RGB content after reload");
  }

  remove_dir(temp_dir);
  return ok;
}

} // namespace
} // namespace multi_camera_people_tracking

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  int failures = 0;

  failures += !multi_camera_people_tracking::test_help_runs(binary);
  failures += !multi_camera_people_tracking::test_missing_config_file_fails_cleanly(binary);
  failures += !multi_camera_people_tracking::test_load_app_config_parses_dynamic_stream_list();
  failures += !multi_camera_people_tracking::test_load_app_config_rejects_missing_streams();
  failures += !multi_camera_people_tracking::test_common_config_yaml_uses_supported_shape();
  failures += !multi_camera_people_tracking::test_tracker_reuses_track_id_for_nearby_detection();
  failures += !multi_camera_people_tracking::test_tracker_drops_track_after_missing_budget();
  failures +=
      !multi_camera_people_tracking::test_make_optiview_tracking_detection_uses_track_id_label_text();
  failures +=
      !multi_camera_people_tracking::test_tensor_mode_session_accepts_fp32_cv_mat_through_session_api();
  failures += !multi_camera_people_tracking::test_save_overlay_frame_converts_rgb_to_bgr_for_jpeg();

  return failures == 0 ? 0 : 1;
}
