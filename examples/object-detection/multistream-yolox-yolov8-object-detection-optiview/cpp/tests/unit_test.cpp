#include "../utils/config_api.cpp"
#include "../utils/model_family_api.cpp"
#include "../utils/pipeline_api.cpp"
#include "../utils/sample_utils_api.cpp"
#include "../utils/workers_api.cpp"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_temp_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace multistream_yolox_yolov8_optiview {
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

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 10000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 10000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_smoke_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--validate-config-only"}, 10000);
  return expect_true(result.exit_code == 0, "validate-config-only exits with code 0") &&
         expect_contains(result.stdout_text, "Config validated",
                         "validate-config-only reports validated config");
}

bool test_load_app_config_parses_runtime_worker_count() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_cfg_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: yolov8\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "  - rtsp://127.0.0.1:8554/src2\n"
         "input:\n"
         "  tcp: true\n"
         "  latency_ms: 120\n"
         "runtime:\n"
         "  worker_count: 4\n"
         "  mailbox_depth: 1\n"
         "  profile: true\n"
         "inference:\n"
         "  frames: 0\n"
         "  fps: 0\n"
         "  min_score: 0.25\n"
         "  nms_iou: 0.45\n"
         "  max_detections: 100\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n"
         "    video_port_base: 9000\n"
         "    json_port_base: 9100\n"
         "    json_offset_ms: 12.5\n"
         "  video_enabled: false\n"
         "  video_mode: clean\n"
         "  debug_dir: null\n"
         "  save_every: 0\n";
  out.close();

  bool ok = true;
  try {
    const AppConfig cfg = load_app_config(config_path);
    ok &= expect_true(cfg.model.path == "assets/models/yolo_v8m_mpk.tar.gz",
                      "config keeps model path");
    ok &= expect_true(cfg.model.family == ModelFamily::YoloV8, "config keeps model family");
    ok &= expect_true(cfg.worker_count == 4, "config keeps worker_count");
    ok &= expect_true(cfg.mailbox_depth == 1, "config keeps mailbox_depth");
    ok &= expect_true(cfg.profile, "config keeps profile=true");
    ok &= expect_true(cfg.optiview_json_offset_ms == 12.5, "config keeps json_offset_ms");
    ok &= expect_true(!cfg.video_enabled, "config keeps video_enabled=false");
    ok &= expect_true(cfg.video_mode == VideoMode::Clean, "config keeps clean video_mode");
    ok &= expect_true(cfg.rtsp_urls.size() == 2, "config keeps all streams");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config should load: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_yolox_family_until_supported() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_yolox_unsupported_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolox_m_mpk.tar.gz\n"
         "  family: yolox\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "YOLOX model packs are not supported yet",
                         "unsupported yolox family error mentions future support state");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_invalid_worker_count() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_bad_worker_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: auto\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 0\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "worker_count", "invalid worker_count error mentions worker_count");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_invalid_video_mode() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_bad_mode_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: auto\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n"
         "  video_mode: purple\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "video_mode", "invalid video_mode error mentions video_mode");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_empty_streams() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_no_streams_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: auto\n"
         "streams: []\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "streams", "empty streams error mentions streams");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_json_output_enabled_is_disabled_for_annotated_video() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_annotated_json_off_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: yolov8\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n"
         "  video_enabled: false\n"
         "  video_mode: annotated\n";
  out.close();

  bool ok = true;
  try {
    const AppConfig cfg = load_app_config(config_path);
    ok &= expect_true(!json_output_enabled(cfg),
                      "annotated video mode suppresses json output");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config should load: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_json_output_enabled_stays_enabled_for_clean_video() {
  const std::string temp_dir = create_temp_dir("multistream_yolox_yolov8_clean_json_on_");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "  family: yolov8\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  optiview:\n"
         "    host: 127.0.0.1\n"
         "  video_enabled: false\n"
         "  video_mode: clean\n";
  out.close();

  bool ok = true;
  try {
    const AppConfig cfg = load_app_config(config_path);
    ok &= expect_true(json_output_enabled(cfg), "clean video mode keeps json output enabled");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config should load: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_parse_model_family_rejects_yolox_until_supported() {
  bool ok = false;
  try {
    static_cast<void>(parse_model_family("yolox"));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "YOLOX model packs are not supported yet",
                         "parse_model_family rejects yolox until support lands");
  }
  return ok;
}

bool test_resolve_model_family_auto_rejects_yolox_until_supported() {
  bool ok = false;
  try {
    static_cast<void>(resolve_model_family("assets/models/yolox_s_mpk.tar.gz", ModelFamily::Auto));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "YOLOX model packs are not supported yet",
                         "auto resolve rejects yolox model path until support lands");
  }
  return ok;
}

bool test_resolve_model_family_auto_for_yolov8() {
  return expect_true(
      resolve_model_family("assets/models/yolo_v8m_mpk.tar.gz", ModelFamily::Auto) ==
          ModelFamily::YoloV8,
      "auto resolves yolo_v8 model path to YoloV8");
}

bool test_parse_bbox_payload_normalizes_yolov8_boxes() {
  const std::vector<std::uint8_t> payload = {
      1, 0, 0, 0,
      246, 255, 255, 255,
      20, 0, 0, 0,
      140, 0, 0, 0,
      50, 0, 0, 0,
      102, 102, 102, 63,
      3, 0, 0, 0,
  };

  const auto boxes = parse_bbox_payload(payload, 100, 80);

  bool ok = true;
  ok &= expect_true(boxes.size() == 1, "bbox payload yields one normalized box");
  if (!boxes.empty()) {
    ok &= expect_true(boxes.front().x1 == 0.0f, "bbox payload clamps x1 to frame");
    ok &= expect_true(boxes.front().y1 == 20.0f, "bbox payload keeps y1");
    ok &= expect_true(boxes.front().x2 == 100.0f, "bbox payload clamps x2 to frame");
    ok &= expect_true(boxes.front().y2 == 70.0f, "bbox payload keeps y2");
    ok &= expect_true(boxes.front().class_id == 3, "bbox payload keeps class_id");
  }
  return ok;
}

bool test_require_detector_output_kind_rejects_yolox_until_supported() {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Bundle;
  sample.payload_tag = "BBOX";
  sample.format = "BBOX";

  bool ok = false;
  try {
    static_cast<void>(require_detector_output_kind(ModelFamily::YoloX, sample));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "YOLOX model packs are not supported yet",
                         "yolox output kind rejects unsupported family until support lands");
  }
  return ok;
}

bool test_require_detector_output_kind_rejects_unsupported_sample_kind() {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Unknown;

  bool ok = false;
  try {
    static_cast<void>(require_detector_output_kind(ModelFamily::YoloV8, sample));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "unsupported detector output",
                         "unsupported detector sample error mentions unsupported detector output");
  }
  return ok;
}

bool test_build_optiview_detection_payload_builds_objects_and_labels() {
  std::vector<Detection> boxes{
      Detection{-5.0f, 10.0f, 35.0f, 50.0f, 0.8f, 2},
  };

  const auto payload =
      build_optiview_detection_payload(boxes, 30, 40, {"person", "bicycle", "car"});

  bool ok = true;
  ok &= expect_true(payload.objects.size() == 1, "detection payload yields one OptiView object");
  ok &= expect_true(payload.labels == std::vector<std::string>({"person", "bicycle", "car"}),
                    "detection payload keeps the full OptiView label table");
  if (!payload.objects.empty()) {
    const auto& object = payload.objects.front();
    ok &= expect_true(object.x == 0, "detection payload clamps x");
    ok &= expect_true(object.y == 10, "detection payload keeps y");
    ok &= expect_true(object.w == 30, "detection payload clamps width");
    ok &= expect_true(object.h == 30, "detection payload clamps height");
    ok &= expect_true(object.class_id == 2, "detection payload keeps class_id");
  }
  return ok;
}

bool test_optiview_frame_id_prefers_detector_sample_frame_id() {
  simaai::neat::Sample sample;
  sample.frame_id = 42;
  return expect_true(optiview_frame_id(sample, 7) == "42",
                     "OptiView frame id prefers detector sample frame_id");
}

bool test_optiview_frame_id_falls_back_to_packet_index() {
  simaai::neat::Sample sample;
  sample.frame_id = -1;
  return expect_true(optiview_frame_id(sample, 7) == "7",
                     "OptiView frame id falls back to packet index");
}

bool test_optiview_timestamp_ms_applies_publish_offset() {
  return expect_true(optiview_timestamp_ms(1.234, 25.0) == 1259,
                     "OptiView timestamp adds json offset to publish time");
}

bool test_latest_frame_mailbox_deduplicates_ready_notifications_and_requeues_after_completion() {
  ReadyStreamQueue ready_queue;
  LatestFrameMailbox<std::string> mailbox(7, 1);

  bool ok = true;
  ok &= expect_true(mailbox.push("frame-0", ready_queue) == 0, "first mailbox push drops nothing");
  ok &= expect_true(mailbox.push("frame-1", ready_queue) == 1, "mailbox keeps latest frame");

  int ready_stream = -1;
  ok &= expect_true(ready_queue.pop_wait(ready_stream, 0), "ready queue receives stream once");
  ok &= expect_true(ready_stream == 7, "ready queue reports the mailbox stream index");
  ok &= expect_true(!ready_queue.pop_wait(ready_stream, 0),
                    "mailbox does not enqueue duplicate ready notifications");

  std::string frame;
  ok &= expect_true(mailbox.take_for_processing(frame), "mailbox yields latest frame");
  ok &= expect_true(frame == "frame-1", "mailbox returns the newest frame");
  ok &= expect_true(mailbox.push("frame-2", ready_queue) == 0,
                    "mailbox accepts a new frame while in flight");
  ok &= expect_true(!ready_queue.pop_wait(ready_stream, 0),
                    "mailbox does not requeue while work is in flight");

  mailbox.complete(ready_queue);
  ok &= expect_true(ready_queue.pop_wait(ready_stream, 0), "mailbox requeues after completion");
  ok &= expect_true(ready_stream == 7, "mailbox requeues the same stream index");
  return ok;
}

bool test_collect_detector_runtime_keys_deduplicates_same_geometry() {
  StreamProbeSpec a{ModelFamily::YoloV8, RtspProbe{640, 480, 30}};
  StreamProbeSpec b{ModelFamily::YoloV8, RtspProbe{640, 480, 25}};
  StreamProbeSpec c{ModelFamily::YoloV8, RtspProbe{1280, 720, 30}};

  const auto keys = collect_detector_runtime_keys({a, b, c});

  bool ok = true;
  ok &= expect_true(keys.size() == 2, "detector runtime keys deduplicate matching geometry");
  if (keys.size() == 2) {
    ok &= expect_true(keys[0].family == ModelFamily::YoloV8 && keys[0].width == 640 &&
                          keys[0].height == 480,
                      "first detector runtime key keeps initial geometry");
    ok &= expect_true(keys[1].family == ModelFamily::YoloV8 && keys[1].width == 1280 &&
                          keys[1].height == 720,
                      "second detector runtime key keeps distinct geometry");
  }
  return ok;
}

bool test_detector_stage_names_cover_yolov8_and_reject_yolox() {
  const auto yolov8 = detector_stage_names(ModelFamily::YoloV8);

  bool ok = true;
  ok &= expect_true(
      yolov8 == std::vector<std::string>{"input", "quant_tess", "mla", "sima_box_decode", "output"},
      "yolov8 stage names match fixed detector graph");
  try {
    static_cast<void>(detector_stage_names(ModelFamily::YoloX));
    ok &= expect_true(false, "yolox stage names should be rejected until support lands");
  } catch (const std::exception& ex) {
    ok &= expect_contains(ex.what(), "YOLOX model packs are not supported yet",
                          "yolox stage names reject unsupported family until support lands");
  }
  return ok;
}

bool test_source_output_every_n_only_decimates_when_target_is_meaningfully_lower() {
  AppConfig cfg;
  RtspProbe probe{640, 480, 30};

  bool ok = true;
  cfg.fps = 0;
  ok &= expect_true(source_output_every_n(cfg, probe) == 1, "source every_n stays 1 when fps is uncapped");
  cfg.fps = 20;
  ok &= expect_true(source_output_every_n(cfg, probe) == 1,
                    "source every_n stays 1 when target fps is near source fps");
  cfg.fps = 12;
  ok &= expect_true(source_output_every_n(cfg, probe) == 2,
                    "source every_n becomes 2 for 30fps source and 12fps target");
  cfg.fps = 10;
  ok &= expect_true(source_output_every_n(cfg, probe) == 3,
                    "source every_n becomes 3 for 30fps source and 10fps target");
  return ok;
}

bool test_producer_emit_period_s_always_paces_when_fps_configured() {
  AppConfig cfg;
  RtspProbe probe{640, 480, 30};

  bool ok = true;
  cfg.fps = 10;
  ok &= expect_true(producer_emit_period_s(cfg, probe) > 0.0,
                    "producer pacing active when fps=10 and source is 30fps");
  cfg.fps = 20;
  ok &= expect_true(producer_emit_period_s(cfg, probe) > 0.0,
                    "producer pacing active when fps=20 and source is 30fps");
  probe.fps = 0;
  cfg.fps = 10;
  ok &= expect_true(producer_emit_period_s(cfg, probe) > 0.0,
                    "producer pacing active when source fps is unknown");
  cfg.fps = 0;
  ok &= expect_true(producer_emit_period_s(cfg, probe) == 0.0,
                    "producer pacing disabled when fps is uncapped");
  return ok;
}

} // namespace
} // namespace multistream_yolox_yolov8_optiview

int main(int argc, char** argv) {
  using namespace multistream_yolox_yolov8_optiview;

  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_smoke_runs(binary);
  ok &= test_load_app_config_parses_runtime_worker_count();
  ok &= test_load_app_config_rejects_yolox_family_until_supported();
  ok &= test_load_app_config_rejects_invalid_worker_count();
  ok &= test_load_app_config_rejects_invalid_video_mode();
  ok &= test_load_app_config_rejects_empty_streams();
  ok &= test_json_output_enabled_is_disabled_for_annotated_video();
  ok &= test_json_output_enabled_stays_enabled_for_clean_video();
  ok &= test_parse_model_family_rejects_yolox_until_supported();
  ok &= test_resolve_model_family_auto_rejects_yolox_until_supported();
  ok &= test_resolve_model_family_auto_for_yolov8();
  ok &= test_parse_bbox_payload_normalizes_yolov8_boxes();
  ok &= test_require_detector_output_kind_rejects_yolox_until_supported();
  ok &= test_require_detector_output_kind_rejects_unsupported_sample_kind();
  ok &= test_build_optiview_detection_payload_builds_objects_and_labels();
  ok &= test_optiview_frame_id_prefers_detector_sample_frame_id();
  ok &= test_optiview_frame_id_falls_back_to_packet_index();
  ok &= test_optiview_timestamp_ms_applies_publish_offset();
  ok &= test_latest_frame_mailbox_deduplicates_ready_notifications_and_requeues_after_completion();
  ok &= test_collect_detector_runtime_keys_deduplicates_same_geometry();
  ok &= test_detector_stage_names_cover_yolov8_and_reject_yolox();
  ok &= test_source_output_every_n_only_decimates_when_target_is_meaningfully_lower();
  ok &= test_producer_emit_period_s_always_paces_when_fps_configured();
  return ok ? 0 : 1;
}
