#include "../utils/config_api.cpp"
#include "../utils/model_family_api.cpp"
#include "../utils/pipeline_api.cpp"
#include "../utils/sample_utils_api.cpp"
#include "../utils/workers_api.cpp"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_output_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace multistream_object_detection_insight {
namespace {

std::string env_or_empty(const char* key) {
  const char* value = std::getenv(key);
  return value == nullptr ? std::string{} : std::string{value};
}

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
  const std::string temp_dir = create_test_output_dir("multistream-object-detection-insight",
                                                      "test_validate_config_only_smoke_runs");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
  out.close();

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 10000);
  const bool ok = expect_true(result.exit_code == 0, "validate-config-only exits with code 0") &&
                  expect_contains(result.stdout_text, "Config validated",
                                  "validate-config-only reports validated config");
  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_parses_runtime_worker_count() {
  const std::string temp_dir = create_test_output_dir(
      "multistream-object-detection-insight", "test_load_app_config_parses_runtime_worker_count");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
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
         "  insight:\n"
         "    host: 127.0.0.1\n"
         "    video_port_base: 9000\n"
         "    metadata_port_base: 9100\n"
         "    metadata_offset_ms: 12.5\n"
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
    ok &= expect_true(cfg.worker_count == 4, "config keeps worker_count");
    ok &= expect_true(cfg.mailbox_depth == 1, "config keeps mailbox_depth");
    ok &= expect_true(cfg.profile, "config keeps profile=true");
    ok &= expect_true(cfg.insight_metadata_offset_ms == 12.5, "config keeps metadata_offset_ms");
    ok &= expect_true(!cfg.video_enabled, "config keeps video_enabled=false");
    ok &= expect_true(cfg.video_mode == VideoMode::Clean, "config keeps clean video_mode");
    ok &= expect_true(cfg.rtsp_urls.size() == 2, "config keeps all streams");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config should load: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_removed_model_family_field() {
  const std::string temp_dir =
      create_test_output_dir("multistream-object-detection-insight",
                             "test_load_app_config_rejects_removed_model_family_field");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/unsupported_mpk.tar.gz\n"
         "  family: yolov8\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "model.family is no longer supported",
                         "stale model family field error mentions removal");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_invalid_worker_count() {
  const std::string temp_dir = create_test_output_dir(
      "multistream-object-detection-insight", "test_load_app_config_rejects_invalid_worker_count");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 0\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
  out.close();

  bool ok = false;
  try {
    static_cast<void>(load_app_config(config_path));
  } catch (const std::exception& ex) {
    ok = expect_contains(ex.what(), "worker_count",
                         "invalid worker_count error mentions worker_count");
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_load_app_config_rejects_invalid_video_mode() {
  const std::string temp_dir = create_test_output_dir(
      "multistream-object-detection-insight", "test_load_app_config_rejects_invalid_video_mode");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
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
  const std::string temp_dir = create_test_output_dir("multistream-object-detection-insight",
                                                      "test_load_app_config_rejects_empty_streams");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams: []\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
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

bool test_metadata_output_enabled_follows_video_mode_contract() {
  const std::string temp_dir =
      create_test_output_dir("multistream-object-detection-insight",
                             "test_metadata_output_enabled_follows_video_mode_contract");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  bool ok = true;
  for (const std::string mode : {"clean", "annotated"}) {
    const fs::path config_path = fs::path(temp_dir) / ("config_" + mode + ".yaml");
    std::ofstream out(config_path);
    out << "model:\n"
           "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
           "streams:\n"
           "  - rtsp://127.0.0.1:8554/src1\n"
           "runtime:\n"
           "  worker_count: 2\n"
           "output:\n"
           "  insight:\n"
           "    host: 127.0.0.1\n"
           "  video_enabled: true\n"
        << "  video_mode: " << mode << "\n";
    out.close();

    try {
      const AppConfig cfg = load_app_config(config_path);
      const bool expected = mode == "clean";
      ok &= expect_true(metadata_output_enabled(cfg) == expected,
                        mode + (expected ? " keeps sidecar metadata enabled for clean video"
                                         : " suppresses sidecar metadata for annotated video"));
    } catch (const std::exception& ex) {
      ok &= expect_true(false, std::string("config should load: ") + ex.what());
    }
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_metadata_output_enabled_stays_enabled_for_metadata_only_mode() {
  const std::string temp_dir =
      create_test_output_dir("multistream-object-detection-insight",
                             "test_metadata_output_enabled_stays_enabled_for_metadata_only_mode");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n"
         "  video_enabled: false\n"
         "  video_mode: annotated\n";
  out.close();

  bool ok = true;
  try {
    const AppConfig cfg = load_app_config(config_path);
    ok &= expect_true(metadata_output_enabled(cfg),
                      "metadata-only mode keeps metadata output enabled");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("config should load: ") + ex.what());
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_metadata_output_enabled_stays_enabled_for_video_disabled_in_any_mode() {
  const std::string temp_dir = create_test_output_dir(
      "multistream-object-detection-insight",
      "test_metadata_output_enabled_stays_enabled_for_video_disabled_in_any_mode");
  if (temp_dir.empty()) {
    return expect_true(false, "created temp directory");
  }

  bool ok = true;
  for (const std::string mode : {"clean", "annotated"}) {
    const fs::path config_path = fs::path(temp_dir) / ("config_" + mode + ".yaml");
    std::ofstream out(config_path);
    out << "model:\n"
           "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
           "streams:\n"
           "  - rtsp://127.0.0.1:8554/src1\n"
           "runtime:\n"
           "  worker_count: 2\n"
           "output:\n"
           "  insight:\n"
           "    host: 127.0.0.1\n"
           "  video_enabled: false\n"
        << "  video_mode: " << mode << "\n";
    out.close();

    try {
      const AppConfig cfg = load_app_config(config_path);
      ok &= expect_true(metadata_output_enabled(cfg),
                        mode + " keeps metadata enabled when video output is disabled");
    } catch (const std::exception& ex) {
      ok &= expect_true(false, std::string("config should load: ") + ex.what());
    }
  }

  remove_dir(temp_dir);
  return ok;
}

bool test_resolve_model_family_auto_for_yolov8() {
  return expect_true(resolve_model_family("assets/models/yolo_v8m_mpk.tar.gz", ModelFamily::Auto) ==
                         ModelFamily::YoloV8,
                     "auto resolves yolo_v8 model path to YoloV8");
}

bool test_parse_bbox_payload_normalizes_yolov8_boxes() {
  const std::vector<std::uint8_t> payload = {
      1, 0, 0,  0, 246, 255, 255, 255, 20,  0,  0, 0, 140, 0,
      0, 0, 50, 0, 0,   0,   102, 102, 102, 63, 3, 0, 0,   0,
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

bool test_build_insight_detection_payload_builds_objects() {
  std::vector<Detection> boxes{
      Detection{-5.0f, 10.0f, 35.0f, 50.0f, 0.8f, 2},
  };

  const auto payload = build_insight_detection_payload(boxes, 30, 40, {"person", "bicycle", "car"});

  bool ok = true;
  ok &= expect_true(payload.objects.size() == 1, "detection payload yields one metadata object");
  if (!payload.objects.empty()) {
    const auto& object = payload.objects.front();
    ok &= expect_true(object.id == "obj_1", "detection payload assigns object id");
    ok &= expect_true(object.label == "car", "detection payload resolves object label");
    ok &= expect_true(object.x == 0, "detection payload clamps x");
    ok &= expect_true(object.y == 10, "detection payload keeps y");
    ok &= expect_true(object.w == 30, "detection payload clamps width");
    ok &= expect_true(object.h == 30, "detection payload clamps height");
    ok &= expect_true(object.confidence == 0.8f, "detection payload keeps confidence");
  }
  return ok;
}

bool test_tensor_rgb_from_sample_converts_nv12_source_frames() {
  simaai::neat::Tensor tensor;
  std::string err;
  if (!sima_examples::make_blank_nv12_tensor(128, 72, tensor, err)) {
    return expect_true(false, "created blank NV12 tensor for sample conversion test");
  }

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = std::move(tensor);

  bool ok = true;
  try {
    const cv::Mat frame = tensor_rgb_from_sample(sample);
    ok &= expect_true(frame.rows == 72, "NV12 sample converts to image with expected height");
    ok &= expect_true(frame.cols == 128, "NV12 sample converts to image with expected width");
    ok &= expect_true(frame.type() == CV_8UC3, "NV12 sample converts to 3-channel uint8 RGB");
  } catch (const std::exception& ex) {
    ok &= expect_true(false, std::string("NV12 sample should convert to RGB frame: ") + ex.what());
  }
  return ok;
}

bool test_insight_frame_id_prefers_detector_sample_frame_id() {
  simaai::neat::Sample sample;
  sample.frame_id = 42;
  return expect_true(insight_frame_id(sample, 7) == "42",
                     "Insight frame id prefers detector sample frame_id");
}

bool test_insight_frame_id_falls_back_to_packet_index() {
  simaai::neat::Sample sample;
  sample.frame_id = -1;
  return expect_true(insight_frame_id(sample, 7) == "7",
                     "Insight frame id falls back to packet index");
}

bool test_insight_timestamp_ms_applies_publish_offset() {
  return expect_true(insight_timestamp_ms(1.234, 25.0) == 1259,
                     "Insight timestamp adds metadata offset to publish time");
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

bool test_format_video_build_error_includes_stream_mode_and_detail() {
  const std::string annotated =
      format_video_build_error(6, VideoMode::Annotated, "Allocate output buffers failed");
  const std::string clean = format_video_build_error(2, VideoMode::Clean, "set_state failure");

  bool ok = true;
  ok &= expect_contains(annotated, "stream 6", "video build error includes stream index");
  ok &= expect_contains(annotated, "annotated", "video build error includes annotated mode");
  ok &= expect_contains(annotated, "Allocate output buffers failed",
                        "video build error keeps underlying detail");
  ok &= expect_contains(clean, "clean", "video build error includes clean mode");
  return ok;
}

bool test_detector_stage_names_cover_yolov8() {
  const auto yolov8 = detector_stage_names(ModelFamily::YoloV8);

  return expect_true(
      yolov8 == std::vector<std::string>{"input", "preproc", "mla", "sima_box_decode", "output"},
      "yolov8 stage names match the RGB preprocess detector graph");
}

bool test_apply_runtime_env_defaults_sets_expected_env_when_unset() {
  unsetenv("SIMA_FORCE_MODEL_NUM_BUFFERS");
  unsetenv("SIMA_FORCE_DECODER_NUM_BUFFERS");
  unsetenv("SIMA_FORCE_DECODER_POOL_BUFFERS");
  unsetenv("SIMA_PULL_TIMEOUT_DIAG");

  apply_runtime_env_defaults();

  bool ok = true;
  ok &= expect_true(env_or_empty("SIMA_FORCE_MODEL_NUM_BUFFERS") == "3",
                    "runtime defaults set model buffer count when unset");
  ok &= expect_true(env_or_empty("SIMA_FORCE_DECODER_NUM_BUFFERS") == "7",
                    "runtime defaults set decoder buffer count when unset");
  ok &= expect_true(env_or_empty("SIMA_FORCE_DECODER_POOL_BUFFERS") == "7",
                    "runtime defaults set decoder pool buffer count when unset");
  ok &= expect_true(env_or_empty("SIMA_PULL_TIMEOUT_DIAG") == "0",
                    "runtime defaults disable noisy pull-timeout diagnostics when unset");
  return ok;
}

bool test_apply_runtime_env_defaults_preserves_explicit_env() {
  setenv("SIMA_FORCE_MODEL_NUM_BUFFERS", "11", 1);
  setenv("SIMA_FORCE_DECODER_NUM_BUFFERS", "12", 1);
  setenv("SIMA_FORCE_DECODER_POOL_BUFFERS", "13", 1);
  setenv("SIMA_PULL_TIMEOUT_DIAG", "1", 1);

  apply_runtime_env_defaults();

  bool ok = true;
  ok &= expect_true(env_or_empty("SIMA_FORCE_MODEL_NUM_BUFFERS") == "11",
                    "runtime defaults preserve explicit model buffer count");
  ok &= expect_true(env_or_empty("SIMA_FORCE_DECODER_NUM_BUFFERS") == "12",
                    "runtime defaults preserve explicit decoder buffer count");
  ok &= expect_true(env_or_empty("SIMA_FORCE_DECODER_POOL_BUFFERS") == "13",
                    "runtime defaults preserve explicit decoder pool buffer count");
  ok &= expect_true(env_or_empty("SIMA_PULL_TIMEOUT_DIAG") == "1",
                    "runtime defaults preserve explicit pull-timeout diag setting");
  return ok;
}

bool test_startup_trace_defaults_to_disabled() {
  unsetenv("SIMA_INSIGHT_STARTUP_TRACE");
  return expect_true(!startup_trace_enabled_from_env(),
                     "startup trace defaults to disabled when env is unset");
}

bool test_startup_trace_accepts_truthy_aliases() {
  bool ok = true;
  setenv("SIMA_INSIGHT_STARTUP_TRACE", "1", 1);
  ok &= expect_true(startup_trace_enabled_from_env(), "startup trace accepts numeric truthy alias");
  setenv("SIMA_INSIGHT_STARTUP_TRACE", "TRUE", 1);
  ok &= expect_true(startup_trace_enabled_from_env(), "startup trace accepts uppercase true alias");
  setenv("SIMA_INSIGHT_STARTUP_TRACE", "banana", 1);
  ok &= expect_true(!startup_trace_enabled_from_env(),
                    "startup trace stays disabled for unknown values");
  unsetenv("SIMA_INSIGHT_STARTUP_TRACE");
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

bool test_build_source_input_group_options_match_working_rtsp_group_contract() {
  AppConfig cfg;
  cfg.tcp = true;
  cfg.latency_ms = 125;
  RtspProbe probe{1280, 720, 30};

  const auto options = build_source_input_group_options(cfg, "rtsp://127.0.0.1:8554/src1", probe);

  bool ok = true;
  ok &= expect_true(options.url == "rtsp://127.0.0.1:8554/src1",
                    "source group options keep the RTSP URL");
  ok &= expect_true(options.tcp, "source group options keep tcp=true");
  ok &= expect_true(options.latency_ms == 125, "source group options keep latency");
  ok &= expect_true(options.insert_queue, "source group options keep the input queue");
  ok &= expect_true(options.auto_caps_from_stream, "source group options use auto caps fixup");
  ok &=
      expect_true(options.fallback_h264_width == 1280, "source group options keep fallback width");
  ok &=
      expect_true(options.fallback_h264_height == 720, "source group options keep fallback height");
  ok &= expect_true(options.fallback_h264_fps == 30, "source group options keep fallback fps");
  ok &= expect_true(options.out_format == "RGB", "source group options decode directly to RGB");
  ok &= expect_true(!options.decoder_raw_output,
                    "source group options use the converted RGB decoder output path");
  ok &= expect_true(options.use_videoscale,
                    "source group options enable videoscale like the working tracking example");
  ok &= expect_true(options.output_caps.enable,
                    "source group options enable explicit RGB output caps");
  ok &=
      expect_true(options.output_caps.format == "RGB", "source group options keep RGB output caps");
  ok &=
      expect_true(options.output_caps.width == 1280, "source group options keep RGB output width");
  ok &=
      expect_true(options.output_caps.height == 720, "source group options keep RGB output height");
  ok &= expect_true(options.output_caps.fps == 30, "source group options keep RGB output fps");
  ok &= expect_true(options.output_caps.memory == simaai::neat::CapsMemory::SystemMemory,
                    "source group options keep RGB output in system memory");
  return ok;
}

bool test_source_producer_contract_matches_working_rtsp_pipeline() {
  bool ok = true;
  ok &= expect_true(
      source_startup_pull_timeout_ms() == 50000,
      "source producer keeps the 50 second startup pull timeout from the working example");
  ok &= expect_true(
      source_pull_timeout_ms() == 10000,
      "source producer keeps the 10 second steady-state pull timeout from the working example");
  ok &= expect_true(source_startup_stagger_s() == 0.5,
                    "source producer startup keeps the working half-second stream stagger");
  return ok;
}

} // namespace
} // namespace multistream_object_detection_insight

int main(int argc, char** argv) {
  using namespace multistream_object_detection_insight;

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
  ok &= test_load_app_config_rejects_removed_model_family_field();
  ok &= test_load_app_config_rejects_invalid_worker_count();
  ok &= test_load_app_config_rejects_invalid_video_mode();
  ok &= test_load_app_config_rejects_empty_streams();
  ok &= test_metadata_output_enabled_follows_video_mode_contract();
  ok &= test_metadata_output_enabled_stays_enabled_for_metadata_only_mode();
  ok &= test_metadata_output_enabled_stays_enabled_for_video_disabled_in_any_mode();
  ok &= test_resolve_model_family_auto_for_yolov8();
  ok &= test_parse_bbox_payload_normalizes_yolov8_boxes();
  ok &= test_require_detector_output_kind_rejects_unsupported_sample_kind();
  ok &= test_build_insight_detection_payload_builds_objects();
  ok &= test_tensor_rgb_from_sample_converts_nv12_source_frames();
  ok &= test_insight_frame_id_prefers_detector_sample_frame_id();
  ok &= test_insight_frame_id_falls_back_to_packet_index();
  ok &= test_insight_timestamp_ms_applies_publish_offset();
  ok &= test_latest_frame_mailbox_deduplicates_ready_notifications_and_requeues_after_completion();
  ok &= test_collect_detector_runtime_keys_deduplicates_same_geometry();
  ok &= test_format_video_build_error_includes_stream_mode_and_detail();
  ok &= test_detector_stage_names_cover_yolov8();
  ok &= test_apply_runtime_env_defaults_sets_expected_env_when_unset();
  ok &= test_apply_runtime_env_defaults_preserves_explicit_env();
  ok &= test_startup_trace_defaults_to_disabled();
  ok &= test_startup_trace_accepts_truthy_aliases();
  ok &= test_producer_emit_period_s_always_paces_when_fps_configured();
  ok &= test_build_source_input_group_options_match_working_rtsp_group_contract();
  ok &= test_source_producer_contract_matches_working_rtsp_pipeline();
  return ok ? 0 : 1;
}
