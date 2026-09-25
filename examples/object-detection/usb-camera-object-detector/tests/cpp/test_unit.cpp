// Unit test for usb-camera-object-detector: CLI handling, configuration validation,
// and the resolved camera fragment. Runs without a camera, a model, or a board.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr char kShippedConfig[] =
    "examples/object-detection/usb-camera-object-detector/src/common/config.yaml";
constexpr char kShippedLabels[] =
    "examples/object-detection/usb-camera-object-detector/src/common/coco_label.txt";

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

bool expect_absent(const std::string& haystack, const std::string& needle,
                   const std::string& message) {
  return expect_true(haystack.find(needle) == std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir("usb-camera-object-detector", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

// A minimal valid config, with one section optionally overridden.
std::string config_body(const std::string& extra_source = "",
                        const std::string& extra_inference = "",
                        const std::string& extra_runtime = "",
                        const std::string& labels = kShippedLabels) {
  return std::string("model:\n")
      .append("  path: models/yolo26m-det-bf16-mla_tess-b1.tar.gz\n")
      .append("  labels: " + labels + "\n")
      .append("source:\n")
      .append("  device: /dev/video16\n")
      .append("  width: 1920\n")
      .append("  height: 1080\n")
      .append("  fps: 30\n")
      .append("  flip: none\n")
      .append(extra_source)
      .append("inference:\n")
      .append("  frames: 0\n")
      .append("  min_score: 0.30\n")
      .append("  nms_iou: 0.50\n")
      .append("  max_detections: 100\n")
      .append(extra_inference)
      .append("runtime:\n")
      .append("  profile: false\n")
      .append("  profile_interval: 100\n")
      .append("  queue_depth: 3\n")
      .append(extra_runtime)
      .append("output:\n")
      .append("  insight:\n")
      .append("    host: 127.0.0.1\n")
      .append("    video_port: 9000\n")
      .append("    metadata_port: 9100\n")
      .append("    bitrate_kbps: 4000\n");
}

bool validate_rejects(const std::string& binary, const std::string& test_name,
                      const std::string& body, const std::string& expected_error,
                      const std::string& message) {
  const fs::path config_path = write_config(test_name, body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, message + " is rejected") &&
      expect_contains(result.stderr_text, expected_error, message + " names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

// Returns the `[validate] fragment=` line for a config body.
std::string fragment_for(const std::string& binary, const std::string& test_name,
                         const std::string& body, bool* ok) {
  const fs::path config_path = write_config(test_name, body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] " << test_name << ": validation exited " << result.exit_code << "\n";
    *ok = false;
    return {};
  }
  const std::string marker = "[validate] fragment=";
  const auto start = result.stdout_text.find(marker);
  if (start == std::string::npos) {
    std::cerr << "[FAIL] " << test_name << ": no fragment line\n";
    *ok = false;
    return {};
  }
  const auto begin = start + marker.size();
  const auto end = result.stdout_text.find('\n', begin);
  return result.stdout_text.substr(begin, end - begin);
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "Usage", "help prints usage") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_unknown_flag_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(result.exit_code == 1, "unknown flag exits with code 1") &&
         expect_contains(result.stderr_text, "unknown argument",
                         "unknown flag error names the argument");
}

bool test_missing_config_value_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config"}, 20000);
  return expect_true(result.exit_code == 1, "bare --config exits with code 1") &&
         expect_contains(result.stderr_text, "--config requires a path",
                         "bare --config explains the missing value");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "/nonexistent/usb-camera.yaml"}, 20000);
  return expect_true(result.exit_code == 1, "missing config exits with code 1") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

// The shipped config is validated through a scratch copy: the harness writes
// process logs next to the --config file, which must not land in src/common/.
bool test_shipped_config_validates(const std::string& binary) {
  std::ifstream shipped(kShippedConfig);
  if (!shipped.good()) {
    std::cerr << "[FAIL] shipped config is missing: " << kShippedConfig << "\n";
    return false;
  }
  std::string body((std::istreambuf_iterator<char>(shipped)), std::istreambuf_iterator<char>());
  // The committed config carries an Insight placeholder rather than a real host.
  const std::string placeholder = "<insight-host-ip>";
  const auto host_pos = body.find(placeholder);
  if (host_pos == std::string::npos) {
    std::cerr << "[FAIL] shipped config no longer uses an Insight host placeholder\n";
    return false;
  }
  body.replace(host_pos, placeholder.size(), "127.0.0.1");

  const fs::path config_path = write_config("test_shipped_config_validates", body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "shipped config validates") &&
                  expect_contains(result.stdout_text, "classes=80",
                                  "validate output reports the 80 COCO classes") &&
                  expect_contains(result.stdout_text, "stream=1920x1080@30",
                                  "validate output reports the capture mode") &&
                  expect_contains(result.stdout_text, "configuration OK",
                                  "validate output confirms the configuration");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_fragment_pins_mjpeg_capture(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_fragment_pins_mjpeg_capture", config_body(), &ok);
  if (!ok) {
    return false;
  }
  return expect_contains(frag, "v4l2src device=/dev/video16",
                         "fragment opens the configured node") &&
         expect_contains(frag, "io-mode=mmap", "fragment uses zero-copy mmap io") &&
         expect_contains(frag, "image/jpeg", "fragment pins MJPEG rather than raw YUYV") &&
         expect_contains(frag, "width=1920,height=1080,framerate=30/1",
                         "fragment carries the capture mode") &&
         // jpegparse breaks UVC MJPEG on GStreamer 1.22 (see camera_fragment).
         expect_absent(frag, "jpegparse", "fragment omits jpegparse") &&
         expect_contains(frag, "neatdecoder", "fragment uses the SiMa hardware decoder") &&
         expect_contains(frag, "dec-type=mjpeg", "hardware decoder is in MJPEG mode") &&
         expect_contains(frag, "dec-fmt=NV12", "hardware decoder publishes NV12") &&
         expect_absent(frag, "jpegdec", "fragment does not decode JPEG on the CPU") &&
         expect_absent(frag, "videoconvert", "hardware decoder needs no CPU conversion") &&
         expect_contains(frag, "leaky=downstream", "fragment queues drop rather than stall");
}

// gst_parse_launch reads a trailing caps string as an element name and fails
// with `no element "video"`, so the fragment must end on a real element.
bool test_fragment_does_not_end_on_caps(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_fragment_does_not_end_on_caps", config_body(), &ok);
  if (!ok) {
    return false;
  }
  const auto last = frag.find_last_of('!');
  const std::string tail = last == std::string::npos ? frag : frag.substr(last + 1);
  const auto first_char = tail.find_first_not_of(' ');
  const std::string trimmed = first_char == std::string::npos ? tail : tail.substr(first_char);
  return expect_true(trimmed.rfind("queue", 0) == 0, "fragment ends on a real element");
}

bool test_fragment_omits_flip_by_default(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_fragment_omits_flip_by_default", config_body(), &ok);
  if (!ok) {
    return false;
  }
  return expect_absent(frag, "videoflip", "no videoflip when source.flip is none");
}

bool test_fragment_inserts_flip(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_fragment_inserts_flip", config_body("  flip: rotate-180\n"), &ok);
  if (!ok) {
    return false;
  }
  const auto flip_pos = frag.find("videoflip method=rotate-180");
  const auto decode_pos = frag.find("neatdecoder");
  return expect_true(flip_pos != std::string::npos, "videoflip is inserted for rotate-180") &&
         expect_true(decode_pos < flip_pos, "videoflip runs after the hardware decode");
}

bool test_fragment_honours_capture_mode(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_fragment_honours_capture_mode",
                   config_body("  width: 1280\n  height: 720\n  fps: 25\n"), &ok);
  if (!ok) {
    return false;
  }
  return expect_contains(frag, "width=1280,height=720,framerate=25/1",
                         "capture caps follow the configured mode") &&
         expect_contains(frag, "image/jpeg,width=1280,height=720,framerate=25/1",
                         "output caps follow the configured mode");
}

bool test_override_replaces_the_camera(const std::string& binary) {
  bool ok = true;
  const std::string frag =
      fragment_for(binary, "test_override_replaces_the_camera",
                   config_body("  override_fragment: \"videotestsrc ! queue\"\n"), &ok);
  if (!ok) {
    return false;
  }
  return expect_true(frag == "videotestsrc ! queue", "override is used verbatim") &&
         expect_absent(frag, "v4l2src", "override suppresses the camera element");
}

// A YAML emitter wraps a long plain scalar onto more-indented continuation lines.
// Python's yaml reads that back as one value, so the C++ parser must fold it the
// same way; before it did, such a config failed with "invalid config line".
bool test_wrapped_override_fragment_is_folded(const std::string& binary) {
  bool ok = true;
  const std::string frag = fragment_for(
      binary, "test_wrapped_override_fragment_is_folded",
      config_body("  override_fragment: videotestsrc pattern=smpte is-live=true\n"
                  "    ! video/x-raw,format=NV12,width=1920,height=1080\n"
                  "    ! queue leaky=downstream max-size-buffers=2\n"),
      &ok);
  if (!ok) {
    return false;
  }
  return expect_true(frag ==
                         "videotestsrc pattern=smpte is-live=true "
                         "! video/x-raw,format=NV12,width=1920,height=1080 "
                         "! queue leaky=downstream max-size-buffers=2",
                     "wrapped override fragment folds into one value");
}

bool test_override_reported_as_source(const std::string& binary) {
  const fs::path config_path =
      write_config("test_override_reported_as_source",
                   config_body("  override_fragment: \"videotestsrc ! queue\"\n"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "config with an override validates") &&
                  expect_contains(result.stdout_text, "source=override",
                                  "validate output flags the overridden source");
  remove_dir(config_path.parent_path().string());
  return ok;
}

// An override supplies its own source, so the device is not required.
bool test_override_allows_empty_device(const std::string& binary) {
  const fs::path config_path =
      write_config("test_override_allows_empty_device",
                   std::string("model:\n")
                       .append("  path: models/yolo26m-det-bf16-mla_tess-b1.tar.gz\n")
                       .append("  labels: " + std::string(kShippedLabels) + "\n")
                       .append("source:\n")
                       .append("  device: \"\"\n")
                       .append("  override_fragment: \"videotestsrc ! queue\"\n")
                       .append("output:\n")
                       .append("  insight:\n")
                       .append("    host: 127.0.0.1\n"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "empty device validates when overridden");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_missing_model_path_is_rejected(const std::string& binary) {
  const fs::path config_path =
      write_config("test_missing_model_path_is_rejected",
                   std::string("model:\n")
                       .append("  labels: " + std::string(kShippedLabels) + "\n")
                       .append("output:\n")
                       .append("  insight:\n")
                       .append("    host: 127.0.0.1\n"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config without model.path is rejected") &&
      expect_contains(result.stderr_text, "model.path", "missing model error names model.path");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_missing_labels_file_is_rejected(const std::string& binary) {
  const fs::path config_path = write_config("test_missing_labels_file_is_rejected",
                                            config_body("", "", "", "/nonexistent/coco.txt"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config with a missing labels file is rejected") &&
      expect_contains(result.stderr_text, "labels file does not exist",
                      "missing labels error names the labels file");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_empty_labels_file_is_rejected(const std::string& binary) {
  const std::string temp_dir =
      create_test_scratch_dir("usb-camera-object-detector", "test_empty_labels_file_is_rejected");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] failed to create temp directory\n";
    return false;
  }
  const fs::path labels_path = fs::path(temp_dir) / "empty_label.txt";
  std::ofstream(labels_path) << "\n \n";

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream(config_path) << config_body("", "", "", labels_path.string());

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config with an empty labels file is rejected") &&
      expect_contains(result.stderr_text, "labels file is empty",
                      "empty labels error names the labels file");
  remove_dir(temp_dir);
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_unknown_flag_is_rejected(binary);
  ok &= test_missing_config_value_is_rejected(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_shipped_config_validates(binary);
  ok &= test_fragment_pins_mjpeg_capture(binary);
  ok &= test_fragment_does_not_end_on_caps(binary);
  ok &= test_fragment_omits_flip_by_default(binary);
  ok &= test_fragment_inserts_flip(binary);
  ok &= test_fragment_honours_capture_mode(binary);
  ok &= test_override_replaces_the_camera(binary);
  ok &= test_wrapped_override_fragment_is_folded(binary);
  ok &= test_override_reported_as_source(binary);
  ok &= test_override_allows_empty_device(binary);
  ok &= test_missing_model_path_is_rejected(binary);
  ok &= test_missing_labels_file_is_rejected(binary);
  ok &= test_empty_labels_file_is_rejected(binary);

  ok &= validate_rejects(binary, "test_rejects_zero_width", config_body("  width: 0\n"),
                         "source.width", "zero source.width");
  ok &= validate_rejects(binary, "test_rejects_zero_height", config_body("  height: 0\n"),
                         "source.height", "zero source.height");
  ok &= validate_rejects(binary, "test_rejects_zero_fps", config_body("  fps: 0\n"), "source.fps",
                         "zero source.fps");
  ok &= validate_rejects(binary, "test_rejects_bad_flip", config_body("  flip: rotate-90\n"),
                         "source.flip", "unsupported source.flip");
  ok &= validate_rejects(binary, "test_rejects_negative_frames", config_body("", "  frames: -1\n"),
                         "inference.frames", "negative inference.frames");
  ok &= validate_rejects(binary, "test_rejects_out_of_range_min_score",
                         config_body("", "  min_score: 1.50\n"), "inference.min_score",
                         "out-of-range inference.min_score");
  ok &= validate_rejects(binary, "test_rejects_out_of_range_nms",
                         config_body("", "  nms_iou: 1.20\n"), "inference.nms_iou",
                         "out-of-range inference.nms_iou");
  ok &= validate_rejects(binary, "test_rejects_zero_max_detections",
                         config_body("", "  max_detections: 0\n"), "inference.max_detections",
                         "zero inference.max_detections");
  ok &= validate_rejects(binary, "test_rejects_zero_profile_interval",
                         config_body("", "", "  profile_interval: 0\n"), "runtime.profile_interval",
                         "zero runtime.profile_interval");
  ok &= validate_rejects(binary, "test_rejects_zero_queue_depth",
                         config_body("", "", "  queue_depth: 0\n"), "runtime.queue_depth",
                         "zero runtime.queue_depth");

  return ok ? 0 : 1;
}
