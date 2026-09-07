// E2E test for usb-camera-object-detector.
// There is no USB camera on the test target, so this drives the same graph from
// the synthetic NV12 source declared under `testing.e2e` in config.yaml. That
// exercises the custom source node, the branch, the video sender, the model,
// the box decode, and the metadata send -- everything but v4l2src itself.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr char kExample[] = "usb-camera-object-detector";
constexpr int kFrames = 30;

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path(kExample, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured detection model not found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }
  const std::string example_dir = fs::path(binary).parent_path().string();
  for (const auto& candidate :
       {std::string("examples/object-detection/usb-camera-object-detector/src/common/"
                    "coco_label.txt"),
        example_dir + "/src/common/coco_label.txt"}) {
    if (labels_file.empty() && fs::exists(candidate)) {
      labels_file = candidate;
    }
  }
  if (labels_file.empty()) {
    return skip_or_fail("src/common/coco_label.txt not found; set "
                        "SIMANEAT_APPS_TEST_LABELS_FILE");
  }

  const auto out_dir = create_test_output_dir(kExample, "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  // write_e2e_config folds `testing.e2e.*` into the runtime keys, so the
  // synthetic source override in config.yaml is applied here automatically.
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config(kExample, config_path,
                   {{"model.path", model_path},
                    {"model.labels", labels_file},
                    {"inference.frames", std::to_string(kFrames)},
                    {"output.insight.host", "127.0.0.1"}});

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (result.stdout_text.find("source=override") == std::string::npos) {
    std::cerr << "[FAIL] run did not report the overridden source\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else if (result.stdout_text.find("processed=" + std::to_string(kFrames)) == std::string::npos) {
    std::cerr << "[FAIL] run did not publish " << kFrames << " frames\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else {
    std::cout << "[OK] published " << kFrames << " frames from the synthetic source\n";
  }

  remove_dir(out_dir);
  return rc;
}
