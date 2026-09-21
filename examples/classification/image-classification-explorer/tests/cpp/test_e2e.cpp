// E2E test for image-classification-explorer.
// Runs the binary against a real image with four real models and verifies the
// report files it produces.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr std::array<const char*, 4> kModelNames = {"resnet_50", "resnet_18", "efficientnet_b0",
                                                    "densenet_121"};

bool file_contains(const fs::path& path, const std::string& needle) {
  std::ifstream in(path);
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str().find(needle) != std::string::npos;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  std::vector<fs::path> model_paths;
  for (const auto* name : kModelNames) {
    model_paths.push_back(fs::path(models_dir) / (std::string(name) + "_mpk.tar.gz"));
  }
  for (const auto& p : model_paths) {
    if (!fs::exists(p)) {
      return skip_or_fail(
          "resnet_50/resnet_18/efficientnet_b0/densenet_121 model packages (.tar.gz) not found "
          "under SIMANEAT_APPS_TEST_MODELS_DIR");
    }
  }

  std::string image_path;
  if (const char* image_env = env_or_null("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE")) {
    image_path = image_env;
  } else {
    image_path = "assets/datasets-test/imagenet/goldfish.jpeg";
  }
  if (!fs::exists(image_path)) {
    env_or_skip("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
                "path to goldfish image for classification e2e (e.g. "
                "assets/datasets-test/imagenet/goldfish.jpeg)");
  }

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto out_dir = create_test_output_dir("image-classification-explorer", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  const fs::path report_dir = fs::path(out_dir) / "report";
  ConfigScalars overrides = {{"io.input", image_path},
                             {"io.fallback_image_url", "null"},
                             {"io.output_dir", report_dir.string()}};
  for (size_t i = 0; i < kModelNames.size(); ++i) {
    overrides[std::string("models.") + kModelNames[i] + ".path"] = model_paths[i].string();
  }
  write_e2e_config("image-classification-explorer", config_path, overrides);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  int failures = 0;
  for (const auto& name : {"report.json", "report.csv", "report.html"}) {
    if (!fs::exists(report_dir / name)) {
      std::cerr << "[FAIL] missing report file: " << (report_dir / name) << "\n";
      ++failures;
    }
  }
  for (const auto* name : kModelNames) {
    if (failures == 0 &&
        !file_contains(report_dir / "report.json", std::string("\"") + name + "\"")) {
      std::cerr << "[FAIL] report.json missing " << name << " predictions\n";
      ++failures;
    }
  }

  if (failures == 0) {
    std::cout << "[OK] classification explorer pipeline completed successfully\n";
  }
  remove_dir(out_dir);
  return failures > 0 ? 1 : 0;
}
