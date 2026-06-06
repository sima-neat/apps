// E2E test for person-re-identification.
// Runs the binary with a real model and image pair, verifies output artifacts.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

std::vector<fs::path> collect_images_recursive(const fs::path& root) {
  std::vector<fs::path> images;
  if (!fs::exists(root)) {
    return images;
  }
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

bool read_json(const fs::path& path, nlohmann::json& out) {
  std::ifstream in(path);
  if (!in.good()) {
    return false;
  }
  in >> out;
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_dir_raw ? fs::path(models_dir_raw) : fs::path("assets/models");

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (const auto& entry : fs::directory_iterator(models_dir)) {
      const std::string name = entry.path().filename().string();
      if (name.find("reid") != std::string::npos && name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    return skip_or_fail("ReID model (.tar.gz with 'reid' in name) not found under models dir");
  }

  const char* images_env = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const fs::path images_root =
      images_env ? fs::path(images_env) : fs::path("assets/images/neat_reid_examples");
  const std::vector<fs::path> images = collect_images_recursive(images_root);
  if (images.size() < 2) {
    return skip_or_fail("need at least 2 images under images dir (found " +
                        std::to_string(images.size()) + ")");
  }

  const fs::path image_a = images[0];
  const fs::path image_b = images[1];
  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const double cosine_threshold =
      e2e_double("person-re-identification", "comparison.cosine", "threshold");
  const double euclidean_threshold =
      e2e_double("person-re-identification", "comparison.euclidean", "threshold");

  int failures = 0;

  // Case 1: cosine metric, default output type (both).
  {
    const std::string out_dir =
        create_test_output_dir("person-re-identification", "test_full_pipeline_cosine");
    if (out_dir.empty()) {
      return 1;
    }
    const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
    {
      std::ofstream config_file(config_path);
      config_file << "model:\n"
                  << "  path: " << model_path << "\n"
                  << "io:\n"
                  << "  image1: " << image_a.string() << "\n"
                  << "  image2: " << image_b.string() << "\n"
                  << "  output_dir: " << out_dir << "\n"
                  << "output:\n"
                  << "  type: both\n"
                  << "comparison:\n"
                  << "  metric: cosine\n"
                  << "  threshold: " << cosine_threshold << "\n"
                  << "runtime:\n"
                  << "  timeout_ms: 5000\n"
                  << "  profile: false\n";
    }

    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

    const fs::path comparison = fs::path(out_dir) / "comparison.jpg";
    const fs::path result_json = fs::path(out_dir) / "result.json";

    if (r.exit_code != 0) {
      std::cerr << "[FAIL] cosine run exit code " << r.exit_code << "\n";
      std::cerr << "stderr:\n" << r.stderr_text << "\n";
      ++failures;
    } else if (!fs::exists(comparison) || fs::is_empty(comparison)) {
      std::cerr << "[FAIL] cosine run missing comparison.jpg\n";
      ++failures;
    } else if (!fs::exists(result_json) || fs::is_empty(result_json)) {
      std::cerr << "[FAIL] cosine run missing result.json\n";
      ++failures;
    } else {
      nlohmann::json payload;
      if (!read_json(result_json, payload)) {
        std::cerr << "[FAIL] could not parse result.json for cosine run\n";
        ++failures;
      } else if (payload.value("metric", "") != "cosine") {
        std::cerr << "[FAIL] cosine run result.json has wrong metric\n";
        ++failures;
      } else {
        std::cout << "[OK] cosine run produced comparison.jpg and result.json\n";
      }
    }

    remove_dir(out_dir);
  }

  // Case 2: euclidean metric, explicit json-only output.
  {
    const std::string out_dir =
        create_test_output_dir("person-re-identification", "test_output_type_json_only");
    if (out_dir.empty()) {
      return 1;
    }
    const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
    {
      std::ofstream config_file(config_path);
      config_file << "model:\n"
                  << "  path: " << model_path << "\n"
                  << "io:\n"
                  << "  image1: " << image_a.string() << "\n"
                  << "  image2: " << image_b.string() << "\n"
                  << "  output_dir: " << out_dir << "\n"
                  << "output:\n"
                  << "  type: json\n"
                  << "comparison:\n"
                  << "  metric: euclidean\n"
                  << "  threshold: " << euclidean_threshold << "\n"
                  << "runtime:\n"
                  << "  timeout_ms: 5000\n"
                  << "  profile: false\n";
    }

    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

    const fs::path comparison = fs::path(out_dir) / "comparison.jpg";
    const fs::path result_json = fs::path(out_dir) / "result.json";

    if (r.exit_code != 0) {
      std::cerr << "[FAIL] euclidean/json run exit code " << r.exit_code << "\n";
      std::cerr << "stderr:\n" << r.stderr_text << "\n";
      ++failures;
    } else if (!fs::exists(result_json) || fs::is_empty(result_json)) {
      std::cerr << "[FAIL] euclidean/json run missing result.json\n";
      ++failures;
    } else if (fs::exists(comparison)) {
      std::cerr << "[FAIL] euclidean/json run should not create comparison.jpg\n";
      ++failures;
    } else {
      nlohmann::json payload;
      if (!read_json(result_json, payload)) {
        std::cerr << "[FAIL] could not parse result.json for euclidean/json run\n";
        ++failures;
      } else if (payload.value("metric", "") != "euclidean") {
        std::cerr << "[FAIL] euclidean/json run result.json has wrong metric\n";
        ++failures;
      } else {
        std::cout << "[OK] euclidean json-only run produced expected artifacts\n";
      }
    }

    remove_dir(out_dir);
  }

  return failures > 0 ? 1 : 0;
}
