// E2E test: run the binary, then assert its detections match the golden reference
// (tests/golden_detections.json), not just that non-empty output images exist.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

Box box_from_json(const nlohmann::json& node) {
  Box b;
  const auto& coords = node.at("box");
  b.x1 = coords.at(0).get<float>();
  b.y1 = coords.at(1).get<float>();
  b.x2 = coords.at(2).get<float>();
  b.y2 = coords.at(3).get<float>();
  b.class_id = node.at("class_id").get<int>();
  if (node.contains("score")) {
    b.score = node.at("score").get<float>();
  }
  return b;
}

float box_iou(const Box& a, const Box& b) {
  const float inter_w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float inter_h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = inter_w * inter_h;
  const float denom = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1) +
                      std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1) - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

std::string describe(const Box& b) {
  return "class=" + std::to_string(b.class_id) + " box=[" + std::to_string(b.x1) + "," +
         std::to_string(b.y1) + "," + std::to_string(b.x2) + "," + std::to_string(b.y2) + "]";
}

nlohmann::json load_json(const fs::path& path, std::string& err) {
  std::ifstream in(path);
  if (!in) {
    err = "failed to open " + path.string();
    return {};
  }
  try {
    return nlohmann::json::parse(in);
  } catch (const std::exception& e) {
    err = "failed to parse " + path.string() + ": " + e.what();
    return {};
  }
}

// Golden file path, resolved for both the ctest (cwd = apps root) and packaged (APPS_ROOT) layouts.
fs::path golden_detections_path() {
  const fs::path relative =
      "examples/object-detection/ssd-mobilenet-object-detector/tests/golden_detections.json";
  if (const char* apps_root = env_or_null("APPS_ROOT")) {
    const fs::path rooted = fs::path(apps_root) / relative;
    if (fs::exists(rooted)) {
      return rooted;
    }
  }
  return relative;
}

// Greedy IoU match of every golden box against the reported detections for one image.
bool match_image(const std::string& image, const std::vector<Box>& expected,
                 const std::vector<Box>& actual, float min_score, float min_iou) {
  std::vector<bool> used(actual.size(), false);
  bool ok = true;
  for (const Box& exp : expected) {
    float best_iou = 0.0f;
    int best_idx = -1;
    for (size_t j = 0; j < actual.size(); ++j) {
      if (used[j] || actual[j].class_id != exp.class_id || actual[j].score < min_score) {
        continue;
      }
      const float iou = box_iou(exp, actual[j]);
      if (iou > best_iou) {
        best_iou = iou;
        best_idx = static_cast<int>(j);
      }
    }
    if (best_idx < 0 || best_iou < min_iou) {
      std::cerr << "[FAIL] " << image << ": golden " << describe(exp)
                << " unmatched (best_iou=" << best_iou << ", "
                << (best_idx >= 0 ? describe(actual[static_cast<size_t>(best_idx)])
                                  : "no candidate")
                << ")\n";
      ok = false;
      continue;
    }
    used[static_cast<size_t>(best_idx)] = true;
  }
  return ok;
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

  const std::string model_path = configured_model_path("ssd-mobilenet-object-detector", models_dir);

  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "SSD-MobileNetV2 model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/datasets-test/coco";

  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory containing test images (defaults to assets/datasets-test/coco)");
    return kSkipCode;
  }

  const std::string out_dir =
      create_test_output_dir("ssd-mobilenet-object-detector", "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path detections_path = run_dir / "detections.json";
  write_e2e_config("ssd-mobilenet-object-detector", config_path,
                   {{"model.path", model_path},
                    {"io.input_dir", input_dir},
                    {"io.output_dir", out_dir},
                    {"io.detections_json", detections_path.string()},
                    {"runtime.num_runs", "1"}});
  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const int output_files = count_output_files(out_dir);
  if (output_files == 0) {
    std::cerr << "[FAIL] expected annotated output images but output directory is empty\n";
    remove_dir(out_dir);
    return 1;
  }
  if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some annotated output images are empty\n";
    remove_dir(out_dir);
    return 1;
  }

  std::string err;
  const nlohmann::json golden = load_json(golden_detections_path(), err);
  if (!err.empty()) {
    std::cerr << "[FAIL] golden detections: " << err << "\n";
    remove_dir(out_dir);
    return 1;
  }
  const nlohmann::json reported = load_json(detections_path, err);
  if (!err.empty()) {
    std::cerr << "[FAIL] reported detections: " << err << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const float min_score = golden.at("match").at("min_score").get<float>();
  const float min_iou = golden.at("match").at("min_iou").get<float>();

  std::map<std::string, std::vector<Box>> actual_by_image;
  std::map<std::string, float> frame_area_by_image;
  for (const auto& entry : reported.at("images")) {
    std::vector<Box> boxes;
    for (const auto& det : entry.at("detections")) {
      boxes.push_back(box_from_json(det));
    }
    const std::string image = entry.at("image").get<std::string>();
    actual_by_image[image] = std::move(boxes);
    frame_area_by_image[image] = entry.at("width").get<float>() * entry.at("height").get<float>();
  }

  int asserted = 0;
  int failures = 0;
  for (const auto& [image, expected_json] : golden.at("images").items()) {
    const auto it = actual_by_image.find(image);
    if (it == actual_by_image.end()) {
      // The harness may point at a different image folder; only assert on images it ran.
      continue;
    }
    std::vector<Box> expected;
    for (const auto& node : expected_json) {
      expected.push_back(box_from_json(node));
    }
    asserted += static_cast<int>(expected.size());
    if (!match_image(image, expected, it->second, min_score, min_iou)) {
      ++failures;
    }
  }

  if (golden.contains("forbidden")) {
    for (const auto& [image, rules] : golden.at("forbidden").items()) {
      const auto detections = actual_by_image.find(image);
      if (detections == actual_by_image.end()) {
        continue;
      }
      const float frame_area = frame_area_by_image.at(image);
      for (const auto& rule : rules) {
        const int class_id = rule.at("class_id").get<int>();
        const float min_area_fraction = rule.at("min_area_fraction").get<float>();
        for (const Box& box : detections->second) {
          const float area = std::max(0.0f, box.x2 - box.x1) * std::max(0.0f, box.y2 - box.y1);
          if (box.class_id == class_id && area / frame_area >= min_area_fraction) {
            std::cerr << "[FAIL] " << image << ": forbidden " << rule.at("label").get<std::string>()
                      << " covers " << (100.0f * area / frame_area)
                      << "% of the frame: " << describe(box) << "\n";
            ++failures;
          }
        }
      }
    }
  }

  if (asserted == 0) {
    std::cerr << "[FAIL] no golden detections were asserted; input folder has none of the "
                 "golden images\n";
    remove_dir(out_dir);
    return 1;
  }
  if (failures > 0) {
    std::cerr << "[FAIL] golden detection mismatch on " << failures << " image(s)\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] ssd-mobilenet-object-detector produced " << output_files
            << " output files and matched " << asserted << " golden detections\n";
  return 0;
}
