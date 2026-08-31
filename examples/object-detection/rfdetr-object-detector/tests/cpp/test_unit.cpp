#define main rfdetr_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_process.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  int failures = 0;

  std::vector<float> scores(305, 0.0F);
  scores[3] = 2.0F;
  scores[4] = 2.0F;
  std::vector<float> proposals(scores.size() * 4U);
  for (std::size_t index = 0; index < scores.size(); ++index) {
    proposals[index * 4U] = static_cast<float>(index);
  }
  const auto [gathered, indices] = stable_topk_gather(scores, proposals);
  if (indices.size() != 300U || gathered.size() != 1200U || indices[0] != 3 || indices[1] != 4 ||
      indices[2] != 0 || gathered[0] != 3.0F || gathered[4] != 4.0F) {
    std::cerr << "[FAIL] TopK/Gather must be stable, descending, and limited to 300 rows\n";
    ++failures;
  }

  std::vector<std::string> labels(91, "unused");
  labels[1] = "person";
  std::vector<float> boxes(1200, 0.0F);
  boxes[0] = 0.5F;
  boxes[1] = 0.5F;
  boxes[2] = 0.5F;
  boxes[3] = 0.25F;
  std::vector<float> logits(27300, -20.0F);
  logits[1] = 10.0F;
  const auto objects = postprocess(boxes, logits, 1920, 1080, labels, 0.5F, 10);
  if (objects.size() != 1U || objects[0].label != "person" ||
      std::abs(objects[0].x - 480.0F) > 0.01F || std::abs(objects[0].y - 405.0F) > 0.01F ||
      std::abs(objects[0].w - 960.0F) > 0.01F || std::abs(objects[0].h - 270.0F) > 0.01F) {
    std::cerr << "[FAIL] postprocessing must preserve sparse COCO IDs and source geometry\n";
    ++failures;
  }

  const std::string temp_dir =
      create_test_scratch_dir("rfdetr-object-detector", "unknown-model-variant");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] could not create config test directory\n";
    ++failures;
  } else {
    const fs::path config_path = fs::path(temp_dir) / "config.yaml";
    std::ofstream config(config_path);
    config << "model:\n  variant: large\nsource: {}\ninference: {}\noutput:\n  insight: {}\n";
    config.close();
    try {
      (void)load_config(config_path);
      std::cerr << "[FAIL] config must reject model variants other than small and medium\n";
      ++failures;
    } catch (const std::exception& error) {
      if (std::string(error.what()).find("small or medium") == std::string::npos) {
        std::cerr << "[FAIL] invalid model variant error must name the supported variants\n";
        ++failures;
      }
    }
    remove_dir(temp_dir);
  }

  const std::string binary = argv[1];
  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  if (help.exit_code != 0 || help.stdout_text.find("--config") == std::string::npos) {
    std::cerr << "[FAIL] help should describe the config-driven CLI\n";
    ++failures;
  }
  const auto no_config = spawn_and_wait(binary, {}, 20000);
  if (no_config.exit_code == 0 ||
      no_config.stderr_text.find("--config is required") == std::string::npos) {
    std::cerr << "[FAIL] the application should require an explicit config path\n";
    ++failures;
  }
  const auto missing =
      spawn_and_wait(binary, {"--config", "/nonexistent/rfdetr-config.yaml"}, 20000);
  if (missing.exit_code == 0 || missing.stderr_text.find("config") == std::string::npos) {
    std::cerr << "[FAIL] a missing config should fail clearly\n";
    ++failures;
  }
  return failures == 0 ? 0 : 1;
}
