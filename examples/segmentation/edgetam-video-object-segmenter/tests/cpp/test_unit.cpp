// Unit test for edgetam-video-object-segmenter: the same tracking-step and config checks as
// tests/python/test_unit.py, plus CLI handling.
#include "../../src/cpp/utils/tracker.h"

#include "support/testing/test_checks.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_e2e_config;

namespace {

constexpr const char* kExampleName = "edgetam-video-object-segmenter";

// Mask decoder outputs whose mask k is k + 1 everywhere and whose pointer k is 10 * (k + 1)
// everywhere.
edgetam::ChosenMask choose(const std::vector<float>& iou_scores, float object_score) {
  std::vector<float> masks(edgetam::kMaskSize * edgetam::kMaskSize * edgetam::kMasks);
  for (std::size_t i = 0; i < masks.size(); ++i) {
    masks[i] = static_cast<float>(i % edgetam::kMasks + 1);
  }
  std::vector<float> pointers(edgetam::kMasks * edgetam::kPointerDim);
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    pointers[i] = static_cast<float>(10 * (i / edgetam::kPointerDim + 1));
  }
  return edgetam::choose_mask(masks, iou_scores, {object_score}, pointers);
}

bool all_equal(const cv::Mat& values, float expected) {
  return cv::countNonZero(values != expected) == 0;
}

bool near(double value, double expected) {
  return std::abs(value - expected) < 1e-5;
}

bool check_prompt_arrays() {
  const edgetam::PromptArrays prompted = edgetam::prompt_arrays(cv::Point2d(0.25, 0.75));
  const edgetam::PromptArrays tracked = edgetam::prompt_arrays(std::nullopt);
  // (0.25 * 1024 + 0.5) / 512 - 1 and (0.75 * 1024 + 0.5) / 512 - 1.
  bool ok =
      expect_true(near(prompted.coords[0], -0.4990234375) && near(prompted.coords[1], 0.5009765625),
                  "the prompt point is a pixel centre in [-1, 1]");
  ok &=
      expect_true(prompted.types == std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                  "the prompt point is foreground and the other slots are padding");
  ok &= expect_true(tracked.coords == std::vector<float>(6, 0.0F) &&
                        tracked.types ==
                            std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                    "a tracked frame's prompt is all padding");
  return ok;
}

bool check_choose_mask() {
  // Mask 0 has the highest score and must still lose: the model's authors use it only for a box.
  const edgetam::ChosenMask best = choose({0.9F, 0.2F, 0.7F, 0.4F}, 1.0F);
  bool ok = expect_true(best.mask.size() == cv::Size(256, 256) && all_equal(best.mask, 3.0F) &&
                            best.pointer == std::vector<float>(256, 30.0F),
                        "choose_mask keeps the best of masks 1 to 3");
  const edgetam::ChosenMask absent = choose({0.1F, 0.8F, 0.3F, 0.2F}, 0.0F);
  ok &= expect_true(all_equal(absent.mask, edgetam::kNoObjectLogit) &&
                        absent.pointer == std::vector<float>(256, 20.0F),
                    "choose_mask reports no object when the object score is not positive");
  return ok;
}

bool check_memory_mask() {
  cv::Mat mask(256, 256, CV_32F, cv::Scalar(-2.0F));
  mask.colRange(128, 256).setTo(2.0F);
  const std::vector<float> prompted = edgetam::memory_mask(mask, true);
  const std::vector<float> tracked = edgetam::memory_mask(mask, false);
  const std::vector<float> absent =
      edgetam::memory_mask(cv::Mat(256, 256, CV_32F, cv::Scalar(edgetam::kNoObjectLogit)), false);
  bool ok =
      expect_true(prompted.size() == 1024 * 1024 && prompted[0] == 0.0F && prompted[499] == 0.0F &&
                      prompted[524] == 1.0F && prompted[1023] == 1.0F,
                  "the prompted frame's memory mask is thresholded");
  ok &= expect_true(near(tracked[0], 1 / (1 + std::exp(2.0))) &&
                        near(tracked[1023], 1 / (1 + std::exp(-2.0))),
                    "a tracked frame's memory mask is a probability");
  ok &= expect_true(absent == std::vector<float>(1024 * 1024, 0.0F),
                    "the no-object logit becomes probability 0");
  return ok;
}

bool check_locate_object() {
  cv::Mat mask(256, 256, CV_32F, cv::Scalar(-1.0F));
  const edgetam::ObjectLocation absent = edgetam::locate_object(mask);
  mask(cv::Range(32, 96), cv::Range(128, 256)).setTo(1.0F);
  const edgetam::ObjectLocation located = edgetam::locate_object(mask);
  bool ok = expect_true(located.visible && located.area == 0.125 && located.center_x == 0.75 &&
                            located.center_y == 0.25,
                        "locate_object gives the area and centre in frame fractions");
  ok &= expect_true(!absent.visible && absent.area == 0.0 && !absent.center_x && !absent.center_y,
                    "locate_object reports an empty mask as not visible");
  return ok;
}

bool check_memory_bank() {
  edgetam::MemoryBank bank;
  bank.add(std::vector<float>(512 * 64, 1.0F), std::vector<float>(256, 1.0F));
  const edgetam::MemoryArrays first = bank.arrays();
  bool ok = expect_true(first.used_memories == std::vector<float>{1, 0, 0, 0, 0, 0, 0} &&
                            first.memories[512 * 64 - 1] == 1.0F &&
                            first.memories[512 * 64] == 0.0F && first.used_pointers[0] == 1.0F &&
                            first.used_pointers[1] == 0.0F && first.pointers[256] == 0.0F,
                        "the memory bank leaves unused slots zero");

  edgetam::MemoryBank filled;
  for (int frame = 0; frame < 20; ++frame) {
    filled.add(std::vector<float>(512 * 64, static_cast<float>(frame)),
               std::vector<float>(256, static_cast<float>(frame)));
  }
  // Frame 0 is the prompted frame; 19 is the newest. Each row holds its frame number.
  const edgetam::MemoryArrays arrays = filled.arrays();
  std::vector<float> memory_frames;
  for (int slot = 0; slot < 7; ++slot) {
    memory_frames.push_back(arrays.memories[slot * 512 * 64]);
  }
  std::vector<float> pointer_frames;
  for (int slot = 0; slot < 16; ++slot) {
    pointer_frames.push_back(arrays.pointers[slot * 256]);
  }
  ok &= expect_true(memory_frames == std::vector<float>{0, 19, 18, 17, 16, 15, 14} &&
                        pointer_frames == std::vector<float>{0, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                             10, 9, 8, 7, 6, 5} &&
                        arrays.used_memories == std::vector<float>(7, 1.0F) &&
                        arrays.used_pointers == std::vector<float>(16, 1.0F),
                    "the memory bank keeps the prompted frame, then the latest, newest first");
  return ok;
}

// Runs the binary on the shipped config with its four model paths pointing at files that exist
// and `overrides` applied, and checks that it stops with `message`.
bool check_config_error(const std::string& binary, const std::string& test_name,
                        std::map<std::string, std::string> overrides, const std::string& message) {
  const fs::path scratch = create_test_scratch_dir(kExampleName, test_name);
  for (const char* name : {"image_encoder", "memory_attention", "mask_decoder", "memory_encoder"}) {
    const fs::path package = scratch / (std::string(name) + ".tar.gz");
    std::ofstream(package) << "";
    overrides.emplace(std::string("model.") + name, package.string());
  }
  const fs::path config_path = write_e2e_config(kExampleName, scratch / "config.yaml", overrides);
  const ProcessResult result = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
  remove_dir(scratch.string());
  return expect_true(result.exit_code == 1, test_name + " exits with 1") &&
         expect_contains(result.stderr_text, message, test_name + " names the setting");
}

bool check_cli(const std::string& binary) {
  const ProcessResult help = spawn_and_wait(binary, {"--help"}, 20000);
  const ProcessResult unknown = spawn_and_wait(binary, {"--bogus"}, 20000);
  const ProcessResult no_path = spawn_and_wait(binary, {"--config"}, 20000);
  const ProcessResult missing = spawn_and_wait(binary, {"--config", "/nonexistent.yaml"}, 20000);
  bool ok = expect_true(help.exit_code == 0, "--help exits with 0") &&
            expect_contains(help.stdout_text, "--config", "--help describes the config option");
  ok &= expect_true(unknown.exit_code == 1, "an unknown argument exits with 1") &&
        expect_contains(unknown.stderr_text, "unknown argument: --bogus",
                        "an unknown argument is named");
  ok &= expect_contains(no_path.stderr_text, "--config requires a path",
                        "--config without a path is reported");
  ok &= expect_contains(missing.stderr_text, "failed to open config",
                        "a missing config file is reported");
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  bool ok = check_prompt_arrays();
  ok &= check_choose_mask();
  ok &= check_memory_mask();
  ok &= check_locate_object();
  ok &= check_memory_bank();
  ok &= check_cli(binary);
  ok &= check_config_error(
      binary, "prompt_x_out_of_range", {{"prompt.x", "1.2"}},
      "prompt.x 1.2 and prompt.y 0.38 must be fractions of the frame, from 0 to 1");
  ok &= check_config_error(
      binary, "prompt_y_out_of_range", {{"prompt.y", "-0.1"}},
      "prompt.x 0.87 and prompt.y -0.1 must be fractions of the frame, from 0 to 1");
  ok &= check_config_error(binary, "negative_frames", {{"inference.frames", "-1"}},
                           "inference.frames must be 0 or more");
  ok &= check_config_error(binary, "negative_save_every", {{"output.save_every", "-1"}},
                           "output.save_every must be 0 or more");
  ok &= check_config_error(binary, "missing_package", {{"model.mask_decoder", "missing.tar.gz"}},
                           "model.mask_decoder missing.tar.gz does not exist");
  ok &= check_config_error(binary, "unreadable_video", {{"source.video_path", "missing.mp4"}},
                           "could not open source.video_path missing.mp4");
  return ok ? 0 : 1;
}
