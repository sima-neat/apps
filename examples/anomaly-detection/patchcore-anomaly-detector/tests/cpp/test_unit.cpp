// Unit tests for the PatchCore anomaly detector: the checks the application leaves to its config,
// and the host-side memory bank on hand-made patch features. tests/python/test_unit.py runs the
// same checks in Python.

#include "../../src/cpp/utils/memory_bank.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

const fs::path kCommonDir = "examples/anomaly-detection/patchcore-anomaly-detector/src/common";

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_and_input() {
  const auto config = sima_examples::Config::load(kCommonDir / "config.yaml");
  const std::string type = config.string("input.type");
  const std::string codec = config.string("input.codec");
  bool ok =
      expect_true(type == "rtsp" || type == "image_dir", "the shipped input.type is supported");
  ok &= expect_true(codec == "h264" || codec == "h265" || codec == "mjpeg",
                    "the shipped codec is supported");
  ok &= expect_true(fs::path(config.string("model.path")).filename() ==
                        "patchcore_wide_resnet50_2_bf16_mla.tar.gz",
                    "model.path names the PatchCore package");
  ok &= expect_true(
      config.integer("input.width") >= 0 && config.integer("input.height") >= 0 &&
          config.integer("input.fps") >= 0 && config.integer("input.rtsp.latency_ms") >= 0 &&
          config.integer("inference.frames") >= 0 && config.integer("inference.max_detections") > 0,
      "the shipped sizes, latency, and counts are in range");
  for (const std::string key :
       {"output.insight.video_port_base", "output.insight.metadata_port_base"}) {
    ok &= expect_true(config.integer(key) >= 1 && config.integer(key) <= 65535, key + " is a port");
  }
  // Core sends to 127.0.0.1 when the host is empty.
  ok &=
      expect_true(!config.string("output.insight.host").empty() && config.boolean("input.rtsp.tcp"),
                  "the shipped Insight host is set and RTSP runs over TCP");
  return ok;
}

bool test_shipped_calibration_settings_are_in_range() {
  const auto config = sima_examples::Config::load(kCommonDir / "config.yaml");
  const fs::path nominal = config.string("app.nominal_images_dir");
  std::size_t images = 0;
  for (const auto& entry : fs::directory_iterator(nominal)) {
    images += entry.path().extension() == ".png" ? 1 : 0;
  }
  bool ok = expect_true(config.number("app.coreset_ratio") > 0.0 &&
                            config.number("app.coreset_ratio") <= 1.0 &&
                            config.number("app.threshold_percentile") >= 0.0 &&
                            config.number("app.threshold_percentile") <= 100.0 &&
                            config.number("app.threshold_margin") >= 1.0,
                        "the shipped calibration settings are in range");
  ok &= expect_true(images >= 5, "the nominal set holds enough images to hold every 5th one out");
  // Outlined images written among the nominal images would be calibrated on next time.
  ok &= expect_true(!fs::equivalent(nominal, ".") &&
                        fs::weakly_canonical(nominal) !=
                            fs::weakly_canonical(config.string("output.dir")) &&
                        fs::weakly_canonical(nominal) !=
                            fs::weakly_canonical(config.string("app.memory_bank_dir")),
                    "output.dir and app.memory_bank_dir are not the nominal set");
  return ok;
}

bool test_anomaly_map_covers_the_frame_and_keeps_the_peak_in_place() {
  patchcore::AnomalyResult result;
  result.score_map.assign(28 * 28, 0.0F);
  result.score_map[27] = 10.0F; // Row 0, the last column.
  const cv::Mat map = patchcore::anomaly_map(result, 28, 28, 224, 4.0);
  cv::Point peak;
  cv::minMaxLoc(map, nullptr, nullptr, nullptr, &peak);
  return expect_true(map.rows == 224 && map.cols == 224 && map.type() == CV_32FC1 && peak.y < 16 &&
                         peak.x > 208,
                     "the anomaly map is 224x224 with the peak in its corner");
}

// A 2x2 grid of 4-channel patches, each a different unit vector.
patchcore::PatchEmbeddings one_hot_patches() {
  patchcore::PatchEmbeddings patches{2, 2, 4, std::vector<float>(16, 0.0F)};
  for (int patch = 0; patch < 4; ++patch) {
    patches.values[patch * 4 + patch] = 1.0F;
  }
  return patches;
}

bool test_identical_patches_score_zero_and_a_far_one_scores_high() {
  const auto normal = one_hot_patches();
  const auto bank = patchcore::MemoryBank::build({normal}, 1.0, 0);
  bool ok = expect_true(bank.size() == 4U, "a ratio of 1 keeps every patch");
  const auto same = bank.score(normal, 1);
  ok &= expect_true(same.image_score < 1e-5F, "the bank's own patches score 0");
  auto defect = normal;
  defect.values[0] = 5.0F;
  const auto far = bank.score(defect, 1);
  ok &= expect_true(std::abs(far.image_score - 4.0F) < 1e-4F && far.score_map[0] == far.image_score,
                    "a patch 4 away from the bank scores 4 at its place");
  return ok;
}

bool test_reweighting_never_raises_the_plain_max_score() {
  const auto normal = one_hot_patches();
  const auto bank = patchcore::MemoryBank::build({normal}, 1.0, 0);
  auto defect = normal;
  defect.values[0] = 5.0F;
  const float reweighted = bank.score(defect, 4).image_score;
  return expect_true(reweighted > 0.0F && reweighted < bank.score(defect, 1).image_score,
                     "the reweighted score lies between 0 and the plain max");
}

// Ten points on a line: the coreset starts at patch seed % n, then takes the farthest.
bool test_coreset_keeps_the_ratio_starting_at_the_seed_patch() {
  std::vector<float> line(10);
  for (std::size_t i = 0; i < line.size(); ++i) {
    line[i] = static_cast<float>(i);
  }
  const auto some = patchcore::greedy_coreset_indices(line, 1, 0.3, 13);
  auto all = patchcore::greedy_coreset_indices(line, 1, 1.0, 0);
  std::sort(all.begin(), all.end());
  const std::vector<std::size_t> every = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  return expect_true(some == std::vector<std::size_t>{3, 9, 0} && all == every,
                     "the coreset keeps the ratio, starting at patch seed % n");
}

bool test_percentile_threshold_interpolates_like_numpy() {
  std::vector<float> scores;
  for (int value = 1; value <= 100; ++value) {
    scores.push_back(static_cast<float>(value));
  }
  return expect_true(std::abs(patchcore::percentile_threshold(scores, 99.0) - 99.01F) < 1e-4F,
                     "the 99th percentile of 1..100 is 99.01");
}

bool test_confidence_is_half_at_the_threshold_and_capped_at_one() {
  return expect_true(std::abs(patchcore::confidence(20.0, 20.0) - 0.5) < 1e-9 &&
                         std::abs(patchcore::confidence(30.0, 20.0) - 0.75) < 1e-9 &&
                         patchcore::confidence(50.0, 20.0) == 1.0 &&
                         patchcore::confidence(-1.0, 20.0) == 0.0,
                     "confidence is 0.5 at the threshold and stays within 0 to 1");
}

bool test_extract_hwc_transposes_channel_first_output() {
  std::vector<float> chw(12);
  for (std::size_t i = 0; i < chw.size(); ++i) {
    chw[i] = static_cast<float>(i);
  }
  const auto hwc = patchcore::extract_hwc({1, 3, 2, 2}, chw, 3);
  const std::vector<float> first(hwc.values.begin(), hwc.values.begin() + 3);
  const std::vector<float> right(hwc.values.begin() + 3, hwc.values.begin() + 6);
  return expect_true(hwc.height == 2 && hwc.width == 2 && hwc.channels == 3 &&
                         first == std::vector<float>{0, 4, 8} &&
                         right == std::vector<float>{1, 5, 9},
                     "a channel-first output is transposed to (H, W, C)");
}

void write_file(const fs::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary) << text;
}

bool test_bank_meta_round_trips() {
  const fs::path dir = fs::temp_directory_path() / "patchcore_unit_bank_meta";
  fs::create_directories(dir);
  write_file(dir / "model.tar.gz", "model.tar.gz");
  patchcore::BankMeta meta;
  meta.model_sha256 = patchcore::sha256_file(dir / "model.tar.gz");
  meta.threshold_value = 12.5;
  meta.has_patch_threshold = true;
  meta.patch_threshold_value = 8.25;
  patchcore::save_bank_meta(dir / "bank_meta.json", meta);
  const auto loaded = patchcore::load_bank_meta(dir / "bank_meta.json");
  fs::remove_all(dir);
  return expect_true(loaded.model_sha256 == meta.model_sha256 && loaded.threshold_value == 12.5 &&
                         loaded.has_patch_threshold && loaded.patch_threshold_value == 8.25,
                     "bank_meta.json keeps the model hash and both thresholds");
}

bool throws(const std::function<void()>& call) {
  try {
    call();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

bool test_a_bank_for_another_model_or_bank_file_is_rejected() {
  const fs::path dir = fs::temp_directory_path() / "patchcore_unit_bank_check";
  fs::create_directories(dir);
  const fs::path model = dir / "model.tar.gz";
  const fs::path bank = dir / "memory_bank.npy";
  write_file(model, "model");
  write_file(bank, "bank");
  patchcore::BankMeta meta;
  meta.model_sha256 = patchcore::sha256_file(model);
  meta.bank_sha256 = patchcore::sha256_file(bank);
  bool ok = expect_true(!throws([&] { patchcore::verify_bank_matches_model(meta, model); }) &&
                            !throws([&] { patchcore::verify_bank_hash(meta, bank); }),
                        "the bank's own model and bank file pass");
  ok &= expect_true(throws([&] { patchcore::verify_bank_matches_model(meta, bank); }) &&
                        throws([&] { patchcore::verify_bank_hash(meta, model); }),
                    "another model or bank file is rejected");
  // A bank_meta.json without bank_sha256 skips the bank check.
  ok &= expect_true(!throws([&] { patchcore::verify_bank_hash({}, dir / "missing.npy"); }),
                    "a meta without bank_sha256 skips the bank check");
  fs::remove_all(dir);
  return ok;
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_and_input();
  ok &= test_shipped_calibration_settings_are_in_range();
  ok &= test_anomaly_map_covers_the_frame_and_keeps_the_peak_in_place();
  ok &= test_identical_patches_score_zero_and_a_far_one_scores_high();
  ok &= test_reweighting_never_raises_the_plain_max_score();
  ok &= test_coreset_keeps_the_ratio_starting_at_the_seed_patch();
  ok &= test_percentile_threshold_interpolates_like_numpy();
  ok &= test_confidence_is_half_at_the_threshold_and_capped_at_one();
  ok &= test_extract_hwc_transposes_channel_first_output();
  ok &= test_bank_meta_round_trips();
  ok &= test_a_bank_for_another_model_or_bank_file_is_rejected();
  return ok ? 0 : 1;
}
