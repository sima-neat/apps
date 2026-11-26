// Unit tests for the PatchCore anomaly detector: the checks the application leaves to its config,
// and the host-side memory bank on hand-made patch features. tests/python/test_unit.py runs the
// same checks in Python.

#include "../../src/cpp/utils/memory_bank.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
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

bool test_identical_patches_score_zero_and_a_far_one_scores_high() {
  patchcore::PatchEmbeddings normal{2, 2, 4, std::vector<float>(16, 0.0F)};
  for (int patch = 0; patch < 4; ++patch) {
    normal.values[patch * 4 + patch] = 1.0F;
  }
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

bool test_bank_meta_round_trips() {
  patchcore::BankMeta meta;
  meta.model_sha256 = std::string(64, 'a');
  meta.threshold_value = 12.5;
  meta.has_patch_threshold = true;
  meta.patch_threshold_value = 8.25;
  const fs::path path = fs::temp_directory_path() / "patchcore_unit_bank_meta.json";
  patchcore::save_bank_meta(path, meta);
  const auto loaded = patchcore::load_bank_meta(path);
  fs::remove(path);
  return expect_true(loaded.model_sha256 == meta.model_sha256 && loaded.threshold_value == 12.5 &&
                         loaded.has_patch_threshold && loaded.patch_threshold_value == 8.25,
                     "bank_meta.json keeps both thresholds");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_and_input();
  ok &= test_shipped_calibration_settings_are_in_range();
  ok &= test_anomaly_map_covers_the_frame_and_keeps_the_peak_in_place();
  ok &= test_identical_patches_score_zero_and_a_far_one_scores_high();
  ok &= test_bank_meta_round_trips();
  return ok ? 0 : 1;
}
