// Unit test for single-stream-instance-segmenter: CLI arg handling and metadata encoding.
#include "../../src/cpp/segmentation_metadata.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

/// Mask head plane whose foreground covers the whole 640x640 letterboxed model input.
cv::Mat full_mask() {
  return cv::Mat(160, 160, CV_8UC1, cv::Scalar(255));
}

instance_seg::MetadataSegment segment_with(const std::string& id, float confidence,
                                           std::size_t points) {
  instance_seg::MetadataSegment segment;
  segment.id = id;
  segment.label = "person";
  segment.confidence = confidence;
  segment.bbox = cv::Rect(0, 0, 64, 64);
  for (std::size_t i = 0; i < points; ++i) {
    segment.polygon.emplace_back(static_cast<int>(i % 64), static_cast<int>(i / 64));
  }
  return segment;
}

bool test_polygon_is_frame_absolute_and_in_bounds() {
  const cv::Size frame_size(1920, 1080);
  const cv::Rect frame_rect(1600, 900, 300, 170);
  const auto polygon = instance_seg::mask_polygon(full_mask(), frame_rect, frame_size, 0.5);

  bool ok = expect_true(polygon.size() >= 3, "polygon has at least three points");
  cv::Rect extent = cv::boundingRect(polygon);
  ok = expect_true(extent.x >= frame_rect.x && extent.y >= frame_rect.y &&
                       extent.br().x <= frame_rect.br().x && extent.br().y <= frame_rect.br().y,
                   "polygon stays inside the detection rectangle") &&
       ok;
  return expect_true(extent.br().x <= frame_size.width && extent.br().y <= frame_size.height,
                     "polygon stays inside the frame") &&
         ok;
}

bool test_polygon_is_empty_without_foreground() {
  const cv::Mat empty_mask(160, 160, CV_8UC1, cv::Scalar(0));
  const auto polygon =
      instance_seg::mask_polygon(empty_mask, cv::Rect(0, 0, 64, 64), cv::Size(640, 640), 0.5);
  return expect_true(polygon.empty(), "a mask below the threshold yields no polygon");
}

bool test_encoded_shape() {
  const auto encoded = instance_seg::encode_segments({segment_with("seg_1", 0.9f, 5)});
  const auto data = json::parse(encoded.data_json);

  bool ok = expect_true(encoded.dropped == 0, "a single segment is not dropped");
  ok = expect_true(data.contains("segments") && data["segments"].size() == 1,
                   "data carries one entry under segments") &&
       ok;
  const auto& entry = data["segments"][0];
  ok = expect_true(entry["mask_format"] == "polygon", "mask_format is polygon") && ok;
  ok = expect_true(entry["bbox"] == json::array({0, 0, 64, 64}), "bbox is [x, y, w, h]") && ok;
  ok = expect_true(entry["mask"].size() == 5 && entry["mask"][0].size() == 2,
                   "mask is a list of [x, y] pairs") &&
       ok;
  return expect_true(entry.contains("id") && entry.contains("label") &&
                         entry.contains("confidence"),
                     "entry carries id, label and confidence") &&
         ok;
}

bool test_budget_drops_lowest_confidence_first() {
  // Several kilobytes of polygon per segment, so only a few of the twelve fit the budget.
  constexpr int kCount = 12;
  std::vector<instance_seg::MetadataSegment> segments;
  for (int i = 0; i < kCount; ++i) {
    segments.push_back(
        segment_with("seg_" + std::to_string(i), 0.5f + 0.01f * static_cast<float>(i), 1000));
  }
  const auto encoded = instance_seg::encode_segments(segments);
  const auto data = json::parse(encoded.data_json);
  const int kept = static_cast<int>(data["segments"].size());

  bool ok = expect_true(encoded.data_json.size() <= instance_seg::kMetadataByteBudget,
                        "payload fits the byte budget");
  ok = expect_true(encoded.dropped > 0 && kept > 0, "an oversized frame drops part of the frame") &&
       ok;
  ok = expect_true(kept + encoded.dropped == kCount,
                   "kept plus dropped accounts for every segment") &&
       ok;

  float lowest_kept = 1.0f;
  for (const auto& entry : data["segments"]) {
    lowest_kept = std::min(lowest_kept, entry["confidence"].get<float>());
  }
  const float expected_lowest = 0.5f + 0.01f * static_cast<float>(kCount - kept);
  return expect_true(std::fabs(lowest_kept - expected_lowest) < 1e-4f,
                     "the kept segments are the highest-confidence ones") &&
         ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  // Test 1: --help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] --help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] --help printed usage\n";
    }
  }

  // Test 2: unknown flag is rejected.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --bogus: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag rejected\n";
    }
  }

  // Test 3: bad config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad config: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path rejected\n";
    }
  }

  failures += test_polygon_is_frame_absolute_and_in_bounds() ? 0 : 1;
  failures += test_polygon_is_empty_without_foreground() ? 0 : 1;
  failures += test_encoded_shape() ? 0 : 1;
  failures += test_budget_drops_lowest_confidence_first() ? 0 : 1;

  return failures > 0 ? 1 : 0;
}
