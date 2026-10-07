// Unit test for efficientsam3-promptable-segmenter: the same token ids, metadata and overlay
// timing as tests/python/test_unit.py, plus CLI handling.
#define main efficientsam3_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_process.h"

#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << (ok ? "[OK] " : "[FAIL] ") << what << "\n";
  failures += ok ? 0 : 1;
}

const std::vector<std::pair<std::string, std::vector<int64_t>>> kTokens = {
    {"person", {49406, 2533, 49407}},
    {"red car", {49406, 736, 1615, 49407}},
    {"it's a dog!", {49406, 585, 568, 320, 1929, 256, 49407}},
    {"café 2 cups", {49406, 15304, 273, 11463, 49407}},
    {"Traffic   Light", {49406, 3399, 1395, 49407}},
    {"a person wearing a yellow safety vest near the road",
     {49406, 320, 2533, 3309, 320, 4481, 3406, 12473, 2252, 518, 1759, 49407}},
};

const std::string kSegmentsJson =
    R"({"segments":[{"id":"seg_1","label":"red car","confidence":0.9123,"bbox":[390,276,420,180],)"
    R"("mask_format":"polygon","mask":[[400,284],[403,449],[799,447],[795,281]]},)"
    R"({"id":"seg_3","label":"red car","confidence":0.4001,"bbox":[1809,1071,111,9],)"
    R"("mask_format":"polygon","mask":[[1800,1063],[1800,1079],[1919,1079],[1919,1063]]}]})";

constexpr int kQueries = 200;
constexpr int kMaskSide = 192;

// Detections and mask logits of one 1920x1080 frame, as the model returns them.
struct FrameResult {
  std::vector<float> detections = std::vector<float>(kQueries * 6, 0.0F);
  std::vector<float> masks = std::vector<float>(kMaskSide * kMaskSide * kQueries, -10.0F);

  FrameResult() {
    detect(7, {205.0F, 258.0F, 425.0F, 425.0F, 0.91234F});
    fill(7, 50, 80, 40, 80);
    detect(3, {10, 10, 100, 100, 0.25F}); // below min_score
    fill(3, 10, 20, 10, 20);
    detect(12, {500, 500, 600, 600, 0.5F}); // empty mask: skipped, still numbered
    detect(30, {950.0F, 1000.0F, 1010.0F, 1015.0F, 0.40005F}); // box past the frame edge
    fill(30, 180, 192, 180, 192);
  }
  void detect(int query, std::array<float, 5> row) {
    std::copy(row.begin(), row.end(), detections.begin() + query * 6);
  }
  void fill(int query, int y0, int y1, int x0, int x1) {
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        masks[(y * kMaskSide + x) * kQueries + query] = 10.0F;
      }
    }
  }
  std::vector<Segment> segments(const Config& cfg) {
    const cv::Mat rows(kQueries, 6, CV_32F, detections.data());
    const cv::Mat planes(kMaskSide, kMaskSide, CV_32FC(kQueries), masks.data());
    return segments_of(rows, planes, cfg, 1920, 1080);
  }
};

neat::Sample frame(int64_t pts_ms, int64_t frame_id) {
  neat::Sample sample;
  sample.pts_ns = pts_ms * 1'000'000;
  sample.frame_id = frame_id;
  return sample;
}

bool same(const std::vector<OverlayClock::Message>& messages,
          const std::vector<std::tuple<std::string, int64_t, std::string>>& expected) {
  if (messages.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& [data, timestamp_ms, frame_id] = expected[i];
    if (messages[i].data != data || messages[i].timestamp_ms != timestamp_ms ||
        messages[i].frame_id != frame_id) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  const fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);

  const Config packaged = load_config(config_path);
  check(packaged.prompt == "person" && packaged.min_score == 0.3 && packaged.max_detections == 20 &&
            packaged.mask_threshold == 0.5 && packaged.metadata_port == 9100,
        "packaged config loads");

  const ClipTokenizer tokenizer(config_path.parent_path() / "bpe_simple_vocab_16e6.txt.gz");
  for (const auto& [prompt, ids] : kTokens) {
    std::vector<int64_t> expected = ids;
    expected.resize(kTextTokens, 0);
    const std::vector<int64_t> actual = tokenizer.encode(prompt, kTextTokens);
    std::string got;
    for (const int64_t id : actual) {
      got += " " + std::to_string(id);
    }
    check(actual == expected, "tokens of '" + prompt + "':" + got);
  }
  std::string twenty_cars;
  for (int i = 0; i < 20; ++i) {
    twenty_cars += "car ";
  }
  const std::vector<int64_t> long_prompt = tokenizer.encode(twenty_cars, kTextTokens);
  check(long_prompt.size() == kTextTokens && long_prompt.front() == 49406 &&
            long_prompt.back() == 49407,
        "long prompts are truncated and keep the end token");

  Config cfg = packaged;
  cfg.prompt = "red car";
  FrameResult result;
  check(segments_json(result.segments(cfg)) == kSegmentsJson, "segments match the Python ones");
  cfg.max_detections = 1;
  const std::vector<Segment> best = result.segments(cfg);
  check(best.size() == 1 && best.front().id == "seg_1", "max_detections keeps the best scores");

  const cv::Mat empty(kMaskSide, kMaskSide, CV_32F, cv::Scalar(-10.0F));
  check(mask_outline(empty, {100, 100, 300, 300}, 1920, 1080, 0.0F).empty(),
        "an empty mask has no outline");

  OverlayClock clock;
  for (const auto& [pts_ms, frame_id] : {std::pair{0, 1}, {33, 2}, {66, 3}, {100, 4}}) {
    clock.add_frame(frame(pts_ms, frame_id));
  }
  const bool first = same(clock.add_result(33, "A"), {{"A", 0, "1"}, {"A", 33, "2"}});
  clock.add_frame(frame(133, 5));
  clock.add_frame(frame(166, 6));
  // Frame 66 is nearer the result at 33, frame 100 nearer the one at 133; frame 166 waits.
  const bool second =
      same(clock.add_result(133, "B"), {{"A", 66, "3"}, {"B", 100, "4"}, {"B", 133, "5"}});
  check(first && second, "every frame is sent the nearest result");

  const ProcessResult help = spawn_and_wait(binary, {"--help"}, 20000);
  check(help.exit_code == 0 && help.stdout_text.find("Usage") != std::string::npos,
        "--help prints usage");
  const ProcessResult bogus = spawn_and_wait(binary, {"--bogus"}, 20000);
  check(bogus.exit_code != 0 && bogus.stderr_text.find("unknown argument") != std::string::npos,
        "unknown flags are rejected");
  const ProcessResult no_path = spawn_and_wait(binary, {"--config"}, 20000);
  check(no_path.exit_code != 0 &&
            no_path.stderr_text.find("--config requires a path") != std::string::npos,
        "--config without a path is rejected");

  return failures > 0 ? 1 : 0;
}
