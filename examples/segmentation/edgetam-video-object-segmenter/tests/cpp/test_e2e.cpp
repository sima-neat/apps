// E2E test for edgetam-video-object-segmenter: the shipped desk video in, the blue book's mask and
// position out. The same assertions as tests/python/test_e2e.py.
#include "support/testing/test_checks.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "edgetam-video-object-segmenter";
constexpr const char* kVideo = "assets/datasets/tum-rgbd/freiburg1-desk.mp4";
constexpr int kFrames = 100;
constexpr int kSaveEvery = 10;
// The shipped prompt point is on a blue book. The camera pans, so the book crosses the frame and
// leaves it at frame 48. Each entry is a frame and the book's centre in it, measured on a Modalix
// DevKit.
struct BookCenter {
  int frame;
  double x;
  double y;
};
constexpr BookCenter kBookCenters[] = {{0, 0.885, 0.388}, {20, 0.479, 0.362}, {40, 0.830, 0.473}};
constexpr double kCenterTolerance = 0.06;
constexpr int kLastFrameInView = 40;
constexpr int kFirstFrameOutOfView = 60;

// The four packages under the test model directory, by config key. Empty when one is missing.
std::map<std::string, std::string> model_files(const std::string& models_dir) {
  std::map<std::string, std::string> files;
  for (const char* name : {"image_encoder", "memory_attention", "mask_decoder", "memory_encoder"}) {
    const fs::path path = fs::path(models_dir) /
                          (std::string("edgetam_w16a16_w8a16_w16a16_w8a8_") + name + "_mpk.tar.gz");
    if (!fs::is_regular_file(path)) {
      return {};
    }
    files[std::string("model.") + name] = path.string();
  }
  return files;
}

// Runs the binary on the shipped config with the overrides and returns the frames of its
// results.json, or null when the run failed.
nlohmann::json tracked_frames(const std::string& binary, const std::string& test_name,
                              std::map<std::string, std::string> overrides,
                              const std::string& out_dir) {
  overrides["source.video_path"] = kVideo;
  overrides["output.dir"] = out_dir;
  const fs::path config_path =
      write_e2e_config(kExampleName, fs::path(out_dir).parent_path() / "config.yaml", overrides);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()},
                     env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000));
  if (!expect_true(result.exit_code == 0, test_name + " exits with 0")) {
    std::cerr << "stdout:\n" << result.stdout_text << "\nstderr:\n" << result.stderr_text << "\n";
    return nullptr;
  }
  std::ifstream results(fs::path(out_dir) / "results.json");
  return nlohmann::json::parse(results, nullptr, false).value("frames", nlohmann::json());
}

bool check_follows_the_book(const std::string& binary, std::map<std::string, std::string> models) {
  const std::string name = "follows_the_book_and_reports_when_it_leaves_the_view";
  const std::string out_dir = create_test_output_dir(kExampleName, name);
  models["inference.frames"] = std::to_string(kFrames);
  models["output.save_every"] = std::to_string(kSaveEvery);
  const nlohmann::json frames = tracked_frames(binary, name, models, out_dir);
  if (!expect_true(frames.is_array() && frames.size() == kFrames,
                   "results.json holds one entry per frame")) {
    return false;
  }

  bool in_view = true;
  for (int index = 0; index <= kLastFrameInView; ++index) {
    // The book covers 1.6% to 3.5% of the frame while it is in view.
    const double area = frames[index].at("area").get<double>();
    in_view &= frames[index].at("frame") == index && frames[index].at("visible") == true &&
               area > 0.008 && area < 0.07;
  }
  bool centred = true;
  for (const BookCenter& center : kBookCenters) {
    const nlohmann::json& frame = frames[center.frame];
    centred &= frame.at("center_x").is_number() &&
               std::abs(frame.at("center_x").get<double>() - center.x) < kCenterTolerance &&
               std::abs(frame.at("center_y").get<double>() - center.y) < kCenterTolerance;
  }
  bool out_of_view = true;
  for (int index = kFirstFrameOutOfView; index < kFrames; ++index) {
    out_of_view &= frames[index] == nlohmann::json{{"frame", index},
                                                   {"visible", false},
                                                   {"area", 0.0},
                                                   {"center_x", nullptr},
                                                   {"center_y", nullptr}};
  }
  bool ok = expect_true(in_view, "the book is visible in frames 0 to 40");
  ok &= expect_true(centred, "the mask's centre follows the book across the frame");
  ok &= expect_true(out_of_view, "no object is reported after the book leaves the view");
  bool overlays = true;
  for (int index = 0; index < kFrames; index += kSaveEvery) {
    const cv::Mat image =
        cv::imread((fs::path(out_dir) / cv::format("frame_%06d.jpg", index)).string());
    overlays &= image.size() == cv::Size(640, 480);
  }
  ok &= expect_true(overlays, "every tenth frame is saved with the video's size");
  // The overlays and results.json.
  ok &= expect_true(count_output_files(out_dir) == kFrames / kSaveEvery + 1,
                    "only the selected overlays are saved");
  remove_dir(out_dir);
  return ok;
}

bool check_another_prompt_point(const std::string& binary,
                                std::map<std::string, std::string> models) {
  const std::string name = "another_prompt_point_follows_another_object";
  const std::string out_dir = create_test_output_dir(kExampleName, name);
  // The point is on the monitor, which is far larger than the book.
  models["prompt.x"] = "0.42";
  models["prompt.y"] = "0.2";
  models["inference.frames"] = "5";
  models["output.save_every"] = "0";
  const nlohmann::json frames = tracked_frames(binary, name, models, out_dir);
  if (!expect_true(frames.is_array() && frames.size() == 5, "results.json holds five frames")) {
    return false;
  }

  bool on_monitor = true;
  for (const nlohmann::json& frame : frames) {
    on_monitor &= frame.at("visible") == true && frame.at("area").get<double>() > 0.08 &&
                  std::abs(frame.at("center_x").get<double>() - 0.42) < 0.15 &&
                  std::abs(frame.at("center_y").get<double>() - 0.2) < 0.15;
  }
  bool ok = expect_true(on_monitor, "the mask covers the monitor under the new point");
  ok &= expect_true(count_output_files(out_dir) == 1, "save_every 0 saves no image");
  remove_dir(out_dir);
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::map<std::string, std::string> models = model_files(models_dir ? models_dir : "models");
  if (models.empty()) {
    return skip_or_fail("an EdgeTAM package is missing under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  bool ok = check_follows_the_book(binary, models);
  ok &= check_another_prompt_point(binary, models);
  return ok ? 0 : 1;
}
