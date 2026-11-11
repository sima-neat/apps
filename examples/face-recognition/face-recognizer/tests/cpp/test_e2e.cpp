// E2E test for face-recognizer with the SCRFD and ArcFace models.
//
// The bundled clip (tests/fixtures/face_clip.mp4) shows the two people of the shared COCO
// images 000000493864.jpg (PersonA) and 000000116439.jpg (PersonB) in every frame:
//   1. Enrolling PersonA from an image folder must name PersonA in every frame of the clip
//      and label the other face Unknown.
//   2. Enrolling PersonB into the same gallery must then name both people in every frame.
//   3. Enrolling from the clip itself must write a gallery with the one named person.
//   4. A live source (RTSP, and a USB camera when SIMANEAT_TEST_USB_CAMERA names one) must
//      reach this test, which listens where Insight would, as RTP video with face metadata
//      that carries the timestamp of a received video frame. Insight pairs a result with
//      its frame by that timestamp.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "face-recognizer";
// Tests run from the apps root.
const fs::path kClip = "examples/face-recognition/face-recognizer/tests/fixtures/face_clip.mp4";
const fs::path kCoco = "assets/datasets/coco";
constexpr int kClipFrames = 89;
constexpr int kPairedMessages = 10;

struct Models {
  std::string scrfd;
  std::string arcface;
};

bool run_ok(const std::string& label, const ProcessResult& result) {
  if (result.exit_code == 0) {
    return true;
  }
  std::cerr << "[FAIL] " << label << " exit code " << result.exit_code << "\n";
  std::cerr << "stdout:\n" << result.stdout_text << "\n";
  std::cerr << "stderr:\n" << result.stderr_text << "\n";
  return false;
}

ConfigScalars base_config(const Models& models, const fs::path& gallery) {
  return {{"model.scrfd", models.scrfd},
          {"model.arcface", models.arcface},
          {"gallery.path", gallery.string()}};
}

// Enrolls `image` as `person` into `gallery`, through a folder named after the person.
bool enroll_image(const std::string& binary, const Models& models, const fs::path& work,
                  const fs::path& gallery, const std::string& person, const std::string& image,
                  int timeout_ms) {
  const fs::path images = work / ("images_" + person);
  fs::create_directories(images / person);
  fs::copy_file(kCoco / image, images / person / image);
  ConfigScalars config = base_config(models, gallery);
  config["enrollment.images_dir"] = images.string();
  const fs::path config_path = work / ("enroll_" + person + ".yaml");
  write_e2e_config(kExampleName, config_path, config);

  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--enroll"}, timeout_ms);
  if (!run_ok("enrolling " + person, result)) {
    return false;
  }
  if (result.stdout_text.find("enrolled=1 skipped=0") == std::string::npos) {
    std::cerr << "[FAIL] enrolling " << person << " did not enroll its image\n"
              << "stdout:\n"
              << result.stdout_text << "\n";
    return false;
  }
  std::cout << "[OK] enrolled " << person << " from " << image << "\n";
  return true;
}

// The names printed for each frame of a video-file run, from lines such as
// "frame 3: PersonA 0.972, Unknown 0.412".
std::vector<std::multiset<std::string>> names_per_frame(const std::string& output) {
  std::vector<std::multiset<std::string>> frames;
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    const auto colon = line.find(": ");
    if (line.rfind("frame ", 0) != 0 || colon == std::string::npos) {
      continue;
    }
    std::multiset<std::string> names;
    std::istringstream faces(line.substr(colon + 2));
    std::string face;
    while (std::getline(faces, face, ',')) {
      const auto start = face.find_first_not_of(' ');
      const auto space = face.rfind(' ');
      if (start != std::string::npos && space != std::string::npos && space > start) {
        names.insert(face.substr(start, space - start));
      }
    }
    frames.push_back(names);
  }
  return frames;
}

// Runs the whole clip and requires exactly `expected` names in every frame.
bool recognize_clip(const std::string& binary, const Models& models, const fs::path& work,
                    const fs::path& gallery, const std::multiset<std::string>& expected,
                    const std::string& label, int timeout_ms) {
  ConfigScalars config = base_config(models, gallery);
  config["source.type"] = "video_file";
  config["source.video_file.path"] = kClip.string();
  const fs::path config_path = work / "recognize.yaml";
  write_e2e_config(kExampleName, config_path, config);

  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);
  if (!run_ok(label, result)) {
    return false;
  }
  const auto frames = names_per_frame(result.stdout_text);
  const auto matching = std::count(frames.begin(), frames.end(), expected);
  if (frames.size() != kClipFrames || matching != kClipFrames) {
    std::cerr << "[FAIL] " << label << ": " << matching << " of " << frames.size()
              << " frames named the expected people (clip has " << kClipFrames << " frames)\n"
              << "stdout:\n"
              << result.stdout_text << "\n";
    return false;
  }
  std::cout << "[OK] " << label << " in " << kClipFrames << " of " << kClipFrames << " frames\n";
  return true;
}

bool enroll_video(const std::string& binary, const Models& models, const fs::path& work,
                  int timeout_ms) {
  const fs::path gallery = work / "gallery_video.bin";
  ConfigScalars config = base_config(models, gallery);
  config["enrollment.video_path"] = kClip.string();
  config["enrollment.name"] = "PersonC";
  config["enrollment.sample_every"] = "10";
  const fs::path config_path = work / "enroll_video.yaml";
  write_e2e_config(kExampleName, config_path, config);

  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--enroll"}, timeout_ms);
  if (!run_ok("enrolling from a video", result)) {
    return false;
  }
  // Every tenth of the clip's 89 frames is sampled.
  if (result.stdout_text.find("enrolled=8 skipped=0 people=1") == std::string::npos ||
      !fs::exists(gallery)) {
    std::cerr << "[FAIL] enrolling from a video did not enroll one person from 8 frames\n"
              << "stdout:\n"
              << result.stdout_text << "\n";
    return false;
  }
  std::cout << "[OK] enrolled one person from 8 frames of a video\n";
  return true;
}

int run_live_case(const std::string& binary, const Models& models, const fs::path& work,
                  const fs::path& gallery, const StreamSourceCase& source_case,
                  const std::string& source, int timeout_ms) {
  // This test is the Insight receiver, so the application publishes to loopback.
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  // The case fills its source type's section; a USB case's "URL" is the camera's stable ID.
  const bool usb = source_case.type == "usb";
  ConfigScalars config = base_config(models, gallery);
  config["source.type"] = source_case.type;
  config["source." + source_case.type + (usb ? ".camera" : ".url")] = source;
  config["output.insight.host"] = "127.0.0.1";
  config["output.insight.video_port"] = std::to_string(video_port);
  config["output.insight.metadata_port"] = std::to_string(metadata_port);
  const fs::path config_path = work / (source_case.name + ".yaml");
  write_e2e_config(kExampleName, config_path, config);

  MetadataJsonListenerOptions metadata_options;
  metadata_options.base_port = metadata_port;
  MetadataJsonListener metadata_listener(metadata_options);
  const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in video_address{};
  video_address.sin_family = AF_INET;
  video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  video_address.sin_port = htons(static_cast<uint16_t>(video_port));
  if (!metadata_listener.ok() || video_socket < 0 ||
      ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address), sizeof(video_address)) !=
          0) {
    if (video_socket >= 0) {
      ::close(video_socket);
    }
    std::cerr << "[FAIL] could not listen on the Insight video and metadata ports\n";
    return 1;
  }

  // An RTSP stream is forwarded as RTP H.264 (payload type 96); a USB camera's own JPEG is
  // forwarded as RTP/JPEG (payload type 26).
  const int payload_type = usb ? 26 : 96;
  std::set<uint32_t> video_timestamps;
  MetadataJsonListenerResult metadata;
  // A message pairs with a frame when its millisecond timestamp, in 90 kHz RTP units, is
  // within one millisecond of that frame's RTP timestamp.
  const auto paired_messages = [&] {
    return std::count_if(metadata.messages.begin(), metadata.messages.end(),
                         [&](const MetadataJsonMessage& message) {
                           if (message.timestamp_ms < 0) {
                             return false;
                           }
                           const auto rtp = static_cast<uint32_t>(message.timestamp_ms * 90);
                           const auto next = video_timestamps.lower_bound(rtp >= 90 ? rtp - 90 : 0);
                           return next != video_timestamps.end() && *next <= rtp + 90;
                         });
  };
  // The application runs until stopped; stop it once enough results are paired.
  const ProcessResult result = spawn_until(
      binary, {"--config", config_path.string()},
      [&] {
        std::array<unsigned char, 2048> packet{};
        for (;;) {
          const auto bytes = ::recv(video_socket, packet.data(), packet.size(), MSG_DONTWAIT);
          if (bytes <= 0) {
            break;
          }
          // RTP version 2; bytes 4 to 7 are the timestamp.
          if (bytes > 12 && (packet[0] >> 6) == 2 && (packet[1] & 0x7F) == payload_type) {
            video_timestamps.insert((uint32_t{packet[4]} << 24) | (uint32_t{packet[5]} << 16) |
                                    (uint32_t{packet[6]} << 8) | uint32_t{packet[7]});
          }
        }
        (void)metadata_listener.poll_messages(metadata, 100);
        return !metadata.error.empty() || paired_messages() >= kPairedMessages;
      },
      timeout_ms);
  ::close(video_socket);

  const std::string problem = exit_problem(result);
  if (!problem.empty() || !metadata.error.empty() || paired_messages() < kPairedMessages) {
    std::cerr << "[FAIL] " << source_case.name << " " << problem << " " << metadata.error
              << " video frames=" << video_timestamps.size()
              << " metadata messages=" << metadata.messages.size()
              << " paired=" << paired_messages() << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    return 1;
  }
  std::cout << "[OK] " << source_case.name << " sent RTP " << (usb ? "JPEG" : "H.264")
            << " video and " << paired_messages()
            << " metadata messages paired with their video frames\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_dir_raw ? models_dir_raw : "models";
  const auto& shipped = example_common_config(kExampleName);
  const Models models{
      (models_dir / fs::path(shipped.string_or("model.scrfd", "")).filename()).string(),
      (models_dir / fs::path(shipped.string_or("model.arcface", "")).filename()).string()};
  if (!fs::exists(models.scrfd) || !fs::exists(models.arcface)) {
    return skip_or_fail("SCRFD or ArcFace model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path work = create_test_scratch_dir(kExampleName, "e2e");
  if (work.empty()) {
    return 1;
  }
  const fs::path gallery = work / "gallery.bin";

  bool ok =
      enroll_image(binary, models, work, gallery, "PersonA", "000000493864.jpg", timeout_ms) &&
      recognize_clip(binary, models, work, gallery, {"PersonA", "Unknown"},
                     "PersonA named and the other face Unknown", timeout_ms) &&
      enroll_image(binary, models, work, gallery, "PersonB", "000000116439.jpg", timeout_ms) &&
      recognize_clip(binary, models, work, gallery, {"PersonA", "PersonB"},
                     "PersonA and PersonB named", timeout_ms) &&
      enroll_video(binary, models, work, timeout_ms);
  if (!ok) {
    remove_dir(work.string());
    return 1;
  }

  const int cases_rc = run_single_stream_source_cases(
      "face recognizer", {{"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true}},
      [&](const StreamSourceCase& source_case, const std::string& url) {
        return run_live_case(binary, models, work, gallery, source_case, url, timeout_ms);
      });
  // USB capture needs a camera on the DevKit, which CI runners do not have, so this case runs
  // only when SIMANEAT_TEST_USB_CAMERA names one (see usb_camera_ids()), even under --strict.
  int usb_rc = 0;
  bool usb_ran = false;
  const StreamSourceCase usb_case{"usb_camera", "SIMANEAT_TEST_USB_CAMERA", "usb", "mjpeg", 30,
                                  true};
  if (const char* camera = env_or_null(usb_case.env_key.c_str())) {
    usb_ran = true;
    usb_rc = run_live_case(binary, models, work, gallery, usb_case, camera, timeout_ms);
  } else {
    std::cerr << "[SKIP] set SIMANEAT_TEST_USB_CAMERA to a USB camera ID to run usb_camera e2e\n";
  }
  remove_dir(work.string());
  if (usb_rc != 0) {
    return 1;
  }
  // A camera-only run on a DevKit is a pass, not "no stream URLs configured".
  return usb_ran && cases_rc == 77 ? 0 : cases_rc;
}
