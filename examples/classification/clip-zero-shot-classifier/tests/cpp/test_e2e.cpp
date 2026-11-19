// E2E test for clip-zero-shot-classifier: an RTSP H.264, H.265 or MJPEG stream in, H.264 video and
// the top prompts out to Insight. The same assertions as tests/python/test_e2e.py.
// main.cpp is compiled in for panel_layout, which places the result panel the test looks for in the
// received video.
#define main clip_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "../../src/cpp/clip_tokenizer.h"
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "clip-zero-shot-classifier";
constexpr int kFrames = 60;
// The same prompt twice, spelled with different case and spacing, must score the same. The stream
// content is not known, so the test does not assert which prompt wins.
const std::vector<std::string> kPrompts = {
    "a photo of people walking", "A Photo of  people   WALKING", "a photo of cars on a street",
    "a photo of a forest",       "a photo of a kitchen",         "a photo of a beach",
};
// Token ids of CLIP ViT-B/16's tokenizer.json, checked against Hugging Face tokenizers.
const std::vector<std::pair<std::string, std::vector<int64_t>>> kRealTokens = {
    {"a photo of a dog.", {320, 1125, 539, 320, 1929, 269}},
    {"A Photo of  TWO   cats!", {320, 1125, 539, 1237, 3989, 256}},
    {"it's route 66, isn't it?", {585, 568, 5261, 277, 277, 267, 2923, 713, 585, 286}},
    {"black-and-white colobus", {1449, 268, 537, 268, 1579, 7300, 2840}},
};

// Empty when every message carries the top prompts the application promises.
std::string top_classes_problem(const std::vector<MetadataJsonMessage>& messages) {
  for (const MetadataJsonMessage& message : messages) {
    const nlohmann::json top = nlohmann::json::parse(message.payload).at("data").at("top_classes");
    if (top.size() != std::min<std::size_t>(3, kPrompts.size())) {
      return "expected 3 top prompts: " + top.dump();
    }
    const nlohmann::json* duplicate = nullptr;
    for (std::size_t i = 0; i < top.size(); ++i) {
      const nlohmann::json& entry = top[i];
      const auto label = entry.at("label").get<std::string>();
      const double confidence = entry.at("confidence").get<double>();
      const double similarity = entry.at("similarity").get<double>();
      if (std::find(kPrompts.begin(), kPrompts.end(), label) == kPrompts.end() ||
          confidence < 0.0 || confidence > 1.0 ||
          (i > 0 && confidence > top[i - 1].at("confidence").get<double>())) {
        return "unsorted or invalid top prompts: " + top.dump();
      }
      // CLIP ViT-B/16 places real images at a cosine of about 0.1 to 0.35 from short captions.
      if (!(similarity > 0.05 && similarity < 0.5)) {
        return "implausible similarity: " + entry.dump();
      }
      if (label == kPrompts[0] || label == kPrompts[1]) {
        if (duplicate != nullptr && (duplicate->at("confidence") != entry.at("confidence") ||
                                     duplicate->at("similarity") != entry.at("similarity"))) {
          return "the same prompt spelled twice scored differently: " + top.dump();
        }
        duplicate = &entry;
      }
    }
    // A uniform result would give every prompt 1/6.
    if (!(top[0].at("confidence").get<double>() > 1.5 / static_cast<double>(kPrompts.size()))) {
      return "the result is close to uniform: " + top.dump();
    }
  }
  return "";
}

// Appends the RTP timestamp of each H.264 video frame's last packet, and the payload of every H.264
// packet, until stop is set.
void read_video_frames(int video_socket, const std::atomic<bool>& stop,
                       std::vector<uint32_t>& frame_timestamps,
                       std::vector<std::vector<unsigned char>>& payloads) {
  const timeval timeout{0, 100000};
  ::setsockopt(video_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  std::array<unsigned char, 65536> packet{};
  while (!stop) {
    const auto bytes = ::recv(video_socket, packet.data(), packet.size(), 0);
    if (bytes > 12 && (packet[0] >> 6) == 2 && (packet[1] & 0x7F) == 96) {
      payloads.emplace_back(packet.begin() + 12 + 4 * (packet[0] & 0x0F), packet.begin() + bytes);
      if ((packet[1] & 0x80) != 0) {
        uint32_t timestamp = 0;
        std::memcpy(&timestamp, packet.data() + 4, sizeof(timestamp));
        frame_timestamps.push_back(ntohl(timestamp));
      }
    }
  }
}

// The last frame of the RTP H.264 payloads, decoded, as I420 rows: Y, then U, then V. Empty when no
// frame decodes.
cv::Mat last_video_frame(const std::vector<std::vector<unsigned char>>& payloads,
                         const fs::path& path) {
  std::ofstream stream(path, std::ios::binary);
  const auto write = [&](const unsigned char* data, std::size_t size, bool start) {
    stream.write(start ? "\0\0\0\1" : "", start ? 4 : 0);
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  };
  for (const std::vector<unsigned char>& payload : payloads) {
    const int kind = payload.empty() ? 0 : payload[0] & 0x1F;
    if (kind == 24) { // STAP-A: NAL units, each after its 2-byte size.
      for (std::size_t at = 1; at + 2 < payload.size();) {
        const std::size_t size = (std::size_t{payload[at]} << 8) | payload[at + 1];
        write(payload.data() + at + 2, std::min(size, payload.size() - at - 2), true);
        at += 2 + size;
      }
    } else if (kind == 28 && payload.size() > 2) {
      // FU-A: one NAL unit split over packets, its header rebuilt from the first.
      const unsigned char header = (payload[0] & 0xE0) | (payload[1] & 0x1F);
      if ((payload[1] & 0x80) != 0) {
        write(&header, 1, true);
      }
      write(payload.data() + 2, payload.size() - 2, false);
    } else if (!payload.empty()) {
      write(payload.data(), payload.size(), true);
    }
  }
  stream.close();
  cv::VideoCapture capture(path.string(), cv::CAP_FFMPEG);
  cv::Mat frame;
  for (cv::Mat image; capture.read(image);) {
    frame = image.clone();
  }
  cv::Mat i420;
  if (!frame.empty()) {
    cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
  }
  return i420;
}

double median(const cv::Mat& area) {
  std::vector<uchar> values;
  for (int y = 0; y < area.rows; ++y) {
    values.insert(values.end(), area.ptr(y), area.ptr(y) + area.cols);
  }
  std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
  return values[values.size() / 2];
}

// Empty when the frame shows the result panel: its box dimmed, and the best row's bar amber where
// the same area above the box is not.
std::string panel_problem(const cv::Mat& i420) {
  const int height = i420.rows * 2 / 3;
  const int width = i420.cols;
  const PanelLayout layout = panel_layout(width, height, 3);
  const cv::Rect& box = layout.box;
  // Dimming leaves luma at most 0.3 x 255 = 77 everywhere but the text and the filled bars, about a
  // sixth of the box. The area above it is no reference: it was as dark in the MJPEG stream's
  // street scene.
  const cv::Mat box_luma = i420(box);
  const double dimmed = static_cast<double>(cv::countNonZero(box_luma <= 90)) / box.area();
  if (dimmed < 0.75) {
    return "only " + cv::format("%.2f", dimmed) + " of the panel box has luma at most 90";
  }
  const cv::Mat u(height / 2, width / 2, CV_8UC1, const_cast<uchar*>(i420.ptr(height)));
  const cv::Mat v(height / 2, width / 2, CV_8UC1,
                  const_cast<uchar*>(i420.ptr(height)) + height / 2 * (width / 2));
  // The best prompt has more than a sixth of the probability, so its bar's first fifth is filled.
  const cv::Rect& bar = layout.rows[0].bar;
  for (const auto& [top, where] :
       {std::pair{bar.y, std::string("bar")},
        std::pair{bar.y - box.height, std::string("area above the box")}}) {
    const cv::Rect area(bar.x / 2, top / 2, (bar.x + bar.width / 5) / 2 - bar.x / 2,
                        (top + bar.height) / 2 - top / 2);
    const double area_u = median(u(area));
    const double area_v = median(v(area));
    const bool amber =
        std::abs(area_u - kAccentYuv[1]) <= 20 && std::abs(area_v - kAccentYuv[2]) <= 20;
    if (amber != (where == "bar")) {
      return "the best row's " + where + " has chroma U " + std::to_string(area_u) + " V " +
             std::to_string(area_v);
    }
  }
  return "";
}

int check_tokenizer(const fs::path& tokenizer_json) {
  const ClipTokenizer tokenizer(tokenizer_json);
  for (const auto& [prompt, ids] : kRealTokens) {
    std::vector<int64_t> expected{49406};
    expected.insert(expected.end(), ids.begin(), ids.end());
    expected.resize(77, 49407);
    if (tokenizer.encode(prompt, 77) != expected) {
      std::cerr << "[FAIL] tokens of '" << prompt << "' differ from CLIP's\n";
      return 1;
    }
  }
  std::cout << "[OK] the tokenizer matches CLIP's token ids\n";
  return 0;
}

int run_case(const std::string& binary, const std::map<std::string, fs::path>& models,
             const StreamSourceCase& source_case, const std::string& rtsp_url) {
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  const std::string output_dir =
      create_test_output_dir(kExampleName, "test_" + source_case.name + "_to_insight");
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path run_dir = fs::path(output_dir).parent_path();
  const fs::path prompts_path = run_dir / "prompts.txt";
  std::ofstream prompts(prompts_path);
  for (const std::string& prompt : kPrompts) {
    prompts << prompt << "\n";
  }
  prompts.close();
  const fs::path config_path = run_dir / "config.yaml";
  write_e2e_config(
      kExampleName, config_path,
      {{"model.image_encoder", fs::absolute(models.at("image_encoder")).string()},
       {"model.text_encoder", fs::absolute(models.at("text_encoder")).string()},
       {"model.tokenizer", fs::absolute(models.at("tokenizer")).string()},
       {"model.token_embeddings", fs::absolute(models.at("token_embeddings")).string()},
       {"prompts.path", prompts_path.string()},
       {"source.rtsp_url", rtsp_url},
       {"source.codec", source_case.codec},
       {"inference.frames", std::to_string(kFrames)},
       {"output.insight.host", "127.0.0.1"},
       {"output.insight.video_port", std::to_string(video_port)},
       {"output.insight.metadata_port", std::to_string(metadata_port)}});

  MetadataJsonListenerOptions options;
  options.host = "127.0.0.1";
  options.base_port = metadata_port;
  options.num_ports = 1;
  options.metadata_type = "classification";
  options.data_array_key = "top_classes";
  MetadataJsonListener listener(options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
    return 1;
  }
  const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in video_address{};
  video_address.sin_family = AF_INET;
  video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  video_address.sin_port = htons(static_cast<uint16_t>(video_port));
  if (video_socket < 0 || ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address),
                                 sizeof(video_address)) != 0) {
    if (video_socket >= 0) {
      ::close(video_socket);
    }
    std::cerr << "[FAIL] could not bind Insight video port\n";
    return 1;
  }

  std::atomic<bool> stop{false};
  std::vector<uint32_t> frame_timestamps;
  std::vector<std::vector<unsigned char>> payloads;
  std::thread reader(read_video_frames, video_socket, std::cref(stop), std::ref(frame_timestamps),
                     std::ref(payloads));
  const ProcessResult process =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);
  stop = true;
  reader.join();
  ::close(video_socket);
  const MetadataJsonListenerResult metadata = listener.drain_pending();
  const std::string context =
      "\nstdout:\n" + process.stdout_text + "\nstderr:\n" + process.stderr_text + "\n";
  if (process.exit_code != 0) {
    std::cerr << "[FAIL] " << source_case.name << " exited with " << process.exit_code << context;
    return 1;
  }
  if (process.stdout_text.find("processed=" + std::to_string(kFrames) + " ") == std::string::npos ||
      process.stdout_text.find(" codec=" + source_case.codec + " prompts=" +
                               std::to_string(kPrompts.size()) + " ") == std::string::npos) {
    std::cerr << "[FAIL] " << source_case.name << " did not classify " << kFrames << " frames"
              << context;
    return 1;
  }
  const std::vector<MetadataJsonMessage>& messages = metadata.messages;
  // UDP on loopback can drop a datagram under load, so most of the frames must arrive, not all.
  if (frame_timestamps.size() < kFrames * 9 / 10) {
    std::cerr << "[FAIL] " << source_case.name << " " << frame_timestamps.size() << " of "
              << kFrames << " H.264 video frames arrived\n";
    return 1;
  }
  if (!metadata.error.empty() || messages.size() < kFrames * 9 / 10) {
    std::cerr << "[FAIL] " << source_case.name << " " << messages.size() << " of " << kFrames
              << " classification messages arrived " << metadata.error << "\n";
    return 1;
  }
  for (std::size_t i = 1; i < messages.size(); ++i) {
    if (std::stoll(messages[i].frame_id) <= std::stoll(messages[i - 1].frame_id) ||
        messages[i].timestamp_ms < messages[i - 1].timestamp_ms) {
      std::cerr << "[FAIL] " << source_case.name << " frame ids or timestamps went backwards\n";
      return 1;
    }
  }
  if (messages.back().timestamp_ms <= messages.front().timestamp_ms) {
    std::cerr << "[FAIL] " << source_case.name << " timestamps did not advance\n";
    return 1;
  }
  // Insight draws a message on the video frame whose 90 kHz RTP timestamp is within 1 ms of the
  // message's.
  const auto matched = std::count_if(messages.begin(), messages.end(), [&](const auto& message) {
    const auto rtp = static_cast<uint32_t>(message.timestamp_ms * 90);
    return std::any_of(frame_timestamps.begin(), frame_timestamps.end(), [&](uint32_t frame) {
      return std::abs(static_cast<int64_t>(frame) - rtp) <= 90;
    });
  });
  if (static_cast<std::size_t>(matched) * 10 < messages.size() * 9) {
    std::cerr << "[FAIL] " << source_case.name << " " << matched << " of " << messages.size()
              << " messages match a video frame\n";
    return 1;
  }
  const cv::Mat i420 = last_video_frame(payloads, run_dir / "video.h264");
  if (i420.empty()) {
    std::cerr << "[FAIL] " << source_case.name << " no frame of the received video decodes\n";
    return 1;
  }
  if (const std::string problem = panel_problem(i420); !problem.empty()) {
    std::cerr << "[FAIL] " << source_case.name << " " << problem << "\n";
    return 1;
  }
  if (const std::string problem = top_classes_problem(messages); !problem.empty()) {
    std::cerr << "[FAIL] " << source_case.name << " " << problem << "\n";
    return 1;
  }
  remove_dir(output_dir);
  std::cout << "[OK] " << source_case.name << " published H.264 video and " << messages.size()
            << " top-prompt messages\n";
  return 0;
}

// The encoders point at missing files, so the codec error can only come first if the stream is
// checked before either model loads.
int run_wrong_codec_case(const std::string& binary, const std::map<std::string, fs::path>& models,
                         const std::string& h264_url) {
  const std::string output_dir = create_test_output_dir(kExampleName, "test_wrong_codec");
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path run_dir = fs::path(output_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  write_e2e_config(
      kExampleName, config_path,
      {{"model.image_encoder", (run_dir / "missing_image_encoder.tar.gz").string()},
       {"model.text_encoder", (run_dir / "missing_text_encoder.tar.gz").string()},
       {"model.tokenizer", fs::absolute(models.at("tokenizer")).string()},
       {"model.token_embeddings", fs::absolute(models.at("token_embeddings")).string()},
       {"prompts.path",
        fs::absolute("examples/classification/clip-zero-shot-classifier/src/common/prompts.txt")
            .string()},
       {"source.rtsp_url", h264_url},
       {"source.codec", "h265"}});

  const ProcessResult process = spawn_and_wait(binary, {"--config", config_path.string()}, 60000);
  const std::string expected =
      h264_url + " is h264, but source.codec is h265; set source.codec to h264";
  if (process.exit_code == 0 || process.stderr_text.find(expected) == std::string::npos) {
    std::cerr << "[FAIL] a wrong source.codec was not rejected before the models load\nstdout:\n"
              << process.stdout_text << "\nstderr:\n"
              << process.stderr_text << "\n";
    return 1;
  }
  remove_dir(output_dir);
  std::cout << "[OK] a wrong source.codec is rejected before the models load\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const char* models_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_raw != nullptr ? models_raw : "models";
  const std::map<std::string, fs::path> models = {
      {"image_encoder", models_dir / "clip_vit_b_16_w8a16_image_encoder_mpk.tar.gz"},
      {"text_encoder", models_dir / "clip_vit_b_16_w8a16_text_encoder_mpk.tar.gz"},
      {"tokenizer", models_dir / "tokenizer.json"},
      {"token_embeddings", models_dir / "token_embeddings.npy"},
  };
  for (const auto& [key, path] : models) {
    if (!fs::exists(path)) {
      return skip_or_fail("missing CLIP ViT-B/16 artifact: " + path.string());
    }
  }
  if (check_tokenizer(models.at("tokenizer")) != 0) {
    return 1;
  }
  const std::vector<StreamSourceCase> source_cases = {
      {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264"},
      {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265"},
      {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg"},
  };
  const std::string binary = argv[1];
  const int rc = run_single_stream_source_cases(
      "CLIP zero-shot classifier", source_cases,
      [&](const StreamSourceCase& source_case, const std::string& rtsp_url) {
        return run_case(binary, models, source_case, rtsp_url);
      });
  if (rc != 0) {
    return rc;
  }
  return run_single_stream_source_cases(
      "CLIP zero-shot classifier with a wrong codec",
      {{"rtsp_h264_as_h265", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h265"}},
      [&](const StreamSourceCase&, const std::string& rtsp_url) {
        return run_wrong_codec_case(binary, models, rtsp_url);
      });
}
