/**
 * @example usb-camera-object-detector.cpp
 * USB (UVC) camera YOLO26 object detection with Insight output.
 *
 * OpenCV captures compressed MJPEG through V4L2. Neat input and decoder nodes
 * turn each encoded sample into NV12 before branching:
 *
 *     Input -> JpegParse -> SimaDecode(MJPEG) -> branch -+-> video_sender -> Insight
 *                                                   `-> model -> detections
 *
 * Both branches stay inside one Run so the encoder and the detections share a
 * GStreamer timeline; Insight correlates the RTP timestamp with the metadata
 * timestamp and cannot render overlays if they drift apart.
 *
 * Usage: usb-camera-object-detector [--config <path>] [--validate-config-only]
 */
#include "neat.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <thread>
#include <mutex>
#include <exception>
#include <opencv2/videoio.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kDefaultWidth = 1920;
constexpr int kDefaultHeight = 1080;
constexpr int kDefaultFps = 30;
constexpr float kDefaultMinScore = 0.30f;
constexpr float kDefaultNmsIou = 0.50f;
constexpr int kDefaultMaxDetections = 100;
constexpr int kDefaultProfileInterval = 100;
constexpr int kDefaultQueueDepth = 3;
constexpr int kDefaultVideoPort = 9000;
constexpr int kDefaultMetadataPort = 9100;
constexpr int kDefaultBitrateKbps = 4000;
constexpr int kPullTimeoutMs = 20000;
constexpr std::size_t kBboxRecordSize = 24;

std::atomic<bool> g_stop{false};

void handle_signal(int) {
  g_stop.store(true);
}

struct Config {
  std::string model_path;
  std::string labels_path;
  std::string device;
  int width = kDefaultWidth;
  int height = kDefaultHeight;
  int fps = kDefaultFps;
  std::string flip = "none";
  std::string override_fragment;
  int frames = 0;
  float min_score = kDefaultMinScore;
  float nms_iou = kDefaultNmsIou;
  int max_detections = kDefaultMaxDetections;
  bool profile = false;
  int profile_interval = kDefaultProfileInterval;
  int queue_depth = kDefaultQueueDepth;
  std::string insight_host;
  int video_port = kDefaultVideoPort;
  int metadata_port = kDefaultMetadataPort;
  int bitrate_kbps = kDefaultBitrateKbps;
};

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = 0;
};

// `videoflip` methods, keyed by the config spelling.
const std::map<std::string, std::string>& flip_methods() {
  static const std::map<std::string, std::string> kMethods = {
      {"none", ""},
      {"rotate-180", "rotate-180"},
      {"horizontal-flip", "horizontal-flip"},
      {"vertical-flip", "vertical-flip"}};
  return kMethods;
}

std::string parse_flip(const std::string& value) {
  std::string lowered = sima_examples::trim_copy(value);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (flip_methods().count(lowered) == 0) {
    throw std::runtime_error(
        "source.flip must be one of horizontal-flip, none, rotate-180, vertical-flip");
  }
  return lowered;
}

Config load_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);

  Config cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels",
                                  "examples/object-detection/usb-camera-object-detector/src/common/"
                                  "coco_label.txt");
  cfg.device = raw.string_or("source.device", "");
  cfg.width = raw.int_or("source.width", kDefaultWidth);
  cfg.height = raw.int_or("source.height", kDefaultHeight);
  cfg.fps = raw.int_or("source.fps", kDefaultFps);
  cfg.flip = parse_flip(raw.string_or("source.flip", "none"));
  cfg.override_fragment = raw.string_or("source.override_fragment", "");
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = static_cast<float>(raw.double_or("inference.min_score", kDefaultMinScore));
  cfg.nms_iou = static_cast<float>(raw.double_or("inference.nms_iou", kDefaultNmsIou));
  cfg.max_detections = raw.int_or("inference.max_detections", kDefaultMaxDetections);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", kDefaultProfileInterval);
  cfg.queue_depth = raw.int_or("runtime.queue_depth", kDefaultQueueDepth);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", kDefaultVideoPort);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", kDefaultMetadataPort);
  cfg.bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", kDefaultBitrateKbps);

  if (cfg.model_path.empty()) {
    throw std::runtime_error("model.path must be set to a compiled model package");
  }
  if (cfg.labels_path.empty()) {
    throw std::runtime_error("model.labels must point to a labels file");
  }
  if (cfg.device.empty() && cfg.override_fragment.empty()) {
    throw std::runtime_error("source.device must be set");
  }
  if (cfg.width <= 0) {
    throw std::runtime_error("source.width must be > 0");
  }
  if (cfg.height <= 0) {
    throw std::runtime_error("source.height must be > 0");
  }
  if (cfg.width % 2 || cfg.height % 2) {
    throw std::runtime_error("source.width and source.height must be even for NV12");
  }
  if (cfg.fps <= 0) {
    throw std::runtime_error("source.fps must be > 0");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("inference.frames must be >= 0");
  }
  if (cfg.min_score < 0.0f || cfg.min_score > 1.0f) {
    throw std::runtime_error("inference.min_score must be in [0.0, 1.0]");
  }
  if (cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
    throw std::runtime_error("inference.nms_iou must be in [0.0, 1.0]");
  }
  if (cfg.max_detections <= 0) {
    throw std::runtime_error("inference.max_detections must be > 0");
  }
  if (cfg.profile_interval <= 0) {
    throw std::runtime_error("runtime.profile_interval must be > 0");
  }
  if (cfg.queue_depth <= 0) {
    throw std::runtime_error("runtime.queue_depth must be > 0");
  }
  if (cfg.insight_host.empty()) {
    throw std::runtime_error("output.insight.host must be set");
  }
  if (cfg.video_port <= 0 || cfg.video_port > 65535) {
    throw std::runtime_error("output.insight.video_port must be in [1, 65535]");
  }
  if (cfg.metadata_port <= 0 || cfg.metadata_port > 65535) {
    throw std::runtime_error("output.insight.metadata_port must be in [1, 65535]");
  }
  if (cfg.bitrate_kbps <= 0) {
    throw std::runtime_error("output.insight.bitrate_kbps must be > 0");
  }
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream input(labels_path);
  if (!input.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = sima_examples::trim_copy(line);
    if (!trimmed.empty()) {
      labels.push_back(trimmed);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

std::string camera_caps(const Config& cfg) {
  return "image/jpeg,width=" + std::to_string(cfg.width) + ",height=" +
         std::to_string(cfg.height) + ",framerate=" + std::to_string(cfg.fps) + "/1";
}

std::string source_description(const Config& cfg) {
  if (!cfg.override_fragment.empty()) return cfg.override_fragment;
  return "V4L2 device=" + cfg.device + " caps=" + camera_caps(cfg) +
         " -> Input -> JpegParse -> SimaDecode(MJPEG,NV12) flip=" + cfg.flip;
}

// Owns capture only. Neat graph construction remains in the application entrypoint.
class UsbCamera {
public:
  explicit UsbCamera(const Config& opt) : capture_(opt.device, cv::CAP_V4L2) {
    if (!capture_.isOpened()) {
      throw std::runtime_error("Cannot open USB camera " + opt.device);
    }
    const auto mjpg = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    capture_.set(cv::CAP_PROP_FOURCC, mjpg);
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, opt.width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, opt.height);
    capture_.set(cv::CAP_PROP_FPS, opt.fps);
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 2);
    if (static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC)) != mjpg ||
        !capture_.set(cv::CAP_PROP_CONVERT_RGB, 0)) {
      throw std::runtime_error("USB camera must provide compressed MJPEG without CPU decode");
    }
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    if (static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH)) != opt.width ||
        static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT)) != opt.height ||
        !std::isfinite(fps) || std::abs(fps - opt.fps) > 0.1) {
      throw std::runtime_error("USB camera did not negotiate the requested resolution/frame rate");
    }
    caps = "image/jpeg,width=" + std::to_string(opt.width) +
           ",height=" + std::to_string(opt.height) + ",framerate=" + std::to_string(opt.fps) + "/1";
  }

  void close() { capture_.release(); }

  std::vector<std::uint8_t> read(std::stop_token stop = {}) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    // Do not feed a truncated JPEG to the request/response decoder: jpegparse
    // can wait for another buffer while the application waits for its output.
    for (int attempt = 0; attempt < 8; ++attempt) {
      cv::Mat frame;
      bool ok = false;
      // waitAny handles the first grab too. A stopped camera cannot strand
      // startup in an unbounded read before the producer thread exists.
      std::vector<int> ready;
      while (!stop.stop_requested() && !g_stop.load()) {
        if (cv::VideoCapture::waitAny({capture_}, ready, 100000000) && !ready.empty()) break;
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("USB camera timed out waiting for a frame");
      }
      if (stop.stop_requested() || g_stop.load()) return {};
      ok = capture_.retrieve(frame);
      if (!ok || frame.empty()) {
        throw std::runtime_error("USB camera stopped delivering frames");
      }
      const auto size = frame.total() * frame.elemSize();
      if (!frame.isContinuous() || size < 2 || frame.data[0] != 0xff || frame.data[1] != 0xd8) {
        throw std::runtime_error("USB camera returned invalid MJPEG data");
      }
      if (frame.data[size - 2] == 0xff && frame.data[size - 1] == 0xd9) {
        return {frame.data, frame.data + size};
      }
      std::cerr << "USB: discarded incomplete MJPEG frame (" << size
                << " bytes; dropped_total=" << ++dropped_frames_ << ")\n";
    }
    throw std::runtime_error("USB camera returned 8 consecutive incomplete MJPEG frames");
  }

  std::string caps;

private:
  std::uint64_t dropped_frames_ = 0;
  cv::VideoCapture capture_;
};

std::vector<Box> parse_bbox_payload(const std::vector<uint8_t>& payload, int img_w, int img_h,
                                    int max_detections, float min_score) {
  std::vector<Box> boxes;
  if (payload.size() < sizeof(uint32_t)) {
    return boxes;
  }

  uint32_t declared = 0;
  std::memcpy(&declared, payload.data(), sizeof(declared));
  std::size_t count =
      std::min<std::size_t>(declared, (payload.size() - sizeof(uint32_t)) / kBboxRecordSize);
  if (max_detections > 0) {
    count = std::min<std::size_t>(count, static_cast<std::size_t>(max_detections));
  }

  boxes.reserve(count);
  const uint8_t* base = payload.data() + sizeof(uint32_t);
  for (std::size_t i = 0; i < count; ++i) {
    int32_t x = 0;
    int32_t y = 0;
    int32_t w = 0;
    int32_t h = 0;
    float score = 0.0f;
    int32_t class_id = 0;
    const uint8_t* record = base + i * kBboxRecordSize;
    std::memcpy(&x, record + 0, sizeof(x));
    std::memcpy(&y, record + 4, sizeof(y));
    std::memcpy(&w, record + 8, sizeof(w));
    std::memcpy(&h, record + 12, sizeof(h));
    std::memcpy(&score, record + 16, sizeof(score));
    std::memcpy(&class_id, record + 20, sizeof(class_id));

    const auto clamp = [](float value, int limit) {
      return std::clamp(value, 0.0f, static_cast<float>(limit));
    };
    Box box;
    box.x1 = clamp(static_cast<float>(x), img_w);
    box.y1 = clamp(static_cast<float>(y), img_h);
    box.x2 = clamp(static_cast<float>(static_cast<int64_t>(x) + w), img_w);
    box.y2 = clamp(static_cast<float>(static_cast<int64_t>(y) + h), img_h);
    box.score = score;
    box.class_id = static_cast<int>(class_id);
    if (!(score >= min_score && score <= 1.0f) || class_id < 0 ||
        box.x2 <= box.x1 || box.y2 <= box.y1) continue;
    boxes.push_back(box);
  }
  return boxes;
}

std::string class_label(int class_id, const std::vector<std::string>& labels) {
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < labels.size()) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  return "unknown";
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

// Insight's object-detection contract: bbox is [x, y, w, h], clamped to the frame.
std::string build_metadata_json(const std::vector<Box>& boxes,
                                const std::vector<std::string>& labels, int frame_w, int frame_h) {
  std::ostringstream out;
  out << "{\"objects\":[";
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const Box& box = boxes[i];
    int x = std::max(0, static_cast<int>(box.x1));
    int y = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x + w > frame_w) {
      w = frame_w - x;
    }
    if (y + h > frame_h) {
      h = frame_h - y;
    }
    if (i > 0) {
      out << ',';
    }
    out << "{\"id\":\"obj_" << (i + 1) << "\",\"label\":\""
        << json_escape(class_label(box.class_id, labels)) << "\",\"confidence\":" << box.score
        << ",\"bbox\":[" << x << ',' << y << ',' << std::max(0, w) << ',' << std::max(0, h) << "]}";
  }
  out << "]}";
  return out.str();
}

std::unique_ptr<neat::Model> make_model(const Config& cfg) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  opt.preprocess.input_max_width = cfg.width;
  opt.preprocess.input_max_height = cfg.height;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = neat::BoxDecodeType::YoloV26;
  opt.score_threshold = cfg.min_score;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.max_detections;
  return std::make_unique<neat::Model>(cfg.model_path, opt);
}

groups::VideoSenderOptions make_video_options(const Config& cfg) {
  auto opt = groups::VideoSenderOptions::H264RtpUdpFromRaw(cfg.width, cfg.height, cfg.fps);
  opt.host = cfg.insight_host;
  opt.channel = 0;
  opt.video_port_base = cfg.video_port;
  opt.encoder.bitrate_kbps = cfg.bitrate_kbps;
  return opt;
}

// Reuses the repository's shared BBOX extractor so every detection example
// unpacks the same Sample shapes the same way.
std::vector<uint8_t> bbox_payload_from_sample(const neat::Sample& sample) {
  std::vector<uint8_t> payload;
  std::string err;
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    neat::Sample tensor_sample = sample;
    tensor_sample.kind = neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    if (!objdet::extract_bbox_payload(tensor_sample, payload, err)) {
      throw std::runtime_error("failed to extract detections: " + err);
    }
    return payload;
  }
  if (!objdet::extract_bbox_payload(sample, payload, err)) {
    throw std::runtime_error("failed to extract detections: " + err);
  }
  return payload;
}

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [--config <path>] [--validate-config-only]\n"
            << "  --config <path>          Path to YAML configuration\n"
            << "  --validate-config-only   Validate the configuration and exit\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  bool validate_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --config requires a path\n";
        return 1;
      }
      config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      validate_only = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Error: unknown argument: " << arg << "\n";
      return 1;
    }
  }

  if (!fs::exists(config_path)) {
    std::cerr << "Error: config file not found: " << config_path.string() << "\n";
    return 1;
  }

  Config cfg;
  std::vector<std::string> labels;
  try {
    cfg = load_config(config_path);
    labels = load_labels(cfg.labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  if (validate_only) {
    const std::string source_label = cfg.override_fragment.empty() ? cfg.device : "override";
    std::cout << "[validate] model=" << cfg.model_path << " classes=" << labels.size()
              << " source=" << source_label << " stream=" << cfg.width << "x" << cfg.height << "@"
              << cfg.fps << " flip=" << cfg.flip << " min_score=" << cfg.min_score
              << " nms_iou=" << cfg.nms_iou << " max_detections=" << cfg.max_detections
              << " queue_depth=" << cfg.queue_depth << " insight=" << cfg.insight_host << ":"
              << cfg.video_port << "/" << cfg.metadata_port << "\n";
    std::cout << "[validate] source_pipeline=" << source_description(cfg) << "\n";
    std::cout << "[validate] configuration OK\n";
    return 0;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  std::signal(SIGHUP, handle_signal);

  try {
    auto model = make_model(cfg);

    neat::Graph video_graph("video");
    video_graph.connect(neat::nodes::Input("video"), groups::VideoSender(make_video_options(cfg)));

    neat::Graph model_graph("model");
    model_graph.connect(neat::nodes::Input("model"), *model);

    neat::Graph detections_graph("detections");
    detections_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));

    // RealtimeLatestByStream: if one branch falls behind, drop its stale frames rather
    // than back-pressuring the camera. The video branch must never stall the MLA.
    neat::GraphLinkOptions live;
    live.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

    std::unique_ptr<UsbCamera> camera;
    neat::Sample seed;
    neat::Graph source_graph("capture_decode");
    if (!cfg.override_fragment.empty()) {
      source_graph.add(neat::nodes::Custom(cfg.override_fragment, neat::InputRole::Source));
    } else {
      camera = std::make_unique<UsbCamera>(cfg);
      neat::InputOptions ingress;
      ingress.payload_type = neat::PayloadType::Encoded;
      ingress.caps_override = camera->caps;
      ingress.memory_policy = neat::InputMemoryPolicy::SystemMemory;
      ingress.do_timestamp = false;
      neat::SimaDecodeOptions decode;
      decode.type = neat::SimaDecodeType::MJPEG;
      decode.raw_output = true;
      decode.out_format = "NV12";
      decode.dec_width = cfg.width;
      decode.dec_height = cfg.height;
      decode.dec_fps = cfg.fps;
      source_graph.add(neat::nodes::Input("jpeg", ingress));
      source_graph.add(neat::nodes::JpegParse());
      source_graph.add(neat::nodes::SimaDecode(decode));
      auto first_frame = camera->read();
      if (g_stop.load()) return 130;
      seed = neat::make_encoded_sample(std::move(first_frame), camera->caps, 0, -1,
                                       1000000000LL / cfg.fps);
    }
    if (cfg.flip != "none") {
      source_graph.add(neat::nodes::Custom("videoflip method=" + flip_methods().at(cfg.flip)));
    }
    auto branch = neat::graphs::Branch("camera", {"video", "model"});
    neat::Graph graph("usb_camera_object_detector");
    graph.connect(source_graph, branch);
    graph.connect(branch, video_graph, live);
    graph.connect(branch, model_graph, live);
    graph.connect(model_graph, detections_graph);

    if (cfg.profile) {
      std::cout << "Backend:\n" << graph.describe_backend() << "\n";
    }

    neat::RunOptions run_options;
    run_options.preset = neat::RunPreset::Realtime;
    run_options.queue_depth = cfg.queue_depth;
    run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;
    run_options.input_timeout_ms = kPullTimeoutMs;
    neat::Run run = camera ? graph.build(seed, run_options) : graph.build(run_options);

    neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = 0;
    metadata_options.metadata_port_base = cfg.metadata_port;
    std::string metadata_error;
    neat::MetadataSender metadata_sender(metadata_options, &metadata_error);
    if (!metadata_sender.ok()) {
      throw std::runtime_error("metadata sender initialization failed: " + metadata_error);
    }

    const std::string source_label = cfg.override_fragment.empty() ? cfg.device : "override";
    std::cout << "source=" << source_label << " stream=" << cfg.width << "x" << cfg.height << "@"
              << cfg.fps << " model=" << cfg.model_path << " insight=" << cfg.insight_host
              << " video=" << cfg.video_port << " metadata=" << metadata_sender.metadata_port()
              << " channel=0\n";

    int processed = 0;
    int detections = 0;
    int window_frames = 0;
    int window_boxes = 0;
    double window_pull_ms = 0.0;
    auto window_start = std::chrono::steady_clock::now();

    std::mutex capture_error_mutex;
    std::exception_ptr capture_error;
    std::jthread producer;
    if (camera) {
      producer = std::jthread([&](std::stop_token stop) {
        const auto capture_start = std::chrono::steady_clock::now();
        int64_t frame_id = 0;
        try {
          while (!stop.stop_requested() && !g_stop.load()) {
            neat::Sample encoded;
            if (frame_id == 0) {
              encoded = seed;
            } else {
              auto bytes = camera->read(stop);
              if (bytes.empty()) break;
              const auto pts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - capture_start).count();
              encoded = neat::make_encoded_sample(std::move(bytes), camera->caps, pts, -1,
                                                  1000000000LL / cfg.fps);
            }
            encoded.frame_id = frame_id++;
            encoded.stream_id = "camera";
            if (stop.stop_requested()) break;
            if (!run.push(encoded)) throw std::runtime_error("USB decoder rejected JPEG input");
          }
        } catch (...) {
          if (!stop.stop_requested()) {
            std::lock_guard lock(capture_error_mutex);
            capture_error = std::current_exception();
          }
        }
        camera->close();
      });
    }
    auto check_capture_error = [&] {
      std::lock_guard lock(capture_error_mutex);
      if (capture_error) std::rethrow_exception(capture_error);
    };
    auto stop_capture = [&] {
      producer.request_stop();
      run.close();
      if (producer.joinable()) producer.join();
    };
    auto last_output = std::chrono::steady_clock::now();
    try {
      while (!g_stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
        check_capture_error();
        neat::Sample sample;
        neat::PullError err;
        const auto pull_start = std::chrono::steady_clock::now();
        const auto status = run.pull("detections", camera ? 200 : kPullTimeoutMs, sample, &err);
        const auto pull_end = std::chrono::steady_clock::now();

        if (status == neat::PullStatus::Timeout) {
          if (pull_end - last_output >= std::chrono::milliseconds(kPullTimeoutMs))
            throw std::runtime_error("timed out waiting for detections");
          continue;
        }
        last_output = pull_end;
        if (status == neat::PullStatus::Closed) {
          std::cout << "pipeline closed\n";
          break;
        }
        if (status != neat::PullStatus::Ok) {
          throw std::runtime_error("pull failed: " + err.message);
        }

        const auto boxes = parse_bbox_payload(bbox_payload_from_sample(sample), cfg.width, cfg.height,
                                              cfg.max_detections, cfg.min_score);
        if (!metadata_sender.send_metadata(
            "object-detection", build_metadata_json(boxes, labels, cfg.width, cfg.height),
            sample.pts_ns >= 0 ? static_cast<int64_t>(sample.pts_ns / 1000000) : -1,
            sample.frame_id >= 0 ? std::to_string(sample.frame_id) : std::string(),
            &metadata_error)) {
          throw std::runtime_error("metadata send failed: " + metadata_error);
        }

        ++processed;
        detections += static_cast<int>(boxes.size());

        if (cfg.profile) {
          using ms = std::chrono::duration<double, std::milli>;
          ++window_frames;
          window_boxes += static_cast<int>(boxes.size());
          window_pull_ms += ms(pull_end - pull_start).count();
          if (window_frames >= cfg.profile_interval) {
            const double elapsed = std::chrono::duration<double>(pull_end - window_start).count();
            std::cout << "[profile] frames=" << window_frames
                      << " output_fps=" << (elapsed > 0.0 ? window_frames / elapsed : 0.0)
                      << " avg_detection_pull_ms=" << window_pull_ms / window_frames
                      << " avg_boxes=" << static_cast<double>(window_boxes) / window_frames << "\n";
            window_frames = 0;
            window_boxes = 0;
            window_pull_ms = 0.0;
            window_start = pull_end;
          }
        }
      }

    } catch (...) {
      stop_capture();
      throw;
    }
    stop_capture();
    check_capture_error();
    std::cout << "processed=" << processed << " detections=" << detections
              << " video_sender=" << cfg.insight_host << ":" << cfg.video_port << "\n";
    return g_stop.load() ? 130 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
