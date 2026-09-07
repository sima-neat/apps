/**
 * @example usb-camera-object-detector.cpp
 * USB (UVC) camera YOLO26 object detection with Insight output.
 *
 * The camera is a plain GStreamer fragment behind Neat's Custom() escape hatch,
 * because Neat has no V4L2 source node. From there the graph is ordinary Neat:
 *
 *     v4l2src (MJPEG) -> neatdecoder (NV12) -> branch -+-> video_sender -> Insight
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
  std::string device = "/dev/video16";
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
  cfg.device = raw.string_or("source.device", "/dev/video16");
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
  if (cfg.video_port <= 0) {
    throw std::runtime_error("output.insight.video_port must be > 0");
  }
  if (cfg.metadata_port <= 0) {
    throw std::runtime_error("output.insight.metadata_port must be > 0");
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

/**
 * GStreamer fragment for the USB camera. Neat has no V4L2 source node, so this
 * goes through the Custom() escape hatch.
 *
 * io-mode=mmap   zero-copy DMA from the UVC driver; io-mode=rw memcpys every frame.
 * image/jpeg     pins MJPEG. Without it v4l2src negotiates raw YUYV, which USB 2.0
 *                bandwidth limits to ~5 fps at 1080p.
 * queue leaky    drop stale frames rather than stall the camera when the MLA is busy.
 * neatdecoder    SiMa hardware MJPEG decode, emitting NV12 straight into SiMaAI memory
 *                for the CVU and the encoder. This is what `nodes::SimaDecode` with
 *                `SimaDecodeType::MJPEG` generates; it is spelled inline because the
 *                whole camera path is already one `custom()` fragment.
 *                It needs no videoconvert (NV12 is native) and no jpegparse ahead of
 *                it: v4l2src delivers one whole JPEG per buffer, and GStreamer 1.22's
 *                jpegparse cannot read the APP0 segment UVC cameras emit -- it warned
 *                once per frame ("Failed to parse app0 segment") then killed the run
 *                with a media-format error. Measured on a Logitech BRIO at 1080p:
 *
 *                    decoder        1080p30 CPU     1080p60 CPU / fps
 *                    jpegdec        141% of a core  204% / 43.4
 *                    neatdecoder     35% of a core   60% / 46.6
 *
 *                Three runs per arm, spread under 1.5 points; every hardware sample
 *                beat every CPU sample. Neither decoder reaches 60 fps -- that ceiling
 *                is downstream, not here. Do not reintroduce jpegdec or jpegparse.
 *
 * The fragment must not end on a bare caps string: gst_parse_launch reads a trailing
 * `video/x-raw,...` as an element name and fails with `no element "video"`. Ending on
 * a real element keeps the caps a capsfilter.
 */
std::string camera_fragment(const Config& cfg) {
  if (!cfg.override_fragment.empty()) {
    return cfg.override_fragment;
  }

  std::ostringstream fragment;
  fragment << "v4l2src device=" << cfg.device << " io-mode=mmap"
           << " ! image/jpeg,width=" << cfg.width << ",height=" << cfg.height
           << ",framerate=" << cfg.fps << "/1"
           << " ! queue leaky=downstream max-size-buffers=2"
           << " ! neatdecoder sima-allocator-type=2 dec-type=mjpeg dec-fmt=NV12";
  // COCO models lose confidence on inverted scenes; correct the mount before inference.
  // videoflip works downstream of the hardware decoder and measured free (34.4% vs
  // 34.5% of a core without it).
  if (cfg.flip != "none") {
    fragment << " ! videoflip method=" << flip_methods().at(cfg.flip);
  }
  fragment << " ! queue leaky=downstream max-size-buffers=2";
  return fragment.str();
}

std::vector<Box> parse_bbox_payload(const std::vector<uint8_t>& payload, int img_w, int img_h,
                                    int max_detections) {
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
    box.x2 = clamp(static_cast<float>(x + w), img_w);
    box.y2 = clamp(static_cast<float>(y + h), img_h);
    box.score = score;
    box.class_id = static_cast<int>(class_id);
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
    std::cout << "[validate] fragment=" << camera_fragment(cfg) << "\n";
    std::cout << "[validate] configuration OK\n";
    return 0;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

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

    // connect() registers the source; add()ing it as well emits the fragment twice and
    // starts two v4l2src elements on the same device.
    auto source = neat::nodes::Custom(camera_fragment(cfg), neat::InputRole::Source);
    auto branch = neat::graphs::Branch("camera", {"video", "model"});

    neat::Graph graph("usb_camera_object_detector");
    graph.connect(source, branch);
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
    neat::Run run = graph.build(run_options);

    neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = 0;
    metadata_options.metadata_port_base = cfg.metadata_port;
    neat::MetadataSender metadata_sender(metadata_options);

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

    while (!g_stop.load() && (cfg.frames <= 0 || processed < cfg.frames)) {
      neat::Sample sample;
      neat::PullError err;
      const auto pull_start = std::chrono::steady_clock::now();
      const auto status = run.pull("detections", kPullTimeoutMs, sample, &err);
      const auto pull_end = std::chrono::steady_clock::now();

      if (status == neat::PullStatus::Timeout) {
        std::cerr << "[warn] timed out waiting for detections\n";
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        std::cout << "pipeline closed\n";
        break;
      }
      if (status != neat::PullStatus::Ok) {
        throw std::runtime_error("pull failed: " + err.message);
      }

      const auto boxes = parse_bbox_payload(bbox_payload_from_sample(sample), cfg.width, cfg.height,
                                            cfg.max_detections);
      metadata_sender.send_metadata(
          "object-detection", build_metadata_json(boxes, labels, cfg.width, cfg.height),
          sample.pts_ns >= 0 ? static_cast<int64_t>(sample.pts_ns / 1000000) : -1,
          sample.frame_id >= 0 ? std::to_string(sample.frame_id) : std::string());

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

    run.close();
    std::cout << "processed=" << processed << " detections=" << detections
              << " video_sender=" << cfg.insight_host << ":" << cfg.video_port << "\n";
    return g_stop.load() ? 130 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
