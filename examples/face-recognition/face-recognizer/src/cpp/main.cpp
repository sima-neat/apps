// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/pull_status.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "align.h"
#include "gallery.h"
#include "match.h"
#include "scrfd_decode.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kPullTimeoutMs = 50;
constexpr const char* kOutputName = "recognizer_output";

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

enum class SourceType { Rtsp, Usb, VideoFile };

struct CliOptions {
  fs::path config_path;
  bool enroll = false;
  bool validate_config_only = false;
};

struct AppConfig {
  std::string scrfd_model;
  std::string arcface_model;
  fs::path gallery_path;
  SourceType source = SourceType::Rtsp;
  std::string rtsp_url;
  bool rtsp_tcp = true;
  int rtsp_latency_ms = 100;
  int rtsp_fps = 0;
  std::string usb_camera;
  int usb_width = 1280;
  int usb_height = 720;
  int usb_fps = 30;
  std::string video_path;
  int frames = 0;
  face_recog::ScrfdConfig detector;
  face_recog::MatchConfig match;
  std::string enroll_images_dir;
  std::string enroll_video_path;
  std::string enroll_name;
  int enroll_sample_every = 5;
  double enroll_min_score = 0.75;
  int enroll_max_per_person = 0;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
};

// The frame geometry of the opened source, known before the graph is built.
struct Source {
  int width = 0;
  int height = 0;
  int fps = 0;
  std::optional<neat::UsbCameraInfo> usb;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--enroll") {
      options.enroll = true;
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--config <path>] [--enroll] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

SourceType parse_source_type(const std::string& value) {
  if (value == "rtsp") {
    return SourceType::Rtsp;
  }
  if (value == "usb") {
    return SourceType::Usb;
  }
  if (value == "video_file") {
    return SourceType::VideoFile;
  }
  throw std::runtime_error("source.type must be rtsp, usb or video_file");
}

bool is_live(const AppConfig& cfg) {
  return cfg.source != SourceType::VideoFile;
}

void validate_config(const AppConfig& cfg, bool enroll) {
  using sima_examples::require;
  require(!cfg.scrfd_model.empty(), "model.scrfd must be set");
  require(!cfg.arcface_model.empty(), "model.arcface must be set");
  require(!cfg.gallery_path.empty(), "gallery.path must be set");
  require(cfg.detector.min_score > 0.0f && cfg.detector.min_score < 1.0f,
          "inference.min_score must be greater than 0 and less than 1");
  require(cfg.detector.nms_iou >= 0.0f && cfg.detector.nms_iou <= 1.0f,
          "inference.nms_iou must be between 0 and 1");
  require(cfg.detector.max_faces > 0, "inference.max_faces must be > 0");

  if (enroll) {
    require(!cfg.enroll_images_dir.empty() || !cfg.enroll_video_path.empty(),
            "enrollment.images_dir or enrollment.video_path must be set");
    require(cfg.enroll_video_path.empty() || !cfg.enroll_name.empty(),
            "enrollment.name must be set with enrollment.video_path");
    require(cfg.enroll_sample_every > 0, "enrollment.sample_every must be > 0");
    require(cfg.enroll_min_score >= 0.0 && cfg.enroll_min_score <= 1.0,
            "enrollment.min_score must be between 0 and 1");
    require(cfg.enroll_max_per_person >= 0, "enrollment.max_per_person must be >= 0");
    return;
  }

  require(cfg.frames >= 0, "inference.frames must be >= 0");
  require(cfg.match.threshold >= 0.0f && cfg.match.threshold <= 1.0f,
          "match.threshold must be between 0 and 1");
  require(cfg.match.margin >= 0.0f && cfg.match.margin <= 1.0f,
          "match.margin must be between 0 and 1");
  switch (cfg.source) {
  case SourceType::Rtsp:
    require(!cfg.rtsp_url.empty(), "source.rtsp.url must be set when source.type=rtsp");
    require(cfg.rtsp_latency_ms >= 0, "source.rtsp.latency_ms must be >= 0");
    require(cfg.rtsp_fps >= 0, "source.rtsp.fps must be >= 0");
    break;
  case SourceType::Usb:
    require(!cfg.usb_camera.empty(), "source.usb.camera must be set when source.type=usb");
    require(cfg.usb_width > 0 && cfg.usb_height > 0 && cfg.usb_fps > 0,
            "source.usb.width, height and fps must be > 0");
    // Insight receives the camera's JPEG as RTP/JPEG, which carries 8-pixel blocks and at
    // most 2040 pixels per side.
    require(cfg.usb_width % 8 == 0 && cfg.usb_height % 8 == 0 && cfg.usb_width <= 2040 &&
                cfg.usb_height <= 2040,
            "source.usb.width and height must be multiples of 8 and at most 2040");
    break;
  case SourceType::VideoFile:
    require(!cfg.video_path.empty(),
            "source.video_file.path must be set when source.type=video_file");
    break;
  }
  if (is_live(cfg)) {
    require(!cfg.insight_host.empty(), "output.insight.host must be set for source.type=rtsp/usb");
    require(cfg.video_port > 0, "output.insight.video_port must be > 0");
    require(cfg.metadata_port > 0, "output.insight.metadata_port must be > 0");
  }
}

AppConfig load_app_config(const fs::path& config_path, bool enroll) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  cfg.scrfd_model = raw.string_or("model.scrfd", "");
  cfg.arcface_model = raw.string_or("model.arcface", "");
  cfg.gallery_path = raw.string_or("gallery.path", "");
  cfg.source = parse_source_type(raw.string_or("source.type", "rtsp"));
  cfg.rtsp_url = raw.string_or("source.rtsp.url", "");
  cfg.rtsp_tcp = raw.bool_or("source.rtsp.tcp", true);
  cfg.rtsp_latency_ms = raw.int_or("source.rtsp.latency_ms", 100);
  cfg.rtsp_fps = raw.int_or("source.rtsp.fps", 0);
  cfg.usb_camera = raw.string_or("source.usb.camera", "");
  cfg.usb_width = raw.int_or("source.usb.width", 1280);
  cfg.usb_height = raw.int_or("source.usb.height", 720);
  cfg.usb_fps = raw.int_or("source.usb.fps", 30);
  cfg.video_path = raw.string_or("source.video_file.path", "");
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.detector.min_score = static_cast<float>(raw.double_or("inference.min_score", 0.65));
  cfg.detector.nms_iou = static_cast<float>(raw.double_or("inference.nms_iou", 0.40));
  cfg.detector.max_faces = raw.int_or("inference.max_faces", 100);
  cfg.match.threshold = static_cast<float>(raw.double_or("match.threshold", 0.55));
  cfg.match.margin = static_cast<float>(raw.double_or("match.margin", 0.12));
  cfg.match.unknown_label = raw.string_or("match.unknown_label", "Unknown");
  cfg.enroll_images_dir = raw.string_or("enrollment.images_dir", "");
  cfg.enroll_video_path = raw.string_or("enrollment.video_path", "");
  cfg.enroll_name = raw.string_or("enrollment.name", "");
  cfg.enroll_sample_every = raw.int_or("enrollment.sample_every", 5);
  cfg.enroll_min_score = raw.double_or("enrollment.min_score", 0.75);
  cfg.enroll_max_per_person = raw.int_or("enrollment.max_per_person", 0);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  validate_config(cfg, enroll);
  return cfg;
}

// ArcFace, and SCRFD for the still images of enrollment, take one float tensor per call.
neat::Run build_tensor_run(const std::string& model_path, int width, int height) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Tensor;
  opt.preprocess.input_max_width = width;
  opt.preprocess.input_max_height = height;
  opt.preprocess.input_max_depth = 3;
  neat::Model model(model_path, opt);

  neat::Graph graph;
  graph.add(neat::nodes::Input(model.input_appsrc_options(true)));
  graph.add(model);
  graph.add(neat::nodes::Output());

  const cv::Mat seed(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
  neat::RunOptions run_options;
  run_options.output_memory = neat::OutputMemory::Owned;
  return graph.build(neat::TensorList{face_recog::tensor_from_hwc_f32(seed)}, run_options);
}

std::vector<neat::Tensor> infer(neat::Run& run, const neat::Tensor& input,
                                const std::string& model_name) {
  if (!run.push(neat::TensorList{input})) {
    throw std::runtime_error(model_name + " push failed: " + run.last_error());
  }
  const auto sample = run.pull();
  if (!sample.has_value()) {
    throw std::runtime_error(model_name + " returned no output");
  }
  return face_recog::collect_tensors(*sample);
}

// Enrollment and recognition both turn a detected face into its embedding here, so a face
// enrolled from a photo and the same face seen live are comparable.
face_recog::Embedding embed_face(neat::Run& arcface, const cv::Mat& bgr,
                                 const face_recog::Landmarks& landmarks) {
  const cv::Mat crop = face_recog::align_face(bgr, landmarks);
  const auto output =
      infer(arcface, face_recog::tensor_from_hwc_f32(face_recog::preprocess_arcface_crop(crop)),
            "ArcFace");
  if (output.empty()) {
    throw std::runtime_error("ArcFace returned no tensors");
  }
  auto embedding = face_recog::tensor_to_f32(output.front());
  face_recog::l2_normalize(embedding);
  return embedding;
}

bool is_image(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

std::vector<fs::path> sorted_entries(const fs::path& directory) {
  std::vector<fs::path> entries;
  for (const auto& entry : fs::directory_iterator(directory)) {
    entries.push_back(entry.path());
  }
  std::sort(entries.begin(), entries.end());
  return entries;
}

void run_enrollment(const AppConfig& cfg) {
  auto scrfd =
      build_tensor_run(cfg.scrfd_model, face_recog::kScrfdInputW, face_recog::kScrfdInputH);
  auto arcface = build_tensor_run(cfg.arcface_model, face_recog::kArcFaceW, face_recog::kArcFaceH);

  // New faces are averaged into the people already enrolled.
  face_recog::Gallery gallery;
  if (fs::exists(cfg.gallery_path)) {
    gallery = face_recog::load_gallery(cfg.gallery_path);
  }

  int enrolled = 0;
  int skipped = 0;
  // Enrolls the most confident face of one image, so each image should show one person.
  const auto enroll_image = [&](const std::string& name, const cv::Mat& bgr,
                                const std::string& image_name) {
    auto [tensor, letterbox] = face_recog::preprocess_scrfd(bgr);
    const auto faces =
        face_recog::decode_scrfd(infer(scrfd, tensor, "SCRFD"), cfg.detector, letterbox);
    if (faces.empty() || faces.front().score < cfg.enroll_min_score) {
      std::cout << "[skip] " << image_name << ": no face scoring at least " << cfg.enroll_min_score
                << "\n";
      ++skipped;
      return;
    }
    gallery.add(name, embed_face(arcface, bgr, faces.front().landmarks));
    ++enrolled;
    std::cout << "[enroll] " << name << ": " << image_name
              << " score=" << cv::format("%.3f", faces.front().score) << "\n";
  };

  if (!cfg.enroll_images_dir.empty()) {
    sima_examples::require(fs::is_directory(cfg.enroll_images_dir),
                           "enrollment.images_dir is not a directory: " + cfg.enroll_images_dir);
    for (const auto& person_dir : sorted_entries(cfg.enroll_images_dir)) {
      if (!fs::is_directory(person_dir)) {
        continue;
      }
      int taken = 0;
      for (const auto& image_path : sorted_entries(person_dir)) {
        if (!is_image(image_path)) {
          continue;
        }
        if (cfg.enroll_max_per_person > 0 && taken >= cfg.enroll_max_per_person) {
          break;
        }
        ++taken;
        const cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
          std::cout << "[skip] " << image_path.filename().string() << ": cannot be read\n";
          ++skipped;
          continue;
        }
        enroll_image(person_dir.filename().string(), bgr, image_path.filename().string());
      }
    }
  }
  if (!cfg.enroll_video_path.empty()) {
    cv::VideoCapture video(cfg.enroll_video_path);
    sima_examples::require(video.isOpened(), "failed to open video: " + cfg.enroll_video_path);
    cv::Mat bgr;
    for (int frame = 1; video.read(bgr); ++frame) {
      if (frame % cfg.enroll_sample_every == 0) {
        enroll_image(cfg.enroll_name, bgr, "frame " + std::to_string(frame));
      }
    }
  }
  scrfd.close();
  arcface.close();
  sima_examples::require(enrolled > 0, "no face was enrolled");

  if (cfg.gallery_path.has_parent_path()) {
    fs::create_directories(cfg.gallery_path.parent_path());
  }
  face_recog::save_gallery(gallery, cfg.gallery_path);
  std::cout << "enrolled=" << enrolled << " skipped=" << skipped
            << " people=" << gallery.people.size() << " gallery=" << cfg.gallery_path.string()
            << "\n";
}

neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg) {
  neat::UsbCameraInputOptions opt;
  opt.camera_name = cfg.usb_camera;
  opt.width = static_cast<std::uint32_t>(cfg.usb_width);
  opt.height = static_cast<std::uint32_t>(cfg.usb_height);
  opt.framerate_num = static_cast<std::uint32_t>(cfg.usb_fps);
  return opt;
}

Source open_source(const AppConfig& cfg) {
  Source source;
  if (cfg.source == SourceType::Usb) {
    // Checks the camera once: connected, free, mode supported, one frame decodes.
    source.usb = neat::usb_camera_info(usb_camera_options(cfg));
    source.width = cfg.usb_width;
    source.height = cfg.usb_height;
    source.fps = cfg.usb_fps;
  } else if (cfg.source == SourceType::Rtsp) {
    sima_examples::RtspProbeOptions probe_options;
    probe_options.rtsp_tcp = cfg.rtsp_tcp;
    sima_examples::RtspStreamInfo probe;
    sima_examples::require(
        sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, probe_options, probe) &&
            (probe.fps > 0 || cfg.rtsp_fps > 0),
        "failed to probe RTSP frame size and rate: " + cfg.rtsp_url);
    source.width = probe.width;
    source.height = probe.height;
    source.fps = cfg.rtsp_fps > 0 ? cfg.rtsp_fps : probe.fps;
  } else {
    cv::VideoCapture probe(cfg.video_path);
    sima_examples::require(probe.isOpened(), "failed to open video: " + cfg.video_path);
    source.width = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
    source.height = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
    source.fps = static_cast<int>(std::lround(probe.get(cv::CAP_PROP_FPS)));
  }
  sima_examples::require(source.width > 0 && source.height > 0 && source.fps > 0,
                         "failed to read the source frame size and rate");
  return source;
}

// SCRFD fed decoded video frames; the EV74 letterboxes, converts and normalizes each one.
neat::Model::Options scrfd_options(const Source& source) {
  const bool nv16 = source.usb.has_value() && source.usb->decoded_format == neat::FormatTag::NV16;
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format =
      nv16 ? neat::PreprocessColorFormat::NV16 : neat::PreprocessColorFormat::NV12;
  opt.preprocess.input_max_width = source.width;
  opt.preprocess.input_max_height = source.height;
  opt.preprocess.input_max_depth = 3;
  // SCRFD expects [0, 1] input and black letterbox bars. Neither is taken from the model
  // package: left on Auto, normalization stays off and the bars are grey.
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 0;
  opt.preprocess.normalize.enable = neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
  return opt;
}

neat::nodes::groups::VideoSenderOptions video_options(const AppConfig& cfg) {
  using neat::nodes::groups::RtspCodec;
  auto opt = neat::nodes::groups::VideoSenderOptions::Passthrough(
      cfg.source == SourceType::Usb ? RtspCodec::MJPEG : RtspCodec::H264);
  opt.host = cfg.insight_host;
  opt.channel = 0;
  opt.video_port_base = cfg.video_port;
  opt.async = false;
  return opt;
}

neat::Graph encoded_source(const AppConfig& cfg, const Source& source) {
  neat::Graph graph("source");
  if (cfg.source == SourceType::Usb) {
    graph.add(neat::nodes::UsbCameraInput(usb_camera_options(cfg), *source.usb));
    return graph;
  }
  neat::nodes::groups::RtspEncodedInputOptions opt;
  opt.url = cfg.rtsp_url;
  opt.tcp = cfg.rtsp_tcp;
  opt.latency_ms = cfg.rtsp_latency_ms;
  opt.source_fps = source.fps;
  opt.fallback_h264_width = source.width;
  opt.fallback_h264_height = source.height;
  graph.add(neat::nodes::groups::RtspEncodedInput(opt));
  return graph;
}

neat::SimaDecodeOptions decoder_options(const AppConfig& cfg, const Source& source) {
  neat::SimaDecodeOptions opt;
  if (cfg.source == SourceType::Usb) {
    opt.type = neat::SimaDecodeType::MJPEG;
    opt.out_format = source.usb->decoded_format;
  } else {
    opt.type = neat::SimaDecodeType::H264;
    opt.out_format = neat::FormatTag::NV12;
  }
  opt.raw_output = true;
  opt.dec_width = source.width;
  opt.dec_height = source.height;
  opt.dec_fps = source.fps;
  return opt;
}

// A live source is sent to Insight as it arrives and decoded once for the model. Each decoded
// frame goes to SCRFD and to a frame tap, and the two are joined by frame so every face is
// cropped from the frame it was found in.
neat::Graph build_graph(const AppConfig& cfg, const Source& source, neat::Model& scrfd) {
  neat::GraphOptions graph_options;
  graph_options.advanced_execution.postprocess_target = "EV74";
  neat::Graph graph(graph_options);
  auto decoded = neat::graphs::Branch("decoded", {"model", "frame"});

  if (is_live(cfg)) {
    neat::Graph encoded = encoded_source(cfg, source);
    neat::Graph decoder("decoder");
    decoder.add(neat::nodes::SimaDecode(decoder_options(cfg, source)));
    graph.connect(encoded, decoder);
    graph.connect(decoder, decoded);

    neat::Graph video("video_sender");
    video.add(neat::nodes::groups::VideoSender(video_options(cfg)));
    neat::GraphLinkOptions latest;
    latest.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
    graph.connect(encoded, video, latest);
  } else {
    neat::nodes::groups::VideoInputGroupOptions file_options;
    file_options.path = cfg.video_path;
    neat::Graph file = neat::nodes::groups::VideoInputGroup(file_options);
    graph.connect(file, decoded);
  }

  neat::Graph model("model");
  model.connect(neat::nodes::Input("model"), scrfd);
  neat::Graph detections("detections");
  detections.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));
  neat::Graph frame("frame");
  frame.add(neat::nodes::Output("frame", neat::OutputOptions::EveryFrame(4)));
  auto joined =
      neat::graphs::Combine({"frame", "detections"}, kOutputName, neat::CombinePolicy::ByFrame);
  graph.connect(decoded, model);
  graph.connect(decoded, frame);
  graph.connect(model, detections);
  graph.connect(frame, joined);
  graph.connect(detections, joined);
  return graph;
}

// A live source keeps its newest frame; a video file is processed frame by frame.
neat::RunOptions run_options(const AppConfig& cfg) {
  neat::RunOptions options;
  if (is_live(cfg)) {
    options.preset = neat::RunPreset::Realtime;
  }
  options.queue_depth = 3;
  options.overflow_policy =
      is_live(cfg) ? neat::OverflowPolicy::KeepLatest : neat::OverflowPolicy::Block;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

const neat::Sample* find_field(const neat::Sample& sample, const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const neat::Sample& joined_field(const neat::Sample& sample, const std::string& label,
                                 std::size_t bundle_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == neat::SampleKind::Bundle && sample.fields.size() > bundle_index) {
    return sample.fields[bundle_index];
  }
  throw std::runtime_error("joined output missing " + label + " field");
}

// H.264 decodes to NV12; a 4:2:2 USB camera decodes to NV16.
cv::Mat frame_bgr(const neat::Sample& sample) {
  const auto tensors = neat::tensors_from_sample(joined_field(sample, "frame", 0U), true);
  sima_examples::require(!tensors.empty(), "joined frame field has no tensor");
  const auto& frame = tensors.front();
  cv::Mat bgr;
  std::string err;
  const bool converted = frame.image_format() == neat::ImageSpec::PixelFormat::NV16
                             ? sima_examples::nv16_to_bgr(frame, bgr, err)
                             : sima_examples::nv12_to_bgr(frame, bgr, err);
  sima_examples::require(converted, "failed to convert the decoded frame: " + err);
  return bgr;
}

nlohmann::json insight_object(int index, const face_recog::Detection& face,
                              const face_recog::Match& match, const Source& source) {
  const float x1 = std::clamp(face.x1, 0.0f, static_cast<float>(source.width));
  const float y1 = std::clamp(face.y1, 0.0f, static_cast<float>(source.height));
  const float x2 = std::clamp(face.x2, x1, static_cast<float>(source.width));
  const float y2 = std::clamp(face.y2, y1, static_cast<float>(source.height));
  return {{"id", "face_" + std::to_string(index)},
          {"label", match.name},
          {"confidence", face.score},
          {"similarity", match.similarity},
          {"bbox", {x1, y1, x2 - x1, y2 - y1}}};
}

void send_metadata(neat::MetadataSender& sender, const neat::Sample& sample,
                   const std::vector<face_recog::Detection>& faces,
                   const std::vector<face_recog::Match>& matches, const Source& source) {
  nlohmann::json objects = nlohmann::json::array();
  for (std::size_t i = 0; i < faces.size(); ++i) {
    objects.push_back(insight_object(static_cast<int>(i) + 1, faces[i], matches[i], source));
  }
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!sender.send_metadata("object-detection", nlohmann::json{{"objects", objects}}.dump(),
                            timestamp_ms, frame_id, &err)) {
    std::cerr << "[warn] metadata send failed: " << err << "\n";
  }
}

void print_frame(int frame, const std::vector<face_recog::Match>& matches) {
  std::cout << "frame " << frame << ":";
  if (matches.empty()) {
    std::cout << " no faces";
  }
  for (std::size_t i = 0; i < matches.size(); ++i) {
    std::cout << (i == 0 ? " " : ", ") << matches[i].name << " "
              << cv::format("%.3f", matches[i].similarity);
  }
  std::cout << "\n";
}

void run_recognition(const AppConfig& cfg) {
  face_recog::Gallery gallery;
  if (fs::exists(cfg.gallery_path)) {
    gallery = face_recog::load_gallery(cfg.gallery_path);
  } else {
    std::cerr << "[warn] no gallery at " << cfg.gallery_path.string() << "; every face is labelled "
              << cfg.match.unknown_label << "\n";
  }

  const Source source = open_source(cfg);
  auto arcface = build_tensor_run(cfg.arcface_model, face_recog::kArcFaceW, face_recog::kArcFaceH);
  neat::Model scrfd(cfg.scrfd_model, scrfd_options(source));
  auto graph = build_graph(cfg, source, scrfd);

  std::unique_ptr<neat::MetadataSender> metadata_sender;
  std::cout << "source=" << source.width << "x" << source.height << "@" << source.fps
            << " people=" << gallery.people.size();
  if (is_live(cfg)) {
    neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = 0;
    metadata_options.metadata_port_base = cfg.metadata_port;
    std::string err;
    metadata_sender = std::make_unique<neat::MetadataSender>(metadata_options, &err);
    sima_examples::require(metadata_sender->ok(), err);
    std::cout << " insight=" << cfg.insight_host << " video=" << video_options(cfg).video_port()
              << " metadata=" << metadata_sender->metadata_port();
  }
  std::cout << std::endl;

  auto run = graph.build(run_options(cfg));
  const auto letterbox = face_recog::letterbox_for(source.width, source.height);
  int processed = 0;
  std::optional<std::chrono::steady_clock::time_point> first_frame;
  const auto previous_sigint = std::signal(SIGINT, request_stop);
  while (g_stop_requested == 0 && (cfg.frames <= 0 || processed < cfg.frames)) {
    neat::Sample sample;
    neat::PullError pull_error;
    const auto status = run.pull(kOutputName, kPullTimeoutMs, sample, &pull_error);
    if (!is_live(cfg) && status == neat::PullStatus::Closed) {
      // The video file has ended.
      break;
    }
    if (!sima_examples::pull_status_has_sample(status, kOutputName, pull_error, run.last_error())) {
      continue;
    }
    if (!first_frame.has_value()) {
      first_frame = std::chrono::steady_clock::now();
    }

    const auto faces = face_recog::decode_scrfd(
        face_recog::collect_tensors(joined_field(sample, "detections", 1U)), cfg.detector,
        letterbox);
    std::vector<face_recog::Match> matches;
    if (!faces.empty()) {
      const cv::Mat bgr = frame_bgr(sample);
      for (const auto& face : faces) {
        matches.push_back(face_recog::match_embedding(embed_face(arcface, bgr, face.landmarks),
                                                      gallery, cfg.match));
      }
    }
    ++processed;
    if (metadata_sender) {
      send_metadata(*metadata_sender, sample, faces, matches, source);
    } else {
      print_frame(processed, matches);
    }
  }
  std::signal(SIGINT, previous_sigint);
  run.close();
  arcface.close();

  const double seconds =
      first_frame.has_value()
          ? std::chrono::duration<double>(std::chrono::steady_clock::now() - *first_frame).count()
          : 0.0;
  std::cout << "processed=" << processed
            << " fps=" << cv::format("%.1f", seconds > 0.0 ? processed / seconds : 0.0) << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }
    const AppConfig cfg = load_app_config(cli.config_path, cli.enroll);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path.string()
                << " (mode=" << (cli.enroll ? "enroll" : "recognize") << ")\n";
      return 0;
    }
    if (cli.enroll) {
      run_enrollment(cfg);
    } else {
      run_recognition(cfg);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
