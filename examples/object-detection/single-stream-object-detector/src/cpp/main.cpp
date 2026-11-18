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

// Single-stream object detector: YOLO26 on one RTSP, HTTP MJPEG or USB camera source, with the
// video and the detections sent to Insight.
//
// main() builds the whole graph. The helpers it calls are in utils/.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "utils/app_config.h"
#include "utils/config.h"
#include "utils/detections.h"
#include "utils/detector.h"
#include "utils/insight.h"
#include "utils/runtime.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// SIGINT ends the pull loop so the run is closed and the counters printed, the
// way the multistream applications already stop. The e2e harness stops the
// application this way and checks that it exits cleanly.
volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = utils::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

// With runtime.profile, prints averages over every profile_interval published frames.
struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  double start_ms = 0.0;
  double detection_pull_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double detection_pull, double metadata_send, int box_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = utils::time_ms();
    ++frames;
    boxes += box_count;
    detection_pull_ms += detection_pull;
    metadata_send_ms += metadata_send;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = utils::time_ms() - start_ms;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    std::cout << "[profile] frames=" << frames << " output_fps=" << output_fps
              << " avg_detection_pull_ms=" << avg(detection_pull_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_boxes=" << static_cast<double>(boxes) / static_cast<double>(frames) << "\n";
    frames = 0;
    boxes = 0;
    start_ms = 0.0;
    detection_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

// What the pull loop needs from the graph main() built.
struct PipelineRuntime {
  simaai::neat::Run& run;
  simaai::neat::MetadataSender& metadata_sender;
  std::vector<std::string> labels;
  /// Run output the loop pulls: the detections alone, or the frame-joined bundle when saving.
  std::string output_name;
  int frame_w = 0;
  int frame_h = 0;
  int video_port = 0;
};

// Pulls each frame's detections, sends them to Insight, and saves sampled frames.
void run_pipeline(PipelineRuntime& runtime, const utils::AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  g_stop_requested = 0;
  std::signal(SIGINT, request_stop);
  while (g_stop_requested == 0 && (cfg.frames <= 0 || processed < cfg.frames)) {
    simaai::neat::Sample detection_sample;
    simaai::neat::PullError pull_error;
    const double pull_start = utils::time_ms();
    const auto status = runtime.run.pull(runtime.output_name, 20000, detection_sample, &pull_error);
    const double pull_end = utils::time_ms();
    if (!utils::pull_status_has_sample(status, runtime.output_name, pull_error,
                                       runtime.run.last_error())) {
      std::cerr << "[warn] timed out waiting for detections\n";
      continue;
    }

    std::vector<std::uint8_t> payload;
    std::string err;
    if (!utils::extract_bbox_payload(detection_sample, payload, err)) {
      throw std::runtime_error("bbox extract failed: " + err);
    }
    const auto boxes =
        utils::parse_boxes_strict(payload, runtime.frame_w, runtime.frame_h, cfg.max_detections);

    const double metadata_start = utils::time_ms();
    const auto objects =
        utils::build_metadata_boxes(boxes, runtime.labels, runtime.frame_w, runtime.frame_h);
    utils::send_metadata(runtime.metadata_sender, "object-detection",
                         utils::metadata_boxes_data_json("objects", objects), detection_sample);
    const double metadata_end = utils::time_ms();

    ++processed;
    utils::maybe_save_debug_frame(cfg, processed, detection_sample, boxes);
    profile.add(pull_end - pull_start, metadata_end - metadata_start,
                static_cast<int>(boxes.size()));
  }

  profile.flush();
  std::cout << "processed=" << processed << " video_sender=" << cfg.insight_host << ":"
            << runtime.video_port << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    const utils::AppConfig cfg = utils::load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      // Reports which key supplied the source, not its value: a URL can carry
      // credentials and this line ends up in terminal and CI logs.
      std::cout << "Config validated: " << cli.config_path << " (source=" << cfg.source_key
                << ")\n";
      return 0;
    }
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }

    if (cfg.profile) {
      setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
      setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
      setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
    }

    // The application graph. Its nodes are created below, and every link between them is made
    // here in main().
    simaai::neat::Graph graph;

    const utils::StreamInfo geometry = utils::resolve_source_geometry(cfg);
    utils::require(geometry.width > 0 && geometry.height > 0,
                   "failed to probe source frame dimensions");
    utils::require(geometry.fps > 0, "failed to resolve source frame rate");

    // A USB camera's JPEG goes to Insight unchanged and is decoded only for the model, in the
    // camera's own sampling: most webcams send 4:2:2 MJPEG, which decodes to NV16, and the model
    // reads NV16 directly. Other sources are decoded once and re-encoded as H.264 for Insight.
    const bool usb = cfg.source_type == utils::SourceType::Usb;
    // The camera is checked once; the same info gives the decoded layout and builds the node.
    const std::optional<simaai::neat::UsbCameraInfo> usb_camera =
        usb ? std::optional(simaai::neat::usb_camera_info(utils::make_usb_camera_options(cfg)))
            : std::nullopt;
    const simaai::neat::FormatTag decoded_format =
        usb ? usb_camera->decoded_format : simaai::neat::FormatTag::NV12;

    // The model resizes and normalizes the decoded frame itself and decodes YOLO26 boxes.
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format =
        decoded_format == simaai::neat::FormatTag::NV16 ? simaai::neat::PreprocessColorFormat::NV16
                                                        : simaai::neat::PreprocessColorFormat::NV12;
    model_opt.preprocess.input_max_width = geometry.width;
    model_opt.preprocess.input_max_height = geometry.height;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = cfg.min_score;
    model_opt.nms_iou_threshold = cfg.nms_iou;
    model_opt.top_k = cfg.max_detections;
    simaai::neat::Model model(cfg.model_path, model_opt);
    const std::vector<std::string> labels = utils::load_labels(cfg.labels_path);

    // Insight video: the USB camera's JPEG as it is, or the decoded frame encoded as H.264.
    auto video_options =
        usb ? simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
                  simaai::neat::nodes::groups::RtspCodec::MJPEG)
            : simaai::neat::nodes::groups::VideoSenderOptions::FromRaw({.fps = geometry.fps});
    video_options.host = cfg.insight_host;
    video_options.channel = 0;
    video_options.video_port_base = cfg.video_port;
    if (!usb) {
      video_options.encoder.bitrate_kbps = 1000;
    }

    // Insight correlates the RTP timestamp with the metadata timestamp, so the video and the
    // detections must stay in one Run and therefore on one GStreamer timeline. The source
    // branches to the video sender, the model and, with output.save_dir, the saved frames.
    const bool save_frames = !cfg.save_dir.empty();
    std::vector<std::string> decoded_outputs{"model"};
    if (save_frames) {
      decoded_outputs.push_back("frame");
    }
    if (!usb) {
      decoded_outputs.insert(decoded_outputs.begin(), "video");
    }
    auto source = utils::make_source_graph(cfg, geometry, usb_camera);
    auto branch = simaai::neat::graphs::Branch(usb ? "decoded" : "source", decoded_outputs);

    simaai::neat::Graph video_graph("video");
    video_graph.connect(simaai::neat::nodes::Input("video"),
                        simaai::neat::nodes::groups::VideoSender(video_options));

    simaai::neat::Graph model_graph("model");
    model_graph.connect(simaai::neat::nodes::Input("model"), model);
    simaai::neat::Graph detections_graph("detections");
    detections_graph.add(
        simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));

    if (usb) {
      // The camera's JPEG splits before decoding: one copy to Insight, one to the decoder.
      simaai::neat::Graph decode_graph("decode");
      decode_graph.connect(
          simaai::neat::nodes::Input("decode"),
          simaai::neat::nodes::SimaDecode(utils::make_usb_decode_options(cfg, decoded_format)));
      simaai::neat::GraphLinkOptions video_link;
      video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      auto split = simaai::neat::graphs::Branch("source", {"video", "decode"});
      graph.connect(source, split);
      graph.connect(split, video_graph, video_link);
      graph.connect(split, decode_graph);
      graph.connect(decode_graph, branch);
    } else {
      graph.connect(source, branch);
      graph.connect(branch, video_graph);
    }
    graph.connect(branch, model_graph);
    graph.connect(model_graph, detections_graph);
    if (save_frames) {
      // Each saved frame is joined with the detections from the same frame.
      simaai::neat::Graph frame_graph("frame");
      frame_graph.add(
          simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));
      auto joined = simaai::neat::graphs::Combine({"frame", "detections"}, "detector_output",
                                                  simaai::neat::CombinePolicy::ByFrame);
      graph.connect(branch, frame_graph);
      graph.connect(frame_graph, joined);
      graph.connect(detections_graph, joined);
    }
    const std::string output_name = save_frames ? "detector_output" : "detections";
    if (cfg.profile) {
      std::cout << "Backend:\n" << graph.describe_backend() << "\n";
    }

    simaai::neat::RunOptions run_options;
    run_options.preset = simaai::neat::RunPreset::Realtime;
    run_options.queue_depth = 3;
    run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
    simaai::neat::Run run = graph.build(run_options);

    simaai::neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = 0;
    metadata_options.metadata_port_base = cfg.metadata_port;
    std::string metadata_err;
    simaai::neat::MetadataSender metadata_sender(metadata_options, &metadata_err);
    utils::require(metadata_sender.ok(), metadata_err);

    std::cout << "source=" << (usb ? cfg.camera : cfg.source_url)
              << " type=" << utils::source_type_name(cfg.source_type)
              << " codec=" << utils::source_codec_name(cfg.source_codec)
              << " stream=" << geometry.width << "x" << geometry.height << "@" << geometry.fps
              << " insight=" << cfg.insight_host << " video=" << video_options.video_port()
              << " metadata=" << metadata_sender.metadata_port() << " channel=0\n";

    PipelineRuntime runtime{run,
                            metadata_sender,
                            labels,
                            output_name,
                            geometry.width,
                            geometry.height,
                            video_options.video_port()};
    run_pipeline(runtime, cfg);
    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
