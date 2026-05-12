#include "examples/tracking/multi-camera-people-detection-and-tracking-insight/cpp/utils/pipeline_api.cpp"

#include "nodes/groups/VideoSender.h"

#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace multi_camera_people_tracking {
namespace {

constexpr double kYoloV8DefaultDetectionThreshold = 0.6;
constexpr double kYoloV8DefaultNmsIouThreshold = 0.5;
constexpr int kYoloV8DefaultTopK = 24;

std::string upper_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

} // namespace

simaai::neat::Run build_tensor_input_run(simaai::neat::Session& session, const cv::Mat& seed,
                                         simaai::neat::RunMode mode,
                                         const simaai::neat::RunOptions& options) {
  return session.build(seed, mode, options);
}

simaai::neat::Sample run_tensor_input_once(simaai::neat::Run& run, const cv::Mat& input,
                                           int timeout_ms) {
  return run.run(input, timeout_ms);
}

int insight_video_port_for_stream(int port_base, int stream_index) {
  return port_base + stream_index;
}

int insight_metadata_port_for_stream(int port_base, int stream_index) {
  return port_base + stream_index;
}

int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe) {
  return cfg.fps > 0 ? cfg.fps : std::max(probe.fps, 10);
}

RtspProbe probe_rtsp(const std::string& url) {
  cv::VideoCapture capture(url);
  if (!capture.isOpened()) {
    throw std::runtime_error("failed to open RTSP source for probing: " + url);
  }

  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  const int fps = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FPS)));
  capture.release();

  if (width <= 0 || height <= 0) {
    throw std::runtime_error("failed to probe RTSP frame size: " + url);
  }
  return RtspProbe{width, height, std::max(0, fps)};
}

std::shared_ptr<simaai::neat::Model> load_detector_model(const AppConfig& cfg) {
  simaai::neat::Model::Options options;
  options.media_type = "application/vnd.simaai.tensor";
  options.format.clear();
  return std::make_shared<simaai::neat::Model>(cfg.model, options);
}

QuantTessCpuPreproc read_preproc_contract(const simaai::neat::Model& model) {
  const simaai::neat::PreprocOptions preproc(model);
  const nlohmann::json config = preproc.config_json.value_or(nlohmann::json::object());

  const int width = config.value("output_width", config.value("input_width", 640));
  const int height = config.value("output_height", config.value("input_height", 640));
  return QuantTessCpuPreproc{
      width,
      height,
      config.value("aspect_ratio", false),
      upper_copy(config.value("padding_type", std::string("CENTER"))),
  };
}

SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = url;
  options.latency_ms = cfg.latency_ms;
  options.tcp = cfg.tcp;
  options.payload_type = 96;
  options.insert_queue = true;
  options.out_format = "RGB";
  options.decoder_raw_output = false;
  options.auto_caps_from_stream = true;
  options.use_videoconvert = false;
  options.use_videoscale = true;
  options.fallback_h264_width = probe.width;
  options.fallback_h264_height = probe.height;
  if (probe.fps > 0) {
    options.fallback_h264_fps = probe.fps;
  }
  options.output_caps.enable = true;
  options.output_caps.format = "RGB";
  options.output_caps.width = probe.width;
  options.output_caps.height = probe.height;
  if (probe.fps > 0) {
    options.output_caps.fps = probe.fps;
  }
  options.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;

  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::groups::RtspDecodedInput(options));
  runtime.session.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(1)));

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 4;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run = runtime.session.build(run_options);
  return runtime;
}

SessionRun build_detection_run(const AppConfig& cfg, const simaai::neat::Model& model,
                               const RtspProbe& probe, const QuantTessCpuPreproc& quant_preproc) {
  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
  runtime.session.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
  runtime.session.add(simaai::neat::nodes::groups::MLA(model));
  runtime.session.add(simaai::neat::nodes::SimaBoxDecode(
      model, "yolov8", probe.width, probe.height,
      cfg.detection_threshold.value_or(kYoloV8DefaultDetectionThreshold),
      cfg.nms_iou_threshold.value_or(kYoloV8DefaultNmsIouThreshold),
      cfg.top_k.value_or(kYoloV8DefaultTopK)));
  runtime.session.add(simaai::neat::nodes::Output());

  cv::Mat seed = cv::Mat::zeros(quant_preproc.height, quant_preproc.width, CV_32FC3);
  runtime.run = build_tensor_input_run(runtime.session, seed);
  return runtime;
}

SessionRun build_insight_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index) {
  simaai::neat::InputOptions input_options;
  input_options.media_type = "video/x-raw";
  input_options.format = "RGB";
  input_options.use_simaai_pool = false;
  input_options.max_width = probe.width;
  input_options.max_height = probe.height;
  input_options.max_depth = 3;

  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::Input(input_options));
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      probe.width, probe.height, effective_writer_fps(cfg, probe));
  video_options.udp.host = cfg.insight_host;
  video_options.udp.port = insight_video_port_for_stream(cfg.insight_video_port_base, stream_index);
  video_options.udp.sync = false;
  video_options.udp.async = false;
  video_options.rtp.payload_type = 96;
  video_options.rtp.config_interval = 1;
  video_options.encoder.bitrate_kbps = cfg.bitrate_kbps;
  video_options.encoder.profile = "baseline";
  video_options.encoder.level = "4.1";
  runtime.session.add(simaai::neat::nodes::groups::VideoSender(video_options));

  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
  simaai::neat::Tensor seed_tensor =
      simaai::neat::from_cv_mat(seed, simaai::neat::ImageSpec::PixelFormat::RGB, true);
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 2;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  runtime.run = runtime.session.build(seed_tensor, simaai::neat::RunMode::Async, run_options);
  return runtime;
}

simaai::neat::MetadataSender build_insight_metadata_output(const AppConfig& cfg, int stream_index) {
  simaai::neat::MetadataSenderOptions options;
  options.host = cfg.insight_host;
  options.channel = stream_index;
  options.metadata_port_base = cfg.insight_metadata_port_base;
  return simaai::neat::MetadataSender(options);
}

} // namespace multi_camera_people_tracking
