#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/utils/pipeline_api.cpp"

#include <opencv2/videoio.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace multi_camera_people_tracking {
namespace {

constexpr double kYoloV8DefaultDetectionThreshold = 0.6;
constexpr double kYoloV8DefaultNmsIouThreshold = 0.5;
constexpr int kYoloV8DefaultTopK = 24;

} // namespace

simaai::neat::Sample run_sample_input_once(simaai::neat::Run& run,
                                           const simaai::neat::Sample& input, int timeout_ms) {
  auto outputs = run.run(simaai::neat::SampleList{input}, timeout_ms);
  if (outputs.empty()) {
    throw std::runtime_error("detector run produced no samples");
  }
  return std::move(outputs.front());
}

int optiview_video_port_for_stream(int port_base, int stream_index) {
  return port_base + stream_index;
}

int optiview_json_port_for_stream(int port_base, int stream_index) {
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

SessionRun build_detection_run(const AppConfig& cfg, const RtspProbe& probe) {
  SessionRun runtime;

  simaai::neat::Model::Options model_options;
  model_options.preprocess.kind = simaai::neat::InputKind::Image;
  model_options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  model_options.preprocess.input_max_width = probe.width;
  model_options.preprocess.input_max_height = probe.height;
  model_options.preprocess.input_max_depth = 3;
  model_options.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  model_options.score_threshold =
      cfg.detection_threshold.value_or(kYoloV8DefaultDetectionThreshold);
  model_options.nms_iou_threshold = cfg.nms_iou_threshold.value_or(kYoloV8DefaultNmsIouThreshold);
  model_options.top_k = cfg.top_k.value_or(kYoloV8DefaultTopK);
  model_options.boxdecode_original_width = probe.width;
  model_options.boxdecode_original_height = probe.height;
  runtime.model = std::make_shared<simaai::neat::Model>(cfg.model, model_options);

  auto input_options = runtime.model->input_appsrc_options(false);
  input_options.media_type = "video/x-raw";
  input_options.format = "RGB";
  input_options.width = probe.width;
  input_options.height = probe.height;
  input_options.depth = 3;
  runtime.session.add(simaai::neat::nodes::Input(input_options));
  runtime.session.add(simaai::neat::nodes::groups::Preprocess(*runtime.model));
  runtime.session.add(simaai::neat::nodes::groups::Infer(*runtime.model));
  runtime.session.add(simaai::neat::nodes::SimaBoxDecode(
      *runtime.model, simaai::neat::BoxDecodeType::YoloV8,
      cfg.detection_threshold.value_or(kYoloV8DefaultDetectionThreshold),
      cfg.nms_iou_threshold.value_or(kYoloV8DefaultNmsIouThreshold),
      cfg.top_k.value_or(kYoloV8DefaultTopK), "", std::nullopt, std::nullopt, probe.width,
      probe.height));
  runtime.session.add(simaai::neat::nodes::Output());

  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run =
      runtime.session.build(std::vector<cv::Mat>{seed}, simaai::neat::RunMode::Async, run_options);
  return runtime;
}

SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe,
                                    int stream_index) {
  simaai::neat::InputOptions input_options;
  input_options.media_type = "video/x-raw";
  input_options.format = "RGB";
  input_options.use_simaai_pool = false;
  input_options.max_width = probe.width;
  input_options.max_height = probe.height;
  input_options.max_depth = 3;

  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::Input(input_options));
  runtime.session.add(simaai::neat::nodes::VideoConvert());
  runtime.session.add(simaai::neat::nodes::H264EncodeSima(probe.width, probe.height,
                                                          effective_writer_fps(cfg, probe),
                                                          cfg.bitrate_kbps, "baseline", "4.1"));

  simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_options;
  udp_options.payload_type = 96;
  udp_options.config_interval = 1;
  udp_options.udp_host = cfg.optiview_host;
  udp_options.udp_port = optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index);
  udp_options.udp_sync = false;
  udp_options.udp_async = false;
  runtime.session.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_options));

  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 2;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  runtime.run =
      runtime.session.build(std::vector<cv::Mat>{seed}, simaai::neat::RunMode::Async, run_options);
  return runtime;
}

sima_examples::OptiViewSender build_optiview_json_output(const AppConfig& cfg, int stream_index) {
  sima_examples::OptiViewOptions options;
  options.host = cfg.optiview_host;
  options.channel = stream_index;
  options.video_port_base = cfg.optiview_video_port_base;
  options.json_port_base = cfg.optiview_json_port_base;
  return sima_examples::OptiViewSender(options);
}

} // namespace multi_camera_people_tracking
