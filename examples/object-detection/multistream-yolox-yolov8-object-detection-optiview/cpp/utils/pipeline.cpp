#include "pipeline_api.cpp"

#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace multistream_yolox_yolov8_optiview {
namespace {

std::string upper_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

} // namespace

int optiview_video_port_for_stream(int port_base, int stream_index) {
  return port_base + stream_index;
}

int optiview_json_port_for_stream(int port_base, int stream_index) {
  return port_base + stream_index;
}

int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe) {
  return cfg.fps > 0 ? cfg.fps : std::max(probe.fps, 10);
}

int source_output_every_n(const AppConfig& cfg, const RtspProbe& probe) {
  if (cfg.fps <= 0 || probe.fps <= 0 || cfg.fps >= probe.fps) {
    return 1;
  }
  return std::max(1, probe.fps / cfg.fps);
}

double producer_emit_period_s(const AppConfig& cfg, const RtspProbe& probe) {
  static_cast<void>(probe);
  if (cfg.fps <= 0) {
    return 0.0;
  }
  return 1.0 / cfg.fps;
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

std::vector<std::string> detector_stage_names(ModelFamily family) {
  switch (family) {
  case ModelFamily::YoloV8:
    return {"input", "quant_tess", "mla", "sima_box_decode", "output"};
  case ModelFamily::YoloX:
    throw std::invalid_argument(yolox_not_supported_message());
  case ModelFamily::Auto:
    break;
  }
  throw std::invalid_argument("unsupported model family for detector graph");
}

std::shared_ptr<simaai::neat::Model> load_detector_model(const AppConfig& cfg) {
  simaai::neat::Model::Options options;
  options.media_type = "application/vnd.simaai.tensor";
  options.format.clear();
  return std::make_shared<simaai::neat::Model>(cfg.model.path, options);
}

QuantTessCpuPreproc read_preproc_contract(const simaai::neat::Model& model) {
  const simaai::neat::PreprocOptions preproc(model);
  const nlohmann::json config = preproc.config_json.value_or(nlohmann::json::object());

  return QuantTessCpuPreproc{
      config.value("output_width", config.value("input_width", 640)),
      config.value("output_height", config.value("input_height", 640)),
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
  runtime.session.add(
      simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(source_output_every_n(cfg, probe))));

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run = runtime.session.build(run_options);
  return runtime;
}

SessionRun build_detection_run(const AppConfig& cfg, const simaai::neat::Model& model,
                               ModelFamily family, const RtspProbe& probe,
                               const QuantTessCpuPreproc& quant_preproc) {
  static_cast<void>(detector_stage_names(family));

  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
  runtime.session.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
  runtime.session.add(simaai::neat::nodes::groups::MLA(model));

  switch (family) {
  case ModelFamily::YoloV8:
    runtime.session.add(simaai::neat::nodes::SimaBoxDecode(
        model, "yolov8", probe.width, probe.height, cfg.min_score, cfg.nms_iou,
        cfg.max_detections));
    break;
  case ModelFamily::YoloX:
    throw std::invalid_argument(yolox_not_supported_message());
  case ModelFamily::Auto:
    throw std::invalid_argument("unsupported model family for detector graph");
  }

  runtime.session.add(simaai::neat::nodes::Output());

  cv::Mat seed = cv::Mat::zeros(quant_preproc.height, quant_preproc.width, CV_32FC3);
  runtime.run = runtime.session.build(seed);
  return runtime;
}

simaai::neat::Sample run_tensor_input_once(simaai::neat::Run& run, const cv::Mat& input,
                                           int timeout_ms) {
  return run.run(input, timeout_ms);
}

SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index) {
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
  runtime.session.add(simaai::neat::nodes::H264EncodeSima(
      probe.width, probe.height, effective_writer_fps(cfg, probe), 2500, "baseline", "4.1"));

  simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_options;
  udp_options.payload_type = 96;
  udp_options.config_interval = 1;
  udp_options.udp_host = cfg.optiview_host;
  udp_options.udp_port = optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index);
  udp_options.udp_sync = false;
  udp_options.udp_async = false;
  runtime.session.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_options));

  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
  simaai::neat::Tensor seed_tensor =
      simaai::neat::from_cv_mat(seed, simaai::neat::ImageSpec::PixelFormat::RGB, true);
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  runtime.run = runtime.session.build(seed_tensor, simaai::neat::RunMode::Async, run_options);
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

} // namespace multistream_yolox_yolov8_optiview
