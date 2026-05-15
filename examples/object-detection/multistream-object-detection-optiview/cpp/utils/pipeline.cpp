#include "pipeline_api.cpp"

#include <cstdlib>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>

namespace multistream_object_detection_optiview {
namespace {

constexpr const char* kDefaultModelNumBuffers = "3";
constexpr const char* kDefaultDecoderNumBuffers = "7";
constexpr const char* kOptiViewVideoEncoderEnv = "SIMA_OPTIVIEW_VIDEO_ENCODER";
constexpr int kSourceProbeDecoderNumBuffers = 7;
constexpr int kSourceRunQueueDepth = 4;
constexpr int kSourceOutputEveryN = 1;
constexpr simaai::neat::RunPreset kRealtimeRunPreset = simaai::neat::RunPreset::Realtime;

void set_env_if_unset(const char* key, const char* value) {
  if (key == nullptr || value == nullptr) {
    return;
  }
  if (std::getenv(key) != nullptr) {
    return;
  }
  setenv(key, value, 0);
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

OptiViewVideoEncoder optiview_video_encoder_from_env() {
  const char* raw = std::getenv(kOptiViewVideoEncoderEnv);
  if (raw == nullptr || *raw == '\0') {
    return OptiViewVideoEncoder::Hardware;
  }

  std::string value(raw);
  for (char& c : value) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  if (value == "sw" || value == "software") {
    return OptiViewVideoEncoder::Software;
  }
  return OptiViewVideoEncoder::Hardware;
}

double producer_emit_period_s(const AppConfig& cfg, const RtspProbe& probe) {
  static_cast<void>(probe);
  if (cfg.fps <= 0) {
    return 0.0;
  }
  return 1.0 / cfg.fps;
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_input_group_options(const AppConfig& cfg, const std::string& url,
                                 const RtspProbe& probe) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = url;
  options.latency_ms = cfg.latency_ms;
  options.tcp = cfg.tcp;
  options.payload_type = 96;
  options.insert_queue = true;
  options.sync_mode = false;
  options.auto_caps_from_stream = true;
  options.fallback_h264_fps = probe.fps;
  options.fallback_h264_width = probe.width;
  options.fallback_h264_height = probe.height;
  options.sima_allocator_type = 2;
  options.out_format = "RGB";
  options.decoder_raw_output = false;
  options.use_videoconvert = false;
  options.use_videoscale = true;
  options.output_caps.enable = true;
  options.output_caps.format = "RGB";
  options.output_caps.width = probe.width;
  options.output_caps.height = probe.height;
  if (probe.fps > 0) {
    options.output_caps.fps = probe.fps;
  }
  options.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;
  return options;
}

RtspProbe probe_rtsp(const AppConfig& cfg, const std::string& url) {
  apply_runtime_env_defaults();

  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.decoder_num_buffers = kSourceProbeDecoderNumBuffers;

  sima_examples::RtspStreamInfo info;
  if (!sima_examples::probe_rtsp_stream_info(url, probe_options, info)) {
    throw std::runtime_error("failed to open RTSP source for probing: " + url);
  }

  RtspProbe probe_result;
  probe_result.width = info.width;
  probe_result.height = info.height;
  probe_result.fps = info.fps;
  if (probe_result.width <= 0 || probe_result.height <= 0) {
    throw std::runtime_error("failed to probe RTSP frame size: " + url);
  }
  return probe_result;
}

std::vector<std::string> detector_stage_names(ModelFamily family) {
  switch (family) {
  case ModelFamily::YoloV8:
    return {"input", "preproc", "mla", "sima_box_decode", "output"};
  case ModelFamily::Auto:
    break;
  }
  throw std::invalid_argument("unsupported model family for detector graph");
}

void apply_runtime_env_defaults() {
  set_env_if_unset("SIMA_FORCE_MODEL_NUM_BUFFERS", kDefaultModelNumBuffers);
  set_env_if_unset("SIMA_FORCE_DECODER_NUM_BUFFERS", kDefaultDecoderNumBuffers);
  set_env_if_unset("SIMA_FORCE_DECODER_POOL_BUFFERS", kDefaultDecoderNumBuffers);
  set_env_if_unset("SIMA_PULL_TIMEOUT_DIAG", "0");
}

SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe) {
  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::groups::RtspDecodedInput(
      build_source_input_group_options(cfg, url, probe)));
  runtime.session.add(
      simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(kSourceOutputEveryN)));

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = kSourceRunQueueDepth;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run = runtime.session.build(run_options);
  return runtime;
}

SessionRun build_detection_run(const AppConfig& cfg, ModelFamily family, const RtspProbe& probe) {
  static_cast<void>(detector_stage_names(family));
  apply_runtime_env_defaults();

  SessionRun runtime;

  simaai::neat::Model::Options model_options;
  model_options.preprocess.kind = simaai::neat::InputKind::Image;
  model_options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  model_options.preprocess.input_max_width = probe.width;
  model_options.preprocess.input_max_height = probe.height;
  model_options.preprocess.input_max_depth = 3;
  model_options.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  model_options.score_threshold = cfg.min_score;
  model_options.nms_iou_threshold = cfg.nms_iou;
  model_options.top_k = cfg.max_detections;
  model_options.boxdecode_original_width = probe.width;
  model_options.boxdecode_original_height = probe.height;
  runtime.model = std::make_shared<simaai::neat::Model>(cfg.model.path, model_options);

  auto input_options = runtime.model->input_appsrc_options(false);
  input_options.media_type = "video/x-raw";
  input_options.format = "RGB";
  input_options.width = probe.width;
  input_options.height = probe.height;
  input_options.depth = 3;
  runtime.session.add(simaai::neat::nodes::Input(input_options));
  runtime.session.add(simaai::neat::nodes::groups::Preprocess(*runtime.model));
  runtime.session.add(simaai::neat::nodes::groups::Infer(*runtime.model));

  switch (family) {
  case ModelFamily::YoloV8:
    runtime.session.add(simaai::neat::nodes::SimaBoxDecode(
        *runtime.model, simaai::neat::BoxDecodeType::YoloV8, cfg.min_score, cfg.nms_iou,
        cfg.max_detections, "", std::nullopt, std::nullopt, probe.width, probe.height));
    break;
  case ModelFamily::Auto:
    throw std::invalid_argument("unsupported model family for detector graph");
  }

  runtime.session.add(simaai::neat::nodes::Output());

  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
  simaai::neat::RunOptions run_options;
  run_options.preset = kRealtimeRunPreset;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run =
      runtime.session.build(std::vector<cv::Mat>{seed}, simaai::neat::RunMode::Async, run_options);
  return runtime;
}

simaai::neat::Sample run_sample_input_once(simaai::neat::Run& run,
                                           const simaai::neat::Sample& input, int timeout_ms) {
  auto outputs = run.run(simaai::neat::SampleList{input}, timeout_ms);
  if (outputs.empty()) {
    throw std::runtime_error("detector run produced no samples");
  }
  return std::move(outputs.front());
}

SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe,
                                    int stream_index) {
  const int writer_fps = effective_writer_fps(cfg, probe);
  const OptiViewVideoEncoder encoder = optiview_video_encoder_from_env();

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
  if (encoder == OptiViewVideoEncoder::Software) {
    runtime.session.add(simaai::neat::nodes::CapsI420(probe.width, probe.height, writer_fps));
    runtime.session.add(simaai::neat::nodes::H264EncodeSW(2500));
  } else {
    runtime.session.add(simaai::neat::nodes::H264EncodeSima(probe.width, probe.height, writer_fps,
                                                            2500, "baseline", "4.1"));
  }

  simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_options;
  udp_options.payload_type = 96;
  udp_options.config_interval = 1;
  udp_options.udp_host = cfg.optiview_host;
  udp_options.udp_port = optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index);
  udp_options.udp_sync = false;
  udp_options.udp_async = false;
  runtime.session.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_options));

  simaai::neat::RunOptions run_options;
  run_options.preset = kRealtimeRunPreset;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  cv::Mat seed = cv::Mat::zeros(probe.height, probe.width, CV_8UC3);
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

} // namespace multistream_object_detection_optiview
