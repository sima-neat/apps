#include "pipeline_api.cpp"

#include "sample_utils_api.cpp"

#include <cstdlib>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace multistream_yolox_yolov8_optiview {
namespace {

constexpr const char* kDefaultModelNumBuffers = "3";
constexpr const char* kDefaultDecoderNumBuffers = "7";

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

int source_output_every_n(const AppConfig& cfg, const RtspProbe& probe) {
  static_cast<void>(cfg);
  static_cast<void>(probe);
  // Keep encoded H.264 intact; producer pacing throttles detector admission after decode.
  return 1;
}

double producer_emit_period_s(const AppConfig& cfg, const RtspProbe& probe) {
  static_cast<void>(probe);
  if (cfg.fps <= 0) {
    return 0.0;
  }
  return 1.0 / cfg.fps;
}

RtspProbe probe_rtsp(const AppConfig& cfg, const std::string& url) {
  apply_graphpipes_runtime_defaults();

  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.decoder_num_buffers = graphpipes_decoder_num_buffers();

  sima_examples::RtspStreamInfo info;
  if (!sima_examples::probe_rtsp_stream_info(url, probe_options, info)) {
    throw std::runtime_error("failed to open RTSP source for probing: " + url);
  }

  simaai::neat::Session probe;
  probe.add(simaai::neat::nodes::RTSPInput(url, cfg.latency_ms, cfg.tcp, /*drop_on_latency=*/true,
                                           /*buffer_mode=*/"none"));
  probe.add(simaai::neat::nodes::H264Depacketize(
      /*payload_type=*/96,
      /*h264_parse_config_interval=*/1,
      /*h264_fps=*/-1,
      /*h264_width=*/-1,
      /*h264_height=*/-1,
      /*enforce_h264_caps=*/false));
  probe.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_options;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run run = probe.build(run_options);

  const auto sample = run.pull(5000);
  run.stop();
  if (!sample.has_value() || sample->caps_string.empty()) {
    throw std::runtime_error("failed to probe RTSP encoded caps: " + url);
  }

  RtspProbe probe_result;
  probe_result.width = info.width;
  probe_result.height = info.height;
  probe_result.fps = info.fps;
  probe_result.encoded_caps = sample->caps_string;
  probe_result.encoded_caps_appsrc = probe_result.encoded_caps;

  int parsed_width = 0;
  int parsed_height = 0;
  int parsed_fps = 0;
  const bool have_width =
      sima_examples::parse_dim_from_caps(probe_result.encoded_caps, "width", parsed_width);
  const bool have_height =
      sima_examples::parse_dim_from_caps(probe_result.encoded_caps, "height", parsed_height);
  if (probe_result.fps <= 0 &&
      sima_examples::parse_fps_from_caps(probe_result.encoded_caps, parsed_fps)) {
    probe_result.fps = parsed_fps;
  }
  if (!have_width) {
    probe_result.encoded_caps_appsrc += ",width=(int)" + std::to_string(probe_result.width);
  }
  if (!have_height) {
    probe_result.encoded_caps_appsrc += ",height=(int)" + std::to_string(probe_result.height);
  }
  if (probe_result.width <= 0 || probe_result.height <= 0) {
    throw std::runtime_error("failed to probe RTSP frame size: " + url);
  }
  return probe_result;
}

std::vector<std::string> detector_stage_names(ModelFamily family) {
  switch (family) {
  case ModelFamily::YoloV8:
    return {"input", "preproc", "mla", "sima_box_decode", "output"};
  case ModelFamily::YoloX:
    throw std::invalid_argument(yolox_not_supported_message());
  case ModelFamily::Auto:
    break;
  }
  throw std::invalid_argument("unsupported model family for detector graph");
}

void apply_graphpipes_runtime_defaults() {
  set_env_if_unset("SIMA_FORCE_MODEL_NUM_BUFFERS", kDefaultModelNumBuffers);
  set_env_if_unset("SIMA_FORCE_DECODER_NUM_BUFFERS", kDefaultDecoderNumBuffers);
  set_env_if_unset("SIMA_FORCE_DECODER_POOL_BUFFERS", kDefaultDecoderNumBuffers);
}

int graphpipes_decoder_num_buffers() {
  return 7;
}

SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe) {
  apply_graphpipes_runtime_defaults();
  SessionRun runtime;
  runtime.session.add(
      simaai::neat::nodes::RTSPInput(url, cfg.latency_ms, cfg.tcp, /*drop_on_latency=*/true,
                                     /*buffer_mode=*/"none"));
  runtime.session.add(simaai::neat::nodes::Queue());
  runtime.session.add(simaai::neat::nodes::H264Depacketize(
      /*payload_type=*/96,
      /*h264_parse_config_interval=*/1,
      /*h264_fps=*/probe.fps > 0 ? probe.fps : -1,
      /*h264_width=*/probe.width,
      /*h264_height=*/probe.height,
      /*enforce_h264_caps=*/true));
  runtime.session.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  runtime.run = runtime.session.build(run_options);
  return runtime;
}

SessionRun build_decode_run(const AppConfig& cfg, const RtspProbe& probe,
                            const simaai::neat::Sample& bootstrap_encoded) {
  static_cast<void>(cfg);
  apply_graphpipes_runtime_defaults();

  simaai::neat::InputOptions input_options;
  input_options.media_type = "video/x-h264";
  input_options.caps_override = probe.encoded_caps_appsrc;
  input_options.use_simaai_pool = false;

  SessionRun runtime;
  runtime.session.add(simaai::neat::nodes::Input(input_options));

  simaai::neat::H264ParseOptions parse_options;
  parse_options.config_interval = 1;
  parse_options.enforce_caps = true;
  parse_options.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
  parse_options.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
  runtime.session.add(simaai::neat::nodes::H264Parse(parse_options));
  runtime.session.add(simaai::neat::nodes::H264Decode(
      /*sima_allocator_type=*/2,
      /*out_format=*/"NV12",
      /*decoder_name=*/"",
      /*raw_output=*/true,
      /*next_element=*/"",
      /*dec_width=*/-1,
      /*dec_height=*/-1,
      /*dec_fps=*/-1,
      /*num_buffers=*/graphpipes_decoder_num_buffers()));
  runtime.session.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run = runtime.session.build(
      deep_copy_encoded_sample(bootstrap_encoded, probe.encoded_caps_appsrc),
      simaai::neat::RunMode::Async, run_options);
  return runtime;
}

SessionRun build_detection_run(const AppConfig& cfg, ModelFamily family, const RtspProbe& probe) {
  static_cast<void>(detector_stage_names(family));
  apply_graphpipes_runtime_defaults();

  SessionRun runtime;

  simaai::neat::Model::Options model_options;
  model_options.media_type = "video/x-raw";
  model_options.format = "NV12";
  model_options.input_max_width = probe.width;
  model_options.input_max_height = probe.height;
  model_options.input_max_depth = 1;
  runtime.model = std::make_shared<simaai::neat::Model>(cfg.model.path, model_options);

  auto input_options = runtime.model->input_appsrc_options(false);
  input_options.media_type = "video/x-raw";
  input_options.format = "NV12";
  input_options.width = probe.width;
  input_options.height = probe.height;
  input_options.depth = 1;
  runtime.session.add(simaai::neat::nodes::Input(input_options));
  runtime.session.add(simaai::neat::nodes::groups::Preprocess(*runtime.model));
  runtime.session.add(simaai::neat::nodes::groups::Infer(*runtime.model));

  switch (family) {
  case ModelFamily::YoloV8:
    runtime.session.add(simaai::neat::nodes::SimaBoxDecode(
        *runtime.model, "yolov8", probe.width, probe.height, cfg.min_score, cfg.nms_iou,
        cfg.max_detections));
    break;
  case ModelFamily::YoloX:
    throw std::invalid_argument(yolox_not_supported_message());
  case ModelFamily::Auto:
    throw std::invalid_argument("unsupported model family for detector graph");
  }

  runtime.session.add(simaai::neat::nodes::Output());

  std::string nv12_error;
  simaai::neat::Tensor seed_tensor;
  if (!sima_examples::make_blank_nv12_tensor(probe.width, probe.height, seed_tensor, nv12_error)) {
    throw std::runtime_error("failed to build NV12 detector seed: " + nv12_error);
  }
  simaai::neat::RunOptions run_options;
  run_options.queue_depth = 1;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  runtime.run = runtime.session.build(seed_tensor, simaai::neat::RunMode::Async, run_options);
  return runtime;
}

simaai::neat::Sample run_sample_input_once(simaai::neat::Run& run,
                                           const simaai::neat::Sample& input, int timeout_ms) {
  return run.run(input, timeout_ms);
}

SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index,
                                    const simaai::neat::Sample* bootstrap_encoded) {
  if (cfg.video_mode == VideoMode::Clean) {
    if (bootstrap_encoded == nullptr) {
      throw std::invalid_argument(
          "clean OptiView video run requires a real encoded bootstrap sample");
    }

    simaai::neat::InputOptions input_options;
    input_options.media_type = "video/x-h264";
    input_options.caps_override = probe.encoded_caps_appsrc;
    input_options.use_simaai_pool = false;

    SessionRun runtime;
    runtime.session.add(simaai::neat::nodes::Input(input_options));

    simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_options;
    udp_options.h264_caps = probe.encoded_caps_appsrc;
    udp_options.payload_type = 96;
    udp_options.config_interval = 1;
    udp_options.udp_host = cfg.optiview_host;
    udp_options.udp_port =
        optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index);
    udp_options.udp_sync = false;
    udp_options.udp_async = false;
    runtime.session.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_options));

    simaai::neat::RunOptions run_options;
    run_options.queue_depth = 1;
    run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
    runtime.run = runtime.session.build(
        deep_copy_encoded_sample(*bootstrap_encoded, probe.encoded_caps_appsrc),
        simaai::neat::RunMode::Async, run_options);
    return runtime;
  }

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
