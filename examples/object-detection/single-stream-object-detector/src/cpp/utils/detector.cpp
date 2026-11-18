#include "detector.h"

#include "frames.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace utils {
namespace {

void require_mjpeg_fps(const AppConfig& cfg, const StreamInfo& geometry) {
  if (cfg.source_codec == SourceCodec::Mjpeg && geometry.fps <= 0) {
    throw std::runtime_error(
        std::string("MJPEG source did not provide a valid frame rate; set source.") +
        source_type_name(cfg.source_type) + ".fps or use a source with probeable FPS metadata");
  }
}

} // namespace

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_rtsp_source_options(const AppConfig& cfg, const StreamInfo& geometry) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.source_fps = geometry.fps;
  opt.codec = cfg.source_codec == SourceCodec::H264 ? simaai::neat::nodes::groups::RtspCodec::H264
              : cfg.source_codec == SourceCodec::H265
                  ? simaai::neat::nodes::groups::RtspCodec::H265
                  : simaai::neat::nodes::groups::RtspCodec::MJPEG;
  if (cfg.source_codec == SourceCodec::H264) {
    opt.payload_type = 96;
    opt.fallback_h264_width = geometry.width;
    opt.fallback_h264_height = geometry.height;
  } else if (cfg.source_codec == SourceCodec::H265) {
    opt.payload_type = 96;
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  } else {
    opt.mjpeg_payload_type = 26;
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  }
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions
make_http_mjpeg_source_options(const AppConfig& cfg, const StreamInfo& geometry) {
  simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.source_fps = geometry.fps;
  opt.ssl_strict = cfg.ssl_strict;
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::UsbCameraInputOptions make_usb_camera_options(const AppConfig& cfg) {
  simaai::neat::UsbCameraInputOptions opt;
  opt.camera_name = cfg.camera;
  opt.width = static_cast<std::uint32_t>(cfg.source_width);
  opt.height = static_cast<std::uint32_t>(cfg.source_height);
  opt.framerate_num = static_cast<std::uint32_t>(cfg.source_fps);
  return opt;
}

simaai::neat::SimaDecodeOptions make_usb_decode_options(const AppConfig& cfg,
                                                        simaai::neat::FormatTag format) {
  simaai::neat::SimaDecodeOptions opt;
  opt.type = simaai::neat::SimaDecodeType::MJPEG;
  opt.raw_output = true;
  opt.out_format = format;
  opt.dec_width = cfg.source_width;
  opt.dec_height = cfg.source_height;
  opt.dec_fps = cfg.source_fps;
  return opt;
}

simaai::neat::Graph
make_source_graph(const AppConfig& cfg, const StreamInfo& geometry,
                  const std::optional<simaai::neat::UsbCameraInfo>& usb_camera) {
  if (cfg.source_type == SourceType::Usb) {
    simaai::neat::Graph camera("usb_camera");
    camera.add(
        simaai::neat::nodes::UsbCameraInput(make_usb_camera_options(cfg), usb_camera.value()));
    return camera;
  }
  if (cfg.source_type == SourceType::Rtsp) {
    return simaai::neat::nodes::groups::RtspDecodedInput(make_rtsp_source_options(cfg, geometry));
  }
  return simaai::neat::nodes::groups::HttpMjpegDecodedInput(
      make_http_mjpeg_source_options(cfg, geometry));
}

StreamInfo resolve_source_geometry(const AppConfig& cfg) {
  if (cfg.source_type == SourceType::Usb) {
    return {cfg.source_width, cfg.source_height, cfg.source_fps};
  }
  if (cfg.source_type == SourceType::Rtsp) {
    StreamInfo geometry = probe_rtsp_stream_info(cfg.source_url, cfg.tcp);
    if (cfg.source_fps > 0) {
      geometry.fps = cfg.source_fps;
    }
    require_mjpeg_fps(cfg, geometry);
    return geometry;
  }
  StreamInfo geometry = probe_ffprobe(cfg.source_url, false, cfg.ssl_strict);
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }
  require_mjpeg_fps(cfg, geometry);
  if (geometry.width <= 0 || geometry.height <= 0) {
    fill_missing(geometry, probe_decoded_size(make_source_graph(cfg, {0, 0, geometry.fps})));
  }
  return geometry;
}

void maybe_save_debug_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample& sample,
                            const std::vector<Box>& boxes) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0) {
    return;
  }

  const auto tensors = simaai::neat::tensors_from_sample(joined_field(sample, "frame", 0U), true);
  cv::Mat bgr = tensor_bgr_from_decoded(tensors.front());
  draw_boxes(bgr, boxes, static_cast<float>(cfg.min_score));
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), bgr)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

} // namespace utils
