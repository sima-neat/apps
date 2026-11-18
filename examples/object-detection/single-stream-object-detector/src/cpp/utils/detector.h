#pragma once

// This example's source setup and saved frames.
//
// The source options and probes follow source.type: RTSP and HTTP MJPEG streams are decoded by
// the source group, and a USB camera's MJPEG is decoded by a separate node so its own JPEG can
// go to Insight unchanged.

#include "app_config.h"
#include "detections.h"
#include "sources.h"

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"

#include <optional>
#include <vector>

namespace utils {

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_rtsp_source_options(const AppConfig& cfg, const StreamInfo& geometry);
simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions
make_http_mjpeg_source_options(const AppConfig& cfg, const StreamInfo& geometry);
simaai::neat::UsbCameraInputOptions make_usb_camera_options(const AppConfig& cfg);
simaai::neat::SimaDecodeOptions make_usb_decode_options(const AppConfig& cfg,
                                                        simaai::neat::FormatTag format);

// The source as one graph: a decoded RTSP or HTTP MJPEG group, or the USB camera node.
simaai::neat::Graph
make_source_graph(const AppConfig& cfg, const StreamInfo& geometry,
                  const std::optional<simaai::neat::UsbCameraInfo>& usb_camera = std::nullopt);

// Width, height and frame rate of the source; source.<type>.fps, when set, wins. A USB camera
// uses its configured mode. An RTSP stream is probed with ffprobe and OpenCV, an HTTP stream
// with ffprobe and, when that gives no size, one decoded frame.
StreamInfo resolve_source_geometry(const AppConfig& cfg);

// Every output.save_every-th frame, writes the frame with its boxes to output.save_dir.
void maybe_save_debug_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample& sample,
                            const std::vector<Box>& boxes);

} // namespace utils
