#pragma once

#include "examples/tracking/multi-camera-people-detection-and-tracking-insight/cpp/utils/config_api.cpp"
#include "support/runtime/example_utils.h"

#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"
#include <nodes/io/MetadataSender.h>

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>

namespace multi_camera_people_tracking {

constexpr int kSourceStartupPullTimeoutMs = 50000;
constexpr int kSourcePullTimeoutMs = 10000;
constexpr double kSourceStartupStaggerS = 0.5;

struct RtspProbe {
  int width = 0;
  int height = 0;
  int fps = 0;
};

struct SessionRun {
  simaai::neat::Session session;
  simaai::neat::Run run;
  std::shared_ptr<simaai::neat::Model> model;
};

simaai::neat::Sample run_sample_input_once(simaai::neat::Run& run,
                                           const simaai::neat::Sample& input, int timeout_ms);
int insight_video_port_for_stream(int port_base, int stream_index);
int insight_metadata_port_for_stream(int port_base, int stream_index);
int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe);
RtspProbe probe_rtsp(const std::string& url);
SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe);
SessionRun build_detection_run(const AppConfig& cfg, const RtspProbe& probe);
SessionRun build_insight_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index);
simaai::neat::MetadataSender build_insight_metadata_output(const AppConfig& cfg, int stream_index);

} // namespace multi_camera_people_tracking
