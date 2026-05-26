#pragma once

#include "config_api.cpp"
#include "model_family_api.cpp"
#include "support/runtime/example_utils.h"

#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"
#include <nodes/io/MetadataSender.h>

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace multi_stream_object_detector {

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

int insight_video_port_for_stream(int port_base, int stream_index);
int insight_metadata_port_for_stream(int port_base, int stream_index);
void apply_runtime_env_defaults();
int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe);
double producer_emit_period_s(const AppConfig& cfg, const RtspProbe& probe);
simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_input_group_options(const AppConfig& cfg, const std::string& url,
                                 const RtspProbe& probe);
RtspProbe probe_rtsp(const AppConfig& cfg, const std::string& url);
std::vector<std::string> detector_stage_names(ModelFamily family);
SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe);
SessionRun build_detection_run(const AppConfig& cfg, ModelFamily family, const RtspProbe& probe);
simaai::neat::Sample run_sample_input_once(simaai::neat::Run& run,
                                           const simaai::neat::Sample& input, int timeout_ms);
SessionRun build_insight_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index);
simaai::neat::MetadataSender build_insight_metadata_output(const AppConfig& cfg, int stream_index);

} // namespace multi_stream_object_detector
