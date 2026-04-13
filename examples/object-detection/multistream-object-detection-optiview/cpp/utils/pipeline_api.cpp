#pragma once

#include "config_api.cpp"
#include "model_family_api.cpp"
#include "support/runtime/example_utils.h"

#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace multistream_object_detection_optiview {

struct RtspProbe {
  int width = 0;
  int height = 0;
  int fps = 0;
};

enum class OptiViewVideoEncoder {
  Hardware,
  Software,
};

struct SessionRun {
  simaai::neat::Session session;
  simaai::neat::Run run;
  std::shared_ptr<simaai::neat::Model> model;
};

int optiview_video_port_for_stream(int port_base, int stream_index);
int optiview_json_port_for_stream(int port_base, int stream_index);
void apply_runtime_env_defaults();
int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe);
OptiViewVideoEncoder optiview_video_encoder_from_env();
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
SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe, int stream_index);
sima_examples::OptiViewSender build_optiview_json_output(const AppConfig& cfg, int stream_index);

} // namespace multistream_object_detection_optiview
