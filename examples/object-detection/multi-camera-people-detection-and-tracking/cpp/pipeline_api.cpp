#pragma once

#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/config_api.cpp"
#include "support/runtime/example_utils.h"

#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"

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

struct QuantTessCpuPreproc {
  int width = 640;
  int height = 640;
  bool aspect_ratio = false;
  std::string padding_type = "CENTER";
};

struct SessionRun {
  simaai::neat::Session session;
  simaai::neat::Run run;
};

simaai::neat::Run build_tensor_input_run(
    simaai::neat::Session& session, const cv::Mat& seed,
    simaai::neat::RunMode mode = simaai::neat::RunMode::Async,
    const simaai::neat::RunOptions& options = {});
simaai::neat::Sample run_tensor_input_once(simaai::neat::Run& run, const cv::Mat& input,
                                           int timeout_ms);
int optiview_video_port_for_stream(int port_base, int stream_index);
int optiview_json_port_for_stream(int port_base, int stream_index);
int effective_writer_fps(const AppConfig& cfg, const RtspProbe& probe);
RtspProbe probe_rtsp(const std::string& url);
std::shared_ptr<simaai::neat::Model> load_detector_model(const AppConfig& cfg);
QuantTessCpuPreproc read_preproc_contract(const simaai::neat::Model& model);
SessionRun build_source_run(const AppConfig& cfg, const std::string& url, const RtspProbe& probe);
SessionRun build_detection_run(const AppConfig& cfg, const simaai::neat::Model& model,
                               const RtspProbe& probe, const QuantTessCpuPreproc& quant_preproc);
SessionRun build_optiview_video_run(const AppConfig& cfg, const RtspProbe& probe,
                                    int stream_index);
sima_examples::OptiViewSender build_optiview_json_output(const AppConfig& cfg, int stream_index);

} // namespace multi_camera_people_tracking
