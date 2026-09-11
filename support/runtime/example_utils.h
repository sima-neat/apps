#pragma once

#include <neat.h>
#include "neat/nodes.h"
#include "neat/models.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace sima_examples {

void require(bool cond, const std::string& msg);
double time_ms();

struct RtspProbeOptions {
  int payload_type = 96;
  int latency_ms = 200;
  bool rtsp_tcp = true;
  bool debug = false;
  int decoder_num_buffers = 7;
};

struct RtspStreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt,
                            RtspStreamInfo& out);

std::filesystem::path default_goldfish_path();
bool download_file(const std::string& url, const std::filesystem::path& out_path);

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h);

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h);

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t);
std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t, const std::string& label);
std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k);
void check_top1(const std::vector<float>& scores, int expected_id, float min_prob,
                const std::string& label);

struct MetadataBox {
  std::string id;
  std::string label;
  float confidence = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

std::string metadata_boxes_data_json(const std::string& array_key,
                                     const std::vector<MetadataBox>& boxes);

} // namespace sima_examples
