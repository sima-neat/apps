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

std::filesystem::path default_goldfish_path();
bool download_file(const std::string& url, const std::filesystem::path& out_path);

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h);

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);
// A decoded 4:2:2 frame (NV16, as most USB webcams decode) as packed 4:2:2 (CV_8UC2, Y0 U0 Y1
// V0), every row keeping its own chroma. Reads the Y and UV planes at their own offsets and
// strides; cv::COLOR_YUV2BGR_YUY2 converts the result.
bool nv16_to_yuy2(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);
// The same frame to BGR on the CPU, for occasional saved frames.
bool nv16_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);

struct ScoredIndex {
  int index = -1;
  float value = 0.0f;
  float prob = 0.0f;
};

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t);
std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k);

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
