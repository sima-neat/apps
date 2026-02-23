#pragma once

#include "neat/session.h"
#include "neat/nodes.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace sima_examples {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out);
bool has_flag(int argc, char** argv, const std::string& key);
bool parse_int_arg(int argc, char** argv, const std::string& key, int& out);
bool parse_float_arg(int argc, char** argv, const std::string& key, float& out);
bool env_flag(const char* key, bool def = false);
void require(bool cond, const std::string& msg);
double time_ms();
int64_t time_ms_i64();

std::filesystem::path default_rtsp_list_path();
std::vector<std::string> read_rtsp_list(const std::filesystem::path& path);

struct RtspProbeOptions {
  int payload_type = 96;
  int latency_ms = 200;
  bool rtsp_tcp = true;
  bool debug = false;
  int decoder_num_buffers = 7;
};

bool parse_dim_from_caps(const std::string& caps, const char* key, int& out);
bool parse_fps_from_caps(const std::string& caps, int& fps_out);
bool probe_rtsp_encoded(const std::string& url, const RtspProbeOptions& opt, int fps, int w, int h,
                        int tries, int timeout_ms, bool enforce_caps);
bool probe_rtsp_decoded_dims(const std::string& url, const RtspProbeOptions& opt, int tries,
                             int timeout_ms, int& out_w, int& out_h);

std::filesystem::path default_goldfish_path();
bool download_file(const std::string& url, const std::filesystem::path& out_path);
std::string resolve_resnet50_tar();
std::string resolve_yolov8s_tar(const std::filesystem::path& root = {});
std::string resolve_yolov8s_tar_local_first(const std::filesystem::path& root = {},
                                            bool skip_download = false);
std::filesystem::path ensure_coco_sample(const std::filesystem::path& root = {});
std::string find_boxdecode_config(const std::filesystem::path& etc_dir);
std::string prepare_yolo_boxdecode_config(const std::string& src_path,
                                          const std::filesystem::path& root, int img_w, int img_h,
                                          float conf = 0.5f, float nms = 0.5f);

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h);

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h);

simaai::neat::Tensor make_dummy_encoded_tensor(size_t bytes);

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);
bool init_nv12_tensor_meta(simaai::neat::Tensor& out, int w, int h, std::string& err);
bool nv12_copy_to_cpu_tensor(const simaai::neat::Tensor& t, simaai::neat::Tensor& out,
                             std::string& err);
bool bgr_to_nv12_tensor(const cv::Mat& bgr, simaai::neat::Tensor& out, std::string& err);
bool make_blank_nv12_tensor(int w, int h, simaai::neat::Tensor& out, std::string& err);

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

simaai::neat::Tensor pull_tensor_with_retry(simaai::neat::Run& run, const std::string& label,
                                            int per_try_ms, int tries);

std::string h264_gst_pipeline(const std::filesystem::path& out_path, int width, int height,
                              double fps, int bitrate_kbps = 4000);

bool open_h264_writer(cv::VideoWriter& writer, const std::filesystem::path& out_path, int width,
                      int height, double fps, int bitrate_kbps = 4000, std::string* err = nullptr);

bool extract_bbox_payload(const simaai::neat::Sample& result, std::vector<uint8_t>& payload,
                          std::string& err);

using OptiViewObject = simaai::neat::OptiViewObject;
using OptiViewOptions = simaai::neat::OptiViewChannelOptions;
using OptiViewSender = simaai::neat::OptiViewJsonOutput;

inline std::vector<std::string> optiview_default_labels() {
  return simaai::neat::OptiViewDefaultLabels();
}

inline std::string optiview_make_json(int64_t timestamp_ms, const std::string& frame_id,
                                      const std::vector<OptiViewObject>& objects,
                                      const std::vector<std::string>& labels) {
  return simaai::neat::OptiViewMakeJson(timestamp_ms, frame_id, objects, labels);
}

} // namespace sima_examples
