#include "example_utils.h"

#include "asset_utils.h"
#include "support/object_detection/obj_detection_utils.h"

#include "neat/session.h"
#include "neat/nodes.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sima_examples {
bool env_int(const char* key, int* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (!end || *end != '\0')
    return false;
  *out = static_cast<int>(val);
  return true;
}

bool env_double(const char* key, double* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  double val = std::strtod(v, &end);
  if (!end || *end != '\0')
    return false;
  *out = val;
  return true;
}

bool env_string(const char* key, std::string* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  *out = v;
  return true;
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

namespace {

std::string gst_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"' || c == '\\')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

std::string trim_copy(const std::string& s) {
  const std::string whitespace = " \t\r\n";
  const size_t start = s.find_first_not_of(whitespace);
  if (start == std::string::npos)
    return "";
  const size_t end = s.find_last_not_of(whitespace);
  return s.substr(start, end - start + 1);
}

} // namespace

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

std::vector<std::string> get_args(int argc, char** argv, const std::string& key) {
  std::vector<std::string> result;
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      result.push_back(argv[i + 1]);
    }
  }
  return result;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

bool parse_int_arg(int argc, char** argv, const std::string& key, int& out) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw))
    return false;
  out = std::stoi(raw);
  return true;
}

bool parse_float_arg(int argc, char** argv, const std::string& key, float& out) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw))
    return false;
  out = std::stof(raw);
  return true;
}

bool env_flag(const char* key, bool def) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def;
  return std::string(v) != "0";
}

void require(bool cond, const std::string& msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

double time_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t time_ms_i64() {
  return static_cast<int64_t>(time_ms());
}

std::vector<std::string> read_rtsp_list(const fs::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open rtsp list: " + path.string());
  }
  std::vector<std::string> urls;
  std::string line;
  while (std::getline(in, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed[0] == '#')
      continue;
    urls.push_back(trimmed);
  }
  if (urls.empty()) {
    throw std::runtime_error("RTSP list is empty: " + path.string());
  }
  return urls;
}

bool parse_dim_from_caps(const std::string& caps, const char* key, int& out) {
  const std::string typed = std::string(key) + "=(int)";
  std::size_t pos = caps.find(typed);
  if (pos == std::string::npos) {
    const std::string plain = std::string(key) + "=";
    pos = caps.find(plain);
    if (pos == std::string::npos)
      return false;
    pos += plain.size();
  } else {
    pos += typed.size();
  }

  while (pos < caps.size() && !std::isdigit(static_cast<unsigned char>(caps[pos]))) {
    ++pos;
  }
  if (pos >= caps.size())
    return false;

  int value = 0;
  while (pos < caps.size() && std::isdigit(static_cast<unsigned char>(caps[pos]))) {
    value = value * 10 + (caps[pos] - '0');
    ++pos;
  }
  if (value <= 0)
    return false;
  out = value;
  return true;
}

bool parse_fps_from_caps(const std::string& caps, int& fps_out) {
  fps_out = -1;
  const std::string needle = "framerate=(fraction)";
  std::size_t pos = caps.find(needle);
  if (pos == std::string::npos)
    return false;
  pos += needle.size();

  while (pos < caps.size() && !std::isdigit(static_cast<unsigned char>(caps[pos]))) {
    ++pos;
  }
  if (pos >= caps.size())
    return false;

  int num = 0;
  while (pos < caps.size() && std::isdigit(static_cast<unsigned char>(caps[pos]))) {
    num = num * 10 + (caps[pos] - '0');
    ++pos;
  }
  if (pos >= caps.size() || caps[pos] != '/')
    return false;
  ++pos;

  int den = 0;
  while (pos < caps.size() && std::isdigit(static_cast<unsigned char>(caps[pos]))) {
    den = den * 10 + (caps[pos] - '0');
    ++pos;
  }

  if (num <= 0 || den <= 0)
    return false;
  if (num % den != 0)
    return false;
  const int fps = num / den;
  if (fps <= 0 || fps > 240)
    return false;
  fps_out = fps;
  return true;
}

bool probe_rtsp_stream_info(const std::string& url, const RtspProbeOptions& opt, RtspStreamInfo& out) {
  out = RtspStreamInfo{};

  simaai::neat::Session probe;
  probe.add(simaai::neat::nodes::RTSPInput(url, opt.latency_ms, opt.rtsp_tcp,
                                           /*drop_on_latency=*/true, /*buffer_mode=*/"none"));
  probe.add(simaai::neat::nodes::H264Depacketize(opt.payload_type,
                                                 /*config_interval=*/1,
                                                 /*fps=*/-1,
                                                 /*w=*/-1,
                                                 /*h=*/-1,
                                                 /*enforce_caps=*/false));
  probe.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.enable_metrics = opt.debug;
  simaai::neat::Run run = probe.build(run_opt);

  simaai::neat::Sample sample;
  simaai::neat::PullError err;
  const auto st = run.pull(5000, sample, &err);
  run.stop();
  if (st == simaai::neat::PullStatus::Ok && !sample.caps_string.empty()) {
    (void)parse_dim_from_caps(sample.caps_string, "width", out.width);
    (void)parse_dim_from_caps(sample.caps_string, "height", out.height);
    (void)parse_fps_from_caps(sample.caps_string, out.fps);
  }

  if ((out.width <= 0 || out.height <= 0) &&
      !probe_rtsp_decoded_dims(url, opt, /*tries=*/8, /*timeout_ms=*/1000, out.width, out.height)) {
    return false;
  }

  return out.width > 0 && out.height > 0;
}

bool probe_rtsp_encoded(const std::string& url, const RtspProbeOptions& opt, int fps, int w, int h,
                        int tries, int timeout_ms, bool enforce_caps) {
  simaai::neat::Session probe;
  probe.add(simaai::neat::nodes::RTSPInput(url, opt.latency_ms, opt.rtsp_tcp,
                                           /*drop_on_latency=*/true, /*buffer_mode=*/"none"));
  probe.add(simaai::neat::nodes::H264Depacketize(opt.payload_type,
                                                 /*config_interval=*/1,
                                                 /*fps=*/fps,
                                                 /*w=*/w,
                                                 /*h=*/h,
                                                 /*enforce_caps=*/enforce_caps));
  probe.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  run_opt.enable_metrics = opt.debug;
  simaai::neat::Run run = probe.build(run_opt);

  bool ok = false;
  for (int i = 0; i < tries; ++i) {
    auto out = run.pull(timeout_ms);
    if (out.has_value()) {
      ok = true;
      break;
    }
  }
  run.stop();
  return ok;
}

bool probe_rtsp_decoded_dims(const std::string& url, const RtspProbeOptions& opt, int tries,
                             int timeout_ms, int& out_w, int& out_h) {
  out_w = 0;
  out_h = 0;

  simaai::neat::Session probe;
  probe.add(simaai::neat::nodes::RTSPInput(url, opt.latency_ms, opt.rtsp_tcp,
                                           /*drop_on_latency=*/true, /*buffer_mode=*/"none"));
  probe.add(simaai::neat::nodes::H264Depacketize(opt.payload_type,
                                                 /*config_interval=*/1,
                                                 /*fps=*/-1,
                                                 /*w=*/-1,
                                                 /*h=*/-1,
                                                 /*enforce_caps=*/false));
  probe.add(simaai::neat::nodes::H264Decode(/*allocator=*/2,
                                            /*out_format=*/"NV12",
                                            /*decoder_name=*/"probe",
                                            /*raw_output=*/true,
                                            /*next_element=*/"",
                                            /*dec_width=*/-1,
                                            /*dec_height=*/-1,
                                            /*dec_fps=*/-1,
                                            /*num_buffers=*/opt.decoder_num_buffers));
  probe.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  run_opt.enable_metrics = opt.debug;
  simaai::neat::Run run = probe.build(run_opt);

  bool ok = false;
  for (int i = 0; i < tries; ++i) {
    auto out = run.pull(timeout_ms);
    if (!out.has_value() || !out->tensor.has_value())
      continue;
    int w = 0;
    int h = 0;
    if (sima_examples::infer_dims(out->tensor.value(), w, h) && w > 0 && h > 0) {
      out_w = w;
      out_h = h;
      ok = true;
      break;
    }
  }
  run.stop();
  return ok;
}

std::string resolve_resnet50_tar() {
  return sima_examples::resolve_resnet50_tar(fs::path{});
}

std::string find_boxdecode_config(const fs::path& etc_dir) {
  if (!fs::exists(etc_dir))
    return "";

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path();
    if (p.extension() == ".json" &&
        lower_copy(p.filename().string()).find("boxdecode") != std::string::npos) {
      return p.string();
    }
  }

  for (const auto& entry : fs::directory_iterator(etc_dir)) {
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path();
    if (p.extension() != ".json")
      continue;
    std::ifstream in(p);
    if (!in.is_open())
      continue;
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (lower_copy(content).find("boxdecode") != std::string::npos) {
      return p.string();
    }
  }

  return "";
}

std::string prepare_yolo_boxdecode_config(const std::string& src_path, const fs::path& root_in,
                                          int img_w, int img_h, float conf, float nms) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  std::ifstream in(src_path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open boxdecode config: " + src_path);
  }

  json j;
  in >> j;

  j["original_width"] = img_w;
  j["original_height"] = img_h;
  j["detection_threshold"] = conf;
  j["nms_iou_threshold"] = nms;
  if (!j.contains("memory") || !j["memory"].is_object()) {
    j["memory"] = json::object();
  }
  std::string override_decode;
  if (env_string("SIMA_BOXDECODE_DECODE_TYPE", &override_decode)) {
    j["decode_type"] = override_decode;
  }
  int override_topk = 0;
  if (env_int("SIMA_BOXDECODE_TOPK", &override_topk)) {
    j["topk"] = override_topk;
  }
  int override_classes = 0;
  if (env_int("SIMA_BOXDECODE_NUM_CLASSES", &override_classes)) {
    j["num_classes"] = override_classes;
  }
  int override_num_in = 0;
  if (env_int("SIMA_BOXDECODE_NUM_IN_TENSOR", &override_num_in)) {
    j["num_in_tensor"] = override_num_in;
  }
  double override_det = 0.0;
  if (env_double("SIMA_BOXDECODE_DETECTION_THRESHOLD", &override_det)) {
    j["detection_threshold"] = override_det;
  }
  double override_nms = 0.0;
  if (env_double("SIMA_BOXDECODE_NMS_IOU", &override_nms)) {
    j["nms_iou_threshold"] = override_nms;
  }
  int override_mem_cpu = 0;
  if (env_int("SIMA_BOXDECODE_MEMORY_CPU", &override_mem_cpu)) {
    j["memory"]["cpu"] = override_mem_cpu;
  }
  int override_mem_next = 0;
  if (env_int("SIMA_BOXDECODE_MEMORY_NEXT_CPU", &override_mem_next)) {
    j["memory"]["next_cpu"] = override_mem_next;
    if (j.contains("next_cpu")) {
      j["next_cpu"] = override_mem_next;
    }
  }
  int override_debug = 0;
  const bool override_debug_set = env_int("SIMA_BOXDECODE_DEBUG", &override_debug);
  if (override_debug_set) {
    j["debug"] = (override_debug != 0);
    if (!j.contains("system") || !j["system"].is_object()) {
      j["system"] = json::object();
    }
    j["system"]["debug"] = override_debug;
  }
  int override_outq = 0;
  if (env_int("SIMA_BOXDECODE_OUT_QUEUE", &override_outq)) {
    if (!j.contains("system") || !j["system"].is_object()) {
      j["system"] = json::object();
    }
    j["system"]["out_buf_queue"] = override_outq;
  }
  int override_coi = 0;
  if (env_int("SIMA_BOXDECODE_CHANNEL", &override_coi)) {
    j["channel_of_interest"] = override_coi;
  }
  if (j.contains("debug") && j["debug"].is_string()) {
    j["debug"] = false;
  }
  if (!override_debug_set && j.contains("system") && j["system"].is_object()) {
    j["system"]["debug"] = 0;
  }

  const fs::path out_path = root / "tmp" / "boxdecode_runtime.json";
  std::ofstream out(out_path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write boxdecode runtime config");
  }
  out << j.dump(2);
  return out_path.string();
}

cv::Mat load_rgb_resized(const std::string& image_path, int w, int h) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  if (w > 0 && h > 0 && (bgr.cols != w || bgr.rows != h)) {
    cv::resize(bgr, bgr, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h) {
  w = t.width();
  h = t.height();
  if ((w <= 0 || h <= 0) && t.shape.size() >= 2) {
    h = static_cast<int>(t.shape[0]);
    w = static_cast<int>(t.shape[1]);
  }
  return (w > 0 && h > 0);
}

simaai::neat::Tensor make_dummy_encoded_tensor(size_t bytes) {
  if (bytes == 0)
    bytes = 1;
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  simaai::neat::Tensor out;
  out.storage = std::move(storage);
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.read_only = true;
  out.dtype = simaai::neat::TensorDType::UInt8;
  out.layout = simaai::neat::TensorLayout::Unknown;
  out.shape = {static_cast<int64_t>(bytes)};
  out.strides_bytes = {1};
  out.semantic.encoded = simaai::neat::EncodedSpec{};
  return out;
}

bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err) {
  if (!t.is_nv12()) {
    err = "expected NV12 tensor";
    return false;
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    err = "invalid tensor dimensions";
    return false;
  }
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty()) {
    err = "NV12 copy failed";
    return false;
  }
  cv::Mat yuv(h + h / 2, w, CV_8UC1, nv12.data());
  cv::cvtColor(yuv, out, cv::COLOR_YUV2BGR_NV12);
  return true;
}

bool init_nv12_tensor_meta(simaai::neat::Tensor& out, int w, int h, std::string& err) {
  if (w <= 0 || h <= 0) {
    err = "invalid NV12 dimensions";
    return false;
  }
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }
  out.dtype = simaai::neat::TensorDType::UInt8;
  out.layout = simaai::neat::TensorLayout::HW;
  out.shape = {h, w};
  out.strides_bytes = {w, 1};
  out.byte_offset = 0;
  out.device = {simaai::neat::DeviceType::CPU, 0};
  out.read_only = true;
  simaai::neat::ImageSpec image;
  image.format = simaai::neat::ImageSpec::PixelFormat::NV12;
  out.semantic.image = image;

  simaai::neat::Plane y;
  y.role = simaai::neat::PlaneRole::Y;
  y.shape = {h, w};
  y.strides_bytes = {w, 1};
  y.byte_offset = 0;

  simaai::neat::Plane uv;
  uv.role = simaai::neat::PlaneRole::UV;
  uv.shape = {h / 2, w};
  uv.strides_bytes = {w, 1};
  uv.byte_offset = static_cast<int64_t>(w) * static_cast<int64_t>(h);

  out.planes.clear();
  out.planes.push_back(std::move(y));
  out.planes.push_back(std::move(uv));
  return true;
}

bool nv12_copy_to_cpu_tensor(const simaai::neat::Tensor& t, simaai::neat::Tensor& out,
                             std::string& err) {
  if (!t.is_nv12()) {
    err = "expected NV12 tensor";
    return false;
  }
  int w = 0;
  int h = 0;
  if (!infer_dims(t, w, h)) {
    err = "invalid tensor dimensions";
    return false;
  }
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty()) {
    err = "NV12 copy failed";
    return false;
  }
  auto storage = simaai::neat::make_cpu_owned_storage(nv12.size());
  std::memcpy(storage->data, nv12.data(), nv12.size());
  out = simaai::neat::Tensor{};
  out.storage = std::move(storage);
  if (!init_nv12_tensor_meta(out, w, h, err))
    return false;
  return true;
}

bool bgr_to_nv12_tensor(const cv::Mat& bgr, simaai::neat::Tensor& out, std::string& err) {
  if (bgr.empty() || bgr.data == nullptr) {
    err = "empty BGR frame";
    return false;
  }
  if (bgr.type() != CV_8UC3) {
    err = "expected CV_8UC3 BGR frame";
    return false;
  }

  const int w = bgr.cols;
  const int h = bgr.rows;
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }

  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous())
    i420 = i420.clone();
  const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
  const size_t bytes = i420.total() * i420.elemSize();
  if (bytes < expected) {
    err = "I420 buffer too small";
    return false;
  }

  auto storage = simaai::neat::make_cpu_owned_storage(expected);
  uint8_t* dst = static_cast<uint8_t*>(storage->data);
  const uint8_t* src = i420.data;
  const size_t y_bytes = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::memcpy(dst, src, y_bytes);

  const uint8_t* src_u = src + y_bytes;
  const uint8_t* src_v = src_u + (static_cast<size_t>(w) / 2) * (static_cast<size_t>(h) / 2);
  uint8_t* dst_uv = dst + y_bytes;
  for (int row = 0; row < h / 2; ++row) {
    const uint8_t* u_row = src_u + row * (w / 2);
    const uint8_t* v_row = src_v + row * (w / 2);
    uint8_t* uv_row = dst_uv + row * w;
    for (int col = 0; col < w / 2; ++col) {
      uv_row[col * 2] = u_row[col];
      uv_row[col * 2 + 1] = v_row[col];
    }
  }

  out = simaai::neat::Tensor{};
  out.storage = std::move(storage);
  if (!init_nv12_tensor_meta(out, w, h, err))
    return false;
  return true;
}

bool make_blank_nv12_tensor(int w, int h, simaai::neat::Tensor& out, std::string& err) {
  if ((w % 2) != 0 || (h % 2) != 0) {
    err = "NV12 requires even width/height";
    return false;
  }
  const size_t y_bytes = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t uv_bytes = y_bytes / 2;
  auto storage = simaai::neat::make_cpu_owned_storage(y_bytes + uv_bytes);
  std::memset(storage->data, 0, y_bytes + uv_bytes);
  out = simaai::neat::Tensor{};
  out.storage = std::move(storage);
  if (!init_nv12_tensor_meta(out, w, h, err))
    return false;
  return true;
}

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor output");
  }
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.empty()) {
    throw std::runtime_error("Tensor output is empty");
  }
  const size_t bytes = raw.size();
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor plane size is not a multiple of float");
  }

  const size_t elems = bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), raw.data(), elems * sizeof(float));
  return out;
}

std::vector<float> scores_from_tensor(const simaai::neat::Tensor& t, const std::string& label) {
  auto scores_full = tensor_to_floats(t);
  if (scores_full.empty()) {
    throw std::runtime_error(label + ": empty tensor output");
  }
  if (scores_full.size() < 1000) {
    throw std::runtime_error(label + ": expected at least 1000 scores, got " +
                             std::to_string(scores_full.size()));
  }
  if (scores_full.size() > 1000) {
    scores_full.resize(1000);
  }
  return scores_full;
}

std::vector<ScoredIndex> topk_with_softmax(const std::vector<float>& v, int k) {
  if (v.empty() || k <= 0)
    return {};
  const int n = static_cast<int>(v.size());
  k = std::min(k, n);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&v](int a, int b) { return v[a] > v[b]; });

  const float maxv = *std::max_element(v.begin(), v.end());
  double sum = 0.0;
  for (float x : v) {
    sum += std::exp(static_cast<double>(x - maxv));
  }

  std::vector<ScoredIndex> out;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    const int id = idx[i];
    const double prob = std::exp(static_cast<double>(v[id] - maxv)) / sum;
    out.push_back(ScoredIndex{id, v[id], static_cast<float>(prob)});
  }
  return out;
}

void check_top1(const std::vector<float>& scores, int expected_id, float min_prob,
                const std::string& label) {
  const auto top = topk_with_softmax(scores, 5);
  std::cout << "[" << label << "] top1 index=" << top[0].index << " score=" << top[0].value
            << " prob=" << top[0].prob << "\n";
  std::cout << "[" << label << "] top5:";
  for (const auto& t : top) {
    std::cout << " " << t.index << ":" << t.prob;
  }
  std::cout << "\n";

  if (expected_id < 0)
    return;

  if (top[0].index != expected_id) {
    throw std::runtime_error(label + ": top-1 mismatch: expected " + std::to_string(expected_id) +
                             " got " + std::to_string(top[0].index));
  }
  if (min_prob > 0.0f && top[0].prob < min_prob) {
    throw std::runtime_error(label + ": top-1 probability too low: " + std::to_string(top[0].prob) +
                             " < " + std::to_string(min_prob));
  }
  std::cout << "[" << label << "] top-1 matches expected class " << expected_id << "\n";
}

simaai::neat::Tensor pull_tensor_with_retry(simaai::neat::Run& run, const std::string& label,
                                            int per_try_ms, int tries) {
  for (int i = 0; i < tries; ++i) {
    auto t = run.pull_tensor(per_try_ms);
    if (t.has_value())
      return *t;
  }
  throw std::runtime_error(label + ": no tensor received (timeout/EOS)");
}

std::string h264_gst_pipeline(const fs::path& out_path, int width, int height, double fps,
                              int bitrate_kbps) {
  const int fps_i = (fps > 0.0) ? static_cast<int>(fps + 0.5) : 30;
  std::ostringstream ss;
  ss << "appsrc ! videoconvert ! video/x-raw,format=I420"
     << ",width=" << width << ",height=" << height << ",framerate=" << fps_i << "/1"
     << " ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=" << bitrate_kbps
     << " key-int-max=" << fps_i << " ! h264parse ! mp4mux ! filesink location=\""
     << gst_escape(out_path.string()) << "\"";
  return ss.str();
}

bool open_h264_writer(cv::VideoWriter& writer, const fs::path& out_path, int width, int height,
                      double fps, int bitrate_kbps, std::string* err) {
  if (width <= 0 || height <= 0) {
    if (err)
      *err = "invalid width/height for video writer";
    return false;
  }
  const double use_fps = (fps > 0.0) ? fps : 30.0;
  const std::string pipeline = h264_gst_pipeline(out_path, width, height, use_fps, bitrate_kbps);
  writer.open(pipeline, cv::CAP_GSTREAMER, 0, use_fps, cv::Size(width, height), true);
  if (!writer.isOpened()) {
    if (err)
      *err = "failed to open H264 writer";
    return false;
  }
  return true;
}

bool extract_bbox_payload(const simaai::neat::Sample& result, std::vector<uint8_t>& payload,
                          std::string& err) {
  return objdet::extract_bbox_payload(result, payload, err);
}

} // namespace sima_examples
