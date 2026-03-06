#include "support/runtime/example_utils.h"

#include "neat/session.h"
#include "neat/models.h"
#include "neat/node_groups.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_s(const Clock::time_point& t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

int get_int_arg(int argc, char** argv, const std::string& key, int def_val) {
  std::string tmp;
  if (!sima_examples::get_arg(argc, argv, key, tmp))
    return def_val;
  try {
    return std::stoi(tmp);
  } catch (...) {
    return def_val;
  }
}

float get_float_arg(int argc, char** argv, const std::string& key, float def_val) {
  std::string tmp;
  if (!sima_examples::get_arg(argc, argv, key, tmp))
    return def_val;
  try {
    return std::stof(tmp);
  } catch (...) {
    return def_val;
  }
}

std::vector<float> parse_floats_csv(const std::string& s) {
  std::vector<float> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) {
        out.push_back(std::stof(cur));
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty())
    out.push_back(std::stof(cur));
  return out;
}

std::string to_upper(std::string s) {
  for (char& c : s) {
    if (c >= 'a' && c <= 'z')
      c = static_cast<char>(c - 'a' + 'A');
  }
  return s;
}

std::string to_lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

struct DepthModelProfile {
  std::string name;
  std::string input_format; // "BGR" or "RGB"
  int default_size = 256;
  bool depth_column_major = false;
};

DepthModelProfile detect_model_profile(const std::string& model_path) {
  const std::string lower = to_lower(fs::path(model_path).filename().string());
  if (lower.find("depth_anything_v2") != std::string::npos) {
    return {"depth_anything_v2_vits", "RGB", 518, true};
  }
  return {"midas_v21_small_256", "BGR", 256, false};
}

std::string resolve_sample_video() {
  const char* env = std::getenv("SIMA_MIDAS_SAMPLE_VIDEO");
  if (env && *env && fs::exists(env))
    return std::string(env);

  const fs::path local = fs::path("tmp") / "midas_sample.mp4";
  if (fs::exists(local))
    return local.string();

  const std::string url = "https://sample-videos.com/video123/mp4/240/"
                          "big_buck_bunny_240p_1mb.mp4";
  if (!sima_examples::download_file(url, local))
    return "";
  return local.string();
}

size_t dtype_bytes(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
  case simaai::neat::TensorDType::Int8:
    return 1;
  case simaai::neat::TensorDType::UInt16:
  case simaai::neat::TensorDType::Int16:
    return 2;
  case simaai::neat::TensorDType::Int32:
  case simaai::neat::TensorDType::Float32:
    return 4;
  case simaai::neat::TensorDType::Float64:
    return 8;
  }
  return 1;
}

float read_elem(const uint8_t* data, size_t idx, simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return static_cast<float>(reinterpret_cast<const uint8_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int8:
    return static_cast<float>(reinterpret_cast<const int8_t*>(data)[idx]);
  case simaai::neat::TensorDType::UInt16:
    return static_cast<float>(reinterpret_cast<const uint16_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int16:
    return static_cast<float>(reinterpret_cast<const int16_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int32:
    return static_cast<float>(reinterpret_cast<const int32_t*>(data)[idx]);
  case simaai::neat::TensorDType::Float32:
    return reinterpret_cast<const float*>(data)[idx];
  case simaai::neat::TensorDType::Float64:
    return static_cast<float>(reinterpret_cast<const double*>(data)[idx]);
  }
  return 0.0f;
}

bool depth_tensor_to_u8(const simaai::neat::Tensor& t, int fallback_w, int fallback_h,
                        bool column_major_order, cv::Mat& depth_u8, float* out_min,
                        float* out_max) {
  if (!t.is_dense())
    return false;

  // Do not rely on Tensor::width()/height() for model outputs with a leading batch
  // dimension (e.g. MiDaS commonly returns 1x256x256x1 with layout=HWC). In that case
  // width()/height()/channels() can be interpreted as w=256,h=1,c=256, which is not a
  // depth image. Match the Python example and infer spatial dims from non-singleton axes.
  std::vector<int> spatial_dims;
  spatial_dims.reserve(t.shape.size());
  for (auto d64 : t.shape) {
    const int d = static_cast<int>(d64);
    if (d > 1)
      spatial_dims.push_back(d);
  }

  int h = fallback_h;
  int w = fallback_w;
  if (spatial_dims.size() >= 2) {
    h = spatial_dims[0];
    w = spatial_dims[1];
  } else if (spatial_dims.size() == 1) {
    h = spatial_dims[0];
    w = spatial_dims[0];
  }
  if (w <= 0 || h <= 0)
    return false;

  const size_t elem_size = dtype_bytes(t.dtype);
  const size_t needed = static_cast<size_t>(w) * static_cast<size_t>(h) * elem_size;
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.size() < needed)
    return false;
  const uint8_t* data = raw.data();
  cv::Mat depth_f(h, w, CV_32FC1);

  float minv = std::numeric_limits<float>::infinity();
  float maxv = -std::numeric_limits<float>::infinity();

  for (int y = 0; y < h; ++y) {
    float* row = depth_f.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const size_t idx = column_major_order
                             ? (static_cast<size_t>(x) * static_cast<size_t>(h) +
                                static_cast<size_t>(y))
                             : (static_cast<size_t>(y) * static_cast<size_t>(w) +
                                static_cast<size_t>(x));
      float v = read_elem(data, idx, t.dtype);
      row[x] = v;
      minv = std::min(minv, v);
      maxv = std::max(maxv, v);
    }
  }

  if (out_min)
    *out_min = minv;
  if (out_max)
    *out_max = maxv;

  if (std::isfinite(minv) && std::isfinite(maxv) && maxv > minv) {
    cv::normalize(depth_f, depth_u8, 0, 255, cv::NORM_MINMAX);
    depth_u8.convertTo(depth_u8, CV_8U);
  } else {
    depth_u8 = cv::Mat(h, w, CV_8U, cv::Scalar(0));
  }
  return true;
}

bool tensor_to_depth_bgr(const simaai::neat::Tensor& t, int fallback_w, int fallback_h,
                         bool column_major_order, bool use_colormap, cv::Mat& bgr_out,
                         float* out_min, float* out_max) {
  cv::Mat depth_u8;
  if (!depth_tensor_to_u8(t, fallback_w, fallback_h, column_major_order, depth_u8, out_min,
                          out_max)) {
    return false;
  }

  if (use_colormap) {
    cv::applyColorMap(depth_u8, bgr_out, cv::COLORMAP_INFERNO);
  } else {
    cv::cvtColor(depth_u8, bgr_out, cv::COLOR_GRAY2BGR);
  }
  return true;
}

const simaai::neat::Tensor* find_first_tensor(const simaai::neat::Sample& s) {
  if (s.kind == simaai::neat::SampleKind::Tensor && s.tensor.has_value()) {
    return &(*s.tensor);
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : s.fields) {
      if (const auto* t = find_first_tensor(field)) {
        return t;
      }
    }
  }
  return nullptr;
}

const simaai::neat::Tensor* find_depth_tensor(const simaai::neat::Sample& s) {
  const simaai::neat::Tensor* first = find_first_tensor(s);
  if (s.kind == simaai::neat::SampleKind::Tensor && s.tensor.has_value()) {
    const auto& t = *s.tensor;
    const bool looks_like_depth =
        !t.semantic.image.has_value() && t.dtype != simaai::neat::TensorDType::UInt8;
    if (looks_like_depth)
      return &t;
    return first;
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : s.fields) {
      if (field.kind != simaai::neat::SampleKind::Tensor || !field.tensor.has_value())
        continue;
      const auto& t = *field.tensor;
      const bool looks_like_depth =
          !t.semantic.image.has_value() && t.dtype != simaai::neat::TensorDType::UInt8;
      if (looks_like_depth)
        return &t;
    }
  }
  return first;
}

bool tensor_to_bgr_mat(const simaai::neat::Tensor& t, cv::Mat& bgr_out) {
  if (!t.is_dense())
    return false;
  if (t.dtype != simaai::neat::TensorDType::UInt8)
    return false;
  if (!t.semantic.image.has_value() ||
      t.semantic.image->format != simaai::neat::ImageSpec::PixelFormat::BGR) {
    return false;
  }
  const int w = t.width();
  const int h = t.height();
  const int c = t.channels();
  if (w <= 0 || h <= 0 || c != 3)
    return false;

  simaai::neat::Mapping map = t.map_read();
  if (!map.data)
    return false;
  const int64_t stride =
      !t.strides_bytes.empty() ? t.strides_bytes[0] : static_cast<int64_t>(w) * c;
  cv::Mat view(h, w, CV_8UC3, const_cast<uint8_t*>(static_cast<const uint8_t*>(map.data)),
               static_cast<size_t>(stride));
  bgr_out = view.clone();
  return true;
}

struct FetchedFrame {
  cv::Mat frame;
  double pull_s = 0.0;
  double tensor_s = 0.0;
  bool skip_profile = false;
};

class StageProfiler {
public:
  explicit StageProfiler(bool enabled) : enabled_(enabled), skip_next_frame_(enabled) {}

  bool begin_frame(bool skip_current) {
    if (!enabled_)
      return false;
    if (skip_next_frame_) {
      skip_next_frame_ = false;
      return false;
    }
    if (skip_current)
      return false;
    return true;
  }

  void skip_next_frame() {
    if (enabled_)
      skip_next_frame_ = true;
  }

  void add_pull(double dt_s, bool include) { add(pull_, dt_s, include); }
  void add_tensor(double dt_s, bool include) { add(tensor_, dt_s, include); }
  void add_model(double dt_s, bool include) { add(model_, dt_s, include); }
  void add_post(double dt_s, bool include) { add(post_, dt_s, include); }
  void add_write(double dt_s, bool include) { add(write_, dt_s, include); }
  void add_frame(double dt_s, bool include) { add(frame_, dt_s, include); }

  void print(const std::string& label) const {
    if (!enabled_)
      return;
    std::cout << "--------------------------\n";
    std::cout << "[PROFILE] " << label << "\n";
    std::cout << "[PROFILE]   stage         avg(ms)   max(ms)   n\n";
    print_row("pull", pull_);
    print_row("tensor", tensor_);
    print_row("model", model_);
    print_row("post", post_);
    print_row("write", write_);
    print_row("frame", frame_);
  }

private:
  struct Stat {
    double sum_s = 0.0;
    double max_s = 0.0;
    int n = 0;
  };

  static void add(Stat& s, double dt_s, bool include) {
    if (!include)
      return;
    s.sum_s += dt_s;
    s.max_s = std::max(s.max_s, dt_s);
    ++s.n;
  }

  static void print_row(const char* name, const Stat& s) {
    if (s.n <= 0)
      return;
    const double avg_ms = 1000.0 * s.sum_s / static_cast<double>(s.n);
    const double max_ms = 1000.0 * s.max_s;
    std::cout << "[PROFILE]   " << std::left << std::setw(12) << name << std::right
              << std::setw(8) << std::fixed << std::setprecision(1) << avg_ms
              << std::setw(9) << std::fixed << std::setprecision(1) << max_ms << std::setw(5)
              << s.n << "\n";
  }

  bool enabled_ = false;
  bool skip_next_frame_ = false;
  Stat pull_;
  Stat tensor_;
  Stat model_;
  Stat post_;
  Stat write_;
  Stat frame_;
};

bool process_stream(simaai::neat::Model& model,
                    const std::function<bool(FetchedFrame&)>& next_frame,
                    cv::VideoWriter& writer, int max_frames, int log_every, bool profile,
                    float alpha, bool use_colormap, const DepthModelProfile& model_profile) {
  simaai::neat::Model::Runner model_run;
  StageProfiler profiler(profile);
  int processed = 0;
  std::optional<Clock::time_point> log_window_start;
  int log_window_start_frames = 0;

  while (processed < max_frames) {
    const auto t_frame0 = Clock::now();

    FetchedFrame fetched;
    if (!next_frame(fetched))
      break;
    cv::Mat frame = fetched.frame;
    if (frame.empty())
      continue;
    if (frame.type() != CV_8UC3) {
      cv::Mat converted;
      frame.convertTo(converted, CV_8UC3);
      frame = std::move(converted);
    }

    const bool profile_frame = profiler.begin_frame(fetched.skip_profile);
    profiler.add_pull(fetched.pull_s, profile_frame);
    profiler.add_tensor(fetched.tensor_s, profile_frame);

    cv::Mat model_input = frame;
    if (model_profile.input_format == "RGB") {
      cv::cvtColor(frame, model_input, cv::COLOR_BGR2RGB);
    }
    if (!model_run) {
      model_run = model.build(model_input);
    }
    const auto t_model0 = Clock::now();
    simaai::neat::Sample out = model_run.run(model_input);
    const double model_dt_s = elapsed_s(t_model0);
    const simaai::neat::Tensor* out_tensor = find_depth_tensor(out);
    if (out_tensor == nullptr) {
      std::cerr << "Model output missing tensor\n";
      return false;
    }
    const simaai::neat::Tensor& t = *out_tensor;

    const auto t_post0 = Clock::now();
    cv::Mat depth_bgr;
    if (!tensor_to_depth_bgr(t, frame.cols, frame.rows, model_profile.depth_column_major,
                             use_colormap, depth_bgr, nullptr, nullptr)) {
      std::cerr << "Failed to convert depth tensor\n";
      return false;
    }
    if (depth_bgr.size() != frame.size()) {
      cv::resize(depth_bgr, depth_bgr, frame.size(), 0, 0, cv::INTER_NEAREST);
    }
    const double a = std::max(0.0f, std::min(1.0f, alpha));
    cv::Mat blended;
    cv::addWeighted(frame, 1.0 - a, depth_bgr, a, 0.0, blended);
    const double post_dt_s = elapsed_s(t_post0);

    const auto t_write0 = Clock::now();
    writer.write(blended);
    const double write_dt_s = elapsed_s(t_write0);

    ++processed;
    profiler.add_model(model_dt_s, profile_frame);
    profiler.add_post(post_dt_s, profile_frame);
    profiler.add_write(write_dt_s, profile_frame);
    profiler.add_frame(elapsed_s(t_frame0), profile_frame);

    if (!log_window_start.has_value())
      log_window_start = Clock::now();
    if (processed % std::max(1, log_every) == 0) {
      const auto now = Clock::now();
      const int interval_frames = processed - log_window_start_frames;
      const double interval_elapsed_s =
          std::max(1e-6, std::chrono::duration<double>(now - *log_window_start).count());
      std::cout << "[PROGRESS] frames=" << processed << " FPS=" << std::fixed
                << std::setprecision(2)
                << (static_cast<double>(interval_frames) / interval_elapsed_s) << " (last "
                << interval_frames << " frames)\n";
      log_window_start = now;
      log_window_start_frames = processed;
      profiler.print("frames=" + std::to_string(processed));
    }
  }

  profiler.print("summary");
  return true;
}

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " --url <rtsp://...> [options]\n"
            << "  " << prog << " --self-test [options]\n"
            << "\n"
            << "Options:\n"
            << "  --model <tar.gz>     Path to supported depth compiled model package "
               "(midas_v21_small_256, depth_anything_v2_vits)\n"
            << "  --output-file <file.mp4>\n"
            << "                     Output mp4 path (alias: --out, default: midas_depth.mp4)\n"
            << "  --frames <n>         Number of frames to record (default: 120)\n"
            << "  --width <n>          Input/output width (default: model-dependent)\n"
            << "  --height <n>         Input/output height (default: model-dependent)\n"
            << "  --fps <n>            Output fps (default: 30)\n"
            << "  --alpha <f>          Overlay alpha (default: 0.6)\n"
            << "  --format <BGR>       Input format (default: BGR)\n"
            << "  --no-colormap        Disable depth colormap overlay (default: inferno on)\n"
            << "  --colormap           Deprecated alias (colormap is on by default)\n"
            << "  --video <file.mp4>   Local video for --self-test\n"
            << "  --normalize          Enable normalization in Model\n"
            << "  --mean <a,b,c>       Channel mean when --normalize is set\n"
            << "  --std <a,b,c>        Channel stddev when --normalize is set\n"
            << "  --latency <ms>       RTSP latency (default: 200)\n"
            << "  --log-every <n>      Print progress/profile every N frames (default: 100)\n"
            << "  --profile            Print simple per-stage timing breakdowns\n"
            << "  --tcp                Force TCP (default)\n"
            << "  --udp                Use UDP\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  // Lifecycle: parse/validate runtime options.
  const bool self_test = has_flag(argc, argv, "--self-test");

  std::string url;
  sima_examples::get_arg(argc, argv, "--url", url);

  if (!self_test && url.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::string tar_gz;
  sima_examples::get_arg(argc, argv, "--model", tar_gz);
  if (tar_gz.empty()) {
    std::cerr << "Missing --model <path/to/model_mpk.tar.gz>.\n";
    return 3;
  }

  std::string out_path = "midas_depth.mp4";
  if (!sima_examples::get_arg(argc, argv, "--output-file", out_path)) {
    sima_examples::get_arg(argc, argv, "--out", out_path);
  }

  const DepthModelProfile model_profile = detect_model_profile(tar_gz);
  const int frames = get_int_arg(argc, argv, "--frames", 120);
  const int width_arg = get_int_arg(argc, argv, "--width", 0);
  const int height_arg = get_int_arg(argc, argv, "--height", 0);
  const int width = width_arg > 0 ? width_arg : model_profile.default_size;
  const int height = height_arg > 0 ? height_arg : model_profile.default_size;
  const int fps = get_int_arg(argc, argv, "--fps", 30);
  const int log_every = get_int_arg(argc, argv, "--log-every", 100);
  const float alpha = get_float_arg(argc, argv, "--alpha", 0.6f);
  const int latency_ms = get_int_arg(argc, argv, "--latency", 200);
  const bool use_colormap = !has_flag(argc, argv, "--no-colormap");
  const bool tcp = !has_flag(argc, argv, "--udp");
  const bool profile = has_flag(argc, argv, "--profile");

  std::string format = "BGR";
  sima_examples::get_arg(argc, argv, "--format", format);
  format = to_upper(format);
  if (format != "BGR") {
    std::cerr << "Only BGR input is supported in this example. Got: " << format << "\n";
    return 9;
  }

  const bool normalize = has_flag(argc, argv, "--normalize");
  std::vector<float> mean;
  std::vector<float> stddev;
  std::string mean_arg;
  std::string std_arg;
  if (sima_examples::get_arg(argc, argv, "--mean", mean_arg)) {
    mean = parse_floats_csv(mean_arg);
  }
  if (sima_examples::get_arg(argc, argv, "--std", std_arg)) {
    stddev = parse_floats_csv(std_arg);
  }

  std::cout << "Using model: " << tar_gz << "\n";
  std::cout << "Model profile: " << model_profile.name << " (input_format=" << model_profile.input_format
            << ", size=" << width << "x" << height << ")\n";
  if (!self_test)
    std::cout << "RTSP url: " << url << "\n";
  std::cout << "Output: " << out_path << "\n";

  // NEAT API boundary: model object creation and preprocessing configuration.
  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = model_profile.input_format;
  model_opt.preproc.input_width = width;
  model_opt.preproc.input_height = height;
  model_opt.input_max_width = width;
  model_opt.input_max_height = height;
  model_opt.input_max_depth = 3;
  model_opt.preproc.normalize = normalize;
  if (!mean.empty()) {
    std::array<float, 3> m{0.0f, 0.0f, 0.0f};
    for (std::size_t i = 0; i < std::min<std::size_t>(3, mean.size()); ++i) {
      m[i] = mean[i];
    }
    model_opt.preproc.channel_mean = m;
  }
  if (!stddev.empty()) {
    std::array<float, 3> s{1.0f, 1.0f, 1.0f};
    for (std::size_t i = 0; i < std::min<std::size_t>(3, stddev.size()); ++i) {
      s[i] = stddev[i];
    }
    model_opt.preproc.channel_stddev = s;
  }

  simaai::neat::Model model(tar_gz, model_opt);

  cv::VideoWriter writer;
  writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height),
              true);
  if (!writer.isOpened()) {
    std::cerr << "Failed to open VideoWriter: " << out_path << "\n";
    return 5;
  }

  bool ok = false;
  if (self_test) {
    std::string video_path;
    sima_examples::get_arg(argc, argv, "--video", video_path);
    if (video_path.empty()) {
      video_path = resolve_sample_video();
    }
    if (video_path.empty()) {
      std::cerr << "Failed to resolve sample video.\n";
      return 6;
    }
    std::cout << "Self-test video: " << video_path << "\n";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open video: " << video_path << "\n";
      return 7;
    }

    auto next_frame = [&](FetchedFrame& out) -> bool {
      const auto t0 = Clock::now();
      cv::Mat bgr;
      if (!cap.read(bgr))
        return false;
      if (bgr.empty())
        return false;
      if (bgr.cols != width || bgr.rows != height) {
        cv::resize(bgr, bgr, cv::Size(width, height), 0, 0, cv::INTER_AREA);
      }
      out.pull_s = elapsed_s(t0);
      out.tensor_s = 0.0;
      out.skip_profile = false;
      out.frame = std::move(bgr);
      return true;
    };

    ok = process_stream(model, next_frame, writer, frames, log_every, profile, alpha, use_colormap,
                        model_profile);
  } else {
    simaai::neat::nodes::groups::RtspDecodedInputOptions ro;
    ro.url = url;
    ro.latency_ms = latency_ms;
    ro.tcp = tcp;
    ro.payload_type = 96;
    ro.insert_queue = true;
    ro.out_format = "BGR";
    ro.decoder_raw_output = false;
    ro.decoder_name = "decoder";
    ro.use_videoconvert = false;
    ro.use_videoscale = true;
    ro.output_caps.enable = true;
    ro.output_caps.format = "BGR";
    ro.output_caps.width = width;
    ro.output_caps.height = height;
    ro.output_caps.fps = fps;
    ro.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;

    std::unique_ptr<simaai::neat::Session> rtsp_session;
    std::unique_ptr<simaai::neat::Run> rtsp_run;

    // Lifecycle: source setup/reconnect stage with fixed RTSP decode options.
    auto build_rtsp = [&]() -> bool {
      auto session = std::make_unique<simaai::neat::Session>();
      session->add(simaai::neat::nodes::groups::RtspDecodedInput(ro));
      session->add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(1)));
      simaai::neat::RunOptions run_opt;
      auto run = std::make_unique<simaai::neat::Run>(session->build(run_opt));
      rtsp_session = std::move(session);
      rtsp_run = std::move(run);
      return true;
    };
    if (!build_rtsp()) {
      std::cerr << "Failed to build RTSP pipeline\n";
      return 7;
    }

    int reconnect_attempts = 0;
    const int max_reconnect_attempts = 8;
    bool skip_profile_next_frame = false;
    auto next_frame = [&](FetchedFrame& out) -> bool {
      while (true) {
        if (!rtsp_run) {
          if (!build_rtsp()) {
            std::cerr << "Failed to rebuild RTSP pipeline\n";
            return false;
          }
        }
        const auto t_pull0 = Clock::now();
        auto ref_opt = rtsp_run->pull_tensor(/*timeout_ms=*/5000);
        const double pull_dt_s = elapsed_s(t_pull0);
        if (!ref_opt.has_value()) {
          if (reconnect_attempts < max_reconnect_attempts) {
            // Contract: reconnect policy/threshold is preserved.
            ++reconnect_attempts;
            std::cerr << "RTSP pull timed out; reconnecting (" << reconnect_attempts << "/"
                      << max_reconnect_attempts << ")\n";
            try {
              rtsp_run->close();
            } catch (...) {
            }
            rtsp_run.reset();
            rtsp_session.reset();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (!build_rtsp()) {
              std::cerr << "Failed to rebuild RTSP pipeline\n";
              return false;
            }
            skip_profile_next_frame = true;
            continue;
          }
          std::cerr << "RTSP pull timed out / stream closed\n";
          return false;
        }

        reconnect_attempts = 0;
        const auto t_tensor0 = Clock::now();
        cv::Mat view;
        if (!tensor_to_bgr_mat(*ref_opt, view))
          return false;
        out.pull_s = pull_dt_s;
        out.tensor_s = elapsed_s(t_tensor0);
        out.skip_profile = skip_profile_next_frame;
        skip_profile_next_frame = false;
        out.frame = std::move(view);
        return true;
      }
    };

    ok = process_stream(model, next_frame, writer, frames, log_every, profile, alpha, use_colormap,
                        model_profile);
    if (rtsp_run) {
      rtsp_run->close();
      rtsp_run.reset();
    }
    rtsp_session.reset();
  }

  // Lifecycle: teardown stage (release writer after processing loop ends).
  writer.release();

  if (!ok) {
    std::cerr << "Processing failed.\n";
    return 8;
  }

  std::cout << "Wrote depth overlay video to: " << out_path << "\n";
  return 0;
}
