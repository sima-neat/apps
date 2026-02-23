#include "support/runtime/example_utils.h"

#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"
#include "builder/ConfigJsonOverride.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::require;

namespace {

using sima_examples::get_arg;
using sima_examples::has_flag;
using sima_examples::parse_float_arg;
using sima_examples::parse_int_arg;

struct Config {
  fs::path root;
  fs::path coco_dir;
  std::string mpk;
  int warm = 100;
  int topk = 100;
  int max_frames = 0; // 0 = all
  float min_score = 0.52f;
  float nms = 0.5f;
  bool forever = false;
  int pipelines = 4;
};

std::atomic<bool> g_interrupt{false};

void handle_sigint(int) {
  g_interrupt.store(true);
}

bool set_json_name(nlohmann::json& j, const std::string& name) {
  bool changed = false;
  if (j.contains("node_name") && j["node_name"].is_string()) {
    j["node_name"] = name;
    changed = true;
  }
  return changed;
}

void override_node_json_name(const std::shared_ptr<simaai::neat::Node>& node,
                             const std::string& name, const std::string& tag) {
  auto* override = dynamic_cast<simaai::neat::ConfigJsonOverride*>(node.get());
  if (!override)
    return;
  override->override_config_json([&](nlohmann::json& j) { set_json_name(j, name); }, tag);
}

void override_group_json_name(simaai::neat::NodeGroup& group, const std::string& name,
                              const std::string& tag) {
  for (auto& node : group.nodes_mut()) {
    override_node_json_name(node, name, tag);
  }
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool is_image_ext(const fs::path& p) {
  const std::string ext = lower_copy(p.extension().string());
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

bool load_and_prepare_image(const fs::path& path, const std::string& fmt, int width, int height,
                            cv::Mat& out, std::string& err) {
  const std::string format = fmt.empty() ? "BGR" : fmt;

  int flags = cv::IMREAD_COLOR;
  if (format == "GRAY" || format == "GRAY8") {
    flags = cv::IMREAD_GRAYSCALE;
  }

  cv::Mat img = cv::imread(path.string(), flags);
  if (img.empty()) {
    err = "imread failed";
    return false;
  }

  if (format == "RGB") {
    if (img.channels() != 3) {
      err = "expected 3-channel image for RGB";
      return false;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  } else if (format == "BGR") {
    if (img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
  } else if (format == "GRAY" || format == "GRAY8") {
    if (img.channels() == 3) {
      cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
  } else {
    err = "unsupported input format: " + format;
    return false;
  }

  if (width > 0 && height > 0 && (img.cols != width || img.rows != height)) {
    cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  }

  if (!img.isContinuous())
    img = img.clone();
  out = std::move(img);
  return true;
}

Config parse_config(int argc, char** argv) {
  Config cfg;

  std::string root_arg;
  if (get_arg(argc, argv, "--root", root_arg)) {
    cfg.root = fs::path(root_arg);
  }

  std::string coco_arg;
  if (get_arg(argc, argv, "--coco", coco_arg)) {
    cfg.coco_dir = fs::path(coco_arg);
  }

  get_arg(argc, argv, "--mpk", cfg.mpk);
  parse_int_arg(argc, argv, "--warm", cfg.warm);
  parse_int_arg(argc, argv, "--topk", cfg.topk);
  parse_int_arg(argc, argv, "--max", cfg.max_frames);
  parse_int_arg(argc, argv, "--pipelines", cfg.pipelines);
  parse_float_arg(argc, argv, "--min-score", cfg.min_score);
  parse_float_arg(argc, argv, "--nms", cfg.nms);
  cfg.forever = has_flag(argc, argv, "--forever") || has_flag(argc, argv, "--loop");

  if (has_flag(argc, argv, "--help")) {
    std::cout << "Usage: yolov8_coco_tput [--root PATH] [--coco PATH] [--mpk PATH]"
              << " [--warm N] [--topk N] [--max N] [--pipelines N] [--min-score F] [--nms F]"
              << " [--forever|--loop]\n";
    std::exit(0);
  }

  return cfg;
}

bool pull_with_timeout(simaai::neat::Run& async, int pull_timeout_ms, int max_timeouts,
                       int& timeout_count, std::optional<simaai::neat::Sample>& out,
                       std::string& err) {
  while (true) {
    simaai::neat::Sample temp;
    simaai::neat::PullError perr;
    const simaai::neat::PullStatus status = async.pull(pull_timeout_ms, temp, &perr);
    if (status == simaai::neat::PullStatus::Ok) {
      out = temp;
      timeout_count = 0;
      return true;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      timeout_count += 1;
      if (max_timeouts > 0 && timeout_count >= max_timeouts) {
        err = "Run::pull: timeout";
        return false;
      }
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      err = "Run::pull: closed";
      return false;
    }
    err = perr.message.empty() ? "Run::pull: error" : perr.message;
    return false;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::signal(SIGINT, handle_sigint);

    Config cfg = parse_config(argc, argv);

    fs::path root = cfg.root.empty() ? fs::current_path() : cfg.root;
    fs::path coco_dir =
        cfg.coco_dir.empty() ? (root / "tmp" / "coco_val2017" / "val2017") : cfg.coco_dir;

    require(fs::exists(coco_dir), "COCO dir not found: " + coco_dir.string());

    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(coco_dir)) {
      if (!entry.is_regular_file())
        continue;
      const fs::path p = entry.path();
      if (is_image_ext(p))
        image_paths.push_back(p);
    }

    std::sort(image_paths.begin(), image_paths.end());

    if (cfg.max_frames > 0 && static_cast<size_t>(cfg.max_frames) < image_paths.size()) {
      image_paths.resize(static_cast<size_t>(cfg.max_frames));
    }

    require(!image_paths.empty(), "No images found in " + coco_dir.string());

    if (cfg.mpk.empty()) {
      cfg.mpk = sima_examples::resolve_yolov8s_tar_local_first(root);
    }
    require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");

    const std::string format = "BGR";

    const int target_w = 640;
    const int target_h = 640;

    cv::Mat first_frame;
    std::string err;
    if (!load_and_prepare_image(image_paths.front(), format, target_w, target_h, first_frame,
                                err)) {
      throw std::runtime_error("Failed to load " + image_paths.front().string() + ": " + err);
    }

    require(target_w > 0 && target_h > 0, "Invalid probe image size");

    std::cout << "Loading " << image_paths.size() << " images from " << coco_dir << " -> "
              << target_w << "x" << target_h << " format=" << format << "\n";

    std::vector<cv::Mat> frames;
    frames.reserve(image_paths.size());

    frames.push_back(std::move(first_frame));
    for (size_t i = 1; i < image_paths.size(); ++i) {
      cv::Mat frame;
      std::string load_err;
      if (!load_and_prepare_image(image_paths[i], format, target_w, target_h, frame, load_err)) {
        throw std::runtime_error("Failed to load " + image_paths[i].string() + ": " + load_err);
      }
      frames.push_back(std::move(frame));
    }

    require(!frames.empty(), "No frames loaded");

    const int pipeline_count = (cfg.pipelines > 0) ? cfg.pipelines : 1;
    std::vector<simaai::neat::Model> models;
    models.reserve(pipeline_count);
    std::vector<std::unique_ptr<simaai::neat::Session>> sessions;
    sessions.reserve(pipeline_count);
    std::vector<simaai::neat::Run> runs;
    runs.reserve(pipeline_count);

    for (int i = 0; i < pipeline_count; ++i) {
      const std::string name_suffix = "_p" + std::to_string(i);
      const std::string decoder_name = "decoder" + name_suffix;
      simaai::neat::Model::Options model_opt;
      model_opt.media_type = "video/x-raw";
      model_opt.format = format;
      model_opt.preproc.input_width = target_w;
      model_opt.preproc.input_height = target_h;
      model_opt.input_max_width = target_w;
      model_opt.input_max_height = target_h;
      model_opt.input_max_depth = 3;
      model_opt.upstream_name = decoder_name;
      model_opt.name_suffix = name_suffix;
      models.emplace_back(cfg.mpk, model_opt);
      auto& model = models.back();

      auto session = std::make_unique<simaai::neat::Session>();
      auto src_opt = model.input_appsrc_options(false);
      src_opt.buffer_name = decoder_name;
      session->add(simaai::neat::nodes::Input(src_opt));

      const std::string box_name = "boxdecode_p" + std::to_string(i);

      auto pre_group = simaai::neat::nodes::groups::Preprocess(model);
      for (auto& node : pre_group.nodes_mut()) {
        auto* override = dynamic_cast<simaai::neat::ConfigJsonOverride*>(node.get());
        if (!override)
          continue;
        override->override_config_json(
            [&](nlohmann::json& j) {
              if (j.contains("input_buffers") && j["input_buffers"].is_array() &&
                  !j["input_buffers"].empty() && j["input_buffers"][0].is_object()) {
                j["input_buffers"][0]["name"] = decoder_name;
              }
            },
            "preproc_decoder_names");
      }
      session->add(pre_group);

      auto mla_group = simaai::neat::nodes::groups::Infer(model);
      session->add(mla_group);

      auto box_node = simaai::neat::nodes::SimaBoxDecode(model, "yolov8", target_w, target_h,
                                                         cfg.min_score, cfg.nms, cfg.topk);
      const std::string mla_out_name = model.infer_output_name();
      auto* override = dynamic_cast<simaai::neat::ConfigJsonOverride*>(box_node.get());
      if (override) {
        override->override_config_json(
            [&](nlohmann::json& j) {
              set_json_name(j, box_name);
              if (!mla_out_name.empty()) {
                if (!j.contains("buffers") || !j["buffers"].is_object()) {
                  j["buffers"] = nlohmann::json::object();
                }
                if (!j["buffers"].contains("input") || !j["buffers"]["input"].is_array()) {
                  j["buffers"]["input"] = nlohmann::json::array();
                }
                if (j["buffers"]["input"].empty()) {
                  j["buffers"]["input"].push_back(nlohmann::json::object());
                } else if (!j["buffers"]["input"][0].is_object()) {
                  j["buffers"]["input"][0] = nlohmann::json::object();
                }
                j["buffers"]["input"][0]["name"] = mla_out_name;
              }
            },
            box_name);
      }
      session->add(box_node);
      session->add(simaai::neat::nodes::Output());

      runs.push_back(session->build(frames[0]));
      sessions.push_back(std::move(session));
    }

    if (cfg.warm > 0) {
      const int warm_timeout_ms = 60000;
      for (auto& run : runs) {
        run.warmup(frames[0], cfg.warm, warm_timeout_ms);
      }
    }

    const int pull_timeout_ms = 1000;
    const int max_timeouts = 3;

    std::mutex err_mu;
    std::string consumer_error;
    std::atomic<bool> stop_requested{false};
    std::atomic<size_t> outputs{0};

    auto set_error = [&](const std::string& msg) {
      {
        std::lock_guard<std::mutex> lock(err_mu);
        if (consumer_error.empty())
          consumer_error = msg;
      }
      stop_requested.store(true);
    };

    const size_t window_size = 100;

    std::mutex window_mu;
    std::deque<std::chrono::steady_clock::time_point> window;
    std::vector<std::thread> consumers;
    consumers.reserve(runs.size());

    auto record_output = [&](size_t done) {
      const auto now = std::chrono::steady_clock::now();
      double fps = 0.0;
      bool do_print = false;
      {
        std::lock_guard<std::mutex> lock(window_mu);
        window.push_back(now);
        if (window.size() > window_size) {
          window.pop_front();
        }
        if (window.size() == window_size && (done % window_size) == 0) {
          const double elapsed =
              std::chrono::duration<double>(window.back() - window.front()).count();
          fps = (elapsed > 0.0) ? (static_cast<double>(window.size()) / elapsed) : 0.0;
          do_print = true;
        }
      }
      if (do_print) {
        std::cout << "TPUT100 outputs=" << done << " avg_fps=" << fps << "\n";
        if (g_interrupt.load()) {
          stop_requested.store(true);
        }
      }
    };

    const size_t total_expected = frames.size();

    for (size_t idx = 0; idx < runs.size(); ++idx) {
      consumers.emplace_back([&, idx]() {
        auto& async = runs[idx];
        int timeout_count = 0;
        while (true) {
          if (stop_requested.load())
            break;
          std::optional<simaai::neat::Sample> out;
          std::string pull_err;
          if (!pull_with_timeout(async, pull_timeout_ms, max_timeouts, timeout_count, out,
                                 pull_err)) {
            if (!stop_requested.load() && !pull_err.empty())
              set_error(pull_err);
            break;
          }
          if (!out.has_value())
            break;
          std::vector<uint8_t> payload;
          std::string extract_err;
          if (!sima_examples::extract_bbox_payload(*out, payload, extract_err)) {
            set_error(extract_err);
            break;
          }
          const size_t done = outputs.fetch_add(1) + 1;
          record_output(done);
          if (!cfg.forever && done >= total_expected) {
            stop_requested.store(true);
            break;
          }
        }
      });
    }

    const auto start = std::chrono::steady_clock::now();
    if (cfg.forever) {
      size_t pipeline_idx = 0;
      while (!stop_requested.load()) {
        if (g_interrupt.load()) {
          stop_requested.store(true);
          break;
        }
        for (size_t i = 0; i < frames.size(); ++i) {
          if (stop_requested.load() || g_interrupt.load()) {
            stop_requested.store(true);
            break;
          }
          runs[pipeline_idx].push(frames[i]);
          pipeline_idx = (pipeline_idx + 1) % runs.size();
        }
      }
    } else {
      size_t pipeline_idx = 0;
      for (size_t i = 0; i < frames.size(); ++i) {
        if (stop_requested.load() || g_interrupt.load())
          break;
        runs[pipeline_idx].push(frames[i]);
        pipeline_idx = (pipeline_idx + 1) % runs.size();
      }
    }
    for (auto& run : runs) {
      run.close_input();
    }
    for (auto& t : consumers) {
      t.join();
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_s = std::chrono::duration<double>(end - start).count();

    const size_t total = cfg.forever ? outputs.load() : total_expected;
    const size_t out = outputs.load();
    const double avg_fps = (elapsed_s > 0.0) ? (static_cast<double>(out) / elapsed_s) : 0.0;

    bool ok = (out == total) && consumer_error.empty();
    if (!ok) {
      for (size_t i = 0; i < runs.size(); ++i) {
        const std::string last_err = runs[i].last_error();
        if (!last_err.empty()) {
          std::cerr << "Async last_error[" << i << "]: " << last_err << "\n";
        }
      }
      if (!consumer_error.empty()) {
        std::cerr << "Consumer error: " << consumer_error << "\n";
      }
    }

    std::cout << "COCO_TPUT frames=" << total << " outputs=" << out << " elapsed_s=" << elapsed_s
              << " avg_fps=" << avg_fps << " ok=" << (ok ? "1" : "0") << "\n";

    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
