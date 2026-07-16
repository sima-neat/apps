// FastSAM (+ MobileCLIP text prompt) across multiple RTSP cameras on one Modalix MLA.
#include "clip/image_encoder.h"
#include "clip/text_features.h"
#include "config.h"
#include "fastsam.h"
#include "pipeline.h"
#include "support/runtime/config_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_sigint);

  try {
    // Load the config.
    const std::string config_path =
        argc > 1 ? argv[1]
                 : sima_examples::default_config_path(FASTSAM_CPP_SOURCE_DIR).string();
    const app::AppConfig cfg = app::load_config(config_path);

    const auto run_opt = app::make_run_options(cfg.queue_depth);
    const std::string label = cfg.text;

    // Load the precomputed CLIP text features.
    std::cout << "[build] loading precomputed CLIP text features ("
              << cfg.clip_text_features_path << ")" << std::endl;
    const auto text_features =
        app::clip::load_text_features(cfg.clip_text_features_path, cfg.text);

    // Probe every RTSP stream for its resolution.
    std::vector<app::StreamInfo> stream_infos;
    stream_infos.reserve(cfg.rtsp_urls.size());
    int max_w = 0;
    int max_h = 0;
    for (const auto& url : cfg.rtsp_urls) {
      const auto info = app::probe_rtsp(url);
      std::cout << "[rtsp] " << url << " " << info.width << "x" << info.height << "@" << info.fps
                << "\n";
      max_w = std::max(max_w, info.width);
      max_h = std::max(max_h, info.height);
      stream_infos.push_back(info);
    }

    // Build the FastSAM runner.
    auto fastsam = std::make_unique<app::Fastsam>(cfg, run_opt, max_w, max_h);
    std::cout << "[build] FastSAM runner (" << max_w << "x" << max_h << ")" << std::endl;

    // Build the CLIP image encoder.
    auto image_encoder = std::make_unique<app::clip::ImageEncoder>(cfg.clip_image_path, run_opt);
    std::cout << "[build] CLIP image encoder (OpenCV threads=" << cv::getNumThreads() << ")"
              << std::endl;

    // Create and start every stream.
    std::vector<std::unique_ptr<app::Stream>> streams;
    streams.reserve(cfg.rtsp_urls.size());
    for (std::size_t i = 0; i < cfg.rtsp_urls.size(); ++i) {
      streams.push_back(std::make_unique<app::Stream>(cfg, static_cast<int>(i), cfg.rtsp_urls[i],
                                                      stream_infos[i]));
    }
    for (auto& s : streams) {
      s->start(cfg);
    }

    // Spawn one frame-reader thread per stream: pull frames until it closes or we stop.
    std::vector<std::thread> source_threads;
    source_threads.reserve(streams.size());
    for (auto& s : streams) {
      source_threads.emplace_back([&stream = *s] {
        try {
          while (g_stop == 0) {
            if (stream.pull_frame() == app::Stream::Pull::Closed) {
              break;
            }
          }
        } catch (const std::exception& ex) {
          std::cerr << "stream " << stream.index() << ": " << ex.what() << "\n";
          g_stop = 1;
        }
        stream.close_source();
      });
    }

    // Run the detector round-robin over every stream until all are finished.
    while (g_stop == 0) {
      bool did_work = false;
      bool all_done = true;
      for (auto& sp : streams) {
        app::Stream& stream = *sp;
        if (stream.frame_limit_reached(cfg) || stream.done()) {
          continue;
        }
        all_done = false;
        did_work |= stream.process(cfg, *fastsam, *image_encoder, text_features, label);
      }
      if (all_done) { break; }
      if (!did_work) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }

    // Stop and join the frame-reader threads.
    g_stop = 1;
    for (auto& source_thread : source_threads) {
      if (source_thread.joinable()) {
        source_thread.join();
      }
    }

    // Flush profiling and close every stream.
    for (auto& s : streams) {
      s->flush_profile();
      s->close();
    }

    // Close the runners.
    fastsam->close();
    image_encoder->close();

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
