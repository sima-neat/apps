// GraphPipesYOLOOptiview.cpp
//
// Minimal strict multi-channel reference:
// 1) RTSP capture (encoded H264)
// 2) Decode for YOLO only
// 3) Forward original encoded stream to OptiView UDP (no re-encode)
// 4) Emit OptiView JSON from YOLO output
// 5) Release video only when exact frame match exists (strict sync)

#include "support/runtime/example_utils.h"
#include "support/optiview/graphpipes_optiview_helpers.h"

#include "neat/graph.h"
#include "neat/models.h"
#include "neat/node_groups.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace gp = sima_examples::graphpipes_optiview;

int main(int argc, char** argv) {
  try {
    gp::force_runtime_defaults();

    gp::Config cfg = gp::parse_config(argc, argv);
    std::cout << "[mode] strict_multichannel_noencode=1 exact_sync=1\n";

    if (cfg.mpk.empty())
      cfg.mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());

    sima_examples::require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");
    sima_examples::require(cfg.frames > 0, "--frames must be > 0");
    sima_examples::require(cfg.streams > 0, "--streams must be > 0");
    sima_examples::require(cfg.stall_timeout_ms > 0, "--stall-timeout-ms must be > 0");

    std::vector<std::string> urls = sima_examples::read_rtsp_list(cfg.rtsp_list);
    if (cfg.streams > 0 && static_cast<size_t>(cfg.streams) < urls.size()) {
      urls.resize(static_cast<size_t>(cfg.streams));
    }
    sima_examples::require(!urls.empty(), "RTSP list resolved to 0 streams");

    std::cout << "rtsp_list=" << cfg.rtsp_list << " streams=" << urls.size()
              << " transport=" << (cfg.rtsp_tcp ? "tcp" : "udp")
              << " latency_ms=" << cfg.rtsp_latency_ms << "\n";

    const gp::ProbeResult probe = gp::probe_inputs(cfg, urls);

    simaai::neat::nodes::groups::OptiViewOutputNodeGroup optiview_group;
    gp::init_optiview_group(cfg, probe, urls.size(), optiview_group);

    auto io_stats = gp::make_io_stats(urls.size());

    auto sync_store = std::make_shared<gp::SyncPendingVideoStore>(urls.size());
    auto release_pacer =
        gp::make_release_pacer(optiview_group.video_runs(), io_stats, probe.stream_fps);

    simaai::neat::graph::Graph g;

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "NV12";
    model_opt.input_max_width = probe.frame_w;
    model_opt.input_max_height = probe.frame_h;
    model_opt.input_max_depth = 1;
    simaai::neat::Model model(cfg.mpk, model_opt);
    const size_t yolo_workers = std::min<size_t>(2, urls.size());
    std::cout << "[mode] yolo_workers=" << yolo_workers << "\n";

    std::vector<std::shared_ptr<gp::YoloTokenStore>> yolo_tokens;
    yolo_tokens.reserve(yolo_workers);
    std::vector<std::string> decoder_names(yolo_workers);
    std::vector<simaai::neat::graph::NodeId> yolo_sched_nodes;
    yolo_sched_nodes.reserve(yolo_workers);
    std::vector<simaai::neat::graph::NodeId> yolo_out_nodes;
    yolo_out_nodes.reserve(yolo_workers);
    for (size_t worker = 0; worker < yolo_workers; ++worker) {
      decoder_names[worker] = "decoder_w" + std::to_string(worker);
      auto worker_tokens = std::make_shared<gp::YoloTokenStore>(urls.size());
      auto [yolo_sched, yolo_out] =
          gp::add_yolo_pipeline(g, model, probe.frame_w, probe.frame_h, decoder_names[worker],
                                worker_tokens, "w" + std::to_string(worker));
      yolo_tokens.push_back(worker_tokens);
      yolo_sched_nodes.push_back(yolo_sched);
      yolo_out_nodes.push_back(yolo_out);
    }

    std::vector<std::string> stream_ids;
    std::unordered_map<std::string, size_t> stream_index;
    gp::build_stream_maps(urls.size(), stream_ids, stream_index);
    for (size_t i = 0; i < urls.size(); ++i) {
      size_t worker_idx = 0;
      if (yolo_workers > 1) {
        const size_t split = (urls.size() + 1) / 2;
        worker_idx = (i < split) ? 0 : 1;
      }
      const std::string stream_decoder_name = "decoder_stream" + std::to_string(i);
      gp::add_stream_branch(g, i, urls[i], cfg, probe, stream_decoder_name, io_stats[i], sync_store,
                            yolo_sched_nodes[worker_idx]);
    }

    simaai::neat::graph::GraphRunOptions run_opt;
    run_opt.edge_queue = 128;
    run_opt.pull_timeout_ms = 50;
    run_opt.pipeline.output_memory = simaai::neat::OutputMemory::ZeroCopy;

    simaai::neat::graph::GraphRun run = simaai::neat::graph::helpers::build(std::move(g), run_opt);

    std::vector<int> processed(urls.size(), 0);
    std::vector<std::unique_ptr<gp::CollectorContext>> collect_ctx_by_worker;
    collect_ctx_by_worker.reserve(yolo_workers);
    for (size_t worker = 0; worker < yolo_workers; ++worker) {
      collect_ctx_by_worker.push_back(std::make_unique<gp::CollectorContext>(
          gp::CollectorContext{cfg, processed, stream_index, io_stats, sync_store,
                               yolo_tokens[worker], release_pacer, optiview_group}));
    }

    std::vector<simaai::neat::graph::GraphRun::Output> outputs;
    std::unordered_map<simaai::neat::graph::NodeId, size_t> output_to_worker;
    for (size_t worker = 0; worker < yolo_workers; ++worker) {
      outputs.push_back(run.output(yolo_out_nodes[worker]));
      output_to_worker.emplace(yolo_out_nodes[worker], worker);
    }
    auto pull = run.collect(outputs);
    pull.per_stream_target(cfg.frames)
        .stall_after_ms(cfg.stall_timeout_ms)
        .timeout_ms(100)
        .expect_streams(stream_ids)
        .on_sample([&](const simaai::neat::Sample& sample, simaai::neat::graph::NodeId out_node) {
          auto it = output_to_worker.find(out_node);
          if (it == output_to_worker.end())
            return;
          gp::on_yolo_sample(sample, *collect_ctx_by_worker[it->second]);
        });

    pull.run();

    if (release_pacer)
      release_pacer->stop();

    run.emit_summary();

    const gp::FinalTotals totals =
        gp::print_stream_summaries(io_stats, sync_store, yolo_tokens, release_pacer);

    const int processed_total = std::accumulate(processed.begin(), processed.end(), 0);
    std::cout << "processed_total=" << processed_total << "\n";

    run.stop();
    optiview_group.stop();

    if (totals.total_fwd_fail > 0 || totals.total_sync_match_miss > 0 ||
        totals.total_sync_release_fail > 0) {
      throw std::runtime_error(
          "Strict sync failure: fwd_fail=" + std::to_string(totals.total_fwd_fail) +
          " sync_match_miss=" + std::to_string(totals.total_sync_match_miss) +
          " sync_exact_miss=" + std::to_string(totals.total_sync_exact_miss) +
          " sync_release_fail=" + std::to_string(totals.total_sync_release_fail));
    }

    const int expected_total = cfg.frames * static_cast<int>(urls.size());
    if (processed_total != expected_total) {
      throw std::runtime_error(
          "Strict completion failure: processed_total=" + std::to_string(processed_total) +
          " expected_total=" + std::to_string(expected_total));
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
