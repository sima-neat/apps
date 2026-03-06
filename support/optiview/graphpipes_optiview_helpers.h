#pragma once

#include "neat/graph.h"
#include "neat/session.h"
#include "neat/models.h"
#include "neat/node_groups.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sima_examples::graphpipes_optiview {

struct Config {
  std::string rtsp_list;
  std::string model_path;

  int frames = 3000;
  int streams = 4;

  std::string optiview_host = "127.0.0.1";
  int optiview_video_port = 9000;
  int optiview_json_port = 9100;

  int rtsp_latency_ms = 200;
  bool rtsp_tcp = true;

  int stall_timeout_ms = 20000;

  bool send_json = true;
  bool debug = false;
  bool rtsp_debug = false;

  int input_payload_type = 96;
};

struct StreamIoStats {
  std::atomic<int64_t> fwd_push_ok{0};
  std::atomic<int64_t> fwd_push_fail{0};

  std::atomic<int64_t> yolo_samples{0};

  std::atomic<int64_t> bbox_extract_fail{0};
  std::atomic<int64_t> bbox_parse_fail{0};

  std::atomic<int64_t> json_ok{0};
  std::atomic<int64_t> json_fail{0};
  std::atomic<int64_t> json_nonempty{0};
  std::atomic<int64_t> json_empty{0};
  std::atomic<int64_t> boxes_total{0};

  std::atomic<int64_t> sync_release_ok{0};
  std::atomic<int64_t> sync_release_fail{0};
  std::atomic<int64_t> sync_release_pace_drop{0};

  std::atomic<int64_t> sync_exact_ok{0};
  std::atomic<int64_t> sync_exact_miss{0};
  std::atomic<int64_t> sync_token_miss{0};
  std::atomic<int64_t> sync_match_miss{0};
};

using SyncPendingVideoStore = simaai::neat::graph::strict_sync::PendingVideoStore;
using YoloTokenStore = simaai::neat::graph::strict_sync::YoloTokenStore;
using SyncReleasePacer = simaai::neat::graph::strict_sync::ReleasePacer;

struct ProbeResult {
  int frame_w = 0;
  int frame_h = 0;
  int stream_fps = -1;
  std::string enc_caps;
  std::string enc_caps_appsrc;
};

struct CollectorContext {
  const Config& cfg;
  std::vector<int>& processed;
  const std::unordered_map<std::string, size_t>& stream_index;
  const std::vector<std::shared_ptr<StreamIoStats>>& io_stats;
  const std::shared_ptr<SyncPendingVideoStore>& sync_store;
  const std::shared_ptr<YoloTokenStore>& yolo_tokens;
  const std::shared_ptr<SyncReleasePacer>& release_pacer;
  simaai::neat::nodes::groups::OptiViewOutputNodeGroup& optiview_group;
};

struct FinalTotals {
  int64_t total_fwd_fail = 0;
  int64_t total_sync_match_miss = 0;
  int64_t total_sync_exact_miss = 0;
  int64_t total_sync_release_fail = 0;
};

void force_runtime_defaults();
Config parse_config(int argc, char** argv);
ProbeResult probe_inputs(const Config& cfg, const std::vector<std::string>& urls);

void init_optiview_group(const Config& cfg, const ProbeResult& probe, size_t streams,
                         simaai::neat::nodes::groups::OptiViewOutputNodeGroup& out_group);

std::vector<std::shared_ptr<StreamIoStats>> make_io_stats(size_t streams);

std::shared_ptr<SyncReleasePacer>
make_release_pacer(const std::vector<std::shared_ptr<simaai::neat::Run>>& forward_runs,
                   const std::vector<std::shared_ptr<StreamIoStats>>& io_stats, int stream_fps);

std::pair<simaai::neat::graph::NodeId, simaai::neat::graph::NodeId>
add_yolo_pipeline(simaai::neat::graph::Graph& g, simaai::neat::Model& model, int frame_w,
                  int frame_h, const std::string& decoder_name,
                  const std::shared_ptr<YoloTokenStore>& yolo_tokens,
                  const std::string& worker_tag);

void add_stream_branch(simaai::neat::graph::Graph& g, size_t idx, const std::string& url,
                       const Config& cfg, const ProbeResult& probe, const std::string& decoder_name,
                       const std::shared_ptr<StreamIoStats>& io,
                       const std::shared_ptr<SyncPendingVideoStore>& sync_store,
                       simaai::neat::graph::NodeId yolo_sched);

void build_stream_maps(size_t streams, std::vector<std::string>& stream_ids,
                       std::unordered_map<std::string, size_t>& stream_index);

void on_yolo_sample(const simaai::neat::Sample& sample, CollectorContext& ctx);

FinalTotals print_stream_summaries(const std::vector<std::shared_ptr<StreamIoStats>>& io_stats,
                                   const std::shared_ptr<SyncPendingVideoStore>& sync_store,
                                   const std::vector<std::shared_ptr<YoloTokenStore>>& yolo_tokens,
                                   const std::shared_ptr<SyncReleasePacer>& release_pacer);

} // namespace sima_examples::graphpipes_optiview
