#include "graphpipes_optiview_helpers.h"

#include "support/runtime/example_utils.h"

#include "builder/ConfigJsonOverride.h"
#include "neat/graph.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/session.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace sima_examples::graphpipes_optiview {
namespace {

constexpr int kOutputPayloadType = 96;
constexpr int kFixedModelNumBuffers = 3;
constexpr int kFixedDecoderNumBuffers = 7;

bool parse_stream_index(const std::string& sid, size_t* out_idx) {
  if (!out_idx)
    return false;
  constexpr const char* kPrefix = "stream";
  if (sid.rfind(kPrefix, 0) != 0)
    return false;
  if (sid.size() <= std::strlen(kPrefix))
    return false;
  const std::string suffix = sid.substr(std::strlen(kPrefix));
  if (suffix.empty())
    return false;
  char* end = nullptr;
  const unsigned long idx = std::strtoul(suffix.c_str(), &end, 10);
  if (!end || *end != '\0')
    return false;
  *out_idx = static_cast<size_t>(idx);
  return true;
}

const char* sample_kind_name(simaai::neat::SampleKind kind) {
  switch (kind) {
  case simaai::neat::SampleKind::Tensor:
    return "tensor";
  case simaai::neat::SampleKind::Bundle:
    return "bundle";
  default:
    return "other";
  }
}

void log_edge(const char* tag, const simaai::neat::Sample& sample) {
  if (std::getenv("SIMA_GRAPH_EDGE_LOG") == nullptr)
    return;
  std::fprintf(stderr,
               "[EDGE] %s stream=%s frame=%lld input_seq=%lld orig_seq=%lld kind=%s caps=%s\n",
               tag ? tag : "edge", sample.stream_id.empty() ? "<empty>" : sample.stream_id.c_str(),
               static_cast<long long>(sample.frame_id), static_cast<long long>(sample.input_seq),
               static_cast<long long>(sample.orig_input_seq), sample_kind_name(sample.kind),
               sample.caps_string.empty() ? "<empty>" : sample.caps_string.c_str());
}

void ensure_encoded_semantic(simaai::neat::Sample& sample) {
  if (sample.kind != simaai::neat::SampleKind::Tensor || !sample.tensor.has_value())
    return;
  auto& tensor = sample.tensor.value();
  if (tensor.semantic.encoded.has_value())
    return;
  if (sample.caps_string.empty())
    return;

  const auto codec = simaai::neat::caps_to_codec(sample.caps_string);
  if (codec == simaai::neat::EncodedSpec::Codec::UNKNOWN)
    return;

  tensor.semantic.encoded = simaai::neat::EncodedSpec{};
  tensor.semantic.encoded->codec = codec;
  if (!sample.payload_tag.empty())
    return;

  switch (codec) {
  case simaai::neat::EncodedSpec::Codec::H264:
  case simaai::neat::EncodedSpec::Codec::RTP_H264:
    sample.payload_tag = "H264";
    break;
  case simaai::neat::EncodedSpec::Codec::H265:
  case simaai::neat::EncodedSpec::Codec::RTP_H265:
    sample.payload_tag = "H265";
    break;
  case simaai::neat::EncodedSpec::Codec::JPEG:
    sample.payload_tag = "JPEG";
    break;
  default:
    break;
  }
}

size_t encoded_sample_bytes_estimate(const simaai::neat::Sample& sample) {
  constexpr size_t kFallbackEncodedBytes = 256 * 1024;
  if (sample.kind != simaai::neat::SampleKind::Tensor || !sample.tensor.has_value()) {
    return kFallbackEncodedBytes;
  }
  const auto& t = sample.tensor.value();
  if (t.storage && t.storage->size_bytes > 0) {
    return static_cast<size_t>(t.storage->size_bytes);
  }
  if (t.shape.size() == 1 && t.shape[0] > 0) {
    return static_cast<size_t>(t.shape[0]);
  }
  return kFallbackEncodedBytes;
}

void require_model_config_exists(const simaai::neat::Model& model, const std::string& label,
                                 const std::string& plugin, const std::string& processor,
                                 const std::string& fallback_plugin = {}) {
  std::string path = model.find_config_path_by_plugin(plugin);
  if (path.empty() && !fallback_plugin.empty()) {
    path = model.find_config_path_by_plugin(fallback_plugin);
  }
  if (path.empty()) {
    path = model.find_config_path_by_processor(processor);
  }
  if (path.empty()) {
    throw std::runtime_error("Model config missing for " + label);
  }
  if (!fs::exists(path)) {
    throw std::runtime_error("Model config not found for " + label + ": " + path);
  }
}

} // namespace

void force_runtime_defaults() {
  const std::string model_buffers = std::to_string(kFixedModelNumBuffers);
  const std::string decoder_buffers = std::to_string(kFixedDecoderNumBuffers);
  setenv("SIMA_FORCE_MODEL_NUM_BUFFERS", model_buffers.c_str(), 1);
  setenv("SIMA_FORCE_DECODER_NUM_BUFFERS", decoder_buffers.c_str(), 1);
  setenv("SIMA_FORCE_DECODER_POOL_BUFFERS", decoder_buffers.c_str(), 1);

  setenv("SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP", "0", 1);
  setenv("SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT", "0", 1);

  setenv("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS", "15000", 1);
  setenv("SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS_2", "15000", 1);
  setenv("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS", "15000", 1);
  setenv("SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS_2", "15000", 1);
}

Config parse_config(int argc, char** argv) {
  Config cfg;
  std::string raw;

  sima_examples::get_arg(argc, argv, "--rtsp-list", cfg.rtsp_list);
  sima_examples::get_arg(argc, argv, "--model", cfg.model_path);

  if (sima_examples::get_arg(argc, argv, "--frames", raw))
    cfg.frames = std::stoi(raw);
  if (sima_examples::get_arg(argc, argv, "--streams", raw))
    cfg.streams = std::stoi(raw);

  if (sima_examples::get_arg(argc, argv, "--optiview-host", raw))
    cfg.optiview_host = raw;
  if (sima_examples::get_arg(argc, argv, "--optiview-video-port", raw)) {
    cfg.optiview_video_port = std::stoi(raw);
  }
  if (sima_examples::get_arg(argc, argv, "--optiview-json-port", raw)) {
    cfg.optiview_json_port = std::stoi(raw);
  }

  if (sima_examples::get_arg(argc, argv, "--rtsp-pt", raw))
    cfg.input_payload_type = std::stoi(raw);
  if (sima_examples::get_arg(argc, argv, "--rtsp-latency-ms", raw)) {
    cfg.rtsp_latency_ms = std::stoi(raw);
  }
  if (sima_examples::get_arg(argc, argv, "--stall-timeout-ms", raw)) {
    cfg.stall_timeout_ms = std::stoi(raw);
  }

  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  cfg.rtsp_debug = sima_examples::has_flag(argc, argv, "--rtsp-debug");

  if (sima_examples::has_flag(argc, argv, "--rtsp-udp"))
    cfg.rtsp_tcp = false;
  if (sima_examples::has_flag(argc, argv, "--rtsp-tcp"))
    cfg.rtsp_tcp = true;
  if (sima_examples::has_flag(argc, argv, "--no-json"))
    cfg.send_json = false;

  return cfg;
}

ProbeResult probe_inputs(const Config& cfg, const std::vector<std::string>& urls) {
  ProbeResult out;

  sima_examples::RtspProbeOptions rtsp_probe_opt;
  rtsp_probe_opt.payload_type = cfg.input_payload_type;
  rtsp_probe_opt.latency_ms = cfg.rtsp_latency_ms;
  rtsp_probe_opt.rtsp_tcp = cfg.rtsp_tcp;
  rtsp_probe_opt.debug = cfg.rtsp_debug;
  rtsp_probe_opt.decoder_num_buffers = kFixedDecoderNumBuffers;

  simaai::neat::Session probe;
  probe.add(simaai::neat::nodes::RTSPInput(urls[0], cfg.rtsp_latency_ms, cfg.rtsp_tcp,
                                           /*drop_on_latency=*/true,
                                           /*buffer_mode=*/"none"));
  probe.add(simaai::neat::nodes::H264Depacketize(cfg.input_payload_type,
                                                 /*config_interval=*/1,
                                                 /*fps=*/-1,
                                                 /*w=*/-1,
                                                 /*h=*/-1,
                                                 /*enforce_caps=*/false));
  probe.add(simaai::neat::nodes::Output());

  simaai::neat::RunOptions probe_opt;
  probe_opt.enable_metrics = cfg.debug;
  simaai::neat::Run probe_run = probe.build(probe_opt);

  simaai::neat::Sample sample;
  simaai::neat::PullError err;
  auto st = probe_run.pull(5000, sample, &err);
  sima_examples::require(st == simaai::neat::PullStatus::Ok,
                         err.message.empty() ? "Failed to pull probe encoded sample" : err.message);
  sima_examples::require(!sample.caps_string.empty(), "Probe encoded sample missing caps_string");
  out.enc_caps = sample.caps_string;
  (void)sima_examples::parse_fps_from_caps(out.enc_caps, out.stream_fps);
  probe_run.stop();

  const bool have_w = sima_examples::parse_dim_from_caps(out.enc_caps, "width", out.frame_w);
  const bool have_h = sima_examples::parse_dim_from_caps(out.enc_caps, "height", out.frame_h);
  if (!have_w || !have_h) {
    std::cerr << "[init] encoded caps missing dims; probing decoded frame\n";
    sima_examples::require(sima_examples::probe_rtsp_decoded_dims(urls[0], rtsp_probe_opt,
                                                                  /*tries=*/8,
                                                                  /*timeout_ms=*/1000, out.frame_w,
                                                                  out.frame_h),
                           "Failed to infer decoded dimensions from RTSP probe");
  }

  out.enc_caps_appsrc = out.enc_caps;
  if (!have_w)
    out.enc_caps_appsrc += ",width=(int)" + std::to_string(out.frame_w);
  if (!have_h)
    out.enc_caps_appsrc += ",height=(int)" + std::to_string(out.frame_h);

  std::cout << "[init] inferred dims=" << out.frame_w << "x" << out.frame_h
            << " fps=" << (out.stream_fps > 0 ? std::to_string(out.stream_fps) : "<auto>")
            << " caps=" << out.enc_caps << "\n";

  for (size_t i = 0; i < urls.size(); ++i) {
    if (!sima_examples::probe_rtsp_encoded(urls[i], rtsp_probe_opt,
                                           /*fps=*/out.stream_fps,
                                           /*w=*/out.frame_w,
                                           /*h=*/out.frame_h,
                                           /*tries=*/12,
                                           /*timeout_ms=*/500,
                                           /*enforce_caps=*/true)) {
      throw std::runtime_error("RTSP probe failed (no output): " + urls[i]);
    }
    std::cout << "[stream_probe] stream=" << i << " ok=1\n";
  }

  return out;
}

void init_optiview_group(const Config& cfg, const ProbeResult& probe, size_t streams,
                         simaai::neat::nodes::groups::OptiViewOutputNodeGroup& out_group) {
  simaai::neat::nodes::groups::OptiViewOutputNodeGroupOptions optiview_opt;
  optiview_opt.udp.h264_caps = probe.enc_caps_appsrc;
  optiview_opt.udp.payload_type = kOutputPayloadType;
  optiview_opt.udp.config_interval = 1;
  optiview_opt.udp.enable_timings = cfg.debug;
  optiview_opt.udp.host = cfg.optiview_host;
  optiview_opt.udp.video_port_base = cfg.optiview_video_port;
  optiview_opt.send_json = cfg.send_json;
  optiview_opt.json_port_base = cfg.optiview_json_port;
  optiview_opt.frame_w = probe.frame_w;
  optiview_opt.frame_h = probe.frame_h;
  optiview_opt.topk = 100;
  optiview_opt.parse_debug = false;

  std::string optiview_err;
  sima_examples::require(out_group.init(optiview_opt, streams, &optiview_err), optiview_err);

  std::cout << "[optiview] host=" << cfg.optiview_host
            << " video_port_base=" << cfg.optiview_video_port
            << " json_port_base=" << cfg.optiview_json_port
            << " send_json=" << (cfg.send_json ? "1" : "0") << " streams=" << streams << "\n";
}

std::vector<std::shared_ptr<StreamIoStats>> make_io_stats(size_t streams) {
  std::vector<std::shared_ptr<StreamIoStats>> io_stats;
  io_stats.reserve(streams);
  for (size_t i = 0; i < streams; ++i) {
    io_stats.push_back(std::make_shared<StreamIoStats>());
  }
  return io_stats;
}

std::shared_ptr<SyncReleasePacer>
make_release_pacer(const std::vector<std::shared_ptr<simaai::neat::Run>>& forward_runs,
                   const std::vector<std::shared_ptr<StreamIoStats>>& io_stats, int stream_fps) {
  const int sync_release_fps = (stream_fps > 0) ? stream_fps : 25;
  const bool log_sync_fail = std::getenv("SIMA_GRAPH_SYNC_DEBUG") != nullptr;
  auto fail_counts = std::make_shared<std::vector<int64_t>>(forward_runs.size(), 0);
  auto pacer = std::make_shared<SyncReleasePacer>(
      forward_runs, sync_release_fps,
      /*max_queue=*/0,
      [io_stats, forward_runs, fail_counts, log_sync_fail](size_t idx, bool ok) {
        if (idx >= io_stats.size() || !io_stats[idx])
          return;
        if (ok) {
          io_stats[idx]->sync_release_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
          io_stats[idx]->sync_release_fail.fetch_add(1, std::memory_order_relaxed);
          if (log_sync_fail && idx < fail_counts->size()) {
            int64_t fail_count = ++((*fail_counts)[idx]);
            if (fail_count <= 3 || (fail_count % 50) == 0) {
              std::string last_err;
              if (idx < forward_runs.size() && forward_runs[idx]) {
                last_err = forward_runs[idx]->last_error();
              }
              std::cerr << "[sync_release] push_failed stream=" << idx << " count=" << fail_count
                        << " run_error=" << (last_err.empty() ? "<empty>" : last_err) << "\n";
            }
          }
        }
      },
      [io_stats](size_t idx, int64_t dropped) {
        if (dropped <= 0 || idx >= io_stats.size() || !io_stats[idx])
          return;
        io_stats[idx]->sync_release_pace_drop.fetch_add(dropped, std::memory_order_relaxed);
      });

  std::cout << "[sync] strict_release=1"
            << " release_fps=" << sync_release_fps << " release_queue=unbounded"
            << "\n";

  return pacer;
}

std::pair<simaai::neat::graph::NodeId, simaai::neat::graph::NodeId>
add_yolo_pipeline(simaai::neat::graph::Graph& g, simaai::neat::Model& model, int frame_w,
                  int frame_h, const std::string& decoder_name,
                  const std::shared_ptr<YoloTokenStore>& yolo_tokens,
                  const std::string& worker_tag) {
  require_model_config_exists(model, "preproc", "process_cvu", "CVU", "preproc");
  require_model_config_exists(model, "mla", "process_mla", "MLA");

  simaai::neat::graph::nodes::StreamSchedulerOptions yolo_sched_opt;
  yolo_sched_opt.per_stream_queue = 0;
  yolo_sched_opt.drop_policy = simaai::neat::graph::nodes::StreamDropPolicy::DropOldest;
  yolo_sched_opt.max_batch = 1;
  const std::string suffix = worker_tag.empty() ? std::string{} : ("_" + worker_tag);
  auto yolo_sched =
      g.add(simaai::neat::graph::nodes::StreamSchedulerNode(yolo_sched_opt, "yolo_sched" + suffix));

  std::vector<std::shared_ptr<simaai::neat::Node>> yolo_nodes;
  yolo_nodes.reserve(8);

  auto src_opt = model.input_appsrc_options(false);
  src_opt.format = "NV12";
  src_opt.width = frame_w;
  src_opt.height = frame_h;
  src_opt.buffer_name = decoder_name;
  yolo_nodes.push_back(simaai::neat::nodes::Input(src_opt));

  auto preproc_group = simaai::neat::nodes::groups::Preprocess(model);
  for (auto& node : preproc_group.nodes_mut()) {
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
        "graphpipes_decoder_name" + suffix);
  }
  for (const auto& node : preproc_group.nodes())
    yolo_nodes.push_back(node);

  auto infer_group = simaai::neat::nodes::groups::Infer(model);
  for (const auto& node : infer_group.nodes())
    yolo_nodes.push_back(node);

  yolo_nodes.push_back(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", frame_w, frame_h,
                                                          /*min_score=*/0.52f,
                                                          /*nms=*/0.5f,
                                                          /*topk=*/100));

  auto yolo_group = simaai::neat::NodeGroup(std::move(yolo_nodes));
  auto yolo_pipe =
      simaai::neat::graph::helpers::add_pipeline(g, std::move(yolo_group), "yolo" + suffix);

  auto yolo_token_in = g.add(simaai::neat::graph::nodes::Map(
      [yolo_tokens](simaai::neat::Sample& sample) {
        size_t idx = 0;
        if (parse_stream_index(sample.stream_id, &idx)) {
          yolo_tokens->enqueue(idx, sample.frame_id);
        }
        sample.input_seq = sample.frame_id;
      },
      "yolo_token_in" + suffix));

  auto yolo_out = g.add(simaai::neat::graph::nodes::Map(
      [yolo_tokens](simaai::neat::Sample& sample) {
        auto ordered = yolo_tokens->take_ordered();
        if (ordered.has_value()) {
          sample.stream_id = "stream" + std::to_string(ordered->stream_idx);
          sample.frame_id = ordered->frame_id;
          sample.input_seq = ordered->frame_id;
          sample.orig_input_seq = ordered->frame_id;
          return;
        }

        if (sample.stream_id.empty())
          sample.stream_id = "unknown";
        if (sample.frame_id < 0 && sample.orig_input_seq >= 0) {
          sample.frame_id = sample.orig_input_seq;
        }
        if (sample.frame_id < 0 && sample.input_seq >= 0) {
          sample.frame_id = sample.input_seq;
        }
      },
      "tag_yolo_out" + suffix));

  simaai::neat::graph::helpers::chain(g, {yolo_sched, yolo_token_in, yolo_pipe, yolo_out});
  return {yolo_sched, yolo_out};
}

void add_stream_branch(simaai::neat::graph::Graph& g, size_t idx, const std::string& url,
                       const Config& cfg, const ProbeResult& probe, const std::string& decoder_name,
                       const std::shared_ptr<StreamIoStats>& io,
                       const std::shared_ptr<SyncPendingVideoStore>& sync_store,
                       simaai::neat::graph::NodeId yolo_sched) {
  const std::string sid = "stream" + std::to_string(idx);
  auto frame_counter = std::make_shared<std::atomic<int64_t>>(0);

  auto cap_group = simaai::neat::NodeGroup({
      simaai::neat::nodes::RTSPInput(url, cfg.rtsp_latency_ms, cfg.rtsp_tcp,
                                     /*drop_on_latency=*/true,
                                     /*buffer_mode=*/"none"),
      simaai::neat::nodes::H264Depacketize(cfg.input_payload_type,
                                           /*config_interval=*/1,
                                           /*fps=*/probe.stream_fps,
                                           /*w=*/probe.frame_w,
                                           /*h=*/probe.frame_h,
                                           /*enforce_caps=*/true),
  });
  auto cap = simaai::neat::graph::helpers::add_pipeline(g, std::move(cap_group), "cap_" + sid);

  auto fwd = g.add(simaai::neat::graph::nodes::Map(
      [sid, idx, frame_counter, io, sync_store](simaai::neat::Sample& sample) {
        sample.stream_id = sid;
        if (sample.frame_id < 0) {
          sample.frame_id = frame_counter->fetch_add(1, std::memory_order_relaxed);
        }

        ensure_encoded_semantic(sample);
        log_edge("cap_out", sample);

        simaai::neat::Sample copy;
        try {
          if (sample.kind != simaai::neat::SampleKind::Tensor || !sample.tensor.has_value()) {
            io->fwd_push_fail.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          std::vector<uint8_t> bytes = sample.tensor->copy_payload_bytes();
          copy =
              simaai::neat::make_encoded_sample(std::move(bytes), sample.caps_string, sample.pts_ns,
                                                sample.dts_ns, sample.duration_ns);
          copy.frame_id = sample.frame_id;
          copy.stream_id = sample.stream_id;
          copy.port_name.clear();
          copy.input_seq = sample.input_seq;
          copy.orig_input_seq = sample.orig_input_seq;
          copy.media_type = sample.media_type;
          if (!sample.payload_tag.empty()) {
            copy.payload_tag = sample.payload_tag;
          }
        } catch (const std::exception& ex) {
          std::cerr << "[fwd_copy] stream=" << sid << " frame=" << sample.frame_id
                    << " deep_copy_failed=" << ex.what() << "\n";
          io->fwd_push_fail.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        ensure_encoded_semantic(copy);
        log_edge("fwd_copy", copy);

        const int64_t cap_ms = sima_examples::time_ms_i64();
        const size_t payload_bytes = encoded_sample_bytes_estimate(copy);
        const bool queued =
            sync_store->enqueue(idx, sample.frame_id, std::move(copy), cap_ms, payload_bytes);
        if (!queued) {
          io->fwd_push_fail.fetch_add(1, std::memory_order_relaxed);
        } else {
          io->fwd_push_ok.fetch_add(1, std::memory_order_relaxed);
        }
      },
      "fwd_" + sid));

  auto dec_in = g.add(simaai::neat::graph::nodes::Map(
      [sid, frame_counter](simaai::neat::Sample& sample) {
        if (sample.stream_id.empty())
          sample.stream_id = sid;
        if (sample.frame_id < 0) {
          sample.frame_id = frame_counter->fetch_add(1, std::memory_order_relaxed);
        }
        sample.input_seq = sample.frame_id;
        log_edge("dec_in", sample);
      },
      "dec_in_" + sid));

  auto dec = simaai::neat::graph::helpers::add_pipeline(
      g,
      simaai::neat::nodes::H264Decode(/*allocator=*/2,
                                      /*out_format=*/"NV12",
                                      /*decoder_name=*/decoder_name,
                                      /*raw_output=*/true,
                                      /*next_element=*/"",
                                      /*dec_width=*/-1,
                                      /*dec_height=*/-1,
                                      /*dec_fps=*/-1,
                                      /*num_buffers=*/kFixedDecoderNumBuffers),
      "dec_" + sid);

  auto dec_out = g.add(simaai::neat::graph::nodes::Map(
      [sid, frame_counter](simaai::neat::Sample& sample) {
        if (sample.stream_id.empty())
          sample.stream_id = sid;
        if (sample.orig_input_seq >= 0) {
          sample.frame_id = sample.orig_input_seq;
        } else if (sample.input_seq >= 0) {
          sample.frame_id = sample.input_seq;
        }
        if (sample.frame_id < 0) {
          sample.frame_id = frame_counter->fetch_add(1, std::memory_order_relaxed);
        }
        log_edge("dec_out", sample);
      },
      "dec_out_" + sid));

  auto mark_yolo_in = g.add(simaai::neat::graph::nodes::Map(
      [sid](simaai::neat::Sample& sample) {
        if (sample.stream_id.empty())
          sample.stream_id = sid;
        sample.input_seq = sample.frame_id;
      },
      "yolo_in_" + sid));

  simaai::neat::graph::helpers::chain(g,
                                      {cap, fwd, dec_in, dec, dec_out, mark_yolo_in, yolo_sched});
}

void build_stream_maps(size_t streams, std::vector<std::string>& stream_ids,
                       std::unordered_map<std::string, size_t>& stream_index) {
  stream_ids.clear();
  stream_ids.reserve(streams);
  stream_index.clear();
  stream_index.reserve(streams);
  for (size_t i = 0; i < streams; ++i) {
    const std::string sid = "stream" + std::to_string(i);
    stream_ids.push_back(sid);
    stream_index.emplace(sid, i);
  }
}

void on_yolo_sample(const simaai::neat::Sample& sample, CollectorContext& ctx) {
  log_edge("yolo_out", sample);

  auto it = ctx.stream_index.find(sample.stream_id);
  if (it == ctx.stream_index.end())
    return;
  const size_t idx = it->second;
  const std::string& sid = sample.stream_id;

  auto& io = *ctx.io_stats[idx];
  io.yolo_samples.fetch_add(1, std::memory_order_relaxed);

  const int64_t yolo_frame = sample.frame_id;
  const int64_t yolo_ms = sima_examples::time_ms_i64();

  auto tok = ctx.yolo_tokens->take(idx);
  if (!tok.has_value()) {
    io.sync_token_miss.fetch_add(1, std::memory_order_relaxed);
  }
  const int64_t token_frame = tok.has_value() ? tok->frame_id : -1;

  const int64_t match_frame = (yolo_frame >= 0) ? yolo_frame : token_frame;
  bool matched_exact = false;
  std::optional<SyncPendingVideoStore::PendingFrame> pending_video;
  if (match_frame >= 0) {
    pending_video = ctx.sync_store->take(idx, match_frame);
    matched_exact = pending_video.has_value();
  }
  if (!pending_video.has_value() && tok.has_value() && token_frame >= 0 &&
      token_frame != match_frame) {
    pending_video = ctx.sync_store->take(idx, token_frame);
    matched_exact = pending_video.has_value();
  }

  if (pending_video.has_value()) {
    if (matched_exact) {
      io.sync_exact_ok.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    io.sync_exact_miss.fetch_add(1, std::memory_order_relaxed);
    io.sync_match_miss.fetch_add(1, std::memory_order_relaxed);
    if (yolo_frame < 3 || yolo_frame % 500 == 0) {
      std::cerr << "[sync_release] exact_miss stream=" << sid << " yolo_frame=" << yolo_frame
                << " token_frame=" << (tok.has_value() ? std::to_string(token_frame) : "<none>")
                << "\n";
    }
    return;
  }

  bool released_video = false;
  auto release_video = [&]() {
    if (released_video || !pending_video.has_value())
      return;

    bool accepted = false;
    if (ctx.release_pacer) {
      accepted = ctx.release_pacer->enqueue(idx, std::move(pending_video->sample));
    } else {
      const bool release_ok = ctx.optiview_group.push_video(idx, pending_video->sample);
      if (!release_ok) {
        io.sync_release_fail.fetch_add(1, std::memory_order_relaxed);
      } else {
        io.sync_release_ok.fetch_add(1, std::memory_order_relaxed);
      }
      accepted = true;
    }

    if (!accepted) {
      io.sync_release_fail.fetch_add(1, std::memory_order_relaxed);
    }

    pending_video.reset();
    released_video = true;
  };

  if (ctx.processed[idx] >= ctx.cfg.frames) {
    release_video();
    return;
  }
  ctx.processed[idx]++;

  if (ctx.cfg.send_json) {
    simaai::neat::nodes::groups::OptiViewJsonInput json_in;
    json_in.stream_idx = idx;
    json_in.stream_id = sid;
    json_in.frame_id = yolo_frame;
    json_in.capture_ms = pending_video.has_value() ? pending_video->cap_ms : -1;
    json_in.yolo_ms = yolo_ms;
    json_in.output_frame_id = ctx.processed[idx] - 1;
    json_in.yolo_sample = &sample;

    simaai::neat::nodes::groups::OptiViewJsonResult json_out;
    if (ctx.optiview_group.emit_json(json_in, &json_out)) {
      io.json_ok.fetch_add(1, std::memory_order_relaxed);
      io.boxes_total.fetch_add(json_out.boxes, std::memory_order_relaxed);
      if (json_out.nonempty) {
        io.json_nonempty.fetch_add(1, std::memory_order_relaxed);
      } else {
        io.json_empty.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      io.json_fail.fetch_add(1, std::memory_order_relaxed);
      if (json_out.error.find("bbox extract failed") != std::string::npos) {
        io.bbox_extract_fail.fetch_add(1, std::memory_order_relaxed);
      } else if (json_out.error.find("bbox parse failed") != std::string::npos) {
        io.bbox_parse_fail.fetch_add(1, std::memory_order_relaxed);
      }
      std::cerr << "[warn] optiview json send failed stream=" << sid << " err=" << json_out.error
                << "\n";
    }
  }

  release_video();
}

FinalTotals print_stream_summaries(const std::vector<std::shared_ptr<StreamIoStats>>& io_stats,
                                   const std::shared_ptr<SyncPendingVideoStore>& sync_store,
                                   const std::vector<std::shared_ptr<YoloTokenStore>>& yolo_tokens,
                                   const std::shared_ptr<SyncReleasePacer>& release_pacer) {
  FinalTotals totals;

  for (size_t i = 0; i < io_stats.size(); ++i) {
    const auto& io = *io_stats[i];

    const int64_t yolo_samples = io.yolo_samples.load(std::memory_order_relaxed);
    const int64_t json_ok = io.json_ok.load(std::memory_order_relaxed);
    const int64_t json_fail = io.json_fail.load(std::memory_order_relaxed);
    const int64_t json_nonempty = io.json_nonempty.load(std::memory_order_relaxed);
    const int64_t json_empty = io.json_empty.load(std::memory_order_relaxed);
    const int64_t boxes_total = io.boxes_total.load(std::memory_order_relaxed);

    const int64_t fwd_ok = io.fwd_push_ok.load(std::memory_order_relaxed);
    const int64_t fwd_fail = io.fwd_push_fail.load(std::memory_order_relaxed);

    const int64_t bbox_extract_fail = io.bbox_extract_fail.load(std::memory_order_relaxed);
    const int64_t bbox_parse_fail = io.bbox_parse_fail.load(std::memory_order_relaxed);

    const int64_t sync_release_ok = io.sync_release_ok.load(std::memory_order_relaxed);
    const int64_t sync_release_fail = io.sync_release_fail.load(std::memory_order_relaxed);
    const int64_t sync_release_pace_drop =
        io.sync_release_pace_drop.load(std::memory_order_relaxed);

    const int64_t sync_exact_ok = io.sync_exact_ok.load(std::memory_order_relaxed);
    const int64_t sync_exact_miss = io.sync_exact_miss.load(std::memory_order_relaxed);
    const int64_t sync_token_miss = io.sync_token_miss.load(std::memory_order_relaxed);
    const int64_t sync_match_miss = io.sync_match_miss.load(std::memory_order_relaxed);

    totals.total_fwd_fail += fwd_fail;
    totals.total_sync_exact_miss += sync_exact_miss;
    totals.total_sync_match_miss += sync_match_miss;
    totals.total_sync_release_fail += sync_release_fail;

    const auto sync_stats = sync_store->stats(i);
    YoloTokenStore::Stats tok_stats{};
    for (const auto& store : yolo_tokens) {
      if (!store)
        continue;
      const auto s = store->stats(i);
      tok_stats.enqueued += s.enqueued;
      tok_stats.dequeued += s.dequeued;
      tok_stats.miss += s.miss;
      tok_stats.depth += s.depth;
      tok_stats.max_depth = std::max(tok_stats.max_depth, s.max_depth);
    }
    const auto pacer_stats = release_pacer ? release_pacer->stats(i) : SyncReleasePacer::Stats{};

    const double avg_boxes =
        json_ok > 0 ? static_cast<double>(boxes_total) / static_cast<double>(json_ok) : 0.0;

    std::cout << "[stream_io_summary]"
              << " stream=" << i << " fwd_ok=" << fwd_ok << " fwd_fail=" << fwd_fail
              << " yolo_samples=" << yolo_samples << " json_ok=" << json_ok
              << " json_fail=" << json_fail << " json_nonempty=" << json_nonempty
              << " json_empty=" << json_empty << " boxes_total=" << boxes_total
              << " avg_boxes_per_json=" << avg_boxes << " bbox_extract_fail=" << bbox_extract_fail
              << " bbox_parse_fail=" << bbox_parse_fail << " sync_release_ok=" << sync_release_ok
              << " sync_release_fail=" << sync_release_fail
              << " sync_release_pace_drop=" << sync_release_pace_drop
              << " sync_exact_ok=" << sync_exact_ok << " sync_exact_miss=" << sync_exact_miss
              << " sync_token_miss=" << sync_token_miss << " sync_match_miss=" << sync_match_miss
              << "\n";

    std::cout << "[sync_queue_summary]"
              << " stream=" << i << " enqueued=" << sync_stats.enqueued
              << " matched=" << sync_stats.matched << " miss=" << sync_stats.miss
              << " pending_depth=" << sync_stats.pending_depth
              << " pending_mb=" << (sync_stats.pending_bytes / (1024ULL * 1024ULL))
              << " max_pending_depth=" << sync_stats.max_pending_depth
              << " max_pending_mb=" << (sync_stats.max_pending_bytes / (1024ULL * 1024ULL))
              << " token_enq=" << tok_stats.enqueued << " token_deq=" << tok_stats.dequeued
              << " token_miss=" << tok_stats.miss << " token_depth=" << tok_stats.depth
              << " token_max_depth=" << tok_stats.max_depth << " pace_sent=" << pacer_stats.sent
              << " pace_dropped=" << pacer_stats.dropped << "\n";
  }

  return totals;
}

} // namespace sima_examples::graphpipes_optiview
