// 4PipesYOLOOptiview_no_restart.cpp
//
// New architecture (no restarts):
// 1) Build RTSP pipelines (per stream)  -> produces encoded H264
// 2) Build UDP forward pipelines (per stream) and keep hot  -> forwards encoded to OptiView
// immediately 3) Build Decoder pipelines (per stream) and keep hot       -> decodes continuously 4)
// Build YOLO once (shared) after first decoded dims are known 5) Start feeding YOLO from decoded
// queue (drop until ready)
//
// Key properties:
// - NO restart_requested / perform_restart / request_restart.
// - Bounded queues everywhere. Drop old frames under load.
// - Header/IDR gate for decoder input after startup.
// - Forward (UDP) never depends on YOLO.

#include "support/runtime/example_utils.h"

#include "support/object_detection/obj_detection_utils.h"
#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"
#include "builder/ConfigJsonOverride.h"

#include <gst/gst.h>
#include <gst/gstmeta.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::infer_dims;
using sima_examples::make_dummy_encoded_tensor;
using sima_examples::time_ms;
using sima_examples::time_ms_i64;

namespace {

// Keep deterministic caps for decoder+forward Input.
constexpr const char* kFixedH264Caps =
    "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
    "parsed=(boolean)true,width=(int)1280,height=(int)720,"
    "framerate=(fraction)30/1";

struct Config {
  std::string rtsp_list;
  std::string mpk;
  int frames = 300;
  bool frames_set = false;
  bool debug = false;
  std::string optiview_host = "127.0.0.1";
  int optiview_video_port = 9000;
  int optiview_json_port = 9100;
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  cfg.rtsp_list = sima_examples::default_rtsp_list_path().string();
  sima_examples::get_arg(argc, argv, "--rtsp-list", cfg.rtsp_list);
  sima_examples::get_arg(argc, argv, "--mpk", cfg.mpk);
  std::string raw;
  if (sima_examples::get_arg(argc, argv, "--frames", raw)) {
    cfg.frames = std::stoi(raw);
    cfg.frames_set = true;
  }
  if (sima_examples::get_arg(argc, argv, "--optiview-host", raw)) {
    cfg.optiview_host = raw;
  }
  if (sima_examples::get_arg(argc, argv, "--optiview-video-port", raw)) {
    cfg.optiview_video_port = std::stoi(raw);
  }
  if (sima_examples::get_arg(argc, argv, "--optiview-json-port", raw)) {
    cfg.optiview_json_port = std::stoi(raw);
  }
  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  return cfg;
}

size_t encoded_sample_bytes(const simaai::neat::Sample& sample) {
  if (!sample.tensor.has_value())
    return 1;
  const auto& neat = *sample.tensor;
  if (neat.shape.size() == 1)
    return static_cast<size_t>(neat.shape[0]);
  if (neat.storage)
    return static_cast<size_t>(neat.storage->size_bytes);
  return 1;
}

std::string sima_meta_buffer_name(const simaai::neat::Sample& sample) {
  if (!sample.tensor.has_value())
    return "<no-neat>";
  const auto& tensor = *sample.tensor;
  if (!tensor.storage || !tensor.storage->holder)
    return "<no-holder>";
  return "<unavailable>";
}

std::optional<simaai::neat::Sample> shallow_copy_sample(const simaai::neat::Sample& sample) {
  if constexpr (std::is_copy_constructible_v<simaai::neat::Sample>) {
    return sample;
  }
  return std::nullopt;
}

// ------------------------ BoundedQueue (drop-oldest) ------------------------

template <class T> class BoundedQueue {
public:
  explicit BoundedQueue(size_t cap) : cap_(cap) {}

  void close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return closed_;
  }

  // Push; if full, drop oldest (unless keep_latest is true, then drop incoming).
  // Returns false if closed.
  bool push_drop_oldest(T item, bool keep_latest = false) {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_)
      return false;
    if (cap_ == 0)
      return false;
    if (q_.size() >= cap_) {
      if (keep_latest) {
        return true; // silently drop incoming
      }
      q_.pop_front();
    }
    q_.push_back(std::move(item));
    cv_.notify_one();
    return true;
  }

  // Pop with timeout (ms). Returns false on timeout or closed+empty.
  bool pop_wait(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timeout_ms < 0) {
      cv_.wait(lk, [&]() { return closed_ || !q_.empty(); });
    } else {
      cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                   [&]() { return closed_ || !q_.empty(); });
    }
    if (q_.empty())
      return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
  }

private:
  size_t cap_ = 0;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
};

// ------------------------ H264 header scan helpers ------------------------

std::string hex_prefix(const std::vector<uint8_t>& bytes, size_t max_bytes) {
  static const char kHex[] = "0123456789ABCDEF";
  const size_t n = bytes.size() < max_bytes ? bytes.size() : max_bytes;
  std::string out;
  out.reserve(n * 3);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t b = bytes[i];
    out.push_back(kHex[(b >> 4) & 0x0F]);
    out.push_back(kHex[b & 0x0F]);
    if (i + 1 < n)
      out.push_back(' ');
  }
  return out;
}

struct AnnexBProbe {
  bool found = false;
  bool prefix_ok = false;
  size_t offset = 0;
  size_t start_code_len = 0;
  std::string prefix_hex;
};

AnnexBProbe probe_annexb(const simaai::neat::Sample& sample) {
  AnnexBProbe out;
  if (!sample.tensor.has_value())
    return out;
  const auto bytes = sample.tensor->copy_payload_bytes();
  if (bytes.empty())
    return out;
  out.prefix_hex = hex_prefix(bytes, 12);
  const size_t max_scan = bytes.size() < 256 ? bytes.size() : 256;
  for (size_t i = 0; i + 2 < max_scan; ++i) {
    if (bytes[i] != 0 || bytes[i + 1] != 0)
      continue;
    if (i + 3 < max_scan && bytes[i + 2] == 0 && bytes[i + 3] == 1) {
      out.found = true;
      out.offset = i;
      out.start_code_len = 4;
      break;
    }
    if (bytes[i + 2] == 1) {
      out.found = true;
      out.offset = i;
      out.start_code_len = 3;
      break;
    }
  }
  out.prefix_ok = out.found && out.offset == 0;
  return out;
}

struct H264NalSummary {
  bool has_sps = false;
  bool has_pps = false;
  bool has_idr = false;
  bool has_aud = false;
  int nal_count = 0;
};

H264NalSummary scan_h264_annexb_bytes(const std::vector<uint8_t>& bytes) {
  H264NalSummary out;
  if (bytes.size() < 4)
    return out;
  struct StartCode {
    size_t offset = 0;
    size_t len = 0;
  };
  std::vector<StartCode> starts;
  starts.reserve(16);
  for (size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] != 0 || bytes[i + 1] != 0)
      continue;
    if (bytes[i + 2] == 1) {
      starts.push_back(StartCode{i, 3});
      i += 2;
      continue;
    }
    if (bytes[i + 2] == 0 && bytes[i + 3] == 1) {
      starts.push_back(StartCode{i, 4});
      i += 3;
      continue;
    }
  }
  for (const auto& sc : starts) {
    const size_t nal_start = sc.offset + sc.len;
    if (nal_start >= bytes.size())
      continue;
    const uint8_t nal_type = bytes[nal_start] & 0x1F;
    out.nal_count += 1;
    if (nal_type == 7)
      out.has_sps = true;
    else if (nal_type == 8)
      out.has_pps = true;
    else if (nal_type == 5)
      out.has_idr = true;
    else if (nal_type == 9)
      out.has_aud = true;
  }
  return out;
}

H264NalSummary scan_h264_annexb_sample(const simaai::neat::Sample& sample) {
  if (!sample.tensor.has_value())
    return H264NalSummary{};
  const auto bytes = sample.tensor->copy_payload_bytes();
  return scan_h264_annexb_bytes(bytes);
}

bool is_headers_and_idr(const simaai::neat::Sample& s) {
  const H264NalSummary n = scan_h264_annexb_sample(s);
  return n.has_sps && n.has_pps && n.has_idr;
}
bool is_idr_or_headers(const simaai::neat::Sample& s) {
  const H264NalSummary n = scan_h264_annexb_sample(s);
  return n.has_idr || n.has_sps || n.has_pps;
}

std::string h264_nal_summary_string(const H264NalSummary& s) {
  std::string out;
  out.reserve(64);
  out += "nals=" + std::to_string(s.nal_count);
  out += " sps=" + std::string(s.has_sps ? "1" : "0");
  out += " pps=" + std::string(s.has_pps ? "1" : "0");
  out += " idr=" + std::string(s.has_idr ? "1" : "0");
  out += " aud=" + std::string(s.has_aud ? "1" : "0");
  return out;
}

// ------------------------ App structs ------------------------

struct FrameItem {
  int stream_idx = 0;
  int index = 0;                // per-stream counter for labeling
  double pull_ts_ms = 0.0;      // when we pulled decoded
  simaai::neat::Sample decoded; // decoded NV12 (for YOLO)
};

struct StreamStats {
  std::atomic<int> rtsp_pulled{0};
  std::atomic<int> fwd_pushed{0};
  std::atomic<int> dec_pushed{0};
  std::atomic<int> dec_pulled{0};
  std::atomic<int> yolo_in{0};
  std::atomic<int> yolo_out{0};
  std::atomic<int> saved{0};

  std::atomic<int64_t> start_ms{0};
  std::atomic<int64_t> last_rtsp_ms{0};
  std::atomic<int64_t> last_dec_ms{0};
  std::atomic<int64_t> last_yolo_ms{0};
};

struct Stream {
  int idx = 0;
  std::string url;
  std::string decoder_name;
  std::string caps_string = kFixedH264Caps;

  size_t dummy_bytes = 1;

  // Gate state
  std::atomic<bool> need_headers{true};
  std::atomic<int64_t> headers_wait_start_ms{0};

  // Pipelines
  simaai::neat::Run rtsp_run;
  simaai::neat::Run fwd_run;
  simaai::neat::Run dec_run;

  // Queues
  BoundedQueue<simaai::neat::Sample> enc_for_fwd{20};
  BoundedQueue<simaai::neat::Sample> enc_for_dec{20};

  // Per-stream frame counter
  std::atomic<int> frame_index{0};
};

// ------------------------ misc ------------------------

double fps_from_count(int count, double elapsed_ms) {
  if (elapsed_ms <= 0.0)
    return 0.0;
  return static_cast<double>(count) * 1000.0 / elapsed_ms;
}

void print_throughput_summary(const std::vector<std::unique_ptr<StreamStats>>& stats,
                              int total_yolo_out, int total_saved, double start_ms, double end_ms) {
  const double elapsed = (end_ms > start_ms) ? (end_ms - start_ms) : 0.0;
  int total_rtsp = 0, total_fwd = 0, total_dec_pull = 0, total_dec_push = 0;
  for (const auto& st : stats) {
    total_rtsp += st->rtsp_pulled.load();
    total_fwd += st->fwd_pushed.load();
    total_dec_push += st->dec_pushed.load();
    total_dec_pull += st->dec_pulled.load();
  }
  std::cout << "[THROUGHPUT] elapsed_ms=" << elapsed << " rtsp_pulled=" << total_rtsp
            << " fwd_pushed=" << total_fwd << " dec_pushed=" << total_dec_push
            << " dec_pulled=" << total_dec_pull << " yolo_out=" << total_yolo_out
            << " saved=" << total_saved << " yolo_fps=" << fps_from_count(total_yolo_out, elapsed)
            << " saved_fps=" << fps_from_count(total_saved, elapsed) << "\n";
}

int min_saved_across_streams(const std::vector<std::unique_ptr<StreamStats>>& stats) {
  int min_saved = std::numeric_limits<int>::max();
  for (const auto& st : stats) {
    min_saved = std::min(min_saved, st->saved.load());
  }
  if (min_saved == std::numeric_limits<int>::max())
    return 0;
  return min_saved;
}

} // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_config(argc, argv);
    if (cfg.mpk.empty())
      cfg.mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());
    sima_examples::require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");
    if (cfg.frames_set && cfg.frames <= 0)
      sima_examples::require(false, "--frames must be > 0");

    const auto urls = sima_examples::read_rtsp_list(cfg.rtsp_list);
    sima_examples::require(urls.size() <= 20, "Max supported streams is 20");
    std::cout << "rtsp_list=" << cfg.rtsp_list << " streams=" << urls.size() << "\n";
    for (size_t i = 0; i < urls.size(); ++i) {
      std::cout << "rtsp[" << i << "]=" << urls[i] << "\n";
    }

    // Tunables (simple and robust)
    const int rtsp_pull_timeout_ms = 200;
    const int dec_pull_timeout_ms = 5;
    const int det_pull_timeout_ms = 50;

    const int64_t kHeaderWaitMs = 10'000;
    const int64_t kFailNoRtspMs = 10'000;
    const int64_t kFailNoDecodeMs = 10'000;
    const int64_t kFailNoYoloMs = 10'000;

    const int topk = 100;
    const float min_score = 0.52f;

    std::atomic<bool> stop{false};
    std::atomic<bool> yolo_ready{false};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;

    auto fail_now = [&](const std::string& msg) {
      bool expected = false;
      if (failed.compare_exchange_strong(expected, true)) {
        std::lock_guard<std::mutex> lk(fail_mu);
        fail_msg = msg;
      }
      stop.store(true);
    };

    // ------------------ Build Streams + RTSP runs ------------------

    std::deque<Stream> streams;
    std::vector<std::unique_ptr<StreamStats>> stats;
    stats.reserve(urls.size());

    for (size_t i = 0; i < urls.size(); ++i) {
      stats.push_back(std::make_unique<StreamStats>());
      streams.emplace_back();
      Stream& s = streams.back();
      s.idx = static_cast<int>(i);
      s.url = urls[i];
      s.decoder_name = "decoder_stream_" + std::to_string(i);
      s.need_headers.store(true);
      s.headers_wait_start_ms.store(time_ms_i64());
      stats.back()->start_ms.store(time_ms_i64());

      // RTSP pipeline (encoded out)
      simaai::neat::Session rtsp;
      rtsp.add(simaai::neat::nodes::RTSPInput(s.url, /*latency_ms=*/800, /*tcp=*/true));
      rtsp.add(
          simaai::neat::nodes::H264Depacketize(/*payload_type=*/96, /*wait_for_keyframe=*/true));
      rtsp.add(simaai::neat::nodes::Custom("h264parse config-interval=1 disable-passthrough=true"));
      rtsp.add(
          simaai::neat::nodes::Custom(std::string("capsfilter caps=\"") + kFixedH264Caps + "\""));
      rtsp.add(simaai::neat::nodes::Output());
      simaai::neat::RunOptions rtsp_opt;
      rtsp_opt.enable_metrics = cfg.debug;
      s.rtsp_run = rtsp.build(rtsp_opt);

      // Pull one encoded to estimate size + show caps
      simaai::neat::Sample first_enc;
      simaai::neat::PullError enc_err;
      auto st = s.rtsp_run.pull(5000, first_enc, &enc_err);
      sima_examples::require(st == simaai::neat::PullStatus::Ok,
                             enc_err.message.empty() ? "Failed to pull first encoded sample"
                                                     : enc_err.message);
      sima_examples::require(first_enc.tensor.has_value(), "First encoded sample missing payload");
      sima_examples::require(first_enc.tensor->semantic.encoded.has_value(),
                             "First encoded sample is not marked as encoded");
      sima_examples::require(!first_enc.caps_string.empty(),
                             "First encoded sample missing caps_string");

      const AnnexBProbe annexb = probe_annexb(first_enc);
      sima_examples::require(annexb.prefix_ok,
                             "First encoded sample not Annex-B at offset 0. prefix=" +
                                 annexb.prefix_hex);

      s.dummy_bytes = encoded_sample_bytes(first_enc);
      stats[i]->last_rtsp_ms.store(time_ms_i64());
      std::cout << "[init] stream=" << i << " first_enc_caps=" << first_enc.caps_string << "\n";
      std::cout << "[init] stream=" << i << " caps_override=" << s.caps_string
                << " encoded_ok bytes=" << s.dummy_bytes << "\n";

      // Seed queues so forward/decode start instantly.
      // Prefer two pulls to avoid copying; fallback to a shallow copy if needed.
      simaai::neat::Sample first_enc2;
      st = s.rtsp_run.pull(1000, first_enc2, &enc_err);
      const bool have_second = (st == simaai::neat::PullStatus::Ok);

      if (have_second) {
        s.enc_for_dec.push_drop_oldest(std::move(first_enc2), /*keep_latest=*/false);
        s.enc_for_fwd.push_drop_oldest(std::move(first_enc), /*keep_latest=*/false);
      } else {
        auto first_fwd = shallow_copy_sample(first_enc);
        if (first_fwd.has_value()) {
          s.enc_for_fwd.push_drop_oldest(std::move(*first_fwd), /*keep_latest=*/false);
        }
        s.enc_for_dec.push_drop_oldest(std::move(first_enc), /*keep_latest=*/false);
      }
    }

    // ------------------ Build UDP Forward runs (HOT) ------------------

    for (auto& s : streams) {
      simaai::neat::Session forward;
      simaai::neat::InputOptions fwd_src;
      fwd_src.use_simaai_pool = false;
      fwd_src.caps_override = s.caps_string;
      forward.add(simaai::neat::nodes::Input(fwd_src));
      forward.add(
          simaai::neat::nodes::Custom("h264parse config-interval=1 disable-passthrough=true"));
      forward.add(simaai::neat::nodes::Custom(
          "capsfilter "
          "caps=\"video/"
          "x-h264,stream-format=(string)byte-stream,alignment=(string)au,parsed=(boolean)true\""));
      forward.add(simaai::neat::nodes::H264Packetize(96, 1));
      simaai::neat::UdpOutputOptions udp;
      udp.host = cfg.optiview_host;
      udp.port = cfg.optiview_video_port + s.idx;
      forward.add(simaai::neat::nodes::UdpOutput(udp));

      simaai::neat::RunOptions opt;
      opt.enable_metrics = cfg.debug;

      simaai::neat::Tensor dummy = make_dummy_encoded_tensor(s.dummy_bytes);
      s.fwd_run = forward.build(dummy, simaai::neat::RunMode::Async, opt);
    }

    // ------------------ Build Decoder runs (HOT) ------------------

    for (auto& s : streams) {
      simaai::neat::Session decoder;
      simaai::neat::InputOptions dec_src;
      dec_src.use_simaai_pool = false;
      dec_src.caps_override = s.caps_string;
      dec_src.buffer_name = s.decoder_name;
      decoder.add(simaai::neat::nodes::Input(dec_src));
      decoder.add(
          simaai::neat::nodes::Custom("h264parse config-interval=1 disable-passthrough=true"));
      decoder.add(
          simaai::neat::nodes::Custom(std::string("capsfilter caps=\"") + kFixedH264Caps + "\""));
      decoder.add(simaai::neat::nodes::H264Decode(/*allocator=*/2, /*format=*/"NV12",
                                                  /*decoder_name=*/s.decoder_name,
                                                  /*raw_output=*/true));
      decoder.add(simaai::neat::nodes::Output());
      simaai::neat::RunOptions opt;
      opt.enable_metrics = cfg.debug;

      simaai::neat::Tensor dummy = make_dummy_encoded_tensor(s.dummy_bytes);
      s.dec_run = decoder.build(dummy, simaai::neat::RunMode::Async, opt);
    }

    // ------------------ OptiView JSON senders ------------------

    std::vector<std::unique_ptr<sima_examples::OptiViewSender>> senders;
    senders.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      sima_examples::OptiViewOptions opt;
      opt.host = cfg.optiview_host;
      opt.channel = static_cast<int>(i);
      opt.video_port_base = cfg.optiview_video_port;
      opt.json_port_base = cfg.optiview_json_port;
      std::string opt_err;
      auto sender = std::make_unique<sima_examples::OptiViewSender>(opt, &opt_err);
      sima_examples::require(sender->ok(), opt_err);
      std::cout << "optiview host=" << sender->host() << " video_port=" << sender->video_port()
                << " json_port=" << sender->json_port() << " channel=" << i << "\n";
      senders.push_back(std::move(sender));
    }

    const auto optiview_labels = sima_examples::optiview_default_labels();

    // ------------------ Global decoded queue (latest-ish) ------------------

    // Keep this small: we prefer latest frames, not backlog.
    BoundedQueue<FrameItem> decoded_q(32);

    std::atomic<bool> dims_ready{false};
    std::mutex dims_mu;
    std::condition_variable dims_cv;
    int frame_w = 0;
    int frame_h = 0;
    std::optional<simaai::neat::Tensor> yolo_template;

    // ------------------ Threads ------------------

    std::vector<std::thread> rtsp_threads;
    std::vector<std::thread> fwd_threads;
    std::vector<std::thread> dec_threads;

    rtsp_threads.reserve(streams.size());
    fwd_threads.reserve(streams.size());
    dec_threads.reserve(streams.size());

    // RTSP puller: pulls encoded, publishes to enc_for_fwd + enc_for_dec (drop-oldest).
    for (size_t i = 0; i < streams.size(); ++i) {
      rtsp_threads.emplace_back([&, i]() {
        Stream& s = streams[i];
        while (!stop.load()) {
          simaai::neat::Sample enc;
          simaai::neat::PullError err;
          auto st = s.rtsp_run.pull(rtsp_pull_timeout_ms, enc, &err);
          if (st == simaai::neat::PullStatus::Timeout)
            continue;
          if (st == simaai::neat::PullStatus::Closed) {
            fail_now("RTSP closed on stream " + std::to_string(s.idx));
            break;
          }
          if (st == simaai::neat::PullStatus::Error) {
            fail_now("RTSP error on stream " + std::to_string(s.idx) + ": " +
                     (err.message.empty() ? "unknown" : err.message));
            break;
          }
          stats[s.idx]->rtsp_pulled.fetch_add(1);
          stats[s.idx]->last_rtsp_ms.store(time_ms_i64());

          // Keep forward hot always. Prefer shallow copy; fallback to a second pull if needed.
          auto fwd_copy = shallow_copy_sample(enc);
          if (fwd_copy.has_value()) {
            (void)s.enc_for_fwd.push_drop_oldest(std::move(*fwd_copy), /*keep_latest=*/false);
          }

          // For decoder: protect headers when we need them.
          const bool need = s.need_headers.load(std::memory_order_relaxed);
          if (!need) {
            (void)s.enc_for_dec.push_drop_oldest(std::move(enc), /*keep_latest=*/false);
          } else {
            // If queue is full and this is not headers/idr, just drop.
            if (!is_idr_or_headers(enc)) {
              // do nothing
            } else {
              (void)s.enc_for_dec.push_drop_oldest(std::move(enc), /*keep_latest=*/false);
            }
          }

          if (!fwd_copy.has_value()) {
            // Move-only Sample fallback: grab another frame for forward path.
            simaai::neat::Sample enc_fwd;
            simaai::neat::PullError fwd_err;
            auto st2 = s.rtsp_run.pull(0, enc_fwd, &fwd_err);
            if (st2 == simaai::neat::PullStatus::Ok) {
              (void)s.enc_for_fwd.push_drop_oldest(std::move(enc_fwd), /*keep_latest=*/false);
            }
          }
        }
        s.enc_for_fwd.close();
        s.enc_for_dec.close();
      });
    }

    // Forward feeder: reads enc_for_fwd and pushes into fwd_run
    for (size_t i = 0; i < streams.size(); ++i) {
      fwd_threads.emplace_back([&, i]() {
        Stream& s = streams[i];
        while (!stop.load()) {
          simaai::neat::Sample enc;
          if (!s.enc_for_fwd.pop_wait(enc, 200))
            continue;
          // never block on forward; drop if busy
          if (s.fwd_run.try_push(enc)) {
            stats[s.idx]->fwd_pushed.fetch_add(1);
          }
        }
      });
    }

    // Decoder worker: reads enc_for_dec, gates until SPS/PPS/IDR, pushes into dec_run,
    // continuously drains dec_run and publishes decoded frames into decoded_q
    for (size_t i = 0; i < streams.size(); ++i) {
      dec_threads.emplace_back([&, i]() {
        Stream& s = streams[i];
        s.headers_wait_start_ms.store(time_ms_i64());
        while (!stop.load()) {
          // 1) Feed decoder
          simaai::neat::Sample enc;
          if (s.enc_for_dec.pop_wait(enc, 50)) {
            bool need = s.need_headers.load(std::memory_order_relaxed);
            if (need) {
              if (!is_headers_and_idr(enc)) {
                const int64_t now = time_ms_i64();
                const int64_t start = s.headers_wait_start_ms.load();
                if (start > 0 && (now - start) > kHeaderWaitMs) {
                  const AnnexBProbe p = probe_annexb(enc);
                  const H264NalSummary n = scan_h264_annexb_sample(enc);
                  fail_now("Timed out waiting for SPS/PPS/IDR on stream " + std::to_string(s.idx) +
                           " last=" + h264_nal_summary_string(n) + " prefix=" + p.prefix_hex);
                  break;
                }
              } else {
                s.need_headers.store(false, std::memory_order_relaxed);
                s.headers_wait_start_ms.store(0);
                need = false;
              }
            }
            if (!need) {
              if (s.dec_run.try_push(enc)) {
                stats[s.idx]->dec_pushed.fetch_add(1);
              }
            }
          }

          // 2) Drain decoder output (always)
          for (int k = 0; k < 8 && !stop.load(); ++k) {
            simaai::neat::Sample dec;
            simaai::neat::PullError derr;
            auto st = s.dec_run.pull(dec_pull_timeout_ms, dec, &derr);
            if (st == simaai::neat::PullStatus::Timeout)
              break;
            if (st == simaai::neat::PullStatus::Closed) {
              fail_now("Decoder closed on stream " + std::to_string(s.idx));
              break;
            }
            if (st == simaai::neat::PullStatus::Error) {
              fail_now("Decoder error on stream " + std::to_string(s.idx) + ": " +
                       (derr.message.empty() ? "unknown" : derr.message));
              break;
            }
            stats[s.idx]->dec_pulled.fetch_add(1);
            stats[s.idx]->last_dec_ms.store(time_ms_i64());

            if (!dims_ready.load(std::memory_order_acquire)) {
              if (dec.tensor.has_value()) {
                int w = 0, h = 0;
                if (infer_dims(*dec.tensor, w, h)) {
                  std::lock_guard<std::mutex> lk(dims_mu);
                  if (!dims_ready.load(std::memory_order_relaxed)) {
                    frame_w = w;
                    frame_h = h;
                    yolo_template = *dec.tensor;
                    dims_ready.store(true, std::memory_order_release);
                    dims_cv.notify_one();
                  }
                }
              }
            }

            // Drop decoded frames until YOLO is ready.
            if (!yolo_ready.load(std::memory_order_relaxed))
              continue;

            FrameItem item;
            item.stream_idx = s.idx;
            item.index = s.frame_index.fetch_add(1);
            item.pull_ts_ms = time_ms(); // not perfect, but fine for demo
            item.decoded = std::move(dec);

            // YOLO runs on decoded frames; video is forwarded independently.
            decoded_q.push_drop_oldest(std::move(item), /*keep_latest=*/false);
          }
        }
      });
    }

    // ------------------ Wait for decoded dims ------------------

    {
      std::unique_lock<std::mutex> lk(dims_mu);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10'000);
      dims_cv.wait_until(lk, deadline, [&]() {
        return stop.load() || dims_ready.load(std::memory_order_acquire);
      });
    }

    if (!dims_ready.load(std::memory_order_acquire) && !failed.load()) {
      fail_now("Failed to obtain decoded dims before timeout");
    }

    simaai::neat::Run det;
    if (!failed.load()) {
      sima_examples::require(dims_ready.load(std::memory_order_acquire),
                             "Decoded dims unavailable");
      sima_examples::require(yolo_template.has_value(), "Decoded template tensor missing");

      // Build YOLO once.
      simaai::neat::Model::Options model_opt;
      model_opt.media_type = "video/x-raw";
      model_opt.format = "NV12";
      model_opt.input_max_width = frame_w;
      model_opt.input_max_height = frame_h;
      model_opt.input_max_depth = 1;
      simaai::neat::Model model(cfg.mpk, model_opt);
      simaai::neat::Session yolo;

      std::string decoder_names;
      for (size_t i = 0; i < streams.size(); ++i) {
        if (i)
          decoder_names += ",";
        decoder_names += streams[i].decoder_name;
      }

      simaai::neat::InputOptions src_opt = model.input_appsrc_options(false);
      src_opt.format = "NV12";
      src_opt.width = frame_w;
      src_opt.height = frame_h;
      src_opt.buffer_name = decoder_names;
      yolo.add(simaai::neat::nodes::Input(src_opt));

      auto preproc_group = simaai::neat::nodes::groups::Preprocess(model);
      for (auto& node : preproc_group.nodes_mut()) {
        auto* override = dynamic_cast<simaai::neat::ConfigJsonOverride*>(node.get());
        if (!override)
          continue;
        override->override_config_json(
            [&](nlohmann::json& j) {
              if (j.contains("input_buffers") && j["input_buffers"].is_array() &&
                  !j["input_buffers"].empty() && j["input_buffers"][0].is_object()) {
                j["input_buffers"][0]["name"] = decoder_names;
              }
            },
            "rtsp_multi_names");
      }
      yolo.add(preproc_group);
      yolo.add(simaai::neat::nodes::groups::Infer(model));
      yolo.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", frame_w, frame_h, min_score,
                                                  0.5f, topk));
      yolo.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions det_opt;
      det_opt.enable_metrics = cfg.debug;
      det = yolo.build(*yolo_template, simaai::neat::RunMode::Async, det_opt);

      std::cout << "[init] dims=" << frame_w << "x" << frame_h << "\n";
      yolo_ready.store(true);
    }

    // ------------------ YOLO consumer thread (single) ------------------

    std::atomic<int64_t> next_frame_id{0};
    std::atomic<int> total_yolo_out{0};
    std::atomic<int> total_saved{0};

    const bool run_forever = !cfg.frames_set;
    const int target_saved = cfg.frames;

    const double start_all_ms = time_ms();
    std::thread yolo_thread;
    std::thread watchdog;

    if (!failed.load()) {
      yolo_thread = std::thread([&]() {
        std::vector<uint8_t> payload;
        std::vector<objdet::Box> boxes;
        std::vector<sima_examples::OptiViewObject> objects;
        payload.reserve(8192);
        boxes.reserve(static_cast<size_t>(topk));
        objects.reserve(static_cast<size_t>(topk));

        while (!stop.load()) {
          FrameItem item;
          if (!decoded_q.pop_wait(item, 200)) {
            continue;
          }
          // Throttle by frames target (saved == json sends), fair across streams.
          if (!run_forever && min_saved_across_streams(stats) >= target_saved) {
            stop.store(true);
            break;
          }

          // Attach IDs
          const int64_t fid = next_frame_id.fetch_add(1);
          item.decoded.frame_id = fid;
          item.decoded.stream_id = streams[item.stream_idx].decoder_name;
          item.decoded.port_name = streams[item.stream_idx].decoder_name;

          if (cfg.debug && item.index < 5) {
            const std::string meta_name = sima_meta_buffer_name(item.decoded);
            std::cerr << "[DBG] yolo_push stream=" << item.stream_idx << " frame=" << item.index
                      << " port_name=" << item.decoded.port_name
                      << " meta_buffer_name=" << meta_name << "\n";
          }

          if (!det.push(item.decoded)) {
            std::cerr << "[warn] yolo push failed\n";
            continue;
          }
          stats[item.stream_idx]->yolo_in.fetch_add(1);

          simaai::neat::Sample out;
          simaai::neat::PullError out_err;
          while (!stop.load()) {
            auto st2 = det.pull(1000, out, &out_err);
            if (st2 == simaai::neat::PullStatus::Timeout)
              continue;
            if (st2 == simaai::neat::PullStatus::Closed) {
              fail_now("YOLO closed");
              break;
            }
            if (st2 == simaai::neat::PullStatus::Error) {
              fail_now(std::string("YOLO error: ") +
                       (out_err.message.empty() ? "unknown" : out_err.message));
              break;
            }
            break;
          }
          if (stop.load())
            break;

          total_yolo_out.fetch_add(1);
          const int sidx = item.stream_idx;

          stats[sidx]->yolo_out.fetch_add(1);
          stats[sidx]->last_yolo_ms.store(time_ms_i64());

          payload.clear();
          std::string err;
          if (!sima_examples::extract_bbox_payload(out, payload, err)) {
            std::cerr << "[warn] bbox extract failed: " << err << "\n";
            continue;
          }

          boxes.clear();
          try {
            objdet::parse_boxes_strict_into(payload, frame_w, frame_h, topk,
                                            /*debug=*/false, boxes);
          } catch (const std::exception& ex) {
            std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
            continue;
          }

          objects.clear();
          for (const auto& b : boxes) {
            int x1 = static_cast<int>(b.x1);
            int y1 = static_cast<int>(b.y1);
            int w = static_cast<int>(b.x2 - b.x1);
            int h = static_cast<int>(b.y2 - b.y1);
            if (x1 < 0)
              x1 = 0;
            if (y1 < 0)
              y1 = 0;
            if (w < 0)
              w = 0;
            if (h < 0)
              h = 0;
            if (x1 + w > frame_w)
              w = frame_w - x1;
            if (y1 + h > frame_h)
              h = frame_h - y1;
            if (w < 0)
              w = 0;
            if (h < 0)
              h = 0;
            sima_examples::OptiViewObject obj;
            obj.x = x1;
            obj.y = y1;
            obj.w = w;
            obj.h = h;
            obj.score = b.score;
            obj.class_id = b.class_id;
            objects.push_back(obj);
          }

          // Send JSON
          const int64_t ts_ms = time_ms_i64();
          char frame_id_buf[32];
          std::snprintf(frame_id_buf, sizeof(frame_id_buf), "%d", stats[sidx]->saved.load());
          std::string json_payload =
              sima_examples::optiview_make_json(ts_ms, frame_id_buf, objects, optiview_labels);

          std::string json_err;
          if (senders[sidx]->send_json(json_payload, &json_err)) {
            stats[sidx]->saved.fetch_add(1);
            total_saved.fetch_add(1);
          } else {
            std::cerr << "[warn] optiview json send failed: " << json_err << "\n";
          }
        }
      });

      // ------------------ Watchdog ------------------

      watchdog = std::thread([&]() {
        while (!stop.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          const int64_t now = time_ms_i64();

          for (size_t si = 0; si < streams.size(); ++si) {
            const int64_t last_rtsp = stats[si]->last_rtsp_ms.load();
            const int64_t last_dec = stats[si]->last_dec_ms.load();
            const int64_t last_yolo = stats[si]->last_yolo_ms.load();
            const int64_t start_ms = stats[si]->start_ms.load();
            const bool dec_seen = stats[si]->dec_pulled.load() > 0;

            if (last_rtsp > 0 && (now - last_rtsp) > kFailNoRtspMs) {
              fail_now("RTSP stalled stream " + std::to_string(si));
              break;
            }
            if (dec_seen && last_dec > 0 && (now - last_dec) > kFailNoDecodeMs) {
              fail_now("Decoder stalled stream " + std::to_string(si));
              break;
            }
            if (!dec_seen && yolo_ready.load() && start_ms > 0 &&
                (now - start_ms) > kFailNoDecodeMs) {
              fail_now("Decoder stalled before first frame stream " + std::to_string(si));
              break;
            }
            if (yolo_ready.load() && last_yolo > 0 && (now - last_yolo) > kFailNoYoloMs) {
              fail_now("YOLO stalled (no outputs) stream " + std::to_string(si));
              break;
            }
          }
        }
      });
    }

    // Wait for completion
    if (yolo_thread.joinable())
      yolo_thread.join();
    decoded_q.close();
    stop.store(true);

    // Join background threads
    for (auto& t : rtsp_threads)
      if (t.joinable())
        t.join();
    for (auto& t : fwd_threads)
      if (t.joinable())
        t.join();
    for (auto& t : dec_threads)
      if (t.joinable())
        t.join();
    if (watchdog.joinable())
      watchdog.join();

    const double end_all_ms = time_ms();
    print_throughput_summary(stats, total_yolo_out.load(), total_saved.load(), start_all_ms,
                             end_all_ms);

    if (failed.load()) {
      std::string msg;
      {
        std::lock_guard<std::mutex> lk(fail_mu);
        msg = fail_msg.empty() ? "unknown failure" : fail_msg;
      }
      std::cerr << "[FAIL] " << msg << "\n";
      return 1;
    }

    std::cout << "saved_total=" << total_saved.load() << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
