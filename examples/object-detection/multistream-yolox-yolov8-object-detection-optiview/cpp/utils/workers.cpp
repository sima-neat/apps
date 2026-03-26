#include "workers_api.cpp"

#include "image_utils_api.cpp"
#include "pipeline_api.cpp"
#include "sample_utils_api.cpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace multistream_yolox_yolov8_optiview {
namespace {

constexpr int kDefaultProfileIntervalFrames = 200;
constexpr std::size_t kPendingDecodeSyncCapacity = 64;
const fs::path kDefaultLabelsPath =
    fs::path(MULTISTREAM_YOLOX_YOLOV8_OPTIVIEW_SOURCE_DIR) / ".." / "common" / "coco_label.txt";

using SteadyClock = std::chrono::steady_clock;

struct StreamMetrics {
  int pulled = 0;
  int processed = 0;
  int detections = 0;
  int saved = 0;
  int mailbox_drops = 0;
  int sync_exact_ok = 0;
  int sync_exact_miss = 0;
  int sync_release_ok = 0;
  int sync_release_fail = 0;
  double source_time_s = 0.0;
  double preproc_time_s = 0.0;
  double detect_time_s = 0.0;
  double video_push_time_s = 0.0;
  double json_time_s = 0.0;
  double publish_time_s = 0.0;
  double total_loop_time_s = 0.0;
  std::optional<double> wall_started_at_s;
  std::optional<double> wall_last_processed_at_s;
  double interval_source_s = 0.0;
  double interval_preproc_s = 0.0;
  double interval_detect_s = 0.0;
  double interval_video_s = 0.0;
  double interval_json_s = 0.0;
  double interval_publish_s = 0.0;
  double interval_loop_s = 0.0;
  int interval_frames = 0;
  int interval_sync_exact_ok = 0;
  int interval_sync_exact_miss = 0;
  int interval_sync_release_ok = 0;
  int interval_sync_release_fail = 0;
  std::optional<double> interval_wall_started_at_s;
};

struct FrameSyncMeta {
  std::int64_t frame_index = -1;
  double source_time_s = 0.0;
  double video_push_time_s = 0.0;
  std::optional<simaai::neat::Sample> matched_encoded;
};

struct FramePacket {
  simaai::neat::Sample decoded;
  std::int64_t frame_index = -1;
  double source_time_s = 0.0;
  double video_push_time_s = 0.0;
  std::optional<simaai::neat::Sample> matched_encoded;
};

struct EncodedPacket {
  simaai::neat::Sample encoded;
  std::int64_t frame_index = -1;
  double source_pull_time_s = 0.0;
  double video_push_time_s = 0.0;
};

struct AnnexBProbe {
  bool found = false;
  bool prefix_ok = false;
  std::size_t offset = 0;
  std::size_t start_code_len = 0;
  std::string prefix_hex;
};

struct H264NalSummary {
  bool has_sps = false;
  bool has_pps = false;
  bool has_idr = false;
  bool has_aud = false;
  int nal_count = 0;
};

class Event {
public:
  void set() {
    std::lock_guard<std::mutex> lock(mu_);
    signaled_ = true;
    cv_.notify_all();
  }

  bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [&] { return signaled_; });
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool signaled_ = false;
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  ModelFamily family = ModelFamily::Auto;
  RtspProbe probe;
  SessionRun source;
  std::optional<SessionRun> decode;
  std::optional<SessionRun> video;
  std::shared_ptr<BoundedQueue<EncodedPacket>> encoded_for_decode;
  bool video_enabled = true;
  std::optional<sima_examples::OptiViewSender> json_sender;
  bool json_enabled = true;
  std::vector<std::string> class_labels;
  std::shared_ptr<std::atomic<bool>> clean_video_bootstrapped;
  StreamMetrics metrics;
  std::string error_message;
};

struct DetectorRuntime {
  DetectorRuntimeKey key;
  SessionRun runtime;
};

struct WorkerContext {
  int index = 0;
  std::vector<DetectorRuntime> detectors;
};

void print_interval_profile(StreamRuntime& stream);

double now_steady_s() {
  return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

std::int64_t now_unix_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::vector<std::string> load_class_labels(const fs::path& path = kDefaultLabelsPath) {
  std::vector<std::string> labels;
  std::ifstream input(path);
  if (!input.good()) {
    return labels;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

std::string hex_prefix(const std::vector<std::uint8_t>& bytes, std::size_t max_bytes) {
  static const char kHex[] = "0123456789ABCDEF";
  const std::size_t n = std::min(bytes.size(), max_bytes);
  std::string out;
  out.reserve(n * 3);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t b = bytes[i];
    out.push_back(kHex[(b >> 4U) & 0x0F]);
    out.push_back(kHex[b & 0x0F]);
    if (i + 1 < n) {
      out.push_back(' ');
    }
  }
  return out;
}

AnnexBProbe probe_annexb(const simaai::neat::Sample& sample) {
  AnnexBProbe out;
  if (!sample.tensor.has_value()) {
    return out;
  }
  const auto bytes = sample.tensor->copy_payload_bytes();
  if (bytes.empty()) {
    return out;
  }
  out.prefix_hex = hex_prefix(bytes, 12);
  const std::size_t max_scan = std::min<std::size_t>(bytes.size(), 256);
  for (std::size_t i = 0; i + 2 < max_scan; ++i) {
    if (bytes[i] != 0 || bytes[i + 1] != 0) {
      continue;
    }
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

H264NalSummary scan_h264_annexb_bytes(const std::vector<std::uint8_t>& bytes) {
  H264NalSummary out;
  if (bytes.size() < 4) {
    return out;
  }

  struct StartCode {
    std::size_t offset = 0;
    std::size_t len = 0;
  };

  std::vector<StartCode> starts;
  starts.reserve(16);
  for (std::size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] != 0 || bytes[i + 1] != 0) {
      continue;
    }
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
    const std::size_t nal_start = sc.offset + sc.len;
    if (nal_start >= bytes.size()) {
      continue;
    }
    const std::uint8_t nal_type = bytes[nal_start] & 0x1F;
    out.nal_count += 1;
    if (nal_type == 7) {
      out.has_sps = true;
    } else if (nal_type == 8) {
      out.has_pps = true;
    } else if (nal_type == 5) {
      out.has_idr = true;
    } else if (nal_type == 9) {
      out.has_aud = true;
    }
  }
  return out;
}

H264NalSummary scan_h264_annexb_sample(const simaai::neat::Sample& sample) {
  if (!sample.tensor.has_value()) {
    return H264NalSummary{};
  }
  return scan_h264_annexb_bytes(sample.tensor->copy_payload_bytes());
}

bool is_headers_and_idr(const simaai::neat::Sample& sample) {
  const H264NalSummary nals = scan_h264_annexb_sample(sample);
  return nals.has_sps && nals.has_pps && nals.has_idr;
}

std::string h264_nal_summary_string(const H264NalSummary& summary) {
  std::string out;
  out.reserve(64);
  out += "nals=" + std::to_string(summary.nal_count);
  out += " sps=" + std::string(summary.has_sps ? "1" : "0");
  out += " pps=" + std::string(summary.has_pps ? "1" : "0");
  out += " idr=" + std::string(summary.has_idr ? "1" : "0");
  out += " aud=" + std::string(summary.has_aud ? "1" : "0");
  return out;
}

DetectorRuntimeKey detector_runtime_key(ModelFamily family, const RtspProbe& probe) {
  return DetectorRuntimeKey{family, probe.width, probe.height};
}

bool same_detector_runtime_key(const DetectorRuntimeKey& lhs, const DetectorRuntimeKey& rhs) {
  return lhs.family == rhs.family && lhs.width == rhs.width && lhs.height == rhs.height;
}

std::vector<DetectorRuntimeKey> collect_detector_runtime_keys_impl(
    const std::vector<StreamProbeSpec>& streams) {
  std::vector<DetectorRuntimeKey> keys;
  for (const auto& stream : streams) {
    const auto key = detector_runtime_key(stream.family, stream.probe);
    const bool seen =
        std::any_of(keys.begin(), keys.end(), [&](const auto& existing) {
          return same_detector_runtime_key(existing, key);
        });
    if (!seen) {
      keys.push_back(key);
    }
  }
  return keys;
}

StreamRuntime create_stream_runtime(int index, const std::string& url, const AppConfig& cfg,
                                    ModelFamily family,
                                    const std::vector<std::string>& class_labels) {
  const RtspProbe probe = probe_rtsp(cfg, url);
  std::optional<SessionRun> video_run;
  if (cfg.video_enabled && cfg.video_mode == VideoMode::Annotated) {
    video_run = build_optiview_video_run(cfg, probe, index);
  }
  const bool json_enabled = json_output_enabled(cfg);

  StreamRuntime runtime;
  runtime.index = index;
  runtime.url = url;
  runtime.family = family;
  runtime.probe = probe;
  runtime.source = build_source_run(cfg, url, probe);
  runtime.video = std::move(video_run);
  runtime.encoded_for_decode =
      std::make_shared<BoundedQueue<EncodedPacket>>(static_cast<std::size_t>(20));
  runtime.video_enabled = cfg.video_enabled;
  runtime.json_sender = json_enabled
                            ? std::optional<sima_examples::OptiViewSender>(
                                  build_optiview_json_output(cfg, index))
                            : std::nullopt;
  runtime.json_enabled = json_enabled;
  runtime.class_labels = class_labels;
  runtime.clean_video_bootstrapped = std::make_shared<std::atomic<bool>>(false);
  return runtime;
}

WorkerContext build_worker_context(int worker_index, const AppConfig& cfg,
                                   const std::vector<DetectorRuntimeKey>& detector_keys) {
  WorkerContext context;
  context.index = worker_index;
  for (const auto& key : detector_keys) {
    const RtspProbe probe{key.width, key.height, 0};
    context.detectors.push_back(DetectorRuntime{key, build_detection_run(cfg, key.family, probe)});
  }
  return context;
}

std::vector<WorkerContext> build_worker_contexts(const AppConfig& cfg, int worker_count,
                                                 const std::vector<DetectorRuntimeKey>& detector_keys) {
  std::vector<WorkerContext> contexts;
  contexts.reserve(static_cast<std::size_t>(std::max(worker_count, 0)));
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    contexts.push_back(build_worker_context(worker_index, cfg, detector_keys));
  }
  return contexts;
}

void close_stream_runtime(StreamRuntime& stream) {
  if (stream.encoded_for_decode != nullptr) {
    stream.encoded_for_decode->close();
  }
  if (stream.video.has_value()) {
    try {
      stream.video->run.close();
    } catch (...) {
    }
  }
  if (stream.decode.has_value()) {
    try {
      stream.decode->run.close();
    } catch (...) {
    }
  }
  try {
    stream.source.run.close();
  } catch (...) {
  }
}

void close_worker_context(WorkerContext& context) {
  for (auto& detector : context.detectors) {
    try {
      detector.runtime.run.close();
    } catch (...) {
    }
  }
}

DetectorRuntime& find_detector_runtime(WorkerContext& context, ModelFamily family,
                                       const RtspProbe& probe) {
  const auto key = detector_runtime_key(family, probe);
  for (auto& detector : context.detectors) {
    if (same_detector_runtime_key(detector.key, key)) {
      return detector;
    }
  }
  throw std::runtime_error("missing detector runtime for stream geometry");
}

bool strict_clean_video_sync_enabled(const StreamRuntime& stream, const AppConfig& cfg) {
  return stream.video_enabled && cfg.video_mode == VideoMode::Clean;
}

std::string canonical_stream_id(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

void canonicalize_sample_identity(simaai::neat::Sample& sample, int stream_index,
                                  std::int64_t frame_index) {
  if (frame_index < 0) {
    return;
  }
  sample.frame_id = frame_index;
  sample.input_seq = frame_index;
  sample.orig_input_seq = frame_index;
  sample.stream_id = canonical_stream_id(stream_index);
}

std::int64_t resolve_sample_frame_index(const simaai::neat::Sample& sample) {
  if (sample.frame_id >= 0) {
    return sample.frame_id;
  }
  if (sample.orig_input_seq >= 0) {
    return sample.orig_input_seq;
  }
  if (sample.input_seq >= 0) {
    return sample.input_seq;
  }
  return -1;
}

int narrow_frame_index_for_api(std::int64_t frame_index, int fallback = 0) {
  if (frame_index < 0 ||
      frame_index > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(frame_index);
}

FrameSyncMeta make_frame_sync_meta(const EncodedPacket& packet, const RtspProbe& probe,
                                   bool retain_matched_encoded) {
  FrameSyncMeta meta;
  meta.frame_index = packet.frame_index;
  meta.source_time_s = packet.source_pull_time_s;
  meta.video_push_time_s = packet.video_push_time_s;
  if (retain_matched_encoded) {
    meta.matched_encoded =
        deep_copy_encoded_sample(packet.encoded, probe.encoded_caps_appsrc);
  }
  return meta;
}

bool is_bootstrap_worthy_matched_sample(const std::optional<FrameSyncMeta>& matched_meta) {
  return matched_meta.has_value() && matched_meta->matched_encoded.has_value() &&
         is_headers_and_idr(*matched_meta->matched_encoded);
}

bool clean_video_bootstrapped(const StreamRuntime& stream) {
  return stream.clean_video_bootstrapped != nullptr &&
         stream.clean_video_bootstrapped->load(std::memory_order_acquire);
}

void source_thread(StreamRuntime& stream, const AppConfig& cfg, std::atomic<bool>& stop_event,
                   Event* startup_ready) {
  std::int64_t frame_index = 0;
  int empty_pulls = 0;

  try {
    while (!stop_event.load()) {
      if (cfg.frames > 0 && frame_index >= cfg.frames) {
        break;
      }

      const double pull_t0 = now_steady_s();
      const int pull_timeout_ms =
          frame_index == 0 ? kSourceStartupPullTimeoutMs : kSourcePullTimeoutMs;
      const auto sample = stream.source.run.pull(pull_timeout_ms);
      const double pull_elapsed = now_steady_s() - pull_t0;
      if (!sample.has_value()) {
        ++empty_pulls;
        if (cfg.frames > 0 && empty_pulls >= 20) {
          throw std::runtime_error("stream " + std::to_string(stream.index) +
                                   " timed out waiting for RTSP frames");
        }
        continue;
      }

      empty_pulls = 0;

      simaai::neat::Sample encoded = *sample;
      canonicalize_sample_identity(encoded, stream.index, frame_index);
      if (!stream.probe.encoded_caps_appsrc.empty()) {
        encoded.caps_string = stream.probe.encoded_caps_appsrc;
      }

      EncodedPacket packet;
      packet.encoded = deep_copy_encoded_sample(encoded, stream.probe.encoded_caps_appsrc);
      packet.frame_index = frame_index;
      packet.source_pull_time_s = pull_elapsed;
      if (!stream.encoded_for_decode->push_drop_oldest(std::move(packet))) {
        throw std::runtime_error("stream " + std::to_string(stream.index) +
                                 " decode queue closed unexpectedly");
      }
      ++frame_index;
    }
  } catch (const std::exception& ex) {
    stream.error_message = ex.what();
    stop_event.store(true);
    if (startup_ready != nullptr) {
      startup_ready->set();
    }
  }

  if (stream.encoded_for_decode != nullptr) {
    stream.encoded_for_decode->close();
  }
}

void decode_thread(StreamRuntime& stream, const AppConfig& cfg,
                   LatestFrameMailbox<FramePacket>& mailbox, ReadyStreamQueue& ready_queue,
                   std::atomic<bool>& stop_event, Event* startup_ready) {
  constexpr std::int64_t kHeaderWaitMs = 10'000;
  // H.264 startup may need substantially more than a handful of access units
  // before the first decoded frame is emitted. Keep warmup permissive, then
  // tighten once decode output has started flowing.
  constexpr int kDecoderWarmupPacketsInFlight = 64;
  constexpr int kDecoderSteadyPacketsInFlight = 8;

  const double emit_period_s = producer_emit_period_s(cfg, stream.probe);
  const bool retain_matched_encoded = strict_clean_video_sync_enabled(stream, cfg);
  std::optional<double> next_allowed_emit_s;
  std::optional<EncodedPacket> deferred_packet;
  PendingFrameStore<FrameSyncMeta> pending_frames(kPendingDecodeSyncCapacity);
  std::int64_t header_wait_started_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(SteadyClock::now().time_since_epoch())
          .count();
  int decoder_packets_inflight = 0;
  bool decoder_input_closed = false;
  bool saw_first_decoded_output = false;
  bool need_headers = true;
  bool startup_signaled = false;

  try {
    while (!stop_event.load()) {
      const auto max_decoder_packets_in_flight = [&]() {
        return saw_first_decoded_output ? kDecoderSteadyPacketsInFlight
                                        : kDecoderWarmupPacketsInFlight;
      };
      auto admit_packet_to_decode = [&](const EncodedPacket& candidate) {
        if (!stream.decode.has_value()) {
          stream.decode = build_decode_run(cfg, stream.probe, candidate.encoded);
        }
        if (!stream.decode->run.push(candidate.encoded)) {
          throw std::runtime_error("stream " + std::to_string(stream.index) +
                                   " decoder input push failed");
        }

        pending_frames.put(candidate.frame_index,
                           make_frame_sync_meta(candidate, stream.probe, retain_matched_encoded));
        decoder_packets_inflight += 1;
      };

      EncodedPacket packet;
      const bool have_packet = stream.encoded_for_decode->pop_wait(packet, 50);
      const bool source_drained =
          !have_packet && stream.encoded_for_decode->drained() && !deferred_packet.has_value();
      if (have_packet) {
        if (need_headers && !is_headers_and_idr(packet.encoded)) {
          const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  SteadyClock::now().time_since_epoch())
                                  .count();
          if ((now_ms - header_wait_started_ms) > kHeaderWaitMs) {
            const AnnexBProbe prefix = probe_annexb(packet.encoded);
            const H264NalSummary summary = scan_h264_annexb_sample(packet.encoded);
            throw std::runtime_error("stream " + std::to_string(stream.index) +
                                     " timed out waiting for SPS/PPS/IDR: " +
                                     h264_nal_summary_string(summary) +
                                     " prefix=" + prefix.prefix_hex);
          }
        } else if (need_headers) {
          need_headers = false;
          header_wait_started_ms = 0;
        }

        if (!need_headers) {
          if (decoder_packets_inflight >= max_decoder_packets_in_flight()) {
            deferred_packet = std::move(packet);
          } else {
            admit_packet_to_decode(packet);
          }
        }
      }

      if (source_drained && stream.decode.has_value() && !decoder_input_closed) {
        stream.decode->run.close_input();
        decoder_input_closed = true;
      }

      if (!stream.decode.has_value()) {
        if (source_drained) {
          break;
        }
        continue;
      }

      bool decoder_fully_drained = false;
      for (int drain = 0; drain < 8 && !stop_event.load(); ++drain) {
        simaai::neat::Sample decoded;
        simaai::neat::PullError pull_error;
        const auto status = stream.decode->run.pull(decode_pull_timeout_ms(), decoded, &pull_error);
        if (status == simaai::neat::PullStatus::Timeout) {
          if (decoder_input_closed && source_drained && decoder_packets_inflight <= 0) {
            decoder_fully_drained = true;
          }
          break;
        }
        if (status == simaai::neat::PullStatus::Closed) {
          if (decoder_input_closed) {
            decoder_packets_inflight = 0;
            decoder_fully_drained = true;
            break;
          }
          throw std::runtime_error("stream " + std::to_string(stream.index) + " decoder closed");
        }
        if (status == simaai::neat::PullStatus::Error) {
          throw std::runtime_error("stream " + std::to_string(stream.index) + " decoder error: " +
                                   (pull_error.message.empty() ? "unknown"
                                                               : pull_error.message));
        }
        saw_first_decoded_output = true;

        const std::int64_t resolved_frame_index = resolve_sample_frame_index(decoded);
        if (resolved_frame_index >= 0) {
          canonicalize_sample_identity(decoded, stream.index, resolved_frame_index);
        }

        const double decode_completed_at_s = now_steady_s();
        bool should_emit = true;
        if (emit_period_s > 0.0) {
          if (!next_allowed_emit_s.has_value()) {
            const int num_streams = static_cast<int>(cfg.rtsp_urls.size());
            const double phase =
                (num_streams > 1) ? emit_period_s * stream.index / num_streams : 0.0;
            next_allowed_emit_s =
                std::ceil(decode_completed_at_s / emit_period_s) * emit_period_s + phase;
          }
          if (decode_completed_at_s < *next_allowed_emit_s) {
            should_emit = false;
          } else {
            while (*next_allowed_emit_s <= decode_completed_at_s) {
              *next_allowed_emit_s += emit_period_s;
            }
          }
        }

        // Standalone decoder runs do not reliably preserve frame_id/input_seq on
        // every output sample at multistream load. Keep exact lookup when the
        // decoder reports a frame id, but fall back to per-stream decode order
        // so each decoded frame is still paired with the next admitted source
        // frame from the same stream.
        std::optional<FrameSyncMeta> matched_meta =
            take_pending_frame_match_or_oldest(pending_frames, resolved_frame_index);
        if (decoder_packets_inflight > 0) {
          decoder_packets_inflight -= 1;
        }
        const std::int64_t canonical_frame_index =
            matched_meta.has_value() ? matched_meta->frame_index : resolved_frame_index;
        if (canonical_frame_index >= 0 && canonical_frame_index != resolved_frame_index) {
          canonicalize_sample_identity(decoded, stream.index, canonical_frame_index);
        }
        const bool force_bootstrap_emit =
            retain_matched_encoded && !clean_video_bootstrapped(stream) &&
            is_bootstrap_worthy_matched_sample(matched_meta);
        if (!startup_signaled) {
          should_emit = true;
        }
        if (force_bootstrap_emit) {
          should_emit = true;
        }
        if (!should_emit) {
          if (deferred_packet.has_value() &&
              decoder_packets_inflight < max_decoder_packets_in_flight()) {
            EncodedPacket deferred = std::move(*deferred_packet);
            deferred_packet.reset();
            admit_packet_to_decode(deferred);
          }
          continue;
        }

        FramePacket frame;
        frame.decoded = std::move(decoded);
        frame.frame_index = canonical_frame_index;
        if (matched_meta.has_value()) {
          frame.source_time_s = matched_meta->source_time_s;
          frame.video_push_time_s = matched_meta->video_push_time_s;
          frame.matched_encoded = std::move(matched_meta->matched_encoded);
        }
        stream.metrics.mailbox_drops += mailbox.push(std::move(frame), ready_queue);
        if (!startup_signaled && startup_ready != nullptr) {
          startup_ready->set();
          startup_signaled = true;
        }
        if (deferred_packet.has_value() &&
            decoder_packets_inflight < max_decoder_packets_in_flight()) {
          EncodedPacket deferred = std::move(*deferred_packet);
          deferred_packet.reset();
          admit_packet_to_decode(deferred);
        }
      }

      if (decoder_fully_drained || (source_drained && decoder_input_closed &&
                                    decoder_packets_inflight <= 0)) {
        break;
      }
    }
  } catch (const std::exception& ex) {
    stream.error_message = ex.what();
    stop_event.store(true);
    if (startup_ready != nullptr) {
      startup_ready->set();
    }
  }

  mailbox.close();
}

cv::Mat render_frame(const StreamRuntime& stream, const AppConfig& cfg, const cv::Mat& frame,
                     const std::vector<Detection>& detections) {
  if (cfg.video_mode == VideoMode::Clean) {
    return frame;
  }
  return draw_detection_boxes(frame.clone(), detections, stream.class_labels);
}

void process_frame(WorkerContext& worker_context, StreamRuntime& stream, const AppConfig& cfg,
                   const FramePacket& packet) {
  const double loop_start = now_steady_s();
  if (!stream.metrics.wall_started_at_s.has_value()) {
    stream.metrics.wall_started_at_s = loop_start;
  }
  if (!stream.metrics.interval_wall_started_at_s.has_value()) {
    stream.metrics.interval_wall_started_at_s = loop_start;
  }

  auto& detector = find_detector_runtime(worker_context, stream.family, stream.probe);
  const double detect_t0 = now_steady_s();
  const simaai::neat::Sample det_sample =
      run_sample_input_once(detector.runtime.run, packet.decoded, 50000);
  const double detect_elapsed = now_steady_s() - detect_t0;
  const double preproc_elapsed = 0.0;

  const auto detections =
      detections_from_detector_sample(stream.family, det_sample, stream.probe.width, stream.probe.height);
  const bool strict_clean_video_sync = strict_clean_video_sync_enabled(stream, cfg);
  const bool needs_output_frame =
      (stream.video_enabled && cfg.video_mode == VideoMode::Annotated) ||
      (cfg.output_dir.has_value() && cfg.save_every > 0);

  const double publish_t0 = now_steady_s();
  double video_elapsed = packet.video_push_time_s;
  cv::Mat frame_out;
  if (needs_output_frame) {
    frame_out = render_frame(stream, cfg, tensor_rgb_from_sample(packet.decoded), detections);
  }
  if (stream.video_enabled && cfg.video_mode == VideoMode::Annotated) {
    const double video_t0 = now_steady_s();
    if (!stream.video.has_value() || !stream.video->run.push(frame_out)) {
      throw std::runtime_error("stream " + std::to_string(stream.index) + " OptiView video push failed");
    }
    video_elapsed += now_steady_s() - video_t0;
  }
  const double publish_wall_time_s =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

  double json_elapsed = 0.0;
  if (stream.json_enabled) {
    const auto payload =
        build_optiview_detection_payload(detections, stream.probe.width, stream.probe.height,
                                         stream.class_labels);
    const double json_t0 = now_steady_s();
    if (!stream.json_sender->send_detection(
            optiview_timestamp_ms(publish_wall_time_s, cfg.optiview_json_offset_ms),
            optiview_frame_id(det_sample, narrow_frame_index_for_api(packet.frame_index)),
            payload.objects, payload.labels)) {
      throw std::runtime_error("stream " + std::to_string(stream.index) + " OptiView JSON send failed");
    }
    json_elapsed = now_steady_s() - json_t0;
  }
  if (strict_clean_video_sync) {
    if (!packet.matched_encoded.has_value()) {
      stream.metrics.sync_exact_miss += 1;
      stream.metrics.interval_sync_exact_miss += 1;
    } else {
      stream.metrics.sync_exact_ok += 1;
      stream.metrics.interval_sync_exact_ok += 1;

      const double video_t0 = now_steady_s();
      bool release_ok = false;
      if (!stream.video.has_value()) {
        if (!is_headers_and_idr(*packet.matched_encoded)) {
          release_ok = false;
        } else {
          try {
            stream.video = build_optiview_video_run(cfg, stream.probe, stream.index,
                                                    &packet.matched_encoded.value());
            if (stream.clean_video_bootstrapped != nullptr) {
              stream.clean_video_bootstrapped->store(true, std::memory_order_release);
            }
            release_ok = true;
          } catch (const std::exception& ex) {
            stream.metrics.sync_release_fail += 1;
            stream.metrics.interval_sync_release_fail += 1;
            throw std::runtime_error("stream " + std::to_string(stream.index) +
                                     " OptiView clean video bootstrap failed: " + ex.what());
          }
        }
      } else {
        try {
          release_ok = stream.video->run.push(*packet.matched_encoded);
        } catch (const std::exception& ex) {
          stream.metrics.sync_release_fail += 1;
          stream.metrics.interval_sync_release_fail += 1;
          throw std::runtime_error("stream " + std::to_string(stream.index) +
                                   " OptiView clean video push failed: " + ex.what());
        }
      }
      video_elapsed += now_steady_s() - video_t0;
      if (!release_ok && stream.video.has_value()) {
        stream.metrics.sync_release_fail += 1;
        stream.metrics.interval_sync_release_fail += 1;
        throw std::runtime_error("stream " + std::to_string(stream.index) +
                                 " OptiView clean video push failed");
      }
      if (release_ok) {
        stream.metrics.sync_release_ok += 1;
        stream.metrics.interval_sync_release_ok += 1;
      }
    }
  }

  if (!frame_out.empty() &&
      save_debug_frame(cfg.output_dir, stream.index,
                       narrow_frame_index_for_api(packet.frame_index), frame_out, cfg.save_every)) {
    stream.metrics.saved += 1;
  }
  const double publish_elapsed = (now_steady_s() - publish_t0) + packet.video_push_time_s;

  stream.metrics.pulled += 1;
  stream.metrics.processed += 1;
  stream.metrics.detections += static_cast<int>(detections.size());
  stream.metrics.source_time_s += packet.source_time_s;
  stream.metrics.interval_source_s += packet.source_time_s;
  stream.metrics.preproc_time_s += preproc_elapsed;
  stream.metrics.interval_preproc_s += preproc_elapsed;
  stream.metrics.detect_time_s += detect_elapsed;
  stream.metrics.interval_detect_s += detect_elapsed;
  stream.metrics.video_push_time_s += video_elapsed;
  stream.metrics.interval_video_s += video_elapsed;
  stream.metrics.json_time_s += json_elapsed;
  stream.metrics.interval_json_s += json_elapsed;
  stream.metrics.publish_time_s += publish_elapsed;
  stream.metrics.interval_publish_s += publish_elapsed;
  stream.metrics.total_loop_time_s +=
      packet.source_time_s + preproc_elapsed + detect_elapsed + publish_elapsed;
  stream.metrics.interval_loop_s +=
      packet.source_time_s + preproc_elapsed + detect_elapsed + publish_elapsed;
  stream.metrics.interval_frames += 1;
  stream.metrics.wall_last_processed_at_s = now_steady_s();
}

bool all_mailboxes_drained(
    const std::vector<std::shared_ptr<LatestFrameMailbox<FramePacket>>>& mailboxes) {
  for (const auto& mailbox : mailboxes) {
    if (!mailbox->drained()) {
      return false;
    }
  }
  return true;
}

void detector_worker(WorkerContext& worker_context, std::vector<StreamRuntime>& streams,
                     const AppConfig& cfg,
                     const std::vector<std::shared_ptr<LatestFrameMailbox<FramePacket>>>& mailboxes,
                     ReadyStreamQueue& ready_queue, std::atomic<bool>& stop_event) {
  try {
    while (true) {
      if (stop_event.load() && all_mailboxes_drained(mailboxes)) {
        return;
      }

      int stream_index = -1;
      if (!ready_queue.pop_wait(stream_index, 100)) {
        if (all_mailboxes_drained(mailboxes)) {
          return;
        }
        continue;
      }

      StreamRuntime& stream = streams[static_cast<std::size_t>(stream_index)];
      auto& mailbox = *mailboxes[static_cast<std::size_t>(stream_index)];
      FramePacket packet;
      if (!mailbox.take_for_processing(packet)) {
        if (all_mailboxes_drained(mailboxes)) {
          return;
        }
        continue;
      }

      try {
        process_frame(worker_context, stream, cfg, packet);
      } catch (const std::exception& ex) {
        stream.error_message = ex.what();
        stop_event.store(true);
      }

      mailbox.complete(ready_queue);
      if (cfg.profile && stream.metrics.processed > 0 &&
          (stream.metrics.processed % kDefaultProfileIntervalFrames) == 0) {
        print_interval_profile(stream);
      }
    }
  } catch (const std::exception& ex) {
    for (auto& stream : streams) {
      if (stream.error_message.empty()) {
        stream.error_message = ex.what();
      }
    }
    stop_event.store(true);
  }
}

double wall_clock_fps(int frame_count, const std::optional<double>& started_at_s,
                      const std::optional<double>& ended_at_s) {
  if (frame_count <= 0 || !started_at_s.has_value() || !ended_at_s.has_value()) {
    return 0.0;
  }
  const double elapsed = *ended_at_s - *started_at_s;
  if (elapsed <= 0.0) {
    return 0.0;
  }
  return frame_count / elapsed;
}

void print_interval_profile(StreamRuntime& stream) {
  const int n = stream.metrics.interval_frames;
  if (n <= 0) {
    return;
  }
  const double src = stream.metrics.interval_source_s * 1000.0 / n;
  const double pre = stream.metrics.interval_preproc_s * 1000.0 / n;
  const double det = stream.metrics.interval_detect_s * 1000.0 / n;
  const double vid = stream.metrics.interval_video_s * 1000.0 / n;
  const double jsn = stream.metrics.interval_json_s * 1000.0 / n;
  const double pub = stream.metrics.interval_publish_s * 1000.0 / n;
  const double loop = stream.metrics.interval_loop_s * 1000.0 / n;
  const double fps =
      wall_clock_fps(n, stream.metrics.interval_wall_started_at_s, stream.metrics.wall_last_processed_at_s);

  std::cout << "  [stream " << stream.index << "] frames "
            << (stream.metrics.processed - n) << "-" << (stream.metrics.processed - 1)
            << " | source=" << src << "ms preproc=" << pre << "ms detect=" << det
            << "ms video=" << vid << "ms json=" << jsn << "ms publish=" << pub
            << "ms loop=" << loop << "ms throughput_fps=" << fps
            << " mailbox_drops=" << stream.metrics.mailbox_drops
            << " sync=" << stream.metrics.interval_sync_exact_ok << "/"
            << stream.metrics.interval_sync_exact_miss
            << " release=" << stream.metrics.interval_sync_release_ok << "/"
            << stream.metrics.interval_sync_release_fail << "\n";

  stream.metrics.interval_source_s = 0.0;
  stream.metrics.interval_preproc_s = 0.0;
  stream.metrics.interval_detect_s = 0.0;
  stream.metrics.interval_video_s = 0.0;
  stream.metrics.interval_json_s = 0.0;
  stream.metrics.interval_publish_s = 0.0;
  stream.metrics.interval_loop_s = 0.0;
  stream.metrics.interval_frames = 0;
  stream.metrics.interval_sync_exact_ok = 0;
  stream.metrics.interval_sync_exact_miss = 0;
  stream.metrics.interval_sync_release_ok = 0;
  stream.metrics.interval_sync_release_fail = 0;
  stream.metrics.interval_wall_started_at_s = stream.metrics.wall_last_processed_at_s;
}

void print_profile_summary(const std::vector<StreamRuntime>& streams) {
  std::cout << "\nProfile summary (averages per frame):\n";
  for (const auto& stream : streams) {
    const int n = std::max(stream.metrics.processed, 1);
    const double src = stream.metrics.source_time_s * 1000.0 / n;
    const double pre = stream.metrics.preproc_time_s * 1000.0 / n;
    const double det = stream.metrics.detect_time_s * 1000.0 / n;
    const double vid = stream.metrics.video_push_time_s * 1000.0 / n;
    const double jsn = stream.metrics.json_time_s * 1000.0 / n;
    const double pub = stream.metrics.publish_time_s * 1000.0 / n;
    const double loop = stream.metrics.total_loop_time_s * 1000.0 / n;
    const double fps =
        wall_clock_fps(stream.metrics.processed, stream.metrics.wall_started_at_s,
                       stream.metrics.wall_last_processed_at_s);
    std::cout << "  [stream " << stream.index << "] " << stream.metrics.processed
              << " frames | source=" << src << "ms preproc=" << pre << "ms detect=" << det
              << "ms video=" << vid << "ms json=" << jsn << "ms publish=" << pub
              << "ms loop=" << loop << "ms throughput_fps=" << fps
              << " mailbox_drops=" << stream.metrics.mailbox_drops
              << " detections=" << stream.metrics.detections
              << " sync=" << stream.metrics.sync_exact_ok << "/" << stream.metrics.sync_exact_miss
              << " release=" << stream.metrics.sync_release_ok << "/"
              << stream.metrics.sync_release_fail << "\n";
  }
}

} // namespace

std::vector<DetectorRuntimeKey> collect_detector_runtime_keys(
    const std::vector<StreamProbeSpec>& streams) {
  return collect_detector_runtime_keys_impl(streams);
}

int run_app(const AppConfig& cfg, ModelFamily family) {
  if (cfg.output_dir.has_value()) {
    fs::create_directories(*cfg.output_dir);
  }

  const auto class_labels = load_class_labels();

  std::vector<StreamRuntime> streams;
  try {
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      streams.push_back(create_stream_runtime(static_cast<int>(index), cfg.rtsp_urls[index], cfg,
                                              family, class_labels));
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error: failed to set up stream runtimes: " << ex.what() << "\n";
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    return 4;
  }

  std::vector<StreamProbeSpec> stream_specs;
  stream_specs.reserve(streams.size());
  for (const auto& stream : streams) {
    stream_specs.push_back(StreamProbeSpec{stream.family, stream.probe});
  }

  std::vector<WorkerContext> worker_contexts;
  try {
    worker_contexts =
        build_worker_contexts(cfg, cfg.worker_count, collect_detector_runtime_keys(stream_specs));
  } catch (const std::exception& ex) {
    std::cerr << "Error: failed to build detector workers: " << ex.what() << "\n";
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    return 4;
  }

  for (const auto& stream : streams) {
    std::cout << "[stream " << stream.index << "] " << stream.probe.width << "x" << stream.probe.height
              << " @" << effective_writer_fps(cfg, stream.probe) << "fps " << stream.url
              << " -> optiview://" << cfg.optiview_host
              << " video=";
    if (stream.video_enabled) {
      std::cout << optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index);
    } else {
      std::cout << "disabled";
    }
    std::cout << " json=";
    if (stream.json_enabled) {
      std::cout << optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index);
    } else {
      std::cout << "disabled";
    }
    std::cout << "\n";
  }

  std::atomic<bool> stop_event{false};
  ReadyStreamQueue ready_queue;
  std::vector<std::shared_ptr<LatestFrameMailbox<FramePacket>>> mailboxes;
  std::vector<std::thread> source_threads;
  std::vector<std::thread> decode_threads;
  std::vector<std::thread> worker_threads;
  std::vector<std::unique_ptr<Event>> startup_ready_events;

  for (auto& stream : streams) {
    mailboxes.push_back(std::make_shared<LatestFrameMailbox<FramePacket>>(stream.index, cfg.mailbox_depth));
    startup_ready_events.push_back(std::make_unique<Event>());
  }

  for (auto& worker_context : worker_contexts) {
    worker_threads.emplace_back(
        [&worker_context, &streams, &cfg, &mailboxes, &ready_queue, &stop_event]() {
          detector_worker(worker_context, streams, cfg, mailboxes, ready_queue, stop_event);
        });
  }

  try {
    const auto startup_timeout = std::chrono::milliseconds(kSourceStartupPullTimeoutMs);
    for (std::size_t index = 0; index < streams.size(); ++index) {
      if (stop_event.load()) {
        break;
      }
      decode_threads.emplace_back(
          [&streams, &cfg, &mailboxes, &ready_queue, &stop_event,
           ready = startup_ready_events[index].get(), index]() {
            decode_thread(streams[index], cfg, *mailboxes[index], ready_queue, stop_event, ready);
          });
      source_threads.emplace_back([&streams, &cfg, &stop_event,
                                   ready = startup_ready_events[index].get(), index]() {
        source_thread(streams[index], cfg, stop_event, ready);
      });
      if (!startup_ready_events[index]->wait_for(startup_timeout)) {
        streams[index].error_message =
            "startup timeout waiting for first decoded frame";
        stop_event.store(true);
        break;
      }
      if (stop_event.load()) {
        break;
      }
      if (index + 1 < streams.size() && kSourceStartupStaggerS > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(kSourceStartupStaggerS));
      }
    }
    for (auto& thread : worker_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : source_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : decode_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  } catch (...) {
    stop_event.store(true);
    ready_queue.close();
    for (auto& mailbox : mailboxes) {
      mailbox->close();
    }
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    for (auto& worker_context : worker_contexts) {
      close_worker_context(worker_context);
    }
    for (auto& thread : worker_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : source_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : decode_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    throw;
  }

  ready_queue.close();
  for (auto& stream : streams) {
    close_stream_runtime(stream);
  }
  for (auto& worker_context : worker_contexts) {
    close_worker_context(worker_context);
  }

  bool failed = false;
  for (const auto& stream : streams) {
    if (!stream.error_message.empty()) {
      failed = true;
      std::cerr << "[stream " << stream.index << "] error: " << stream.error_message << "\n";
    }
  }
  if (failed) {
    return 5;
  }
  if (cfg.profile) {
    print_profile_summary(streams);
  }
  return 0;
}

} // namespace multistream_yolox_yolov8_optiview
