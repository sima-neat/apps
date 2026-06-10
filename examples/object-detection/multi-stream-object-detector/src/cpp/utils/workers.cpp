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

namespace multi_stream_object_detector {
namespace {

constexpr int kDefaultProfileIntervalFrames = 200;
const fs::path kDefaultLabelsPath =
    fs::path(MULTISTREAM_OBJECT_DETECTION_INSIGHT_SOURCE_DIR) / ".." / "common" / "coco_label.txt";

using SteadyClock = std::chrono::steady_clock;

struct StreamMetrics {
  int processed = 0;
  int detections = 0;
  int saved = 0;
  int mailbox_drops = 0;
  double source_time_s = 0.0;
  double preproc_time_s = 0.0;
  double detect_time_s = 0.0;
  double video_push_time_s = 0.0;
  double metadata_time_s = 0.0;
  double publish_time_s = 0.0;
  double total_loop_time_s = 0.0;
  std::optional<double> wall_started_at_s;
  std::optional<double> wall_last_processed_at_s;
  double interval_source_s = 0.0;
  double interval_preproc_s = 0.0;
  double interval_detect_s = 0.0;
  double interval_video_s = 0.0;
  double interval_metadata_s = 0.0;
  double interval_publish_s = 0.0;
  double interval_loop_s = 0.0;
  int interval_frames = 0;
  std::optional<double> interval_wall_started_at_s;
};

struct FramePacket {
  simaai::neat::Sample decoded;
  std::int64_t frame_index = -1;
  double source_time_s = 0.0;
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  ModelFamily family = ModelFamily::Auto;
  RtspProbe probe;
  GraphRun source;
  std::optional<GraphRun> video;
  bool video_enabled = true;
  std::optional<simaai::neat::MetadataSender> metadata_sender;
  bool metadata_enabled = true;
  std::vector<std::string> class_labels;
  StreamMetrics metrics;
  std::string error_message;

  bool saw_first_source_frame = false;
  bool first_mailbox_push_logged = false;
  std::int64_t next_source_frame_index = 0;
  std::optional<double> next_allowed_emit_s;
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

struct DetectorRuntime {
  DetectorRuntimeKey key;
  GraphRun graph_run;
};

struct WorkerContext {
  int index = 0;
  std::vector<DetectorRuntime> detectors;
};

std::mutex& startup_trace_mutex() {
  static std::mutex mu;
  return mu;
}

void emit_startup_trace(int stream_index, const std::string& message) {
  if (!startup_trace_enabled_from_env()) {
    return;
  }
  std::lock_guard<std::mutex> lock(startup_trace_mutex());
  std::cerr << "[startup trace stream " << stream_index << "] " << message << "\n";
}

double now_steady_s() {
  return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

const char* video_mode_name(VideoMode video_mode) {
  switch (video_mode) {
  case VideoMode::Clean:
    return "clean";
  case VideoMode::Annotated:
    return "annotated";
  }
  return "unknown";
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

DetectorRuntimeKey detector_runtime_key(ModelFamily family, const RtspProbe& probe) {
  return DetectorRuntimeKey{family, probe.width, probe.height};
}

bool same_detector_runtime_key(const DetectorRuntimeKey& lhs, const DetectorRuntimeKey& rhs) {
  return lhs.family == rhs.family && lhs.width == rhs.width && lhs.height == rhs.height;
}

std::vector<DetectorRuntimeKey>
collect_detector_runtime_keys_impl(const std::vector<StreamProbeSpec>& streams) {
  std::vector<DetectorRuntimeKey> keys;
  for (const auto& stream : streams) {
    const auto key = detector_runtime_key(stream.family, stream.probe);
    const bool seen = std::any_of(keys.begin(), keys.end(), [&](const auto& existing) {
      return same_detector_runtime_key(existing, key);
    });
    if (!seen) {
      keys.push_back(key);
    }
  }
  return keys;
}

void initialize_stream_runtime(StreamRuntime& runtime, int index, const std::string& url,
                               const AppConfig& cfg, ModelFamily family,
                               const std::vector<std::string>& class_labels) {
  runtime.index = index;
  runtime.url = url;
  runtime.family = family;
  runtime.probe = probe_rtsp(cfg, url);
  runtime.source = build_source_run(cfg, url, runtime.probe);
  runtime.video_enabled = cfg.video_enabled;
  runtime.metadata_enabled = metadata_output_enabled(cfg);
  runtime.class_labels = class_labels;
  if (runtime.metadata_enabled) {
    runtime.metadata_sender = build_insight_metadata_output(cfg, index);
  }
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

std::vector<WorkerContext>
build_worker_contexts(const AppConfig& cfg, int worker_count,
                      const std::vector<DetectorRuntimeKey>& detector_keys) {
  std::vector<WorkerContext> contexts;
  contexts.reserve(static_cast<std::size_t>(std::max(worker_count, 0)));
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    contexts.push_back(build_worker_context(worker_index, cfg, detector_keys));
  }
  return contexts;
}

void close_stream_runtime(StreamRuntime& stream) {
  if (stream.video.has_value()) {
    try {
      stream.video->run.close();
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
      detector.graph_run.run.close();
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

int narrow_frame_index_for_api(std::int64_t frame_index, int fallback = 0) {
  if (frame_index < 0 || frame_index > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(frame_index);
}

void producer_thread(StreamRuntime& stream, const AppConfig& cfg,
                     const std::shared_ptr<LatestFrameMailbox<FramePacket>>& mailbox,
                     ReadyStreamQueue& ready_queue, std::atomic<bool>& stop_event,
                     Event* startup_ready) {
  try {
    emit_startup_trace(stream.index, "source thread started");
    while (!stop_event.load()) {
      if (cfg.frames > 0 && stream.next_source_frame_index >= cfg.frames) {
        break;
      }

      const int pull_timeout_ms = stream.first_mailbox_push_logged
                                      ? source_pull_timeout_ms()
                                      : source_startup_pull_timeout_ms();
      const double pull_t0 = now_steady_s();
      const auto sample = stream.source.run.pull(pull_timeout_ms);
      const double pull_elapsed = now_steady_s() - pull_t0;
      if (!sample.has_value()) {
        const std::string last_error = stream.source.run.last_error();
        bool source_running = false;
        try {
          source_running = stream.source.run.running();
        } catch (...) {
          source_running = false;
        }
        if (!last_error.empty() || !source_running) {
          emit_startup_trace(stream.index,
                             "source pull returned empty last_error=" +
                                 (last_error.empty() ? std::string("<empty>") : last_error));
          throw std::runtime_error(last_error.empty() ? std::string("source run stopped")
                                                      : last_error);
        }
        continue;
      }

      if (!stream.saw_first_source_frame) {
        emit_startup_trace(stream.index, "first decoded frame pulled");
        stream.saw_first_source_frame = true;
      }

      simaai::neat::Sample decoded = *sample;
      canonicalize_sample_identity(decoded, stream.index, stream.next_source_frame_index);

      const double source_completed_at_s = now_steady_s();
      bool should_emit = true;
      const double emit_period_s = producer_emit_period_s(cfg, stream.probe);
      if (emit_period_s > 0.0) {
        if (!stream.next_allowed_emit_s.has_value()) {
          const int num_streams = static_cast<int>(cfg.rtsp_urls.size());
          const double phase = num_streams > 1 ? emit_period_s * stream.index / num_streams : 0.0;
          stream.next_allowed_emit_s =
              std::ceil(source_completed_at_s / emit_period_s) * emit_period_s + phase;
        }
        if (source_completed_at_s < *stream.next_allowed_emit_s) {
          should_emit = false;
        } else {
          while (*stream.next_allowed_emit_s <= source_completed_at_s) {
            *stream.next_allowed_emit_s += emit_period_s;
          }
        }
      }

      if (!stream.first_mailbox_push_logged) {
        should_emit = true;
      }
      if (!should_emit) {
        ++stream.next_source_frame_index;
        continue;
      }

      FramePacket packet;
      packet.decoded = std::move(decoded);
      packet.frame_index = stream.next_source_frame_index;
      packet.source_time_s = pull_elapsed;
      stream.metrics.mailbox_drops += mailbox->push(std::move(packet), ready_queue);
      if (!stream.first_mailbox_push_logged) {
        emit_startup_trace(stream.index, "first mailbox push complete");
        stream.first_mailbox_push_logged = true;
        if (startup_ready != nullptr) {
          startup_ready->set();
        }
      }
      ++stream.next_source_frame_index;
    }
  } catch (const std::exception& ex) {
    stream.error_message = ex.what();
    stop_event.store(true);
    if (startup_ready != nullptr) {
      startup_ready->set();
    }
  }

  mailbox->close();
  try {
    stream.source.run.close();
  } catch (...) {
  }
}

bool any_stream_failed(const std::vector<StreamRuntime>& streams) {
  return std::any_of(streams.begin(), streams.end(),
                     [](const auto& stream) { return !stream.error_message.empty(); });
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
      run_sample_input_once(detector.graph_run.run, packet.decoded, 50000);
  const double detect_elapsed = now_steady_s() - detect_t0;
  const double preproc_elapsed = 0.0;

  const auto detections = detections_from_detector_sample(stream.family, det_sample,
                                                          stream.probe.width, stream.probe.height);

  const bool needs_saved_frame = cfg.output_dir.has_value() && cfg.save_every > 0;
  const bool needs_rendered_frame =
      needs_saved_frame || (stream.video_enabled && cfg.video_mode == VideoMode::Annotated);

  cv::Mat frame_out;
  if (needs_rendered_frame) {
    frame_out = render_frame(stream, cfg, tensor_rgb_from_sample(packet.decoded), detections);
  }

  const double publish_t0 = now_steady_s();
  double video_elapsed = 0.0;
  if (stream.video_enabled) {
    const double video_t0 = now_steady_s();
    if (!stream.video.has_value()) {
      try {
        stream.video = build_insight_video_run(cfg, stream.probe, stream.index);
      } catch (const std::exception& ex) {
        throw std::runtime_error(format_video_build_error(stream.index, cfg.video_mode, ex.what()));
      }
    }
    if (cfg.video_mode == VideoMode::Clean) {
      if (!stream.video->run.push(packet.decoded)) {
        throw std::runtime_error("stream " + std::to_string(stream.index) +
                                 " Insight clean video push failed");
      }
    } else if (!stream.video->run.push(std::vector<cv::Mat>{frame_out})) {
      throw std::runtime_error("stream " + std::to_string(stream.index) +
                               " Insight video push failed");
    }
    video_elapsed = now_steady_s() - video_t0;
  }
  const double publish_wall_time_s =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

  double metadata_elapsed = 0.0;
  if (stream.metadata_enabled) {
    const auto payload = build_insight_detection_payload(detections, stream.probe.width,
                                                         stream.probe.height, stream.class_labels);
    const double metadata_t0 = now_steady_s();
    const std::string data_json =
        sima_examples::metadata_boxes_data_json("objects", payload.objects);
    std::string metadata_error;
    if (!stream.metadata_sender->send_metadata(
            "object-detection", data_json,
            insight_timestamp_ms(publish_wall_time_s, cfg.insight_metadata_offset_ms),
            insight_frame_id(det_sample, narrow_frame_index_for_api(packet.frame_index)),
            &metadata_error)) {
      throw std::runtime_error("stream " + std::to_string(stream.index) +
                               " Insight metadata send failed: " + metadata_error);
    }
    metadata_elapsed = now_steady_s() - metadata_t0;
  }

  const int processed_frame_index = stream.metrics.processed + 1;
  if (needs_saved_frame && save_debug_frame(cfg.output_dir, stream.index, processed_frame_index,
                                            frame_out, cfg.save_every)) {
    stream.metrics.saved += 1;
  }

  const double publish_elapsed = now_steady_s() - publish_t0;

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
  stream.metrics.metadata_time_s += metadata_elapsed;
  stream.metrics.interval_metadata_s += metadata_elapsed;
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
        const int n = stream.metrics.interval_frames;
        if (n > 0) {
          const double src = stream.metrics.interval_source_s * 1000.0 / n;
          const double pre = stream.metrics.interval_preproc_s * 1000.0 / n;
          const double det = stream.metrics.interval_detect_s * 1000.0 / n;
          const double vid = stream.metrics.interval_video_s * 1000.0 / n;
          const double meta = stream.metrics.interval_metadata_s * 1000.0 / n;
          const double pub = stream.metrics.interval_publish_s * 1000.0 / n;
          const double loop = stream.metrics.interval_loop_s * 1000.0 / n;
          const double fps = (stream.metrics.interval_wall_started_at_s.has_value() &&
                              stream.metrics.wall_last_processed_at_s.has_value() &&
                              *stream.metrics.wall_last_processed_at_s >
                                  *stream.metrics.interval_wall_started_at_s)
                                 ? n / (*stream.metrics.wall_last_processed_at_s -
                                        *stream.metrics.interval_wall_started_at_s)
                                 : 0.0;

          std::cout << "  [stream " << stream.index << "] frames " << (stream.metrics.processed - n)
                    << "-" << (stream.metrics.processed - 1) << " | source=" << src
                    << "ms preproc=" << pre << "ms detect=" << det << "ms video=" << vid
                    << "ms metadata=" << meta << "ms publish=" << pub << "ms loop=" << loop
                    << "ms throughput_fps=" << fps
                    << " mailbox_drops=" << stream.metrics.mailbox_drops << "\n";

          stream.metrics.interval_source_s = 0.0;
          stream.metrics.interval_preproc_s = 0.0;
          stream.metrics.interval_detect_s = 0.0;
          stream.metrics.interval_video_s = 0.0;
          stream.metrics.interval_metadata_s = 0.0;
          stream.metrics.interval_publish_s = 0.0;
          stream.metrics.interval_loop_s = 0.0;
          stream.metrics.interval_frames = 0;
          stream.metrics.interval_wall_started_at_s = stream.metrics.wall_last_processed_at_s;
        }
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

void print_profile_summary(const std::vector<StreamRuntime>& streams) {
  std::cout << "\nProfile summary (averages per frame):\n";
  for (const auto& stream : streams) {
    const int n = std::max(stream.metrics.processed, 1);
    const double src = stream.metrics.source_time_s * 1000.0 / n;
    const double pre = stream.metrics.preproc_time_s * 1000.0 / n;
    const double det = stream.metrics.detect_time_s * 1000.0 / n;
    const double vid = stream.metrics.video_push_time_s * 1000.0 / n;
    const double meta = stream.metrics.metadata_time_s * 1000.0 / n;
    const double pub = stream.metrics.publish_time_s * 1000.0 / n;
    const double loop = stream.metrics.total_loop_time_s * 1000.0 / n;
    const double fps = wall_clock_fps(stream.metrics.processed, stream.metrics.wall_started_at_s,
                                      stream.metrics.wall_last_processed_at_s);
    std::cout << "  [stream " << stream.index << "] " << stream.metrics.processed
              << " frames | source=" << src << "ms preproc=" << pre << "ms detect=" << det
              << "ms video=" << vid << "ms metadata=" << meta << "ms publish=" << pub
              << "ms loop=" << loop << "ms throughput_fps=" << fps
              << " mailbox_drops=" << stream.metrics.mailbox_drops
              << " detections=" << stream.metrics.detections << "\n";
  }
}

} // namespace

std::string format_video_build_error(int stream_index, VideoMode video_mode,
                                     const std::string& detail) {
  return "stream " + std::to_string(stream_index) + " failed to build Insight " +
         video_mode_name(video_mode) + " video run: " + detail;
}

std::vector<DetectorRuntimeKey>
collect_detector_runtime_keys(const std::vector<StreamProbeSpec>& streams) {
  return collect_detector_runtime_keys_impl(streams);
}

int run_app(const AppConfig& cfg, ModelFamily family) {
  if (cfg.output_dir.has_value()) {
    fs::create_directories(*cfg.output_dir);
  }

  const auto class_labels = load_class_labels();

  std::vector<StreamRuntime> streams(cfg.rtsp_urls.size());
  try {
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      initialize_stream_runtime(streams[index], static_cast<int>(index), cfg.rtsp_urls[index], cfg,
                                family, class_labels);
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
  const auto detector_runtime_keys = collect_detector_runtime_keys(stream_specs);

  for (const auto& stream : streams) {
    std::cout << "[stream " << stream.index << "] " << stream.probe.width << "x"
              << stream.probe.height << " @" << effective_writer_fps(cfg, stream.probe) << "fps "
              << stream.url << " -> insight://" << cfg.insight_host << " video=";
    if (stream.video_enabled) {
      std::cout << insight_video_port_for_stream(cfg.insight_video_port_base, stream.index);
    } else {
      std::cout << "disabled";
    }
    std::cout << " metadata=";
    if (stream.metadata_enabled) {
      std::cout << insight_metadata_port_for_stream(cfg.insight_metadata_port_base, stream.index);
    } else {
      std::cout << "disabled";
    }
    std::cout << "\n";
  }

  ReadyStreamQueue ready_queue;
  std::vector<std::shared_ptr<LatestFrameMailbox<FramePacket>>> mailboxes;
  for (auto& stream : streams) {
    mailboxes.push_back(
        std::make_shared<LatestFrameMailbox<FramePacket>>(stream.index, cfg.mailbox_depth));
  }

  std::atomic<bool> stop_event{false};
  std::vector<WorkerContext> worker_contexts;
  std::vector<std::thread> worker_threads;
  std::vector<std::thread> producer_threads;
  std::vector<std::shared_ptr<Event>> startup_events;
  try {
    startup_events.reserve(streams.size());
    producer_threads.reserve(streams.size());
    for (std::size_t index = 0; index < streams.size(); ++index) {
      auto startup_ready = std::make_shared<Event>();
      startup_events.push_back(startup_ready);
      producer_threads.emplace_back([&stream = streams[index], &cfg, mailbox = mailboxes[index],
                                     &ready_queue, &stop_event, startup_ready]() {
        producer_thread(stream, cfg, mailbox, ready_queue, stop_event, startup_ready.get());
      });
      if (!startup_ready->wait_for(std::chrono::milliseconds(source_startup_pull_timeout_ms()))) {
        emit_startup_trace(streams[index].index,
                           "startup wait_for timed out waiting for first decoded frame");
        streams[index].error_message = "startup timeout waiting for first decoded frame";
        stop_event.store(true);
        try {
          streams[index].source.run.close();
        } catch (...) {
        }
        break;
      }
      if (any_stream_failed(streams)) {
        stop_event.store(true);
        break;
      }
      if (index + 1 < streams.size()) {
        std::this_thread::sleep_for(std::chrono::duration<double>(source_startup_stagger_s()));
      }
    }

    if (!stop_event.load() && !any_stream_failed(streams)) {
      worker_contexts = build_worker_contexts(cfg, cfg.worker_count, detector_runtime_keys);
      for (auto& worker_context : worker_contexts) {
        worker_threads.emplace_back(
            [&worker_context, &streams, &cfg, &mailboxes, &ready_queue, &stop_event]() {
              detector_worker(worker_context, streams, cfg, mailboxes, ready_queue, stop_event);
            });
      }
    }

    for (auto& thread : producer_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  } catch (const std::exception& ex) {
    stop_event.store(true);
    for (auto& mailbox : mailboxes) {
      mailbox->close();
    }
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    for (auto& worker_context : worker_contexts) {
      close_worker_context(worker_context);
    }
    for (auto& thread : producer_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : worker_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    std::cerr << "Error: runtime setup failed: " << ex.what() << "\n";
    return 4;
  } catch (...) {
    stop_event.store(true);
    for (auto& mailbox : mailboxes) {
      mailbox->close();
    }
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    for (auto& worker_context : worker_contexts) {
      close_worker_context(worker_context);
    }
    for (auto& thread : producer_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto& thread : worker_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    throw;
  }

  stop_event.store(true);
  for (auto& thread : worker_threads) {
    if (thread.joinable()) {
      thread.join();
    }
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

} // namespace multi_stream_object_detector
