#include "workers_api.cpp"

#include "image_utils_api.cpp"
#include "pipeline_api.cpp"
#include "sample_utils_api.cpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace multistream_yolox_yolov8_optiview {
namespace {

constexpr int kDefaultProfileIntervalFrames = 200;
const fs::path kDefaultLabelsPath =
    fs::path(MULTISTREAM_YOLOX_YOLOV8_OPTIVIEW_SOURCE_DIR) / ".." / "common" / "coco_label.txt";

using SteadyClock = std::chrono::steady_clock;

struct StreamMetrics {
  int pulled = 0;
  int processed = 0;
  int detections = 0;
  int saved = 0;
  int mailbox_drops = 0;
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
  std::optional<double> interval_wall_started_at_s;
};

struct FramePacket {
  cv::Mat frame;
  int frame_index = 0;
  double source_time_s = 0.0;
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
  QuantTessCpuPreprocState quant_preproc_state;
  SessionRun source;
  SessionRun video;
  bool video_enabled = true;
  sima_examples::OptiViewSender json_sender;
  std::vector<std::string> class_labels;
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
                                    const QuantTessCpuPreproc& quant_preproc,
                                    const std::vector<std::string>& class_labels) {
  const RtspProbe probe = probe_rtsp(url);
  SessionRun video_run;
  if (cfg.video_enabled) {
    video_run = build_optiview_video_run(cfg, probe, index);
  }
  return StreamRuntime{
      index,
      url,
      family,
      probe,
      build_cpu_quanttess_preproc_state(quant_preproc, probe.width, probe.height),
      build_source_run(cfg, url, probe),
      std::move(video_run),
      cfg.video_enabled,
      build_optiview_json_output(cfg, index),
      class_labels,
  };
}

WorkerContext build_worker_context(int worker_index, const AppConfig& cfg,
                                   const simaai::neat::Model& model,
                                   const QuantTessCpuPreproc& quant_preproc,
                                   const std::vector<DetectorRuntimeKey>& detector_keys) {
  WorkerContext context;
  context.index = worker_index;
  for (const auto& key : detector_keys) {
    const RtspProbe probe{key.width, key.height, 0};
    context.detectors.push_back(
        DetectorRuntime{key, build_detection_run(cfg, model, key.family, probe, quant_preproc)});
  }
  return context;
}

std::vector<WorkerContext> build_worker_contexts(
    const AppConfig& cfg, const simaai::neat::Model& model,
    const QuantTessCpuPreproc& quant_preproc, int worker_count,
    const std::vector<DetectorRuntimeKey>& detector_keys) {
  std::vector<WorkerContext> contexts;
  contexts.reserve(static_cast<std::size_t>(std::max(worker_count, 0)));
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    contexts.push_back(
        build_worker_context(worker_index, cfg, model, quant_preproc, detector_keys));
  }
  return contexts;
}

void close_stream_runtime(StreamRuntime& stream) {
  if (stream.video_enabled) {
    try {
      stream.video.run.close();
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

void producer_thread(StreamRuntime& stream, const AppConfig& cfg,
                     LatestFrameMailbox<FramePacket>& mailbox, ReadyStreamQueue& ready_queue,
                     std::atomic<bool>& stop_event, Event* startup_ready) {
  int frame_index = 0;
  int empty_pulls = 0;
  const double emit_period_s = cfg.fps > 0 ? (1.0 / cfg.fps) : 0.0;
  std::optional<double> next_allowed_emit_s;

  try {
    while (!stop_event.load()) {
      if (cfg.frames > 0 && frame_index >= cfg.frames) {
        break;
      }
      if (emit_period_s > 0.0) {
        const double now = now_steady_s();
        if (!next_allowed_emit_s.has_value()) {
          next_allowed_emit_s = now;
        }
        if (now < *next_allowed_emit_s) {
          std::this_thread::sleep_for(std::chrono::duration<double>(*next_allowed_emit_s - now));
          continue;
        }
        while (*next_allowed_emit_s <= now) {
          *next_allowed_emit_s += emit_period_s;
        }
      }
      const double t0 = now_steady_s();
      const int pull_timeout_ms =
          frame_index == 0 ? kSourceStartupPullTimeoutMs : kSourcePullTimeoutMs;
      const auto sample = stream.source.run.pull(pull_timeout_ms);
      const double elapsed = now_steady_s() - t0;
      if (!sample.has_value()) {
        ++empty_pulls;
        if (cfg.frames > 0 && empty_pulls >= 20) {
          throw std::runtime_error("stream " + std::to_string(stream.index) +
                                   " timed out waiting for RTSP frames");
        }
        continue;
      }

      empty_pulls = 0;

      FramePacket packet;
      packet.frame = tensor_rgb_from_sample(*sample);
      packet.frame_index = frame_index;
      packet.source_time_s = elapsed;
      stream.metrics.mailbox_drops += mailbox.push(std::move(packet), ready_queue);
      if (startup_ready != nullptr && frame_index == 0) {
        startup_ready->set();
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

  const double preproc_t0 = now_steady_s();
  cv::Mat quant_input = cpu_quanttess_input(packet.frame, stream.quant_preproc_state);
  const double preproc_elapsed = now_steady_s() - preproc_t0;

  auto& detector = find_detector_runtime(worker_context, stream.family, stream.probe);
  const double detect_t0 = now_steady_s();
  const simaai::neat::Sample det_sample = run_tensor_input_once(detector.runtime.run, quant_input, 50000);
  const double detect_elapsed = now_steady_s() - detect_t0;

  const auto detections =
      detections_from_detector_sample(stream.family, det_sample, stream.probe.width, stream.probe.height);
  const bool needs_output_frame =
      stream.video_enabled || (cfg.output_dir.has_value() && cfg.save_every > 0);
  const cv::Mat frame_out = needs_output_frame ? render_frame(stream, cfg, packet.frame, detections)
                                               : packet.frame;

  const double publish_t0 = now_steady_s();
  const double video_t0 = now_steady_s();
  if (stream.video_enabled && !stream.video.run.push(frame_out)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) + " OptiView video push failed");
  }
  const double video_elapsed = now_steady_s() - video_t0;
  const double publish_wall_time_s =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

  const auto payload =
      build_optiview_detection_payload(detections, stream.probe.width, stream.probe.height,
                                       stream.class_labels);
  const double json_t0 = now_steady_s();
  if (!stream.json_sender.send_detection(
          optiview_timestamp_ms(publish_wall_time_s, cfg.optiview_json_offset_ms),
          optiview_frame_id(det_sample, packet.frame_index),
                                         payload.objects, payload.labels)) {
    throw std::runtime_error("stream " + std::to_string(stream.index) + " OptiView JSON send failed");
  }
  const double json_elapsed = now_steady_s() - json_t0;

  if (save_debug_frame(cfg.output_dir, stream.index, packet.frame_index, frame_out, cfg.save_every)) {
    stream.metrics.saved += 1;
  }
  const double publish_elapsed = now_steady_s() - publish_t0;

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
            << " mailbox_drops=" << stream.metrics.mailbox_drops << "\n";

  stream.metrics.interval_source_s = 0.0;
  stream.metrics.interval_preproc_s = 0.0;
  stream.metrics.interval_detect_s = 0.0;
  stream.metrics.interval_video_s = 0.0;
  stream.metrics.interval_json_s = 0.0;
  stream.metrics.interval_publish_s = 0.0;
  stream.metrics.interval_loop_s = 0.0;
  stream.metrics.interval_frames = 0;
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
              << " detections=" << stream.metrics.detections << "\n";
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
  std::shared_ptr<simaai::neat::Model> model;
  QuantTessCpuPreproc quant_preproc;
  try {
    model = load_detector_model(cfg);
    quant_preproc = read_preproc_contract(*model);
  } catch (const std::exception& ex) {
    std::cerr << "Error: failed to build model: " << ex.what() << "\n";
    return 3;
  }

  std::vector<StreamRuntime> streams;
  try {
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      streams.push_back(create_stream_runtime(static_cast<int>(index), cfg.rtsp_urls[index], cfg,
                                              family, quant_preproc, class_labels));
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
    worker_contexts = build_worker_contexts(cfg, *model, quant_preproc, cfg.worker_count,
                                            collect_detector_runtime_keys(stream_specs));
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
    std::cout
              << " json=" << optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index)
              << "\n";
  }

  std::atomic<bool> stop_event{false};
  ReadyStreamQueue ready_queue;
  std::vector<std::shared_ptr<LatestFrameMailbox<FramePacket>>> mailboxes;
  std::vector<std::thread> producer_threads;
  std::vector<std::thread> worker_threads;
  std::vector<std::unique_ptr<Event>> producer_ready_events;

  for (auto& stream : streams) {
    mailboxes.push_back(std::make_shared<LatestFrameMailbox<FramePacket>>(stream.index, cfg.mailbox_depth));
    producer_ready_events.push_back(std::make_unique<Event>());
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
      producer_threads.emplace_back(
          [&streams, &cfg, &mailboxes, &ready_queue, &stop_event,
           ready = producer_ready_events[index].get(), index]() {
            producer_thread(streams[index], cfg, *mailboxes[index], ready_queue, stop_event, ready);
          });
      if (!producer_ready_events[index]->wait_for(startup_timeout)) {
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
    for (auto& thread : producer_threads) {
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
    for (auto& thread : producer_threads) {
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
