#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/workers_api.cpp"

#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/image_utils_api.cpp"
#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/pipeline_api.cpp"
#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/sample_utils_api.cpp"
#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/tracker_api.cpp"

#include "pipeline/TensorOpenCV.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace multi_camera_people_tracking {
namespace {

constexpr int kDefaultProfileIntervalFrames = 200;

using SteadyClock = std::chrono::steady_clock;

struct StreamMetrics {
  int pulled = 0;
  int processed = 0;
  int detections = 0;
  int saved = 0;
  int frame_q_drops = 0;
  int result_q_drops = 0;
  int frame_q_peak = 0;
  int result_q_peak = 0;
  double source_time_s = 0.0;
  double preproc_time_s = 0.0;
  double pull_wait_s = 0.0;
  double track_time_s = 0.0;
  double overlay_time_s = 0.0;
  double write_time_s = 0.0;
  double total_loop_time_s = 0.0;
  std::optional<double> wall_started_at_s;
  std::optional<double> wall_last_processed_at_s;
  double interval_source_s = 0.0;
  double interval_preproc_s = 0.0;
  double interval_pull_s = 0.0;
  double interval_output_s = 0.0;
  double interval_loop_s = 0.0;
  int interval_frames = 0;
  int interval_frame_q_drops = 0;
  int interval_result_q_drops = 0;
  std::optional<double> interval_wall_started_at_s;
};

struct FramePacket {
  cv::Mat frame;
  int frame_index = 0;
  double source_time_s = 0.0;
};

struct ResultPacket {
  cv::Mat frame;
  int frame_index = 0;
  std::vector<std::uint8_t> bbox_payload;
  double source_time_s = 0.0;
  double preproc_time_s = 0.0;
  double pull_wait_s = 0.0;
};

template <typename T> class KeepLatestQueue {
public:
  explicit KeepLatestQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

  int push_keep_latest(T item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return 0;
    }
    int dropped = 0;
    while (queue_.size() >= capacity_) {
      queue_.pop_front();
      ++dropped;
    }
    queue_.push_back(std::move(item));
    cv_.notify_one();
    return dropped;
  }

  bool pop_wait(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    const auto ready = [&] { return closed_ || !queue_.empty(); };
    if (timeout_ms < 0) {
      cv_.wait(lock, ready);
    } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
      return false;
    }
    if (queue_.empty()) {
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.empty();
  }

  bool is_closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

private:
  std::size_t capacity_ = 1;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  bool closed_ = false;
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
  RtspProbe probe;
  std::shared_ptr<simaai::neat::Model> model;
  QuantTessCpuPreprocState quant_preproc_state;
  simaai::neat::Session source_session;
  simaai::neat::Run source_run;
  simaai::neat::Session detect_session;
  simaai::neat::Run detect_run;
  simaai::neat::Session video_session;
  simaai::neat::Run video_run;
  sima_examples::OptiViewSender json_sender;
  PeopleTracker tracker;
  StreamMetrics metrics;
  std::string error_message;
};

double now_steady_s() {
  return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

std::int64_t now_unix_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

StreamRuntime create_stream_runtime(int index, const std::string& url, const AppConfig& cfg,
                                    const std::shared_ptr<simaai::neat::Model>& model,
                                    const QuantTessCpuPreproc& quant_preproc) {
  const RtspProbe probe = probe_rtsp(url);
  const auto quant_preproc_state =
      build_cpu_quanttess_preproc_state(quant_preproc, probe.width, probe.height);
  auto source = build_source_run(cfg, url, probe);
  auto detect = build_detection_run(cfg, *model, probe, quant_preproc);
  auto video = build_optiview_video_run(cfg, probe, index);
  auto sender = build_optiview_json_output(cfg, index);

  StreamRuntime runtime{
      index,
      url,
      probe,
      model,
      quant_preproc_state,
      std::move(source.session),
      std::move(source.run),
      std::move(detect.session),
      std::move(detect.run),
      std::move(video.session),
      std::move(video.run),
      std::move(sender),
      PeopleTracker(cfg.tracker_iou_threshold, cfg.tracker_max_missing),
  };
  return runtime;
}

void close_stream_runtime(StreamRuntime& stream) {
  for (auto* run : {&stream.video_run, &stream.detect_run, &stream.source_run}) {
    try {
      run->close();
    } catch (...) {
    }
  }
}

template <typename T>
void record_queue_depth(StreamMetrics& metrics, const std::string& queue_name,
                        const KeepLatestQueue<T>& queue) {
  const int size = static_cast<int>(queue.size());
  if (queue_name == "frame") {
    metrics.frame_q_peak = std::max(metrics.frame_q_peak, size);
  } else {
    metrics.result_q_peak = std::max(metrics.result_q_peak, size);
  }
}

template <typename T>
void put_keep_latest(KeepLatestQueue<T>& queue, T item, StreamRuntime& stream,
                     const std::string& queue_name) {
  record_queue_depth(stream.metrics, queue_name, queue);
  const int dropped = queue.push_keep_latest(std::move(item));
  if (queue_name == "frame") {
    stream.metrics.frame_q_drops += dropped;
    stream.metrics.interval_frame_q_drops += dropped;
  } else {
    stream.metrics.result_q_drops += dropped;
    stream.metrics.interval_result_q_drops += dropped;
  }
  record_queue_depth(stream.metrics, queue_name, queue);
}

void producer_thread(StreamRuntime& stream, const AppConfig& cfg,
                     KeepLatestQueue<FramePacket>& frame_queue, std::atomic<bool>& stop_event,
                     Event* startup_ready) {
  int frame_index = 0;
  int empty_pulls = 0;
  const double emit_period_s = cfg.fps > 0 ? (1.0 / cfg.fps) : 0.0;
  std::optional<double> next_allowed_emit_s;

  try {
    while (!stop_event.load()) {
      if (cfg.frames > 0 && frame_index >= cfg.frames) {
        break;
      }

      const double t0 = now_steady_s();
      const int pull_timeout_ms =
          frame_index == 0 ? kSourceStartupPullTimeoutMs : kSourcePullTimeoutMs;
      const auto sample = stream.source_run.pull(pull_timeout_ms);
      const double elapsed = now_steady_s() - t0;
      if (!sample.has_value()) {
        empty_pulls += 1;
        if (cfg.frames > 0 && empty_pulls >= 20) {
          throw std::runtime_error("stream " + std::to_string(stream.index) +
                                   " timed out waiting for RTSP frames");
        }
        continue;
      }
      empty_pulls = 0;

      if (emit_period_s > 0.0) {
        const double now = now_steady_s();
        if (!next_allowed_emit_s.has_value()) {
          next_allowed_emit_s = now;
        }
        if (now < *next_allowed_emit_s) {
          continue;
        }
        while (*next_allowed_emit_s <= now) {
          *next_allowed_emit_s += emit_period_s;
        }
      }

      FramePacket packet;
      packet.frame = tensor_rgb_from_sample(*sample);
      packet.frame_index = frame_index;
      packet.source_time_s = elapsed;
      put_keep_latest(frame_queue, std::move(packet), stream, "frame");
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

  frame_queue.close();
}

void infer_thread(StreamRuntime& stream, KeepLatestQueue<FramePacket>& frame_queue,
                  KeepLatestQueue<ResultPacket>& result_queue, std::atomic<bool>& stop_event) {
  try {
    while (!stop_event.load()) {
      FramePacket packet;
      if (!frame_queue.pop_wait(packet, 100)) {
        if (frame_queue.is_closed() && frame_queue.empty()) {
          break;
        }
        continue;
      }

      const double preproc_t0 = now_steady_s();
      const cv::Mat quant_input = cpu_quanttess_input(packet.frame, stream.quant_preproc_state);
      const double preproc_elapsed = now_steady_s() - preproc_t0;

      const double roundtrip_t0 = now_steady_s();
      const simaai::neat::Sample det_sample =
          run_tensor_input_once(stream.detect_run, quant_input, 50000);
      const double roundtrip_elapsed = now_steady_s() - roundtrip_t0;

      ResultPacket result;
      result.frame = std::move(packet.frame);
      result.frame_index = packet.frame_index;
      result.bbox_payload = extract_bbox_payload(det_sample);
      result.source_time_s = packet.source_time_s;
      result.preproc_time_s = preproc_elapsed;
      result.pull_wait_s = roundtrip_elapsed;
      put_keep_latest(result_queue, std::move(result), stream, "result");
    }
  } catch (const std::exception& ex) {
    stream.error_message = ex.what();
    stop_event.store(true);
  }

  result_queue.close();
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

void print_interval_profile(const StreamRuntime& stream) {
  const StreamMetrics& metrics = stream.metrics;
  const int n = metrics.interval_frames;
  if (n <= 0) {
    return;
  }

  const double src_ms = metrics.interval_source_s * 1000.0 / n;
  const double pre_ms = metrics.interval_preproc_s * 1000.0 / n;
  const double pull_ms = metrics.interval_pull_s * 1000.0 / n;
  const double out_ms = metrics.interval_output_s * 1000.0 / n;
  const double loop_ms = metrics.interval_loop_s * 1000.0 / n;
  const double throughput_fps = wall_clock_fps(
      n, metrics.interval_wall_started_at_s, metrics.wall_last_processed_at_s);

  std::cout << "  [stream " << stream.index << "] frames "
            << (metrics.processed - n) << "-" << (metrics.processed - 1) << " | src=" << src_ms
            << "ms  preproc=" << pre_ms << "ms  pull_wait=" << pull_ms
            << "ms  output=" << out_ms << "ms  loop=" << loop_ms
            << "ms  throughput_fps=" << throughput_fps << "  frame_q(drops="
            << metrics.interval_frame_q_drops << ",peak=" << metrics.frame_q_peak
            << ")  result_q(drops=" << metrics.interval_result_q_drops
            << ",peak=" << metrics.result_q_peak << ")\n";
}

void print_profile_summary(const std::vector<StreamRuntime>& streams) {
  std::cout << "\nProfile summary (averages per frame):\n";
  for (const auto& stream : streams) {
    const StreamMetrics& metrics = stream.metrics;
    const int n = std::max(metrics.processed, 1);
    const double src = metrics.source_time_s * 1000.0 / n;
    const double pre = metrics.preproc_time_s * 1000.0 / n;
    const double pull = metrics.pull_wait_s * 1000.0 / n;
    const double track = metrics.track_time_s * 1000.0 / n;
    const double overlay = metrics.overlay_time_s * 1000.0 / n;
    const double write = metrics.write_time_s * 1000.0 / n;
    const double out = track + overlay + write;
    const double loop = metrics.total_loop_time_s * 1000.0 / n;
    const double throughput_fps =
        wall_clock_fps(metrics.processed, metrics.wall_started_at_s, metrics.wall_last_processed_at_s);

    std::cout << "  [stream " << stream.index << "] " << metrics.processed
              << " frames | src=" << src << "ms  preproc=" << pre << "ms  pull_wait=" << pull
              << "ms  output=" << out << "ms (track=" << track << " overlay=" << overlay
              << " write=" << write << ")  loop=" << loop << "ms  throughput_fps="
              << throughput_fps << "  frame_q(total_drops=" << metrics.frame_q_drops
              << ",peak=" << metrics.frame_q_peak << ")  result_q(total_drops="
              << metrics.result_q_drops << ",peak=" << metrics.result_q_peak << ")\n";
  }
}

void publish_thread(StreamRuntime& stream, const AppConfig& cfg,
                    KeepLatestQueue<ResultPacket>& result_queue, std::atomic<bool>& stop_event) {
  const std::optional<fs::path> output_dir =
      cfg.output_dir.has_value() ? std::optional<fs::path>(fs::path(*cfg.output_dir)) : std::nullopt;
  const int profile_every =
      cfg.save_every > 0 ? cfg.save_every : kDefaultProfileIntervalFrames;

  try {
    while (!stop_event.load()) {
      ResultPacket packet;
      if (!result_queue.pop_wait(packet, 100)) {
        if (result_queue.is_closed() && result_queue.empty()) {
          break;
        }
        continue;
      }

      const double loop_start = now_steady_s();
      if (!stream.metrics.wall_started_at_s.has_value()) {
        stream.metrics.wall_started_at_s = loop_start;
      }
      if (!stream.metrics.interval_wall_started_at_s.has_value()) {
        stream.metrics.interval_wall_started_at_s = loop_start;
      }

      stream.metrics.source_time_s += packet.source_time_s;
      stream.metrics.interval_source_s += packet.source_time_s;
      stream.metrics.preproc_time_s += packet.preproc_time_s;
      stream.metrics.pull_wait_s += packet.pull_wait_s;
      stream.metrics.interval_preproc_s += packet.preproc_time_s;
      stream.metrics.interval_pull_s += packet.pull_wait_s;
      stream.metrics.pulled += 1;

      std::vector<Detection> boxes =
          parse_bbox_payload(packet.bbox_payload, packet.frame.cols, packet.frame.rows);
      boxes = filter_person_detections(boxes, cfg.person_class_id);

      const double track_t0 = now_steady_s();
      const auto tracked = stream.tracker.update(boxes, packet.frame_index);
      stream.metrics.track_time_s += now_steady_s() - track_t0;
      stream.metrics.detections += static_cast<int>(tracked.size());

      const double write_t0 = now_steady_s();
      if (!stream.video_run.push(packet.frame)) {
        throw std::runtime_error("stream " + std::to_string(stream.index) +
                                 " OptiView video push failed");
      }

      const auto payload = make_optiview_tracking_detection(tracked);
      const std::string frame_id = std::to_string(stream.metrics.processed);
      if (!stream.json_sender.send_detection(now_unix_ms(), frame_id, payload.objects,
                                             payload.labels)) {
        throw std::runtime_error("stream " + std::to_string(stream.index) +
                                 " OptiView JSON send failed");
      }

      if (output_dir.has_value() && cfg.save_every > 0 &&
          (packet.frame_index % cfg.save_every) == 0) {
        const double overlay_t0 = now_steady_s();
        const cv::Mat overlay = draw_tracked_people(packet.frame.clone(), tracked);
        stream.metrics.overlay_time_s += now_steady_s() - overlay_t0;
        if (save_overlay_frame(output_dir, stream.index, packet.frame_index, overlay,
                               cfg.save_every)) {
          stream.metrics.saved += 1;
        }
      }
      stream.metrics.write_time_s += now_steady_s() - write_t0;

      const double completed_at = now_steady_s();
      const double output_elapsed = completed_at - loop_start;
      stream.metrics.interval_output_s += output_elapsed;
      stream.metrics.processed += 1;
      const double per_frame =
          packet.source_time_s + packet.preproc_time_s + packet.pull_wait_s + output_elapsed;
      stream.metrics.total_loop_time_s += per_frame;
      stream.metrics.interval_loop_s += per_frame;
      stream.metrics.interval_frames += 1;
      stream.metrics.wall_last_processed_at_s = completed_at;

      if (cfg.profile && stream.metrics.interval_frames >= profile_every) {
        print_interval_profile(stream);
        stream.metrics.interval_source_s = 0.0;
        stream.metrics.interval_preproc_s = 0.0;
        stream.metrics.interval_pull_s = 0.0;
        stream.metrics.interval_output_s = 0.0;
        stream.metrics.interval_loop_s = 0.0;
        stream.metrics.interval_frames = 0;
        stream.metrics.interval_frame_q_drops = 0;
        stream.metrics.interval_result_q_drops = 0;
        stream.metrics.interval_wall_started_at_s = stream.metrics.wall_last_processed_at_s;
      }
    }
  } catch (const std::exception& ex) {
    stream.error_message = ex.what();
    stop_event.store(true);
  }
}

} // namespace

int run_app(const AppConfig& cfg) {
  if (cfg.output_dir.has_value()) {
    fs::create_directories(*cfg.output_dir);
  }

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
    streams.reserve(cfg.rtsp_urls.size());
    for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
      streams.push_back(create_stream_runtime(static_cast<int>(index), cfg.rtsp_urls[index], cfg,
                                              model, quant_preproc));
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error: failed to set up stream runtimes: " << ex.what() << "\n";
    for (auto& stream : streams) {
      close_stream_runtime(stream);
    }
    return 4;
  }

  for (const auto& stream : streams) {
    std::cout << "[stream " << stream.index << "] " << stream.probe.width << "x"
              << stream.probe.height << " @" << effective_writer_fps(cfg, stream.probe)
              << "fps " << stream.url << " -> optiview://" << cfg.optiview_host
              << " video="
              << optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index)
              << " json="
              << optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index) << "\n";
  }

  std::atomic<bool> stop_event{false};
  std::vector<std::shared_ptr<KeepLatestQueue<FramePacket>>> frame_queues;
  std::vector<std::shared_ptr<KeepLatestQueue<ResultPacket>>> result_queues;
  std::vector<std::shared_ptr<Event>> startup_events;
  std::vector<std::thread> worker_threads;
  std::vector<std::function<void()>> producer_jobs;
  std::vector<std::thread> producer_threads;

  frame_queues.reserve(streams.size());
  result_queues.reserve(streams.size());
  startup_events.reserve(streams.size());
  producer_jobs.reserve(streams.size());

  for (auto& stream : streams) {
    auto frame_queue = std::make_shared<KeepLatestQueue<FramePacket>>(4);
    auto result_queue = std::make_shared<KeepLatestQueue<ResultPacket>>(4);
    auto startup_ready = std::make_shared<Event>();

    frame_queues.push_back(frame_queue);
    result_queues.push_back(result_queue);
    startup_events.push_back(startup_ready);

    worker_threads.emplace_back([&stream, &cfg, frame_queue, result_queue, &stop_event] {
      infer_thread(stream, *frame_queue, *result_queue, stop_event);
    });
    worker_threads.emplace_back([&stream, &cfg, result_queue, &stop_event] {
      publish_thread(stream, cfg, *result_queue, stop_event);
    });
    producer_jobs.push_back([&stream, &cfg, frame_queue, startup_ready, &stop_event] {
      producer_thread(stream, cfg, *frame_queue, stop_event, startup_ready.get());
    });
  }

  for (std::size_t index = 0; index < producer_jobs.size(); ++index) {
    if (stop_event.load()) {
      break;
    }
    producer_threads.emplace_back(producer_jobs[index]);
    if (!startup_events[index]->wait_for(std::chrono::milliseconds(kSourceStartupPullTimeoutMs))) {
      stop_event.store(true);
      break;
    }
    if (stop_event.load()) {
      break;
    }
    if (index + 1 < producer_jobs.size()) {
      std::this_thread::sleep_for(
          std::chrono::duration<double>(kSourceStartupStaggerS));
    }
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

  for (auto& stream : streams) {
    close_stream_runtime(stream);
  }

  bool failed = false;
  for (const auto& stream : streams) {
    if (!stream.error_message.empty()) {
      std::cerr << "[stream " << stream.index << "] error: " << stream.error_message << "\n";
      failed = true;
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

} // namespace multi_camera_people_tracking
