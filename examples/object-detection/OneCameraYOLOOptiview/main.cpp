#include "support/runtime/example_utils.h"

#include "support/object_detection/obj_detection_utils.h"
#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"

#include <gst/gst.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h> // getpid()

namespace fs = std::filesystem;

using sima_examples::parse_int_arg;
using sima_examples::time_ms;

namespace {

struct Config {
  std::string url;
  std::string mpk;
  int frames = 300;
  bool frames_set = false;
  bool debug = false;
  bool udp = false;
  bool forever = false;
  bool optiview = false;
  std::string optiview_host = "127.0.0.1";
  int optiview_video_port = 9000;
  int optiview_json_port = 9100;
  double optiview_offset_ms = 0.0;
  bool optiview_fifo = false;
};

bool parse_double_arg(int argc, char** argv, const std::string& key, double& out) {
  std::string raw;
  if (!sima_examples::get_arg(argc, argv, key, raw))
    return false;
  out = std::stod(raw);
  return true;
}

Config parse_config(int argc, char** argv) {
  Config cfg;
  sima_examples::get_arg(argc, argv, "--rtsp", cfg.url);
  sima_examples::get_arg(argc, argv, "--mpk", cfg.mpk);
  cfg.frames_set = parse_int_arg(argc, argv, "--frames", cfg.frames);
  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  cfg.udp = sima_examples::has_flag(argc, argv, "--udp");
  cfg.forever = sima_examples::has_flag(argc, argv, "--forever");
  cfg.optiview = sima_examples::has_flag(argc, argv, "--optiview");
  sima_examples::get_arg(argc, argv, "--optiview-host", cfg.optiview_host);
  if (cfg.optiview_host.empty())
    cfg.optiview_host = "127.0.0.1";
  parse_int_arg(argc, argv, "--optiview-video-port", cfg.optiview_video_port);
  parse_int_arg(argc, argv, "--optiview-json-port", cfg.optiview_json_port);
  parse_double_arg(argc, argv, "--optiview-offset-ms", cfg.optiview_offset_ms);
  cfg.optiview_fifo = sima_examples::has_flag(argc, argv, "--optiview-fifo");
  return cfg;
}

using sima_examples::bgr_to_nv12_tensor;
using sima_examples::infer_dims;
using sima_examples::init_nv12_tensor_meta;
using sima_examples::make_blank_nv12_tensor;
using sima_examples::nv12_copy_to_cpu_tensor;
using sima_examples::nv12_to_bgr;

void print_time(const char* label, double ms, bool enabled) {
  if (!enabled)
    return;
  std::cout << label << " " << ms << "\n";
}

void enable_optiview_diagnostics(bool enabled) {
  if (!enabled)
    return;
  setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
  setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
  setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
}

void print_pipeline_report(const char* label, const simaai::neat::Run& run, bool enabled) {
  if (!enabled)
    return;
  simaai::neat::RunReportOptions opt;
  opt.include_pipeline = false;
  opt.include_stage_timings = false;
  opt.include_element_timings = true;
  opt.include_boundaries = false;
  opt.include_flow_stats = true;
  opt.include_node_reports = true;
  opt.include_next_cpu = false;
  opt.include_queue_depth = false;
  opt.include_num_buffers = false;
  opt.include_run_stats = true;
  opt.include_input_stats = true;
  opt.include_system_info = false;
  std::cout << "[TIMING] " << label << "\n" << run.report(opt);
}

void print_stream_summary(const char* label, const simaai::neat::Run& run, bool enabled) {
  if (!enabled)
    return;
  const auto stats = run.input_stats();
  std::cout << "[STREAM] " << label << " avg_push_us=" << stats.avg_push_us
            << " avg_pull_wait_us=" << stats.avg_pull_wait_us
            << " avg_decode_us=" << stats.avg_decode_us << " avg_copy_us=" << stats.avg_copy_us
            << " push_failures=" << stats.push_failures << "\n";
}

double fps_from_count(int count, double elapsed_ms) {
  if (elapsed_ms <= 0.0)
    return 0.0;
  return static_cast<double>(count) * 1000.0 / elapsed_ms;
}

void print_throughput_summary(int produced, int det_outputs, int saved, double producer_start_ms,
                              double producer_end_ms, double consumer_start_ms,
                              double consumer_end_ms) {
  const double producer_elapsed_ms =
      (producer_end_ms > producer_start_ms) ? (producer_end_ms - producer_start_ms) : 0.0;
  const double consumer_elapsed_ms =
      (consumer_end_ms > consumer_start_ms) ? (consumer_end_ms - consumer_start_ms) : 0.0;
  std::cout << "[THROUGHPUT] produced=" << produced
            << " fps=" << fps_from_count(produced, producer_elapsed_ms)
            << " yolo_out=" << det_outputs
            << " fps=" << fps_from_count(det_outputs, consumer_elapsed_ms) << " saved=" << saved
            << " fps=" << fps_from_count(saved, consumer_elapsed_ms)
            << " producer_ms=" << producer_elapsed_ms << " consumer_ms=" << consumer_elapsed_ms
            << "\n";
}

struct FrameItem {
  int index = 0;
  simaai::neat::Tensor frame;
  double pull_ts_ms = 0.0;
};

struct PendingFrame {
  int index = 0;
  double pull_ts_ms = 0.0;
  simaai::neat::Tensor frame;
  cv::Mat bgr;
};

struct FrameQueue {
  explicit FrameQueue(size_t max_size_in) : max_size(max_size_in) {}

  bool push(FrameItem item) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || items.size() < max_size; });
    if (closed)
      return false;
    items.push_back(std::move(item));
    lock.unlock();
    cond.notify_all();
    return true;
  }

  bool pop(FrameItem& out) {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [&]() { return closed || !items.empty(); });
    if (items.empty())
      return false;
    out = std::move(items.front());
    items.pop_front();
    lock.unlock();
    cond.notify_all();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cond.notify_all();
  }

private:
  size_t max_size = 0;
  std::mutex mu;
  std::condition_variable cond;
  std::deque<FrameItem> items;
  bool closed = false;
};

struct ProducerTiming {
  int count = 0;
  double rtsp_pull_sum = 0.0;
  double rtsp_pull_max = 0.0;
  double queue_push_sum = 0.0;
  double queue_push_max = 0.0;

  void add_rtsp_pull(double ms) {
    rtsp_pull_sum += ms;
    if (ms > rtsp_pull_max)
      rtsp_pull_max = ms;
  }

  void add_queue_push(double ms) {
    queue_push_sum += ms;
    if (ms > queue_push_max)
      queue_push_max = ms;
  }

  void print() const {
    if (count <= 0)
      return;
    std::cout << "producer_avg_rtsp_pull_ms " << (rtsp_pull_sum / count)
              << " producer_max_rtsp_pull_ms " << rtsp_pull_max << " producer_avg_queue_push_ms "
              << (queue_push_sum / count) << " producer_max_queue_push_ms " << queue_push_max
              << "\n";
  }
};

struct ConsumerTiming {
  int count = 0;
  double queue_pop_sum = 0.0;
  double queue_pop_max = 0.0;
  double convert_sum = 0.0;
  double convert_max = 0.0;
  double udp_convert_sum = 0.0;
  double udp_convert_max = 0.0;
  double yolo_push_sum = 0.0;
  double yolo_push_max = 0.0;
  double yolo_pull_sum = 0.0;
  double yolo_pull_max = 0.0;
  double udp_push_sum = 0.0;
  double udp_push_max = 0.0;
  double bbox_extract_sum = 0.0;
  double bbox_extract_max = 0.0;
  double bbox_parse_sum = 0.0;
  double bbox_parse_max = 0.0;
  double overlay_sum = 0.0;
  double overlay_max = 0.0;
  double write_sum = 0.0;
  double write_max = 0.0;
  double e2e_sum = 0.0;
  double e2e_max = 0.0;

  void add_queue_pop(double ms) {
    queue_pop_sum += ms;
    if (ms > queue_pop_max)
      queue_pop_max = ms;
  }

  void add_convert(double ms) {
    convert_sum += ms;
    if (ms > convert_max)
      convert_max = ms;
  }

  void add_udp_convert(double ms) {
    udp_convert_sum += ms;
    if (ms > udp_convert_max)
      udp_convert_max = ms;
  }

  void add_yolo_push(double ms) {
    yolo_push_sum += ms;
    if (ms > yolo_push_max)
      yolo_push_max = ms;
  }

  void add_yolo_pull(double ms) {
    yolo_pull_sum += ms;
    if (ms > yolo_pull_max)
      yolo_pull_max = ms;
  }

  void add_udp_push(double ms) {
    udp_push_sum += ms;
    if (ms > udp_push_max)
      udp_push_max = ms;
  }

  void add_bbox_extract(double ms) {
    bbox_extract_sum += ms;
    if (ms > bbox_extract_max)
      bbox_extract_max = ms;
  }

  void add_bbox_parse(double ms) {
    bbox_parse_sum += ms;
    if (ms > bbox_parse_max)
      bbox_parse_max = ms;
  }

  void add_overlay(double ms) {
    overlay_sum += ms;
    if (ms > overlay_max)
      overlay_max = ms;
  }

  void add_write(double ms) {
    write_sum += ms;
    if (ms > write_max)
      write_max = ms;
  }

  void add_e2e(double ms) {
    e2e_sum += ms;
    if (ms > e2e_max)
      e2e_max = ms;
  }

  void print() const {
    if (count <= 0)
      return;
    std::cout << "consumer_avg_queue_pop_ms " << (queue_pop_sum / count)
              << " consumer_max_queue_pop_ms " << queue_pop_max << " consumer_avg_convert_ms "
              << (convert_sum / count) << " consumer_max_convert_ms " << convert_max
              << " consumer_avg_udp_convert_ms " << (udp_convert_sum / count)
              << " consumer_max_udp_convert_ms " << udp_convert_max << " consumer_avg_yolo_push_ms "
              << (yolo_push_sum / count) << " consumer_max_yolo_push_ms " << yolo_push_max
              << " consumer_avg_yolo_pull_ms " << (yolo_pull_sum / count)
              << " consumer_max_yolo_pull_ms " << yolo_pull_max << " consumer_avg_udp_push_ms "
              << (udp_push_sum / count) << " consumer_max_udp_push_ms " << udp_push_max
              << " consumer_avg_bbox_extract_ms " << (bbox_extract_sum / count)
              << " consumer_max_bbox_extract_ms " << bbox_extract_max
              << " consumer_avg_bbox_parse_ms " << (bbox_parse_sum / count)
              << " consumer_max_bbox_parse_ms " << bbox_parse_max << " consumer_avg_overlay_ms "
              << (overlay_sum / count) << " consumer_max_overlay_ms " << overlay_max
              << " consumer_avg_write_ms " << (write_sum / count) << " consumer_max_write_ms "
              << write_max << " consumer_avg_e2e_ms " << (e2e_sum / count)
              << " consumer_max_e2e_ms " << e2e_max << "\n";
  }
};

} // namespace

int main(int argc, char** argv) {
  try {

    Config cfg = parse_config(argc, argv);
    sima_examples::require(!cfg.url.empty(), "Missing --rtsp <url>");
    if (cfg.mpk.empty())
      cfg.mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());
    sima_examples::require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");

    enable_optiview_diagnostics(cfg.optiview);

    simaai::neat::Session camera;
    simaai::neat::nodes::groups::RtspDecodedInputOptions cam_opt;
    cam_opt.url = cfg.url;
    cam_opt.decoder_name = "decoder";
    // RTSP servers often omit framerate in caps; provide a fallback for negotiation.
    cam_opt.fallback_h264_fps = 30;
    camera.add(simaai::neat::nodes::groups::RtspDecodedInput(cam_opt));
    camera.add(simaai::neat::nodes::Output());
    simaai::neat::RunOptions cam_run_opt;
    cam_run_opt.enable_metrics = cfg.optiview;
    auto cam = camera.build(cam_run_opt);

    const double first_pull_start = time_ms();
    simaai::neat::Tensor first;
    try {
      first = cam.pull_tensor_or_throw(5000);
    } catch (const std::exception& e) {
      std::string msg = e.what();
      if (msg.find("timeout") != std::string::npos) {
        throw std::runtime_error(
            "Framerate could not be derived from SDP or timestamps, please set it up for Rtsp");
      }
      throw;
    }
    const double first_pull_end = time_ms();
    const double first_pull_ms = first_pull_end - first_pull_start;
    const double first_pull_ts = first_pull_end;
    int frame_w = 0;
    int frame_h = 0;
    sima_examples::require(infer_dims(first, frame_w, frame_h), "first frame missing dimensions");
    if (frame_w == 1280 && frame_h == 720 && cam_opt.h264_width <= 0 && cam_opt.h264_height <= 0 &&
        cam_opt.fallback_h264_width <= 0 && cam_opt.fallback_h264_height <= 0) {
      std::fprintf(stderr, "[WARN] deriving width=1280 and height=720 from SDP or timestamp\n");
    }

    const bool use_optiview = cfg.optiview;
    const bool use_udp = cfg.udp || cfg.optiview;
    const std::string udp_host = use_optiview ? cfg.optiview_host : "127.0.0.1";
    const int udp_port = use_optiview ? cfg.optiview_video_port : 5000;
    fs::path out_path;
    cv::VideoWriter writer;
    simaai::neat::Run udp_run;
    std::unique_ptr<sima_examples::OptiViewSender> optiview_sender;
    std::vector<std::string> optiview_labels;

    if (use_udp) {
      simaai::neat::Session udp;
      simaai::neat::InputOptions udp_src;
      udp_src.format = "NV12";
      udp_src.width = frame_w;
      udp_src.height = frame_h;
      udp_src.caps_override = "video/x-raw,format=NV12,width=" + std::to_string(frame_w) +
                              ",height=" + std::to_string(frame_h) + ",framerate=30/1";
      udp_src.use_simaai_pool = false;
      udp.add(simaai::neat::nodes::Input(udp_src));
      udp.add(simaai::neat::nodes::H264EncodeSima(frame_w, frame_h, 30, 4000));
      udp.add(simaai::neat::nodes::H264Parse());
      udp.add(simaai::neat::nodes::H264Packetize(96, 1));
      simaai::neat::UdpOutputOptions udp_opt;
      udp_opt.host = udp_host;
      udp_opt.port = udp_port;
      udp.add(simaai::neat::nodes::UdpOutput(udp_opt));

      simaai::neat::Tensor udp_dummy;
      std::string udp_err;
      sima_examples::require(make_blank_nv12_tensor(frame_w, frame_h, udp_dummy, udp_err), udp_err);
      simaai::neat::RunOptions udp_run_opt;
      udp_run_opt.enable_metrics = cfg.optiview;
      udp_run = udp.build(udp_dummy, simaai::neat::RunMode::Async, udp_run_opt);
      std::cout << "udp=" << udp_host << ":" << udp_port << "\n";
    } else {
      out_path = fs::path("out") / "stream_0.mp4";
      std::error_code out_ec;
      fs::create_directories(out_path.parent_path(), out_ec);
      fs::remove(out_path, out_ec);
      std::string writer_err;
      sima_examples::require(sima_examples::open_h264_writer(writer, out_path, frame_w, frame_h,
                                                             30.0, 4000, &writer_err),
                             writer_err);
    }

    if (use_optiview) {
      sima_examples::OptiViewOptions opt;
      opt.host = cfg.optiview_host;
      opt.channel = 0;
      opt.video_port_base = cfg.optiview_video_port;
      opt.json_port_base = cfg.optiview_json_port;
      std::string opt_err;
      optiview_sender = std::make_unique<sima_examples::OptiViewSender>(opt, &opt_err);
      sima_examples::require(optiview_sender->ok(), opt_err);
      optiview_labels = sima_examples::optiview_default_labels();
      std::cout << "optiview host=" << optiview_sender->host()
                << " video_port=" << optiview_sender->video_port()
                << " json_port=" << optiview_sender->json_port() << " channel=0"
                << " offset_ms=" << cfg.optiview_offset_ms
                << " fifo=" << (cfg.optiview_fifo ? "1" : "0") << "\n";
    }

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "NV12";
    model_opt.input_max_width = frame_w;
    model_opt.input_max_height = frame_h;
    model_opt.input_max_depth = 1;
    simaai::neat::Model model(cfg.mpk, model_opt);
    simaai::neat::Session yolo;
    simaai::neat::Model::SessionOptions session_opt;
    session_opt.include_appsrc = true;
    session_opt.include_appsink = false;
    yolo.add(model.session(session_opt));
    const int topk = 100;
    const float min_score = 0.52f;
    yolo.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", frame_w, frame_h, min_score, 0.5f,
                                                topk));
    yolo.add(simaai::neat::nodes::Output());
    simaai::neat::RunOptions det_run_opt;
    det_run_opt.enable_metrics = cfg.optiview;
    auto det = yolo.build(first, simaai::neat::RunMode::Async, det_run_opt);

    if (!use_udp && cfg.forever) {
      sima_examples::require(false, "Video output cannot run forever; pass --frames N");
    }
    if (cfg.frames_set && cfg.frames <= 0) {
      sima_examples::require(false, "--frames must be > 0");
    }

    std::optional<int> frame_limit;
    if (cfg.frames_set) {
      frame_limit = cfg.frames;
    } else if (cfg.forever || use_udp) {
      frame_limit = std::nullopt;
    } else {
      frame_limit = cfg.frames;
    }
    const char* mode = use_optiview ? "optiview" : (use_udp ? "udp" : "video");
    std::cout << "mode=" << mode
              << " frame_limit=" << (frame_limit ? std::to_string(*frame_limit) : "inf")
              << " frames_set=" << (cfg.frames_set ? "1" : "0")
              << " forever=" << (cfg.forever ? "1" : "0") << "\n";

    FrameQueue queue(300);
    ProducerTiming producer_stats;
    ConsumerTiming consumer_stats;
    std::atomic<bool> stop{false};
    std::atomic<int> saved{0};
    std::atomic<int> det_outputs{0};
    double producer_start_ms = 0.0;
    double producer_end_ms = 0.0;
    double consumer_start_ms = 0.0;
    double consumer_end_ms = 0.0;

    std::thread producer([&]() {
      producer_start_ms = time_ms();
      int produced = 0;
      bool use_first = true;
      while (!stop.load() && (!frame_limit || produced < *frame_limit)) {
        simaai::neat::Tensor frame;
        double pull_ms = 0.0;
        double pull_ts = 0.0;
        if (use_first) {
          frame = std::move(first);
          use_first = false;
          pull_ms = first_pull_ms;
          pull_ts = first_pull_ts;
        } else {
          const double t0 = time_ms();
          auto frame_opt = cam.pull_tensor();
          if (!frame_opt.has_value())
            continue;
          const double t1 = time_ms();
          frame = std::move(*frame_opt);
          pull_ms = t1 - t0;
          pull_ts = t1;
        }
        print_time("rtsp_pull_ms", pull_ms, cfg.debug);
        producer_stats.add_rtsp_pull(pull_ms);

        FrameItem item;
        item.index = produced;
        item.frame = std::move(frame);
        item.pull_ts_ms = pull_ts;

        const double q0 = time_ms();
        if (!queue.push(std::move(item)))
          break;
        const double q1 = time_ms();
        const double queue_ms = q1 - q0;
        print_time("queue_push_ms", queue_ms, cfg.debug);
        producer_stats.add_queue_push(queue_ms);

        produced += 1;
        producer_stats.count = produced;
      }
      queue.close();
      producer_end_ms = time_ms();
    });

    std::thread consumer([&]() {
      consumer_start_ms = time_ms();
      int out_pulls = 0;
      const bool use_fifo = cfg.optiview_fifo;
      std::deque<PendingFrame> inflight;
      while (!stop.load() && (!frame_limit || saved.load() < *frame_limit)) {
        FrameItem item;
        const double q0 = time_ms();
        if (!queue.pop(item))
          break;
        const double q1 = time_ms();
        const double queue_ms = q1 - q0;
        print_time("queue_pop_ms", queue_ms, cfg.debug);
        consumer_stats.add_queue_pop(queue_ms);

        PendingFrame pending_current;
        pending_current.index = item.index;
        pending_current.pull_ts_ms = item.pull_ts_ms;
        pending_current.frame = std::move(item.frame);

        if (!use_optiview) {
          const double t_convert0 = time_ms();
          std::string bgr_err;
          if (!nv12_to_bgr(pending_current.frame, pending_current.bgr, bgr_err)) {
            std::cerr << "[warn] nv12->bgr failed: " << bgr_err << "\n";
            continue;
          }
          const double t_convert1 = time_ms();
          const double convert_ms = t_convert1 - t_convert0;
          print_time("nv12_to_bgr_ms", convert_ms, cfg.debug);
          consumer_stats.add_convert(convert_ms);
        }

        const double t_push0 = time_ms();
        const bool pushed = det.push(pending_current.frame);
        const double t_push1 = time_ms();
        const double push_ms = t_push1 - t_push0;
        print_time("yolo_push_ms", push_ms, cfg.debug);
        consumer_stats.add_yolo_push(push_ms);
        if (!pushed) {
          std::cerr << "[warn] push failed\n";
          continue;
        }
        if (use_fifo) {
          inflight.push_back(std::move(pending_current));
        }

        const double t_pull0 = time_ms();
        auto out_opt = det.pull();
        const double t_pull1 = time_ms();
        const double pull_ms = t_pull1 - t_pull0;
        print_time("yolo_pull_ms", pull_ms, cfg.debug);
        consumer_stats.add_yolo_pull(pull_ms);
        if (!out_opt.has_value())
          continue;
        out_pulls += 1;
        det_outputs.store(out_pulls);
        if (cfg.debug && use_fifo) {
          std::cout << "[dbg] inflight=" << inflight.size() << "\n";
        }
        if (cfg.debug) {
          std::cout << "[dbg] det pull=" << out_pulls << " kind=" << static_cast<int>(out_opt->kind)
                    << " tag=" << out_opt->payload_tag << " format=" << out_opt->format
                    << " frame_id=" << out_opt->frame_id << " input_seq=" << out_opt->input_seq
                    << "\n";
        }

        PendingFrame pending;
        if (use_fifo) {
          if (inflight.empty()) {
            std::cerr << "[warn] inflight empty on output\n";
            continue;
          }
          pending = std::move(inflight.front());
          inflight.pop_front();
        } else {
          pending = std::move(pending_current);
        }

        const double t_extract0 = time_ms();
        std::vector<uint8_t> payload;
        std::string err;
        if (!sima_examples::extract_bbox_payload(*out_opt, payload, err)) {
          std::cerr << "[warn] bbox extract failed: " << err << "\n";
          continue;
        }
        const double t_extract1 = time_ms();
        const double extract_ms = t_extract1 - t_extract0;
        print_time("bbox_extract_ms", extract_ms, cfg.debug);
        consumer_stats.add_bbox_extract(extract_ms);

        const double t_parse0 = time_ms();
        std::vector<objdet::Box> boxes;
        try {
          boxes = objdet::parse_boxes_strict(payload, frame_w, frame_h, topk, false);
        } catch (const std::exception& ex) {
          std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
          continue;
        }
        const double t_parse1 = time_ms();
        const double parse_ms = t_parse1 - t_parse0;
        print_time("bbox_parse_ms", parse_ms, cfg.debug);
        consumer_stats.add_bbox_parse(parse_ms);

        std::vector<sima_examples::OptiViewObject> optiview_objects;
        if (use_optiview) {
          optiview_objects.reserve(boxes.size());
          for (const auto& box : boxes) {
            int x1 = static_cast<int>(box.x1);
            int y1 = static_cast<int>(box.y1);
            int w = static_cast<int>(box.x2 - box.x1);
            int h = static_cast<int>(box.y2 - box.y1);
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
            obj.score = box.score;
            obj.class_id = box.class_id;
            optiview_objects.push_back(obj);
          }
        } else {
          const double t_overlay0 = time_ms();
          objdet::draw_boxes(pending.bgr, boxes, min_score, cv::Scalar(0, 255, 0), "det");
          const double t_overlay1 = time_ms();
          const double overlay_ms = t_overlay1 - t_overlay0;
          print_time("overlay_ms", overlay_ms, cfg.debug);
          consumer_stats.add_overlay(overlay_ms);
        }

        if (cfg.debug) {
          std::cout << "boxes=" << boxes.size() << "\n";
        }

        double output_ts = 0.0;
        if (use_udp) {
          const double t_udp_conv0 = time_ms();
          simaai::neat::Tensor nv12_frame;
          std::string nv12_err;
          bool converted = false;
          if (use_optiview) {
            converted = nv12_copy_to_cpu_tensor(pending.frame, nv12_frame, nv12_err);
          } else {
            converted = bgr_to_nv12_tensor(pending.bgr, nv12_frame, nv12_err);
          }
          if (!converted) {
            std::cerr << "[warn] udp convert failed: " << nv12_err << "\n";
            continue;
          }
          const double t_udp_conv1 = time_ms();
          const double udp_conv_ms = t_udp_conv1 - t_udp_conv0;
          print_time(use_optiview ? "nv12_copy_ms" : "bgr_to_nv12_ms", udp_conv_ms, cfg.debug);
          consumer_stats.add_udp_convert(udp_conv_ms);

          const double t_udp_push0 = time_ms();
          if (!udp_run.push(nv12_frame)) {
            std::cerr << "[warn] udp push failed\n";
            continue;
          }
          const double t_udp_push1 = time_ms();
          const double udp_push_ms = t_udp_push1 - t_udp_push0;
          print_time("udp_push_ms", udp_push_ms, cfg.debug);
          consumer_stats.add_udp_push(udp_push_ms);
          output_ts = t_udp_push1;
          if (use_optiview) {
            const int64_t fid =
                out_opt->frame_id >= 0 ? out_opt->frame_id : static_cast<int64_t>(pending.index);
            const int64_t ts_ms = static_cast<int64_t>(output_ts + cfg.optiview_offset_ms);
            std::string json_payload = sima_examples::optiview_make_json(
                ts_ms, std::to_string(fid), optiview_objects, optiview_labels);
            std::string json_err;
            if (!optiview_sender->send_json(json_payload, &json_err)) {
              std::cerr << "[warn] optiview json send failed: " << json_err << "\n";
            }
          }
        } else {
          const double t_write0 = time_ms();
          writer.write(pending.bgr);
          const double t_write1 = time_ms();
          const double write_ms = t_write1 - t_write0;
          print_time("write_ms", write_ms, cfg.debug);
          consumer_stats.add_write(write_ms);
          output_ts = t_write1;
        }

        const double e2e_ms = output_ts - pending.pull_ts_ms;
        print_time("e2e_ms", e2e_ms, cfg.debug);
        consumer_stats.add_e2e(e2e_ms);

        const int saved_now = saved.fetch_add(1) + 1;
        consumer_stats.count = saved_now;
      }
      stop.store(true);
      queue.close();
      consumer_end_ms = time_ms();
    });

    if (producer.joinable())
      producer.join();
    if (consumer.joinable())
      consumer.join();

    if (!use_udp) {
      writer.release();
      std::cout << "saved=" << saved.load() << " video=" << out_path << "\n";
    } else {
      std::cout << "saved=" << saved.load() << " udp=" << udp_host << ":" << udp_port << "\n";
    }
    print_throughput_summary(producer_stats.count, det_outputs.load(), saved.load(),
                             producer_start_ms, producer_end_ms, consumer_start_ms,
                             consumer_end_ms);
    if (cfg.optiview) {
      print_stream_summary("rtsp", cam, true);
      print_stream_summary("yolo", det, true);
      print_pipeline_report("yolo", det, true);
      if (use_udp) {
        print_stream_summary("udp", udp_run, true);
        print_pipeline_report("udp", udp_run, true);
      }
    }
    producer_stats.print();
    consumer_stats.print();
    std::cerr << "[HOLD] pid=" << getpid() << " (sleeping 20s)\n";
    udp_run = simaai::neat::Run{};
    cam = simaai::neat::Run{};
    det = simaai::neat::Run{}; // drop pipelines/plugins
    std::this_thread::sleep_for(std::chrono::seconds(20));
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
