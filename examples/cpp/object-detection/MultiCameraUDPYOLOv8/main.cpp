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
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

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
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  sima_examples::get_arg(argc, argv, "--rtsp", cfg.url);
  sima_examples::get_arg(argc, argv, "--mpk", cfg.mpk);
  cfg.frames_set = parse_int_arg(argc, argv, "--frames", cfg.frames);
  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  cfg.udp = sima_examples::has_flag(argc, argv, "--udp");
  cfg.forever = sima_examples::has_flag(argc, argv, "--forever");
  return cfg;
}

using sima_examples::bgr_to_nv12_tensor;
using sima_examples::infer_dims;
using sima_examples::init_nv12_tensor_meta;
using sima_examples::make_blank_nv12_tensor;
using sima_examples::nv12_to_bgr;

void print_time(const char* label, double ms, bool enabled) {
  if (!enabled)
    return;
  std::cout << label << " " << ms << "\n";
}

struct FrameItem {
  int index = 0;
  simaai::neat::Tensor frame;
  double pull_ts_ms = 0.0;
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
    gst_init(nullptr, nullptr);

    Config cfg = parse_config(argc, argv);
    sima_examples::require(!cfg.url.empty(), "Missing --rtsp <url>");
    if (cfg.mpk.empty())
      cfg.mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());
    sima_examples::require(!cfg.mpk.empty(), "Failed to locate yolo_v8s MPK tarball");

    simaai::neat::Session camera;
    simaai::neat::nodes::groups::RtspDecodedInputOptions cam_opt;
    cam_opt.url = cfg.url;
    cam_opt.decoder_name = "decoder";
    camera.add(simaai::neat::nodes::groups::RtspDecodedInput(cam_opt));
    camera.add(simaai::neat::nodes::Output());
    auto cam = camera.build();

    const double first_pull_start = time_ms();
    simaai::neat::Tensor first = cam.pull_tensor_or_throw(5000);
    const double first_pull_end = time_ms();
    const double first_pull_ms = first_pull_end - first_pull_start;
    const double first_pull_ts = first_pull_end;
    int frame_w = 0;
    int frame_h = 0;
    sima_examples::require(infer_dims(first, frame_w, frame_h), "first frame missing dimensions");

    const bool use_udp = cfg.udp;
    const std::string udp_host = "127.0.0.1";
    const int udp_port = 5000;
    fs::path out_path;
    cv::VideoWriter writer;
    simaai::neat::Run udp_run;

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
      udp_run = udp.build(udp_dummy, simaai::neat::RunMode::Async);
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
    auto det = yolo.build(first, simaai::neat::RunMode::Async);

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
    std::cout << "mode=" << (use_udp ? "udp" : "video")
              << " frame_limit=" << (frame_limit ? std::to_string(*frame_limit) : "inf")
              << " frames_set=" << (cfg.frames_set ? "1" : "0")
              << " forever=" << (cfg.forever ? "1" : "0") << "\n";

    FrameQueue queue(300);
    ProducerTiming producer_stats;
    ConsumerTiming consumer_stats;
    std::atomic<bool> stop{false};
    std::atomic<int> saved{0};

    std::thread producer([&]() {
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
    });

    std::thread consumer([&]() {
      int out_pulls = 0;
      while (!stop.load() && (!frame_limit || saved.load() < *frame_limit)) {
        FrameItem item;
        const double q0 = time_ms();
        if (!queue.pop(item))
          break;
        const double q1 = time_ms();
        const double queue_ms = q1 - q0;
        print_time("queue_pop_ms", queue_ms, cfg.debug);
        consumer_stats.add_queue_pop(queue_ms);

        const double t_convert0 = time_ms();
        cv::Mat bgr;
        std::string bgr_err;
        if (!nv12_to_bgr(item.frame, bgr, bgr_err)) {
          std::cerr << "[warn] nv12->bgr failed: " << bgr_err << "\n";
          continue;
        }
        const double t_convert1 = time_ms();
        const double convert_ms = t_convert1 - t_convert0;
        print_time("nv12_to_bgr_ms", convert_ms, cfg.debug);
        consumer_stats.add_convert(convert_ms);

        const double t_push0 = time_ms();
        const bool pushed = det.push(item.frame);
        const double t_push1 = time_ms();
        const double push_ms = t_push1 - t_push0;
        print_time("yolo_push_ms", push_ms, cfg.debug);
        consumer_stats.add_yolo_push(push_ms);
        if (!pushed) {
          std::cerr << "[warn] push failed\n";
          continue;
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
        if (cfg.debug) {
          std::cout << "[dbg] det pull=" << out_pulls << " kind=" << static_cast<int>(out_opt->kind)
                    << " tag=" << out_opt->payload_tag << " format=" << out_opt->format
                    << " frame_id=" << out_opt->frame_id << " input_seq=" << out_opt->input_seq
                    << "\n";
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

        const double t_overlay0 = time_ms();
        objdet::draw_boxes(bgr, boxes, min_score, cv::Scalar(0, 255, 0), "det");
        const double t_overlay1 = time_ms();
        const double overlay_ms = t_overlay1 - t_overlay0;
        print_time("overlay_ms", overlay_ms, cfg.debug);
        consumer_stats.add_overlay(overlay_ms);

        std::cout << "boxes=" << boxes.size() << "\n";

        double output_ts = 0.0;
        if (use_udp) {
          const double t_udp_conv0 = time_ms();
          simaai::neat::Tensor nv12_frame;
          std::string nv12_err;
          if (!bgr_to_nv12_tensor(bgr, nv12_frame, nv12_err)) {
            std::cerr << "[warn] bgr->nv12 failed: " << nv12_err << "\n";
            continue;
          }
          const double t_udp_conv1 = time_ms();
          const double udp_conv_ms = t_udp_conv1 - t_udp_conv0;
          print_time("bgr_to_nv12_ms", udp_conv_ms, cfg.debug);
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
        } else {
          const double t_write0 = time_ms();
          writer.write(bgr);
          const double t_write1 = time_ms();
          const double write_ms = t_write1 - t_write0;
          print_time("write_ms", write_ms, cfg.debug);
          consumer_stats.add_write(write_ms);
          output_ts = t_write1;
        }

        const double e2e_ms = output_ts - item.pull_ts_ms;
        print_time("e2e_ms", e2e_ms, cfg.debug);
        consumer_stats.add_e2e(e2e_ms);

        const int saved_now = saved.fetch_add(1) + 1;
        consumer_stats.count = saved_now;
      }
      stop.store(true);
      queue.close();
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
    producer_stats.print();
    consumer_stats.print();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
