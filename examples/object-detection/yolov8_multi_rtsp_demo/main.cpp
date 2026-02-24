#include "support/runtime/example_utils.h"
#include "support/object_detection/obj_detection_utils.h"
#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"
#include <gst/gst.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace {
struct Config {
  std::string rtsp_list;
  int frames = 3000;
  bool debug = false;
  std::string optiview_host = "127.0.0.1";
  int optiview_video_port = 9000;
  int optiview_json_port = 9100;
};
struct Inflight {
  int64_t frame_id = -1;
  simaai::neat::Sample enc;
};
struct Pending {
  int stream = -1;
  simaai::neat::Sample enc;
};
struct Stream {
  int idx = 0;
  std::string url, stream_id;
  simaai::neat::Run enc, dec, fwd;
  std::unique_ptr<sima_examples::OptiViewSender> ov;
  std::deque<Inflight> inflight;
  size_t pushed = 0;
  bool closed = false;
  simaai::neat::Sample first;
};

static void die(bool ok, const std::string& msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
static void usage() {
  std::cerr << "Usage: ./yolov8_multi_rtsp_demo --rtsp-list "
            << sima_examples::default_rtsp_list_path().string() << " --frames 3000 [--debug]"
            << " [--optiview-host 127.0.0.1]"
            << " [--optiview-video-port 9000]"
            << " [--optiview-json-port 9100]\n";
}

static Config parse_config(int argc, char** argv) {
  Config cfg;
  cfg.rtsp_list = sima_examples::default_rtsp_list_path().string();
  std::string raw;
  if (sima_examples::get_arg(argc, argv, "--rtsp-list", raw))
    cfg.rtsp_list = raw;
  if (sima_examples::get_arg(argc, argv, "--frames", raw))
    cfg.frames = std::stoi(raw);
  if (sima_examples::get_arg(argc, argv, "--optiview-host", raw))
    cfg.optiview_host = raw;
  if (sima_examples::get_arg(argc, argv, "--optiview-video-port", raw))
    cfg.optiview_video_port = std::stoi(raw);
  if (sima_examples::get_arg(argc, argv, "--optiview-json-port", raw))
    cfg.optiview_json_port = std::stoi(raw);
  cfg.debug = sima_examples::has_flag(argc, argv, "--debug");
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rtsp-list" || arg == "--frames" || arg == "--optiview-host" ||
        arg == "--optiview-video-port" || arg == "--optiview-json-port") {
      if (i + 1 < argc)
        ++i;
      continue;
    }
    if (arg == "--debug")
      continue;
    throw std::runtime_error("Unknown arg: " + arg);
  }
  die(cfg.frames > 0, "--frames must be > 0");
  return cfg;
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static bool init_nv12_meta(simaai::neat::Tensor& t, int w, int h) {
  if (w <= 0 || h <= 0 || (w % 2) || (h % 2))
    return false;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {h, w};
  t.strides_bytes = {w, 1};
  t.byte_offset = 0;
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  simaai::neat::ImageSpec img;
  img.format = simaai::neat::ImageSpec::PixelFormat::NV12;
  t.semantic.image = img;
  simaai::neat::Plane y{.role = simaai::neat::PlaneRole::Y,
                        .shape = {h, w},
                        .strides_bytes = {w, 1},
                        .byte_offset = 0};
  simaai::neat::Plane uv{.role = simaai::neat::PlaneRole::UV,
                         .shape = {h / 2, w},
                         .strides_bytes = {w, 1},
                         .byte_offset = static_cast<int64_t>(w) * h};
  t.planes = {y, uv};
  return true;
}

static bool copy_nv12_cpu(const simaai::neat::Tensor& t, simaai::neat::Tensor& out) {
  if (!t.is_nv12())
    return false;
  int w = t.width();
  int h = t.height();
  if (w <= 0 || h <= 0)
    return false;
  std::vector<uint8_t> nv12 = t.copy_nv12_contiguous();
  if (nv12.empty())
    return false;
  auto storage = simaai::neat::make_cpu_owned_storage(nv12.size());
  std::memcpy(storage->data, nv12.data(), nv12.size());
  out = simaai::neat::Tensor{};
  out.storage = std::move(storage);
  return init_nv12_meta(out, w, h);
}

static bool find_bbox_payload(const simaai::neat::Sample& s, std::vector<uint8_t>& out) {
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& f : s.fields)
      if (find_bbox_payload(f, out))
        return true;
    return false;
  }
  if (s.kind != simaai::neat::SampleKind::Tensor || !s.tensor.has_value())
    return false;
  const std::string tag = !s.payload_tag.empty() ? s.payload_tag : s.format;
  if (tag != "BBOX" && tag != "bbox")
    return false;
  out = s.tensor->copy_payload_bytes();
  return !out.empty();
}

static simaai::neat::Tensor make_dummy_encoded(const simaai::neat::Tensor& ref) {
  size_t bytes = (ref.shape.size() == 1 && ref.shape[0] > 0)
                     ? static_cast<size_t>(ref.shape[0])
                     : static_cast<size_t>(ref.storage->size_bytes);
  if (bytes == 0)
    bytes = 1;
  simaai::neat::Tensor t;
  t.storage = simaai::neat::make_cpu_owned_storage(bytes);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {static_cast<int64_t>(bytes)};
  t.strides_bytes = {1};
  t.semantic.encoded = simaai::neat::EncodedSpec{};
  return t;
}
} // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_config(argc, argv);
    if (cfg.debug) {
      const char* p = std::getenv("GST_PLUGIN_PATH");
      const char* p1 = std::getenv("GST_PLUGIN_PATH_1_0");
      const char* r1 = std::getenv("GST_REGISTRY_1_0");
      std::cout << "[ENV] GST_PLUGIN_PATH=" << (p ? p : "") << "\n";
      std::cout << "[ENV] GST_PLUGIN_PATH_1_0=" << (p1 ? p1 : "") << "\n";
      std::cout << "[ENV] GST_REGISTRY_1_0=" << (r1 ? r1 : "") << "\n";
    }
    const auto urls = sima_examples::read_rtsp_list(cfg.rtsp_list);
    die(urls.size() <= 20, "Max supported streams is 20");

    const std::string mpk = sima_examples::resolve_yolov8s_tar(fs::current_path());
    die(!mpk.empty(), "Failed to locate yolo_v8s MPK tarball");

    const int kW = 1280, kH = 720;
    const std::string kHost = cfg.optiview_host;
    const int kVideoBase = cfg.optiview_video_port;
    const int kJsonBase = cfg.optiview_json_port;
    std::vector<Stream> streams;
    streams.reserve(urls.size());
    std::string ref_caps;
    simaai::neat::Tensor dummy_encoded;

    for (size_t i = 0; i < urls.size(); ++i) {
      Stream s;
      s.idx = static_cast<int>(i);
      s.url = urls[i];
      s.stream_id = "rtsp" + std::to_string(i);
      if (cfg.debug)
        std::cout << "[INIT] stream=" << s.idx << " url=" << s.url << "\n";

      simaai::neat::Session enc;
      enc.add(simaai::neat::nodes::RTSPInput(s.url, 200, true));
      enc.add(simaai::neat::nodes::H264Depacketize(96, -1, -1, -1, -1, false));
      enc.add(simaai::neat::nodes::H264CapsFixup(30, kW, kH));
      enc.add(simaai::neat::nodes::Output());
      s.enc = enc.build();
      if (cfg.debug)
        std::cout << "[PIPE] enc stream=" << s.idx << " " << enc.last_pipeline() << "\n";

      simaai::neat::PullError perr;
      auto st = s.enc.pull(5000, s.first, &perr);
      die(st == simaai::neat::PullStatus::Ok,
          perr.message.empty() ? "Failed to pull first encoded sample" : perr.message);
      die(s.first.tensor.has_value() && s.first.tensor->semantic.encoded.has_value() &&
              !s.first.caps_string.empty(),
          "First sample invalid");
      if (ref_caps.empty())
        ref_caps = s.first.caps_string;
      die(s.first.caps_string == ref_caps, "All streams must have identical caps");
      if (!dummy_encoded.storage)
        dummy_encoded = make_dummy_encoded(*s.first.tensor);

      const bool is_rtp = (ref_caps.find("application/x-rtp") != std::string::npos);
      simaai::neat::Session fwd;
      simaai::neat::InputOptions fsrc;
      fsrc.use_simaai_pool = false;
      fsrc.caps_override = ref_caps;
      fwd.add(simaai::neat::nodes::Input(fsrc));
      if (!is_rtp) {
        fwd.add(simaai::neat::nodes::H264Parse());
        fwd.add(simaai::neat::nodes::H264Packetize(96, 1));
      }
      simaai::neat::UdpOutputOptions udp_opt;
      udp_opt.host = kHost;
      udp_opt.port = kVideoBase + s.idx;
      fwd.add(simaai::neat::nodes::UdpOutput(udp_opt));
      s.fwd = fwd.build(dummy_encoded, simaai::neat::RunMode::Async);
      if (cfg.debug)
        std::cout << "[PIPE] fwd stream=" << s.idx << " " << fwd.last_pipeline() << "\n";

      simaai::neat::Session dec;
      simaai::neat::InputOptions dsrc;
      dsrc.use_simaai_pool = false;
      dsrc.caps_override = ref_caps;
      dec.add(simaai::neat::nodes::Input(dsrc));
      dec.add(simaai::neat::nodes::H264Parse());
      dec.add(simaai::neat::nodes::H264Decode(2, "NV12", "decoder", true));
      dec.add(simaai::neat::nodes::Output());

      simaai::neat::RunOptions run_opt;
      run_opt.queue_depth = 4;
      run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
      s.dec = dec.build(dummy_encoded, simaai::neat::RunMode::Async, run_opt);
      if (cfg.debug)
        std::cout << "[PIPE] dec stream=" << s.idx << " " << dec.last_pipeline() << "\n";

      sima_examples::OptiViewOptions opt;
      opt.host = kHost;
      opt.channel = s.idx;
      opt.video_port_base = kVideoBase;
      opt.json_port_base = kJsonBase;
      std::string opt_err;
      s.ov = std::make_unique<sima_examples::OptiViewSender>(opt, &opt_err);
      die(s.ov->ok(), opt_err);
      std::cout << "optiview host=" << s.ov->host() << " video_port=" << s.ov->video_port()
                << " json_port=" << s.ov->json_port() << " channel=" << s.idx << " fifo=1\n";

      streams.push_back(std::move(s));
    }

    const auto labels = sima_examples::optiview_default_labels();
    const int topk = 100;
    const float min_score = 0.52f;
    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "NV12";
    model_opt.input_max_width = kW;
    model_opt.input_max_height = kH;
    model_opt.input_max_depth = 1;
    simaai::neat::Model model(mpk, model_opt);

    simaai::neat::Session yolo;
    simaai::neat::InputOptions ysrc = model.input_appsrc_options(false);
    ysrc.format = "NV12";
    ysrc.width = kW;
    ysrc.height = kH;
    yolo.add(simaai::neat::nodes::Input(ysrc));
    yolo.add(simaai::neat::nodes::groups::Preprocess(model));
    yolo.add(simaai::neat::nodes::groups::Infer(model));
    yolo.add(simaai::neat::nodes::SimaBoxDecode(model, "yolov8", kW, kH, min_score, 0.5f, topk));
    yolo.add(simaai::neat::nodes::Output());

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 4;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
    simaai::neat::Tensor dummy_nv12;
    dummy_nv12.storage = simaai::neat::make_cpu_owned_storage(static_cast<size_t>(kW) *
                                                              static_cast<size_t>(kH) * 3 / 2);
    die(init_nv12_meta(dummy_nv12, kW, kH), "Invalid NV12 dims");
    auto det = yolo.build(dummy_nv12, simaai::neat::RunMode::Async, run_opt);
    if (cfg.debug)
      std::cout << "[PIPE] yolo " << yolo.last_pipeline() << "\n";

    const size_t inflight_max = 4;
    std::unordered_map<int64_t, Pending> pending;
    double start_ms = now_ms();
    double last_report_ms = start_ms;
    int64_t next_frame_id = 0;
    size_t det_out = 0;

    auto push_encoded = [&](Stream& s, simaai::neat::Sample&& enc) {
      if (s.closed || s.pushed >= static_cast<size_t>(cfg.frames)) {
        s.closed = true;
        return;
      }
      if (s.inflight.size() >= inflight_max)
        return;
      enc.frame_id = next_frame_id++;
      enc.stream_id = s.stream_id;
      if (!s.dec.try_push(enc))
        return;
      s.inflight.push_back({enc.frame_id, std::move(enc)});
      s.pushed++;
    };

    for (auto& s : streams)
      if (s.first.tensor.has_value()) {
        push_encoded(s, std::move(s.first));
        s.first = simaai::neat::Sample{};
      }

    auto all_done = [&]() {
      if (!pending.empty())
        return false;
      for (const auto& s : streams)
        if (!s.closed || !s.inflight.empty())
          return false;
      return true;
    };

    while (true) {
      for (auto& s : streams) {
        if (s.closed)
          continue;
        if (s.pushed >= static_cast<size_t>(cfg.frames)) {
          s.closed = true;
          continue;
        }
        if (s.inflight.size() >= inflight_max)
          continue;
        simaai::neat::Sample enc_msg;
        simaai::neat::PullError perr;
        auto st = s.enc.pull(0, enc_msg, &perr);
        if (st == simaai::neat::PullStatus::Timeout)
          continue;
        if (st == simaai::neat::PullStatus::Closed) {
          s.closed = true;
          continue;
        }
        die(st == simaai::neat::PullStatus::Ok,
            perr.message.empty() ? "Encoded pull failed" : perr.message);
        push_encoded(s, std::move(enc_msg));
      }

      for (auto& s : streams) {
        while (true) {
          simaai::neat::Sample dec_out;
          simaai::neat::PullError derr;
          auto st = s.dec.pull(0, dec_out, &derr);
          if (st == simaai::neat::PullStatus::Timeout || st == simaai::neat::PullStatus::Closed)
            break;
          die(st == simaai::neat::PullStatus::Ok,
              derr.message.empty() ? "Decode pull failed" : derr.message);
          if (s.inflight.empty()) {
            std::cerr << "[warn] decoder output with empty inflight (stream=" << s.idx << ")\n";
            continue;
          }
          Inflight in = std::move(s.inflight.front());
          s.inflight.pop_front();

          simaai::neat::Sample ymsg = dec_out;
          ymsg.frame_id = in.frame_id;
          ymsg.stream_id = s.stream_id;
          simaai::neat::Tensor nv12_copy;
          if (!copy_nv12_cpu(*ymsg.tensor, nv12_copy)) {
            std::cerr << "[warn] nv12 copy failed\n";
            continue;
          }
          ymsg.tensor = std::move(nv12_copy);
          ymsg.owned = true;
          if (ymsg.tensor->storage)
            ymsg.tensor->storage->holder.reset();
          if (det.try_push(ymsg))
            pending[ymsg.frame_id] = Pending{s.idx, std::move(in.enc)};
        }
      }

      while (true) {
        simaai::neat::Sample out_msg;
        simaai::neat::PullError derr;
        auto st = det.pull(0, out_msg, &derr);
        if (st == simaai::neat::PullStatus::Timeout || st == simaai::neat::PullStatus::Closed)
          break;
        die(st == simaai::neat::PullStatus::Ok,
            derr.message.empty() ? "Det pull failed" : derr.message);
        det_out++;

        auto it = pending.find(out_msg.frame_id);
        if (it == pending.end())
          continue;
        int sidx = it->second.stream;
        simaai::neat::Sample enc_sample = std::move(it->second.enc);
        pending.erase(it);
        if (sidx >= 0 && sidx < static_cast<int>(streams.size()))
          (void)streams[sidx].fwd.try_push(enc_sample);

        std::vector<uint8_t> payload;
        if (!find_bbox_payload(out_msg, payload)) {
          std::cerr << "[warn] bbox extract failed\n";
          continue;
        }
        std::vector<objdet::Box> boxes;
        try {
          boxes = objdet::parse_boxes_strict(payload, kW, kH, topk, false);
        } catch (const std::exception& ex) {
          std::cerr << "[warn] bbox parse failed: " << ex.what() << "\n";
          continue;
        }

        if (sidx >= 0 && sidx < static_cast<int>(streams.size())) {
          auto& s = streams[sidx];
          std::vector<sima_examples::OptiViewObject> objs;
          objs.reserve(boxes.size());
          for (const auto& b : boxes) {
            sima_examples::OptiViewObject obj;
            obj.x = static_cast<int>(b.x1);
            obj.y = static_cast<int>(b.y1);
            obj.w = static_cast<int>(b.x2 - b.x1);
            obj.h = static_cast<int>(b.y2 - b.y1);
            obj.score = b.score;
            obj.class_id = b.class_id;
            objs.push_back(obj);
          }
          std::string json = sima_examples::optiview_make_json(
              static_cast<int64_t>(now_ms()), std::to_string(out_msg.frame_id), objs, labels);
          std::string json_err;
          if (!s.ov->send_json(json, &json_err))
            std::cerr << "[warn] optiview json send failed: " << json_err << "\n";
        }
      }

      if (all_done())
        break;
      if (cfg.debug && (now_ms() - last_report_ms) > 2000.0) {
        std::cout << "[DBG] t_ms=" << (now_ms() - start_ms) << " det_out=" << det_out
                  << " pending=" << pending.size() << "\n";
        last_report_ms = now_ms();
      }
    }

    std::cout << "done det_out=" << det_out << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[ERR] " << ex.what() << "\n";
    usage();
    return 1;
  }
}
