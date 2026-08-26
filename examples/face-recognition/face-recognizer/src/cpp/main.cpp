#include "neat.h"
#include "support/runtime/config_utils.h"

#include "scrfd_decode.h"
#include "align.h"
#include "gallery.h"
#include "match.h"
#include "overlay.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "neat/node_groups.h"
#include "neat/nodes.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;
using Ms     = std::chrono::duration<double, std::milli>;

// ── config ────────────────────────────────────────────────────────────────────

struct AppConfig {
    std::string scrfd_model  = "models/scrfd_2.5g_bnkps.mla_mpk.tar.gz";
    std::string arcface_model= "models/w600k_mbf.surgery_mpk.tar.gz";
    std::string gallery_path = "gallery.bin";
    std::string input_uri;          // RTSP URL, video file path, or empty for webcam 0
    std::string output_sink;        // file path, "display", or empty (no output)
    std::string stream_host;        // host IP for overlay UDP stream; empty = no streaming
    int  stream_port         = 5000;
    int  max_frames          = 0;   // 0 = unlimited
    int  rtsp_fps            = -1;  // decoder caps fps hint; -1 = auto-detect from stream
    int  supported_fps       = 45;  // max input FPS the pipeline can sustain (warn above this)
    int  timeout_ms          = 20000;
    int  queue_depth         = 8;
    int  recog_interval      = 5;
    bool is_live             = false; // inferred from URI
    bool test_mode           = false;
    bool show_display        = false;
    bool force_cpu_preproc   = false; // --cpu-preproc: use A65 NEON preproc even for RTSP (A/B vs EV74 CVU)
    bool output_sink_explicit = false; // true when --output/--test/--no-display set output_sink from CLI

    face_recog::ScrfdConfig  scrfd;
    face_recog::MatchConfig  match;
    face_recog::OverlayConfig overlay;
};

static bool looks_live(const std::string& uri) {
    return uri.rfind("rtsp://", 0) == 0 || uri.rfind("rtsps://", 0) == 0 ||
           uri == "0" || uri.empty();
}

static AppConfig load_config(const fs::path& path) {
    const auto raw = sima_examples::ScalarConfig::load(path);
    AppConfig cfg;

    cfg.scrfd_model   = raw.string_or("scrfd.model",   cfg.scrfd_model);
    cfg.arcface_model = raw.string_or("arcface.model", cfg.arcface_model);
    cfg.gallery_path  = raw.string_or("gallery.path",  cfg.gallery_path);
    cfg.input_uri     = raw.string_or("input.uri",     "");
    cfg.output_sink   = raw.string_or("output.sink",   "");
    cfg.timeout_ms      = raw.int_or("runtime.timeout_ms",     cfg.timeout_ms);
    cfg.queue_depth     = raw.int_or("runtime.queue_depth",    cfg.queue_depth);
    cfg.recog_interval  = raw.int_or("runtime.recog_interval", cfg.recog_interval);
    cfg.supported_fps   = raw.int_or("runtime.supported_fps",  cfg.supported_fps);

    cfg.scrfd.conf_threshold  = static_cast<float>(raw.double_or("scrfd.conf_threshold", cfg.scrfd.conf_threshold));
    cfg.scrfd.nms_iou         = static_cast<float>(raw.double_or("scrfd.nms_iou",        cfg.scrfd.nms_iou));
    cfg.scrfd.top_k           = raw.int_or("scrfd.top_k",      cfg.scrfd.top_k);
    cfg.scrfd.keep_top_k      = raw.int_or("scrfd.keep_top_k", cfg.scrfd.keep_top_k);
    cfg.scrfd.cls_per_anchor  = raw.int_or("scrfd.cls_per_anchor",  cfg.scrfd.cls_per_anchor);
    cfg.scrfd.num_anchors     = raw.int_or("scrfd.num_anchors",     cfg.scrfd.num_anchors);
    cfg.scrfd.scale_by_stride = raw.bool_or("scrfd.scale_by_stride", cfg.scrfd.scale_by_stride);

    cfg.match.threshold     = static_cast<float>(raw.double_or("match.threshold", cfg.match.threshold));
    cfg.match.margin        = static_cast<float>(raw.double_or("match.margin",    cfg.match.margin));
    cfg.match.unknown_label = raw.string_or("match.unknown_label", cfg.match.unknown_label);

    cfg.overlay.draw_landmarks = raw.bool_or("overlay.draw_landmarks", cfg.overlay.draw_landmarks);
    cfg.overlay.draw_score     = raw.bool_or("overlay.draw_score",     cfg.overlay.draw_score);

    cfg.is_live = looks_live(cfg.input_uri);
    return cfg;
}

static AppConfig parse_args(int argc, char** argv) {
    fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
    AppConfig cfg;  // defaults; overridden by config then CLI flags

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--input"      && i + 1 < argc) { cfg.input_uri      = argv[++i]; }
        else if (arg == "--gallery"      && i + 1 < argc) { cfg.gallery_path   = argv[++i]; }
        else if (arg == "--scrfd-model"  && i + 1 < argc) { cfg.scrfd_model    = argv[++i]; }
        else if (arg == "--arcface-model"&& i + 1 < argc) { cfg.arcface_model  = argv[++i]; }
        else if (arg == "--output"       && i + 1 < argc) { cfg.output_sink = argv[++i]; cfg.output_sink_explicit = true; }
        else if (arg == "--stream-host"  && i + 1 < argc) { cfg.stream_host    = argv[++i]; }
        else if (arg == "--stream-port"  && i + 1 < argc) { cfg.stream_port    = std::stoi(argv[++i]); }
        else if (arg == "--max-frames"   && i + 1 < argc) { cfg.max_frames     = std::stoi(argv[++i]); }
        else if (arg == "--rtsp-fps"     && i + 1 < argc) { cfg.rtsp_fps       = std::stoi(argv[++i]); }
        else if (arg == "--cpu-preproc") { cfg.force_cpu_preproc = true; }
        else if (arg == "--test")   { cfg.test_mode = true; cfg.show_display = false; cfg.output_sink = ""; cfg.output_sink_explicit = true; }
        else if (arg == "--no-display") { cfg.show_display = false; cfg.output_sink = ""; cfg.output_sink_explicit = true; }
        else if (arg == "--display")    { cfg.show_display = true;  }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: face-recognizer [--config <path>] [--input <uri>]\n"
                      << "       [--gallery <path>] [--scrfd-model <path>] [--arcface-model <path>]\n"
                      << "       [--output <sink>] [--stream-host <host>] [--stream-port N]\n"
                      << "       [--test] [--display] [--no-display]\n"
                      << "\n"
                      << "  --stream-host HOST  Burn overlay onto frames and stream H.264 over UDP.\n"
                      << "                      View with: ffplay -fflags nobuffer udp://@:PORT\n"
                      << "                               or vlc udp://@:PORT\n"
                      << "  --stream-port N     UDP port for the overlay stream (default 5000).\n"
                      << "  --rtsp-fps N        Optional decoder FPS override. Omit to auto-detect\n"
                      << "                      the source rate from the stream (recommended).\n"
                      << "  --cpu-preproc       Force A65 NEON preproc for RTSP instead of the EV74\n"
                      << "                      CVU path. Slower, but useful to A/B detection accuracy.\n";
            std::exit(0);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    auto yaml_cfg = load_config(config_path);
    if (cfg.input_uri.empty())    cfg.input_uri    = yaml_cfg.input_uri;
    if (cfg.gallery_path.empty() || cfg.gallery_path == "gallery.bin")
        cfg.gallery_path = yaml_cfg.gallery_path;
    if (!cfg.output_sink_explicit && cfg.output_sink.empty()) cfg.output_sink = yaml_cfg.output_sink;
    if (cfg.scrfd_model.empty())   cfg.scrfd_model   = yaml_cfg.scrfd_model;
    if (cfg.arcface_model.empty()) cfg.arcface_model = yaml_cfg.arcface_model;
    cfg.timeout_ms      = yaml_cfg.timeout_ms;
    cfg.queue_depth     = yaml_cfg.queue_depth;
    cfg.recog_interval  = yaml_cfg.recog_interval;
    cfg.supported_fps   = yaml_cfg.supported_fps;
    cfg.scrfd         = yaml_cfg.scrfd;
    cfg.match         = yaml_cfg.match;
    cfg.overlay       = yaml_cfg.overlay;
    cfg.is_live       = looks_live(cfg.input_uri);
    if (cfg.show_display && cfg.output_sink.empty()) cfg.output_sink = "display";
    return cfg;
}

// ── per-stage timing accumulator ─────────────────────────────────────────────

struct Timings {
    std::vector<double> preproc_ms;
    std::vector<double> pull_ms;            // blocking wait for next frame from source
    std::vector<double> preproc_compute_ms; // actual copy+resize+normalize (CPU work)
    std::vector<double> scrfd_ms;
    std::vector<double> decode_ms;
    std::vector<double> align_ms;
    std::vector<double> arcface_ms;
    std::vector<double> match_ms;
    std::vector<double> overlay_ms;
    std::vector<double> e2e_ms;
};

static void print_stage(const char* name, const std::vector<double>& v) {
    if (v.empty()) { printf("  %-18s  — no samples —\n", name); return; }
    const double sum = std::accumulate(v.begin(), v.end(), 0.0);
    const double mean = sum / v.size();
    auto sorted = v;
    std::sort(sorted.begin(), sorted.end());
    printf("  %-18s  mean=%6.2f ms  p50=%6.2f ms  p95=%6.2f ms  p99=%6.2f ms\n",
           name, mean,
           sorted[sorted.size() / 2],
           sorted[(sorted.size() - 1) * 95 / 100],
           sorted[(sorted.size() - 1) * 99 / 100]);
}

static void print_timings(const Timings& t, int frames, double total_s) {
    printf("\n═══ Performance report (%d frames, %.2f s) ═══\n", frames, total_s);
    printf("  End-to-end FPS: %.1f\n\n", frames / total_s);
    print_stage("Preproc(SCRFD)",    t.preproc_ms);
    print_stage("  ├ pull (wait)",   t.pull_ms);
    print_stage("  └ compute",       t.preproc_compute_ms);
    print_stage("SCRFD infer",       t.scrfd_ms);
    print_stage("SCRFD decode+NMS",  t.decode_ms);
    print_stage("Align",             t.align_ms);
    print_stage("ArcFace infer",     t.arcface_ms);
    print_stage("Match",             t.match_ms);
    print_stage("Overlay",           t.overlay_ms);
    print_stage("Frame E2E",         t.e2e_ms);
}

// ── build inference pipelines ─────────────────────────────────────────────────

// use_ev74=true  (RTSP path): Preproc node runs on CVU — resize + colorconvert +
//   normalize + quantize + tessellate all happen on EV74 hardware.  The NV12
//   tensor produced by neatdecoder is passed directly; no CPU copy or conversion.
//   seed_nv12 must be a real NV12 tensor (from an already-running RTSP source run)
//   so graph.build() can negotiate GStreamer caps from actual hardware buffer format.
//
// use_ev74=false (file/webcam): classic QuantTess path; caller preprocesses to
//   float32 RGB on CPU and quantizes on EV74 via QuantTess.
static simaai::neat::Run build_scrfd_run(const AppConfig& cfg, bool use_ev74,
                                         const simaai::neat::Tensor* seed_nv12 = nullptr) {
    simaai::neat::Model::Options opt;

    if (use_ev74) {
        opt.preprocess.kind   = simaai::neat::InputKind::Image;
        opt.preprocess.enable = simaai::neat::AutoFlag::On;
        opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
        // Upper bounds on SOURCE frame size (RTSP fallback = 1280×720).
        opt.preprocess.input_max_width  = seed_nv12 ? seed_nv12->width()  : 1280;
        opt.preprocess.input_max_height = seed_nv12 ? seed_nv12->height() : 720;
        opt.preprocess.input_max_depth  = 3;

        // ── These two MUST be pinned to match the A65 CPU path and the model's
        //    calibration.  Leaving either on Auto silently degrades detection.
        //
        // normalize: Auto resolves to OFF here, so the CVU handed the MLA raw
        //   [0,255] RGB instead of [0,1] — a 255x input-scale error, despite
        //   0_preproc.json baking in "normalize": true.  This is the dominant bug:
        //   spurious detections outscored the real face in every frame.
        // pad_value: ResizeSpec defaults to 114 (YOLO gray), but SCRFD was calibrated
        //   with BLACK bars and the CPU NEON path memsets padding to 0.  At
        //   1280x720 -> 640x640 the bars are 280 of 640 rows (~44% of the image).
        //
        // Measured on a fixed 1000-frame clip (>1-face rate; 1 real face present):
        //   pad=114 norm=auto (was)  50.8%      pad=0 norm=auto   56.0%
        //   pad=114 norm=On          19.7%      pad=0  norm=On    11.5%
        //   A65 CPU NEON reference   11.4%
        opt.preprocess.resize.enable    = simaai::neat::AutoFlag::On;
        opt.preprocess.resize.mode      = simaai::neat::ResizeMode::Letterbox;
        opt.preprocess.resize.pad_value = 0;
        opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
        opt.preprocess.normalize.mean   = {0.0f, 0.0f, 0.0f};
        opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
    } else {
        opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
        opt.preprocess.input_max_width  = cfg.scrfd.infer_w;
        opt.preprocess.input_max_height = cfg.scrfd.infer_h;
        opt.preprocess.input_max_depth  = 3;
    }

    simaai::neat::Model model(cfg.scrfd_model, opt);

    simaai::neat::Graph graph;

    if (use_ev74) {
        // Use graph.add(model) so the SDK handles the full route:
        //   neatprocesscvu (CVU Preproc, NV12→INT8) + MLA + neatprocesscvu (DetessDequant).
        // Manually adding Preproc + MLA + DetessDequant nodes with InputKind::Image
        // causes "unsupported model-bound post request" at graph.build() — the SDK
        // requires the model to own its full routing when InputKind::Image is set.
        graph.add(simaai::neat::nodes::Input(model.input_appsrc_options(false)));
        graph.add(model);
    } else {
        graph.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
        graph.add(model);
    }

    graph.add(simaai::neat::nodes::Output());

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth     = cfg.queue_depth;
    run_opt.overflow_policy = cfg.is_live
        ? simaai::neat::OverflowPolicy::DropIncoming
        : simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory   = simaai::neat::OutputMemory::Owned;

    if (use_ev74) {
        if (!seed_nv12)
            throw std::runtime_error("build_scrfd_run(ev74): seed_nv12 must not be null");
        // Use the real NV12 tensor from neatdecoder as the build seed so the SDK
        // negotiates GStreamer caps from the actual HW buffer format — avoids the
        // from_cv_mat() NV12 limitation entirely.
        return graph.build(simaai::neat::TensorList{*seed_nv12}, run_opt);
    } else {
        cv::Mat dummy(cfg.scrfd.infer_h, cfg.scrfd.infer_w, CV_32FC3, cv::Scalar(0, 0, 0));
        return graph.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(dummy)}, run_opt);
    }
}

static simaai::neat::Run build_arcface_run(const AppConfig& cfg) {
    simaai::neat::Model::Options opt;
    opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    opt.preprocess.input_max_width  = face_recog::kArcFaceW;
    opt.preprocess.input_max_height = face_recog::kArcFaceH;
    opt.preprocess.input_max_depth  = 3;

    simaai::neat::Model model(cfg.arcface_model, opt);

    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    graph.add(model);
    graph.add(simaai::neat::nodes::Output());

    cv::Mat dummy(face_recog::kArcFaceH, face_recog::kArcFaceW, CV_32FC3, cv::Scalar(0, 0, 0));
    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth   = 4;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    return graph.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(dummy)}, run_opt);
}

// ── ArcFace inference helper ──────────────────────────────────────────────────

static face_recog::Embedding run_arcface(
    simaai::neat::Run& arc_run,
    const cv::Mat& crop_bgr,
    int timeout_ms)
{
    const cv::Mat f32 = face_recog::preprocess_arcface_crop(crop_bgr);
    const auto tensor = face_recog::tensor_from_hwc_f32(f32);

    if (!arc_run.push(simaai::neat::TensorList{tensor}))
        throw std::runtime_error("arcface push failed: " + arc_run.last_error());

    const auto sample = arc_run.pull(timeout_ms);
    if (!sample.has_value())
        throw std::runtime_error("arcface pull timeout");

    const auto tensors = face_recog::collect_tensors(*sample);
    if (tensors.empty())
        throw std::runtime_error("arcface returned no tensors");

    // Output is [1, 512] or [512]; flatten to vector.
    return face_recog::tensor_to_f32(tensors[0]);
}

// ── main loop ─────────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};

int main(int argc, char** argv) {
    std::signal(SIGINT,  [](int){ g_stop = true; });
    std::signal(SIGTERM, [](int){ g_stop = true; });
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    // Required for neatdecoder NV12 buffers to be readable from EV74/CPU
    ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 1);
    ::setenv("GST_PLUGIN_PATH_1_0",
             "/usr/lib/aarch64-linux-gnu/neat/gst-plugins", 1);

    AppConfig cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (!fs::exists(cfg.scrfd_model))
        throw std::runtime_error("SCRFD model not found: " + cfg.scrfd_model);
    if (!fs::exists(cfg.arcface_model))
        throw std::runtime_error("ArcFace model not found: " + cfg.arcface_model);

    // gallery may not exist yet during enroll phase
    face_recog::Gallery gallery;
    if (fs::exists(cfg.gallery_path)) {
        gallery = face_recog::load_gallery(cfg.gallery_path);
        std::cout << "Gallery: " << gallery.entries.size() << " identities from "
                  << cfg.gallery_path << "\n";
    } else {
        std::cerr << "Warning: gallery not found at " << cfg.gallery_path
                  << " — all faces will be labelled Unknown\n";
    }

    const bool is_rtsp = cfg.input_uri.rfind("rtsp://",  0) == 0 ||
                         cfg.input_uri.rfind("rtsps://", 0) == 0;
    // For RTSP sources the HW decoder already produces NV12 buffers; route them
    // directly to the CVU Preproc node so resize+normalize+quantize+tessellate
    // run on EV74 instead of the A65.  File/webcam sources fall back to CPU preproc.
    const bool use_ev74_preproc = is_rtsp && !cfg.force_cpu_preproc;  // InputKind::Image → RoutePlanner sets use_preproc=true → no cast_0; CVU preproc (0_preproc.json) feeds EVXX_BFLOAT16 directly to MLA

    // RTSP: Neat SDK RtspDecodedInput + HW H.264 decode → NV12 frames.
    // File/webcam: cv::VideoCapture as before.
    // Overlay stream: SiMa HW H264EncodeSima + UdpH264OutputGroup → udpsink.
    std::optional<simaai::neat::Run>    rtsp_src_run;
    std::vector<uint8_t>                rtsp_nv12_buf;       // NV12 bytes for the frame being read in Phase A
    std::vector<uint8_t>                curr_nv12_buf;       // NV12 bytes for the frame being processed in Phase C
    std::optional<simaai::neat::Tensor> rtsp_nv12_tensor; // latest HW NV12 tensor (shared_ptr keeps HW buf alive)
    // Source-frame metadata from the most recent RTSP pull (for input-FPS detection).
    // pts_ns / frame_id count dropped frames too, so they recover the true source rate
    // even once the pipeline starts dropping; duration_ns is the exact per-frame period.
    int64_t last_pull_pts_ns      = -1;
    int64_t last_pull_duration_ns = -1;
    int64_t last_pull_frame_id    = -1;
    // Source frame rate, measured below. The H264 encoder and its caps must be told
    // this — a stale/hardcoded value makes the encoder's rate control and the RTP
    // timestamps disagree with the real delivery rate, which the receiver sees as a
    // low frame rate plus growing latency.
    double  detected_fps          = -1.0;
    cv::VideoCapture cap;
    std::function<bool(cv::Mat&)> read_frame;

    if (is_rtsp) {
        // Builds (or rebuilds) the RTSP source pipeline — called on startup and on reconnect.
        auto build_rtsp_src = [&]() {
            simaai::neat::nodes::groups::RtspDecodedInputOptions src_opt;
            src_opt.url                   = cfg.input_uri;
            src_opt.latency_ms            = 50;
            src_opt.tcp                   = true;
            src_opt.payload_type          = 96;
            src_opt.out_format            = simaai::neat::FormatTag::NV12;
            src_opt.decoder_name          = "decoder";
            src_opt.decoder_raw_output    = true;
            src_opt.auto_caps_from_stream = true;
            src_opt.fallback_h264_width   = 1280;
            src_opt.fallback_h264_height  = 720;
            // -1 (default) → let the decoder derive the framerate from the live stream
            // caps (auto_caps_from_stream).  Injecting a fixed fps that mismatches the
            // stream (e.g. 30 against a 60 FPS source) fails caps negotiation, so only
            // set a fallback when the user explicitly overrides via --rtsp-fps.
            src_opt.fallback_h264_fps     = (cfg.rtsp_fps > 0) ? cfg.rtsp_fps : -1;

            // Flat linear graph (no named subgraph) so Graph::build() produces a
            // "simple linear plan" that uses direct appsink pull instead of the SDK's
            // internal GraphSinkQueue (capacity=256).  The named-output connect() path
            // routes through that queue and fires "sink backpressure timeout" after
            // exactly 256 frames regardless of drop policy on the appsink.
            simaai::neat::OutputOptions live_out;
            live_out.max_buffers = 1;
            live_out.drop        = true;
            live_out.sync        = false;

            simaai::neat::Graph src_graph;
            src_graph.add(simaai::neat::nodes::groups::RtspDecodedInput(src_opt));
            src_graph.add(simaai::neat::nodes::Output(live_out));

            simaai::neat::RunOptions src_run_opt;
            src_run_opt.preset                              = simaai::neat::RunPreset::Realtime;
            src_run_opt.queue_depth                         = 4;
            src_run_opt.overflow_policy                     = simaai::neat::OverflowPolicy::KeepLatest;
            src_run_opt.output_memory                       = simaai::neat::OutputMemory::ZeroCopy;
            src_run_opt.advanced.prepare_output_cpu_visible = true;

            if (rtsp_src_run.has_value()) rtsp_src_run->close();
            rtsp_src_run.emplace(src_graph.build(src_run_opt));
        };

        std::cout << "[BUILD] Building RTSP source pipeline...\n";
        build_rtsp_src();
        std::cout << "[BUILD] RTSP source ready.\n";

        // BGR conversion deferred to Phase C so it overlaps SCRFD_next on MLA.
        // `f` is a size-only placeholder; BGR is produced later via nv12_to_bgr().
        auto pull_rtsp_frame = [&](cv::Mat& f) -> bool {
            simaai::neat::Sample sample;
            simaai::neat::PullError pull_err;
            auto status = rtsp_src_run->pull(cfg.timeout_ms, sample, &pull_err);

            if (status == simaai::neat::PullStatus::Closed) {
                std::cerr << "[RTSP] Stream closed.\n";
                return false;
            }
            if (status == simaai::neat::PullStatus::Error &&
                pull_err.code == "runtime.pull") {
                std::cerr << "[RTSP] Stream ended (EOS flush timeout).\n";
                return false;
            }
            if (status != simaai::neat::PullStatus::Ok) {
                const char* sname =
                    status == simaai::neat::PullStatus::Timeout ? "Timeout" :
                    status == simaai::neat::PullStatus::Error   ? "Error"   : "Unknown";
                std::cerr << "[RTSP] pull status=" << sname
                          << " code=" << pull_err.code
                          << " msg="  << pull_err.message << "\n";
                return false;
            }

            // Capture source-frame metadata for input-FPS detection.
            last_pull_pts_ns      = sample.pts_ns;
            last_pull_duration_ns = sample.duration_ns;
            last_pull_frame_id    = sample.frame_id;

            auto tensors = simaai::neat::tensors_from_sample(sample, false);
            if (tensors.empty()) return false;
            const auto& t = tensors[0];

            const int h = t.height();
            const int w = t.width();
            if (h <= 0 || w <= 0) return false;

            rtsp_nv12_tensor = t;

            try {
                rtsp_nv12_buf = t.copy_nv12_contiguous();
            } catch (const std::exception& e) {
                std::cerr << "[RTSP] copy_nv12_contiguous failed: " << e.what() << "\n";
                return false;
            }
            if (rtsp_nv12_buf.size() < static_cast<size_t>(w) * h * 3 / 2) return false;

            // Return a size-only placeholder so callers can query frame dimensions
            // without paying for the BGR conversion in Phase A.
            f = cv::Mat(h, w, CV_8UC1);
            return true;
        };

        read_frame = pull_rtsp_frame;

        // ── Detect the true source frame rate ───────────────────────────────────
        // No --rtsp-fps needed: derive the input rate from decoder metadata.
        //   1. Preferred: sample.duration_ns (exact per-frame period from caps).
        //   2. Fallback:  frame_id / PTS deltas over a short window — robust to
        //      dropped frames because frame_id counts every source frame.
        // Warn (but still run) if the input exceeds runtime.supported_fps: the
        // decoder will drop frames (KeepLatest), and a sustained overrun can build
        // up buffers and destabilise very long runs.
        {
            cv::Mat tmp;
            // Warm up: let the stream stabilise past the initial keyframe wait.
            for (int i = 0; i < 5; ++i) read_frame(tmp);

            if (read_frame(tmp) && last_pull_duration_ns > 0) {
                detected_fps = 1e9 / static_cast<double>(last_pull_duration_ns);
            } else {
                const int64_t pts0 = last_pull_pts_ns;
                const int64_t fid0 = last_pull_frame_id;
                int pulled = 0;
                for (int i = 0; i < 30; ++i) {
                    if (!read_frame(tmp)) break;
                    ++pulled;
                }
                const int64_t pts1 = last_pull_pts_ns;
                const int64_t fid1 = last_pull_frame_id;
                if (pts1 > pts0) {
                    const double dt_s   = (pts1 - pts0) / 1e9;
                    const double frames = (fid1 > fid0) ? static_cast<double>(fid1 - fid0)
                                                        : static_cast<double>(pulled);
                    if (dt_s > 0) detected_fps = frames / dt_s;
                }
            }

            if (detected_fps > 0) {
                std::printf("[FPS] Detected input stream rate: %.1f FPS (pipeline supports ~%d FPS)\n",
                            detected_fps, cfg.supported_fps);
                if (detected_fps > cfg.supported_fps + 0.5) {
                    std::fprintf(stderr,
                        "\n"
                        "  ****************************** WARNING ******************************\n"
                        "  * Input stream %.1f FPS EXCEEDS pipeline capacity (~%d FPS).\n"
                        "  * The decoder will drop frames (KeepLatest policy). A sustained\n"
                        "  * overrun can accumulate buffers and lead to instability or\n"
                        "  * undefined behaviour over long runs.\n"
                        "  * Fix: lower the source FPS, or raise runtime.supported_fps only\n"
                        "  * if the pipeline can actually keep up. Running anyway...\n"
                        "  *********************************************************************\n\n",
                        detected_fps, cfg.supported_fps);
                }
            } else {
                std::printf("[FPS] Could not detect input stream rate (no timing metadata); "
                            "proceeding without a rate check.\n");
            }
        }

        // Pull one seed frame so we have a real NV12 tensor for SCRFD graph.build().
        // Tensor::from_cv_mat() does not support NV12; using a real decoder tensor
        // lets the SDK negotiate GStreamer caps from the actual HW buffer format.
        // Retry up to 5 times: the neatdecoder may reject the first few packets from
        // a freshly-restarted RTSP stream (invalid keyframe / non-UTF-8 codec errors).
        if (use_ev74_preproc) {
            std::cout << "[BUILD] Pulling seed NV12 frame for SCRFD caps negotiation...\n";
            constexpr int SEED_RETRIES = 5;
            bool seed_ok = false;
            for (int attempt = 0; attempt < SEED_RETRIES && !seed_ok; ++attempt) {
                if (attempt > 0) {
                    std::cout << "[BUILD] Seed pull attempt " << (attempt + 1) << "...\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                cv::Mat seed_bgr;
                seed_ok = read_frame(seed_bgr) && rtsp_nv12_tensor.has_value() && !seed_bgr.empty();
                if (seed_ok)
                    std::cout << "[BUILD] Seed: " << seed_bgr.cols << "x" << seed_bgr.rows << " NV12\n";
            }
            if (!seed_ok)
                throw std::runtime_error("Failed to pull seed NV12 frame from RTSP source after retries");
        }

    } else if (cfg.input_uri.empty() || cfg.input_uri == "0") {
        cap.open(0);
        if (!cap.isOpened())
            throw std::runtime_error("Failed to open webcam");
        read_frame = [&](cv::Mat& f) { return cap.read(f); };
    } else {
        cap.open(cfg.input_uri);
        if (!cap.isOpened())
            throw std::runtime_error("Failed to open video source: " + cfg.input_uri);
        read_frame = [&](cv::Mat& f) { return cap.read(f); };
    }

    // Build inference pipelines.  SCRFD is built after source setup so the EV74 path
    // can use a real NV12 tensor from the decoder as the graph.build() seed (avoids
    // the Tensor::from_cv_mat() NV12 limitation).
    std::cout << "[BUILD] Building SCRFD pipeline ("
              << (use_ev74_preproc ? "EV74 CVU preproc" : "CPU preproc") << ")...\n";
    auto scrfd_run = build_scrfd_run(cfg, use_ev74_preproc,
                                     use_ev74_preproc ? &*rtsp_nv12_tensor : nullptr);
    std::cout << "[BUILD] Building ArcFace pipeline...\n";
    auto arcface_run = build_arcface_run(cfg);
    std::cout << "[BUILD] Pipelines ready.\n";

    // Attach SDK measurement only in test mode — always-on measurement limits
    // the pipeline to the measurement window duration (256 outputs) even on live streams.
    std::optional<simaai::neat::MeasureScope> scrfd_scope;
    std::optional<simaai::neat::MeasureScope> arcface_scope;
    if (cfg.test_mode) {
        simaai::neat::MeasureOptions measure_opt;
        measure_opt.title = "face-recognizer-scrfd";
        scrfd_scope = scrfd_run.start_measurement(measure_opt);
        simaai::neat::MeasureOptions arc_measure_opt;
        arc_measure_opt.title = "face-recognizer-arcface";
        arcface_scope = arcface_run.start_measurement(arc_measure_opt);
    }

    const auto sink_ext = [](const std::string& s) -> std::string {
        auto dot = s.rfind('.');
        if (dot == std::string::npos) return "";
        std::string e = s.substr(dot);
        for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return e;
    };
    const std::string out_ext = sink_ext(cfg.output_sink);
    const bool write_image = !cfg.output_sink.empty() && cfg.output_sink != "display" &&
                             (out_ext == ".jpg" || out_ext == ".jpeg" || out_ext == ".png");
    const bool write_video = !cfg.output_sink.empty() && cfg.output_sink != "display" && !write_image;
    cv::VideoWriter writer;
    bool writer_failed = false;
    if (write_video && !is_rtsp) {
        const double fps_src = cap.get(cv::CAP_PROP_FPS);
        const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(cfg.output_sink,
                    cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                    fps_src > 0 ? fps_src : 25.0,
                    cv::Size(W, H));
        if (!writer.isOpened())
            throw std::runtime_error("Failed to open output: " + cfg.output_sink);
    }

    // Overlay UDP stream (RTP/H264 via SiMa HW encoder) — lazy-init on first frame.
    // NV12 frames are pushed directly; no NV12→BGR conversion required for streaming.
    // Receive on any host:
    //   gst-launch-1.0 udpsrc port=PORT caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" \
    //     ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
    //   ffplay -fflags nobuffer -protocol_whitelist file,udp,rtp <stream.sdp>
    std::optional<simaai::neat::Run> enc_run;
    bool   enc_run_failed = false;
    long   enc_push_ok    = 0;   // frames accepted by the encoder input queue
    long   enc_push_drop  = 0;   // frames dropped because that queue was full
    double enc_push_ms    = 0.0; // cumulative time spent in try_push
    if (!cfg.stream_host.empty()) {
        std::cout << "[stream] Will send RTP/H264 overlay stream (SiMa HW encoder) -> udp://"
                  << cfg.stream_host << ":" << cfg.stream_port << "\n"
                  << "[stream] Receive:  gst-launch-1.0 udpsrc port=" << cfg.stream_port
                  << " caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\""
                  << " ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false\n";
    }

    // Async display thread — decouples imshow/waitKey from the pipeline loop.
    // cv::waitKey(1) on X11/GTK can stall 4-8 ms per frame while the compositor
    // flushes; running it on a dedicated thread lets Phase A start immediately.
    std::mutex disp_mutex;
    std::condition_variable disp_cv;
    cv::Mat disp_frame;        // either 3-ch BGR or 1-ch NV12 (rows=H*3/2, cols=W)
    bool disp_ready = false;
    bool disp_stop  = false;
    std::thread disp_thread;
    if (cfg.show_display || cfg.output_sink == "display") {
        disp_thread = std::thread([&]() {
            while (true) {
                cv::Mat f;
                {
                    std::unique_lock<std::mutex> lk(disp_mutex);
                    disp_cv.wait(lk, [&]{ return disp_ready || disp_stop; });
                    if (disp_stop && !disp_ready) break;
                    f = std::move(disp_frame);
                    disp_ready = false;
                }
                // If 1-channel, it's NV12 (H*3/2 × W): convert to BGR here,
                // off the pipeline hot path.
                if (f.channels() == 1) {
                    cv::Mat bgr;
                    cv::cvtColor(f, bgr, cv::COLOR_YUV2BGR_NV12);
                    f = std::move(bgr);
                }
                cv::imshow("face-recognizer", f);
                if (cv::waitKey(1) == 27) g_stop = true;
                if (disp_stop) break;
            }
        });
    }

    Timings timings;
    int frame_count = 0;
    const auto loop_start = Clock::now();

    // ── pipeline prime ────────────────────────────────────────────────────────
    // Read frame 0, preprocess it, and push to SCRFD so the MLA is already
    // running when the main loop begins.  From this point every loop iteration
    // starts with SCRFD running on `frame`; Phase A (read+preproc of the NEXT
    // frame) overlaps that MLA work on the CPU.
    cv::Mat frame;
    int curr_nv12_w = 0, curr_nv12_h = 0;
    face_recog::PadMeta pad_meta_curr;
    {
        cv::Mat frame0;
        if (!read_frame(frame0) || frame0.empty())
            throw std::runtime_error("Failed to read first frame from source");

        simaai::neat::Tensor t0;
        if (use_ev74_preproc && rtsp_nv12_tensor.has_value()) {
            // EV74 path: push HW NV12 tensor directly — CVU Preproc does
            // resize+colorconvert+normalize on EV74 hardware.
            // NV12 bytes are still needed on CPU for align_face_nv12 + draw_overlay_nv12.
            curr_nv12_buf = rtsp_nv12_buf;
            curr_nv12_w   = frame0.cols;
            curr_nv12_h   = frame0.rows;
            t0 = *rtsp_nv12_tensor;
            pad_meta_curr = face_recog::compute_pad_meta_only(
                frame0.cols, frame0.rows, cfg.scrfd.infer_w, cfg.scrfd.infer_h);
        } else if (is_rtsp && !rtsp_nv12_buf.empty()) {
            // RTSP NV12 fast path: fused NV12→resize→RGB_FP32 (no BGR intermediate).
            curr_nv12_buf = rtsp_nv12_buf;
            curr_nv12_w   = frame0.cols;
            curr_nv12_h   = frame0.rows;
            auto [t, p] = face_recog::preprocess_scrfd_nv12(
                curr_nv12_buf.data(), curr_nv12_w, curr_nv12_h,
                cfg.scrfd.infer_w, cfg.scrfd.infer_h);
            t0 = std::move(t);
            pad_meta_curr = p;
            // frame will be filled from curr_nv12_buf in Phase C of iteration 1
        } else {
            auto [t, p] = face_recog::preprocess_scrfd(frame0, cfg.scrfd.infer_w, cfg.scrfd.infer_h);
            t0 = std::move(t);
            pad_meta_curr = p;
            frame = std::move(frame0);
        }
        if (!scrfd_run.push(simaai::neat::TensorList{t0}))
            throw std::runtime_error("SCRFD push failed (prime): " + scrfd_run.last_error());
    }

    // Detection-quality tally, printed at shutdown. Useful for A/B-ing preproc paths
    // (EV74 CVU vs A65 NEON) on a fixed clip where the true face count is known.
    struct DetStats {
        int    frames = 0;
        int    zero_det = 0, one_det = 0, multi_det = 0;
        int    extra_dets = 0;          // detections beyond the first, summed
        double top_score_sum = 0.0;
        int    named = 0, unknown = 0;  // top-face recognition outcome
    } det_stats;

    const int RECOG_INTERVAL = cfg.recog_interval;
    static std::vector<face_recog::MatchResult> cached_matches;
    static std::vector<face_recog::Detection>   cached_detections; // boxes from last ArcFace run
    int last_recog_frame = -RECOG_INTERVAL;

    // ── main loop ─────────────────────────────────────────────────────────────
    // Parallel-execution layout every iteration:
    //
    //   MLA  |←── SCRFD_curr (7 ms) ──→|←── SCRFD_next (7 ms) ─────────────→|
    //   CPU  |← Phase A (12 ms) →|← Phase B+C (pull/push + decode/ArcFace/IO) →|
    //
    // Phase A overlaps MLA inference: CPU is never idle waiting for MLA, and
    // MLA is never idle waiting for CPU (push_next happens right after pull_curr).
    while (!g_stop) {
        if (cfg.max_frames > 0 && frame_count >= cfg.max_frames) break;
        const auto t_frame = Clock::now();

        // ── Phase A: read + preprocess NEXT frame (CPU, overlaps SCRFD on MLA) ─
        // RTSP NV12 path: copy_nv12_contiguous() + fused NV12→resize→RGB_FP32.
        //   No NV12→BGR conversion here — deferred to Phase C so it overlaps
        //   SCRFD_next on MLA instead of blocking Phase A.
        // EV74 path: only NV12 tensor wrap; CVU Preproc handles the rest on EV74.
        // File/webcam path: standard BGR letterbox + NEON normalize on A65.
        const auto ta0 = Clock::now();
        cv::Mat next_frame;
        int next_nv12_w = 0, next_nv12_h = 0;
        simaai::neat::Tensor next_tensor;
        face_recog::PadMeta  next_pad{};
        bool got_next = false;
        auto t_pull_end = ta0;
        {
            const bool at_limit = cfg.max_frames > 0 &&
                                  (frame_count + 1 >= cfg.max_frames);
            if (!at_limit && !g_stop) {
                cv::Mat nf;
                got_next = read_frame(nf) && !nf.empty();
                t_pull_end = Clock::now();   // pull (blocking wait) done; compute begins
                if (got_next) {
                    if (use_ev74_preproc && rtsp_nv12_tensor.has_value()) {
                        next_nv12_w = nf.cols;
                        next_nv12_h = nf.rows;
                        next_tensor = *rtsp_nv12_tensor;
                        next_pad = face_recog::compute_pad_meta_only(
                            nf.cols, nf.rows, cfg.scrfd.infer_w, cfg.scrfd.infer_h);
                        next_frame = std::move(nf);
                    } else if (is_rtsp && !rtsp_nv12_buf.empty()) {
                        // Fused NV12→resize→RGB_FP32: skips NV12→BGR entirely.
                        // NV12→BGR for overlay/ArcFace is deferred to Phase C.
                        next_nv12_w = nf.cols;
                        next_nv12_h = nf.rows;
                        auto [nt, np] = face_recog::preprocess_scrfd_nv12(
                            rtsp_nv12_buf.data(), next_nv12_w, next_nv12_h,
                            cfg.scrfd.infer_w, cfg.scrfd.infer_h);
                        next_tensor = std::move(nt);
                        next_pad    = np;
                        // next_frame intentionally left empty — Phase C converts NV12
                    } else {
                        auto [nt, np] = face_recog::preprocess_scrfd(
                            nf, cfg.scrfd.infer_w, cfg.scrfd.infer_h);
                        next_tensor = std::move(nt);
                        next_pad    = np;
                        next_frame  = std::move(nf);
                    }
                }
            }
        }
        const auto ta1 = Clock::now();

        // ── Phase B: pull SCRFD_curr (already done — 7ms < 12ms Phase A) ────────
        const auto scrfd_sample = scrfd_run.pull(cfg.timeout_ms);
        if (!scrfd_sample.has_value())
            throw std::runtime_error("SCRFD pull timeout");

        // Immediately push SCRFD_next so MLA stays busy during Phase C.
        if (got_next) {
            if (!scrfd_run.push(simaai::neat::TensorList{next_tensor}))
                throw std::runtime_error("SCRFD push failed: " + scrfd_run.last_error());
        }
        const auto tb1 = Clock::now();

        // ── Phase C: decode + ArcFace + overlay + IO (SCRFD_next runs concurrently) ─

        // When RTSP NV12 is available, skip the full-frame NV12→BGR conversion.
        // ArcFace uses ROI-only conversion (face crop only); overlay draws directly on
        // the NV12 buffer. The display thread converts NV12→BGR asynchronously.
        const bool use_nv12_path =
            is_rtsp && !curr_nv12_buf.empty() && curr_nv12_w > 0;

        std::vector<face_recog::Detection> detections;
        try {
            const auto scrfd_tensors = face_recog::collect_tensors(*scrfd_sample);

            // One-time: compare the letterbox the preproc stage ACTUALLY applied
            // (recorded in Tensor::semantic.preprocess) against compute_pad_meta_only's
            // assumption.  A mismatch means detections are remapped with the wrong
            // scale/offset — boxes land on the wrong objects and scores degrade.
            static bool pp_meta_logged = false;
            if (!pp_meta_logged) {
                pp_meta_logged = true;
                for (const auto& t : scrfd_tensors) {
                    if (!t.semantic.preprocess.has_value()) continue;
                    const auto& pp = *t.semantic.preprocess;
                    std::cout << "[preproc-meta] actual: orig=" << pp.original_width << "x"
                              << pp.original_height
                              << " resized=" << pp.resized_width << "x" << pp.resized_height
                              << " scaled=" << pp.scaled_width << "x" << pp.scaled_height
                              << " pad L/R/T/B=" << pp.pad_left << "/" << pp.pad_right
                              << "/" << pp.pad_top << "/" << pp.pad_bottom
                              << " mode=" << pp.resize_mode
                              << " color " << pp.color_in << "->" << pp.color_out
                              << " norm=" << pp.normalize
                              << " affine scale=(" << pp.affine_scale_x << "," << pp.affine_scale_y
                              << ") offset=(" << pp.affine_offset_x << "," << pp.affine_offset_y << ")\n";
                    std::cout << "[preproc-meta] assumed: pad L/T=" << pad_meta_curr.pad_left
                              << "/" << pad_meta_curr.pad_top
                              << " scale=" << std::min(
                                     (float)cfg.scrfd.infer_w / pad_meta_curr.orig_w,
                                     (float)cfg.scrfd.infer_h / pad_meta_curr.orig_h)
                              << " (orig " << pad_meta_curr.orig_w << "x" << pad_meta_curr.orig_h << ")\n";
                    break;
                }
            }

            detections = face_recog::decode_scrfd(scrfd_tensors, cfg.scrfd, pad_meta_curr);
        } catch (const std::exception& e) {
            std::cerr << "[SCRFD] decode error (skipping frame): " << e.what() << "\n";
            ++frame_count;
            if (is_rtsp) {
                std::swap(curr_nv12_buf, rtsp_nv12_buf);
                curr_nv12_w = next_nv12_w;
                curr_nv12_h = next_nv12_h;
            } else {
                frame = std::move(next_frame);
            }
            pad_meta_curr  = next_pad;
            if (!got_next) break;
            continue;
        }
        const auto tc1 = Clock::now();

        // Determine whether to re-run ArcFace recognition this frame.
        // Always rerun when count changes. When count is the same, check if
        // detection boxes have drifted significantly from the cached positions —
        // SCRFD sorts by score, so a confidence crossover between two faces
        // silently reorders them and would attach the wrong cached label.
        // A centroid shift > half a box width signals a likely reorder or
        // new face and forces fresh recognition.
        const bool face_count_changed = detections.size() != cached_matches.size();
        bool boxes_reordered = false;
        if (!face_count_changed && detections.size() > 1) {
            for (size_t di = 0; di < detections.size() && !boxes_reordered; ++di) {
                const auto& cur = detections[di];
                const auto& prv = cached_detections[di];
                const float cx = (cur.x1 + cur.x2) * 0.5f, cy = (cur.y1 + cur.y2) * 0.5f;
                const float px = (prv.x1 + prv.x2) * 0.5f, py = (prv.y1 + prv.y2) * 0.5f;
                const float half_w = (prv.x2 - prv.x1) * 0.5f;
                boxes_reordered = (std::abs(cx - px) + std::abs(cy - py)) > half_w;
            }
        }
        const bool recog_due = face_count_changed || boxes_reordered ||
                               (frame_count - last_recog_frame >= RECOG_INTERVAL);
        if (recog_due) {
            cached_matches.clear();
            cached_matches.reserve(detections.size());
            cached_detections = detections;
            for (const auto& det : detections) {
                const auto ta0i = Clock::now();
                // NV12 path: convert only the face-ROI region (e.g. 200×200) to BGR
                // instead of the full 1280×720 frame — ~40× fewer pixels to convert.
                const cv::Mat crop = use_nv12_path
                    ? face_recog::align_face_nv12(curr_nv12_buf.data(),
                                                  curr_nv12_w, curr_nv12_h,
                                                  det.landmarks)
                    : face_recog::align_face(frame, det.landmarks);
                const auto ta1i = Clock::now();
                const auto emb  = run_arcface(arcface_run, crop, cfg.timeout_ms);
                const auto ta2i = Clock::now();
                const auto res  = face_recog::match_embedding(emb, gallery, cfg.match);
                const auto ta3i = Clock::now();
                cached_matches.push_back(res);
                if (cfg.test_mode) {
                    timings.align_ms.push_back(Ms(ta1i - ta0i).count());
                    timings.arcface_ms.push_back(Ms(ta2i - ta1i).count());
                    timings.match_ms.push_back(Ms(ta3i - ta2i).count());
                }
            }
            last_recog_frame = frame_count;
        }
        const auto& matches = cached_matches;
        const auto tc2 = Clock::now();

        // NV12 path: annotate the NV12 buffer in place (fast). A full-frame NV12→BGR
        // conversion is produced only when a file writer actually needs one; the encoder
        // does its own NV12→BGR conversion separately before pushing to VideoSender.
        if (use_nv12_path) {
            face_recog::draw_overlay_nv12(curr_nv12_buf.data(), curr_nv12_w, curr_nv12_h,
                                          detections, matches, cfg.overlay);
            if (write_video || write_image) {
                cv::Mat nv12_mat(curr_nv12_h * 3 / 2, curr_nv12_w, CV_8UC1,
                                 curr_nv12_buf.data());
                cv::cvtColor(nv12_mat, frame, cv::COLOR_YUV2BGR_NV12);
            }
        } else {
            face_recog::draw_overlay(frame, detections, matches, cfg.overlay);
        }
        const auto tc3 = Clock::now();

        // Lazy-init the HW encoder Run on first frame when dimensions are known.
        if (!cfg.stream_host.empty() && !enc_run.has_value() && !enc_run_failed &&
            curr_nv12_w > 0 && curr_nv12_h > 0) {
            try {
                simaai::neat::InputOptions enc_in_opt;
                enc_in_opt.payload_type   = simaai::neat::PayloadType::Image;
                enc_in_opt.memory_policy  = simaai::neat::InputMemoryPolicy::SystemMemory;
                enc_in_opt.is_live        = true;
                enc_in_opt.do_timestamp   = true;
                enc_in_opt.block          = false;

                // Tell the encoder the ACTUAL delivery rate. VideoSender wires this
                // into CapsRaw on both sides of VideoConvert and into H264EncodeSima's
                // rate control; a hardcoded 30 while the app pushes ~56 fps makes the
                // receiver report ~45 fps and accumulate lag.
                const int enc_fps = (detected_fps > 0)
                    ? std::clamp((int)std::lround(detected_fps), 1, 120)
                    : 60;

                // Feed BGR to VideoSender so videoconvert performs a real BGR→NV12
                // conversion (allocating a fresh buffer from neatencoder's DMA pool).
                // Passing NV12 directly causes videoconvert to degenerate to a
                // passthrough (NV12→NV12, same buffer), which leaves neatencoder with
                // system-malloc memory that the HW encoder rejects with "Failed to
                // prepare input buffer".
                simaai::neat::nodes::groups::VideoSenderOptions vsopt =
                    simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
                        curr_nv12_w, curr_nv12_h, enc_fps);
                vsopt.host            = cfg.stream_host;
                vsopt.video_port_base = cfg.stream_port;
                vsopt.channel         = 0;

                // VideoSenderEncoderOptions defaults to profile "baseline", which forces
                // CAVLC entropy coding and forbids B-frames — a large efficiency loss that
                // shows up first as blocking/smearing on motion. neatencoder's own default
                // is "main"; "high" is supported and strictly better at a given bitrate.
                vsopt.encoder.profile = "high";

                // Budget bits from actual pixel rate rather than a flat constant.
                // 6 Mbps at 720p60 is only ~0.11 bits/pixel, which starves motion;
                // ~0.2 bpp is a reasonable target for live 720p. Level 4.0 tops out
                // around 20-25 Mbps for High profile, so the clamp stays legal.
                const long long pixel_rate =
                    (long long)curr_nv12_w * curr_nv12_h * enc_fps;
                vsopt.encoder.bitrate_kbps = (int)std::clamp(
                    pixel_rate * 20 / 100 / 1000, 4000LL, 20000LL);

                simaai::neat::Graph enc_graph("encoder");
                enc_graph.add(simaai::neat::nodes::Input(enc_in_opt));
                enc_graph.add(simaai::neat::nodes::groups::VideoSender(vsopt));

                simaai::neat::RunOptions enc_run_opt;
                // 16-frame queue absorbs H264 keyframe spikes (keyframe can take 5-10×
                // longer than a P-frame; at 45 fps that's ~200 ms of backlog headroom).
                // depth=2 caused try_push() drops during every keyframe, which triggered
                // H264 decoder resets on the receiver and appeared as periodic pauses.
                enc_run_opt.queue_depth     = 16;
                enc_run_opt.overflow_policy = simaai::neat::OverflowPolicy::DropIncoming;

                // Seed with a cv::Mat so GStreamer negotiates caps from the OpenCV type
                // (single-channel NV12 layout for the NV12 path, 3-channel BGR otherwise).
                // Seeding with a raw Tensor was causing VideoConvert to NOT degenerate to a
                // passthrough because GStreamer saw an ambiguous [H*3/2, W] shape instead
                // of a recognised single-channel NV12 buffer.
                cv::Mat seed_bgr(curr_nv12_h, curr_nv12_w, CV_8UC3, cv::Scalar(0, 0, 0));
                enc_run = enc_graph.build(std::vector<cv::Mat>{seed_bgr}, enc_run_opt);
                std::cout << "[stream] HW encoder opened -> udp://"
                          << cfg.stream_host << ":" << cfg.stream_port
                          << "  (" << curr_nv12_w << "x" << curr_nv12_h << " @ " << enc_fps
                          << " fps, " << vsopt.encoder.bitrate_kbps << " kbps, "
                          << vsopt.encoder.profile << " profile)\n";
            } catch (const std::exception& ex) {
                std::cerr << "[stream] Failed to open HW encoder: " << ex.what() << "\n";
                enc_run_failed = true;
            }
        }

        // Push the annotated frame (as BGR) to the HW encoder (fire-and-forget, no pull).
        // try_push returning false means the encoder's input queue was full and the
        // frame was dropped (OverflowPolicy::DropIncoming) — that is the gap between
        // the loop FPS and what the receiver actually sees.
        if (enc_run.has_value()) {
            const auto te0 = Clock::now();
            bool pushed = false;
            if (use_nv12_path && !curr_nv12_buf.empty()) {
                // Convert annotated NV12→BGR for the encoder. videoconvert inside
                // VideoSender then does BGR→NV12 using neatencoder's DMA pool,
                // which is required because neatencoder rejects system-memory buffers.
                cv::Mat nv12_mat(curr_nv12_h * 3 / 2, curr_nv12_w, CV_8UC1,
                                 curr_nv12_buf.data());
                cv::Mat bgr_enc;
                cv::cvtColor(nv12_mat, bgr_enc, cv::COLOR_YUV2BGR_NV12);
                pushed = enc_run->try_push(std::vector<cv::Mat>{bgr_enc});
            } else if (!frame.empty()) {
                pushed = enc_run->try_push(std::vector<cv::Mat>{frame});
            }
            if (pushed) ++enc_push_ok; else ++enc_push_drop;
            enc_push_ms += Ms(Clock::now() - te0).count();
        }

        if (write_video && is_rtsp && !writer.isOpened() && !writer_failed) {
            if (!writer.open(cfg.output_sink,
                             cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                             25.0, cv::Size(frame.cols, frame.rows))) {
                std::cerr << "[OUTPUT] Failed to open video writer: " << cfg.output_sink << "\n";
                writer_failed = true;
            }
        }
        if (write_video && writer.isOpened()) writer.write(frame);
        if (write_image && frame_count == 0) cv::imwrite(cfg.output_sink, frame);

        if (cfg.test_mode) {
            for (size_t i = 0; i < matches.size(); ++i)
                printf("  face[%zu] → %-20s  similarity=%.4f\n",
                       i, matches[i].name.c_str(), matches[i].score);
        }
        if (cfg.show_display || cfg.output_sink == "display") {
            {
                std::lock_guard<std::mutex> lk(disp_mutex);
                if (!frame.empty()) {
                    // BGR frame with overlay already drawn — display thread shows directly.
                    disp_frame = frame.clone();
                } else if (use_nv12_path) {
                    // NV12-only path (no stream/write): display thread converts to BGR.
                    cv::Mat nv12_view(curr_nv12_h * 3 / 2, curr_nv12_w,
                                      CV_8UC1, curr_nv12_buf.data());
                    nv12_view.copyTo(disp_frame);
                } else {
                    disp_frame = frame.clone();
                }
                disp_ready = true;
            }
            disp_cv.notify_one();
        }

        ++frame_count;

        if (cfg.test_mode) {
            // preproc_ms = Phase A wall time (includes read + resize/normalize).
            // scrfd_ms   = time from Phase A end to SCRFD pull: ideally ~0 since
            //              SCRFD (7ms) finishes before Phase A (12ms) completes.
            timings.preproc_ms.push_back(Ms(ta1 - ta0).count());
            timings.pull_ms.push_back(Ms(t_pull_end - ta0).count());
            timings.preproc_compute_ms.push_back(Ms(ta1 - t_pull_end).count());
            timings.scrfd_ms.push_back(Ms(tb1 - ta1).count());
            timings.decode_ms.push_back(Ms(tc1 - tb1).count());
            timings.overlay_ms.push_back(Ms(tc3 - tc2).count());
            timings.e2e_ms.push_back(Ms(tc3 - t_frame).count());
        }
        ++det_stats.frames;
        if (detections.empty())            ++det_stats.zero_det;
        else if (detections.size() == 1)   ++det_stats.one_det;
        else                             { ++det_stats.multi_det;
                                           det_stats.extra_dets += (int)detections.size() - 1; }
        if (!detections.empty()) det_stats.top_score_sum += detections[0].score;
        if (!matches.empty()) {
            if (matches[0].index >= 0) ++det_stats.named; else ++det_stats.unknown;
        }

        if (frame_count % 100 == 0) {
            std::cout << "[" << frame_count << "] faces=" << detections.size();
            // det=SCRFD confidence + box, so a bad preproc shows up as low/misplaced dets
            // independently of the ArcFace match score.
            for (size_t i = 0; i < detections.size(); ++i)
                printf("  det[%zu]=%.3f@(%.0f,%.0f,%.0f,%.0f)",
                       i, detections[i].score, detections[i].x1, detections[i].y1,
                       detections[i].x2, detections[i].y2);
            for (size_t i = 0; i < matches.size(); ++i)
                printf("  face[%zu]=%s(%.3f) 2nd=%s(%.3f) margin=%.3f",
                       i, matches[i].name.c_str(), matches[i].score,
                       matches[i].second_name.c_str(), matches[i].second_score,
                       matches[i].score - matches[i].second_score);
            if (enc_run.has_value())
                printf("  enc_ok=%ld drop=%ld", enc_push_ok, enc_push_drop);
            std::cout << "\n";
        }

        if (is_rtsp) {
            // Swap NV12 buffers: next frame's NV12 becomes current for Phase C next iteration.
            std::swap(curr_nv12_buf, rtsp_nv12_buf);
            curr_nv12_w = next_nv12_w;
            curr_nv12_h = next_nv12_h;
        } else {
            frame = std::move(next_frame);
        }
        pad_meta_curr = next_pad;
        if (!got_next) break;
    }

    // ── shutdown ──────────────────────────────────────────────────────────────
    const auto loop_end = Clock::now();
    const double total_s = std::chrono::duration<double>(loop_end - loop_start).count();

    if (cfg.test_mode && scrfd_scope && arcface_scope) {
        const auto scrfd_report   = scrfd_scope->stop();
        const auto arcface_report = arcface_scope->stop();
        std::cout << "\n[SDK SCRFD]\n"   << scrfd_report.to_text()   << "\n";
        std::cout << "[SDK ArcFace]\n"  << arcface_report.to_text() << "\n";
        print_timings(timings, frame_count, total_s > 0 ? total_s : 1.0);
    }

    if (disp_thread.joinable()) {
        { std::lock_guard<std::mutex> lk(disp_mutex); disp_stop = true; }
        disp_cv.notify_one();
        disp_thread.join();
    }

    scrfd_run.close();
    arcface_run.close();
    if (rtsp_src_run) rtsp_src_run->close();
    if (enc_run.has_value()) enc_run->close();
    if (write_video) writer.release();
    if (!is_rtsp) cap.release();

    if (det_stats.frames > 0) {
        const int scored = det_stats.one_det + det_stats.multi_det;
        printf("\n═══ Detection quality (%d frames) ═══\n", det_stats.frames);
        printf("  0 faces : %5d (%.1f%%)   <- misses\n",
               det_stats.zero_det, 100.0 * det_stats.zero_det / det_stats.frames);
        printf("  1 face  : %5d (%.1f%%)\n",
               det_stats.one_det, 100.0 * det_stats.one_det / det_stats.frames);
        printf("  >1 face : %5d (%.1f%%)   <- extra dets total: %d\n",
               det_stats.multi_det, 100.0 * det_stats.multi_det / det_stats.frames,
               det_stats.extra_dets);
        printf("  mean top-det score: %.3f\n",
               scored > 0 ? det_stats.top_score_sum / scored : 0.0);
        printf("  recognized/unknown: %d / %d\n\n", det_stats.named, det_stats.unknown);
    }

    if (enc_push_ok + enc_push_drop > 0) {
        const long tot = enc_push_ok + enc_push_drop;
        printf("═══ Stream encoder (%ld pushes) ═══\n", tot);
        printf("  accepted: %6ld (%.1f%%)   -> ~%.1f FPS at the receiver\n",
               enc_push_ok, 100.0 * enc_push_ok / tot,
               enc_push_ok / std::max(total_s, 0.001));
        printf("  dropped : %6ld (%.1f%%)   <- encoder input queue full\n",
               enc_push_drop, 100.0 * enc_push_drop / tot);
        printf("  mean try_push: %.2f ms\n\n", enc_push_ms / tot);
    }

    std::cout << "Done: " << frame_count << " frames in "
              << cv::format("%.2f", total_s) << " s ("
              << cv::format("%.1f", frame_count / std::max(total_s, 0.001)) << " FPS)\n";
    return 0;
}
