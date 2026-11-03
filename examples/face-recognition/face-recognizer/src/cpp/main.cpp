#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/io/MetadataSender.h>
#include <nlohmann/json.hpp>

#include "scrfd_decode.h"
#include "align.h"
#include "gallery.h"
#include "match.h"
#include "overlay.h"

// Enrollment mode — compiled from enroll.cpp, dispatched when --enroll
// is the first argument so both recognition and enrollment ship in one binary.
int run_enrollment_mode(int argc, char** argv);

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
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;
using Ms     = std::chrono::duration<double, std::milli>;

// ── config ────────────────────────────────────────────────────────────────────

struct AppConfig {
    std::string scrfd_model;    // set by --scrfd-model or yaml; no hardcoded default
    std::string arcface_model;  // set by --arcface-model or yaml; no hardcoded default
    std::string gallery_path = "gallery.bin";
    bool        gallery_path_set = false;  // true when --gallery was explicitly passed
    std::string input_uri;          // RTSP URL or H.264 video file path
    bool        input_uri_set    = false;  // true when --input was explicitly passed
    // input.usb: a USB (UVC) MJPEG camera by its usb_camera_ids() ID, at one of its exact
    // MJPEG modes. Used instead of input.uri when camera is set.
    std::string usb_camera;
    int         usb_width  = 1280;
    int         usb_height = 720;
    int         usb_fps    = 30;
    std::string output_sink;        // annotated recording (.mp4/.avi or .jpg/.png), or empty
    std::string insight_host;             // Insight host IP; read from output.insight.host
    int  insight_video_port    = 9000;   // Insight video UDP port; read from output.insight.video_port
    int  insight_metadata_port = 9100;   // Insight metadata UDP port; read from output.insight.metadata_port
    int  max_frames          = 0;   // 0 = unlimited
    int  rtsp_fps            = -1;  // decoder caps fps hint; -1 = auto-detect from stream
    int  supported_fps       = 45;  // max input FPS the pipeline can sustain (warn above this)
    int  timeout_ms          = 20000;
    int  recog_interval      = 5;
    bool test_mode           = false;
    bool output_sink_explicit = false; // true when --output/--test set output_sink from CLI

    face_recog::ScrfdConfig  scrfd;
    face_recog::MatchConfig  match;
    face_recog::OverlayConfig overlay;
};

static bool is_rtsp_uri(const std::string& uri) {
    return uri.rfind("rtsp://", 0) == 0 || uri.rfind("rtsps://", 0) == 0;
}

static AppConfig load_config(const fs::path& path) {
    const auto raw = sima_examples::ScalarConfig::load(path);
    AppConfig cfg;

    // config.yaml paths are relative to the face-recognizer package root
    // (3 levels up from src/common/config.yaml). Normalise first so that the
    // default path (src/cpp/../common/config.yaml) resolves correctly, then
    // walk up to the package root. This lets the binary run from any CWD —
    // including prebuilt-apps/ in the installed bundle.
    const fs::path pkg_root = fs::absolute(path).lexically_normal()
                                                 .parent_path()  // src/common/
                                                 .parent_path()  // src/
                                                 .parent_path(); // face-recognizer/
    auto resolve = [&](const std::string& p) -> std::string {
        if (p.empty() || fs::path(p).is_absolute()) return p;
        return (pkg_root / p).lexically_normal().string();
    };

    cfg.scrfd_model   = resolve(raw.string_or("scrfd.model",   cfg.scrfd_model));
    cfg.arcface_model = resolve(raw.string_or("arcface.model", cfg.arcface_model));
    cfg.gallery_path  = resolve(raw.string_or("gallery.path",  cfg.gallery_path));
    // For file-based input/output, resolve relative paths from the package root just like
    // model and gallery paths. Skip URIs (contain "://").
    auto resolve_media = [&](const std::string& p) -> std::string {
        if (p.empty() || fs::path(p).is_absolute()) return p;
        if (p.find("://") != std::string::npos) return p;
        return (pkg_root / p).lexically_normal().string();
    };
    cfg.input_uri          = resolve_media(raw.string_or("input.uri",   ""));
    cfg.usb_camera         = raw.string_or("input.usb.camera", "");
    cfg.usb_width          = raw.int_or("input.usb.width",  cfg.usb_width);
    cfg.usb_height         = raw.int_or("input.usb.height", cfg.usb_height);
    cfg.usb_fps            = raw.int_or("input.usb.fps",    cfg.usb_fps);
    cfg.output_sink        = resolve_media(raw.string_or("output.sink",  ""));
    cfg.insight_host          = raw.string_or("output.insight.host",            "");
    cfg.insight_video_port    = raw.int_or   ("output.insight.video_port",       9000);
    cfg.insight_metadata_port = raw.int_or   ("output.insight.metadata_port",    9100);
    cfg.timeout_ms      = raw.int_or("runtime.timeout_ms",     cfg.timeout_ms);
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

    return cfg;
}

static AppConfig parse_args(int argc, char** argv) {
    fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
    AppConfig cfg;  // defaults; overridden by config then CLI flags

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--input"      && i + 1 < argc) { cfg.input_uri = argv[++i]; cfg.input_uri_set = true; }
        else if (arg == "--gallery"      && i + 1 < argc) { cfg.gallery_path = argv[++i]; cfg.gallery_path_set = true; }
        else if (arg == "--scrfd-model"  && i + 1 < argc) { cfg.scrfd_model    = argv[++i]; }
        else if (arg == "--arcface-model"&& i + 1 < argc) { cfg.arcface_model  = argv[++i]; }
        else if (arg == "--output"       && i + 1 < argc) { cfg.output_sink = argv[++i]; cfg.output_sink_explicit = true; }
        else if (arg == "--max-frames"   && i + 1 < argc) { cfg.max_frames     = std::stoi(argv[++i]); }
        else if (arg == "--rtsp-fps"     && i + 1 < argc) { cfg.rtsp_fps       = std::stoi(argv[++i]); }
        else if (arg == "--test")   { cfg.test_mode = true; cfg.output_sink = ""; cfg.output_sink_explicit = true; }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: face-recognizer [--config <path>] [--input <uri>]\n"
                      << "       [--gallery <path>] [--scrfd-model <path>] [--arcface-model <path>]\n"
                      << "       [--output <sink>] [--test]\n"
                      << "\n"
                      << "  input.uri / input.usb.camera (config.yaml):\n"
                      << "                      An RTSP URL or H.264 video file, or a USB camera ID\n"
                      << "                      from usb_camera_ids(). --input replaces both.\n"
                      << "\n"
                      << "  output.insight.host / output.insight.video_port (config.yaml):\n"
                      << "                      Send the video and face metadata to Insight.\n"
                      << "                      Set output.insight.host to the Insight host IP and\n"
                      << "                      open the Insight Web UI to view the faces.\n"
                      << "\n"
                      << "  --rtsp-fps N        Optional decoder FPS override. Omit to auto-detect\n"
                      << "                      the source rate from the stream (recommended).\n";
            std::exit(0);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    auto yaml_cfg = load_config(config_path);
    if (!cfg.input_uri_set) {
        cfg.input_uri  = yaml_cfg.input_uri;
        cfg.usb_camera = yaml_cfg.usb_camera;  // --input replaces both input sources
    }
    cfg.usb_width  = yaml_cfg.usb_width;
    cfg.usb_height = yaml_cfg.usb_height;
    cfg.usb_fps    = yaml_cfg.usb_fps;
    if (!cfg.gallery_path_set)
        cfg.gallery_path = yaml_cfg.gallery_path;
    if (!cfg.output_sink_explicit && cfg.output_sink.empty()) cfg.output_sink = yaml_cfg.output_sink;
    if (cfg.scrfd_model.empty())   cfg.scrfd_model   = yaml_cfg.scrfd_model;
    if (cfg.arcface_model.empty()) cfg.arcface_model = yaml_cfg.arcface_model;
    cfg.timeout_ms          = yaml_cfg.timeout_ms;
    cfg.recog_interval      = yaml_cfg.recog_interval;
    cfg.supported_fps       = yaml_cfg.supported_fps;
    cfg.insight_host           = yaml_cfg.insight_host;
    cfg.insight_video_port     = yaml_cfg.insight_video_port;
    cfg.insight_metadata_port  = yaml_cfg.insight_metadata_port;
    cfg.scrfd         = yaml_cfg.scrfd;
    cfg.match         = yaml_cfg.match;
    cfg.overlay       = yaml_cfg.overlay;
    if (cfg.test_mode) cfg.insight_host.clear();
    return cfg;
}

// ── per-stage timing accumulator ─────────────────────────────────────────────

struct Timings {
    std::vector<double> pull_ms;     // wait for the next joined {frame, detections} sample
    std::vector<double> decode_ms;   // SCRFD decode + NMS on the CPU
    std::vector<double> align_ms;
    std::vector<double> arcface_ms;
    std::vector<double> match_ms;
    std::vector<double> record_ms;   // --output: NV12→BGR + overlay + write
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
    print_stage("Pull (graph)",      t.pull_ms);
    print_stage("SCRFD decode+NMS",  t.decode_ms);
    print_stage("Align",             t.align_ms);
    print_stage("ArcFace infer",     t.arcface_ms);
    print_stage("Match",             t.match_ms);
    if (!t.record_ms.empty()) print_stage("Record (--output)", t.record_ms);
    print_stage("Frame E2E",         t.e2e_ms);
}

// ── build inference pipelines ─────────────────────────────────────────────────

// SCRFD with EV74 CVU preproc: decoded NV12 frames in (NV16 from a 4:2:2 USB camera);
// resize + colorconvert + normalize + quantize + tessellate run on EV74 hardware.
static simaai::neat::Model::Options scrfd_image_options(int max_w, int max_h, bool nv16_frames) {
    simaai::neat::Model::Options opt;
    opt.preprocess.kind   = simaai::neat::InputKind::Image;
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.color_convert.input_format = nv16_frames
                                                    ? simaai::neat::PreprocessColorFormat::NV16
                                                    : simaai::neat::PreprocessColorFormat::NV12;
    opt.preprocess.input_max_width  = max_w;
    opt.preprocess.input_max_height = max_h;
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
    return opt;
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

// ── shared helpers ────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};

// Detection-quality tally, printed at shutdown (the E2E test parses this summary).
struct DetStats {
    int    frames = 0;
    int    zero_det = 0, one_det = 0, multi_det = 0;
    int    extra_dets = 0;          // detections beyond the first, summed
    double top_score_sum = 0.0;
    int    named = 0, unknown = 0;  // top-face recognition outcome

    void add(const std::vector<face_recog::Detection>& dets,
             const std::vector<face_recog::MatchResult>& matches) {
        ++frames;
        if (dets.empty())          ++zero_det;
        else if (dets.size() == 1) ++one_det;
        else                     { ++multi_det; extra_dets += (int)dets.size() - 1; }
        if (!dets.empty()) top_score_sum += dets[0].score;
        if (!matches.empty()) {
            if (matches[0].index >= 0) ++named; else ++unknown;
        }
    }

    void print() const {
        if (frames == 0) return;
        const int scored = one_det + multi_det;
        printf("\n═══ Detection quality (%d frames) ═══\n", frames);
        printf("  0 faces : %5d (%.1f%%)   <- misses\n", zero_det, 100.0 * zero_det / frames);
        printf("  1 face  : %5d (%.1f%%)\n", one_det, 100.0 * one_det / frames);
        printf("  >1 face : %5d (%.1f%%)   <- extra dets total: %d\n",
               multi_det, 100.0 * multi_det / frames, extra_dets);
        printf("  mean top-det score: %.3f\n", scored > 0 ? top_score_sum / scored : 0.0);
        printf("  recognized/unknown: %d / %d\n\n", named, unknown);
    }
};

// Re-run ArcFace when the face count changes, when a box's nearest cached box sits at
// another index (SCRFD's score-sorted order flipped, which would swap labels), when a
// box moved more than half its width (a new face), or every `interval` frames.
static bool recognition_due(const std::vector<face_recog::Detection>& dets,
                            const std::vector<face_recog::Detection>& cached,
                            int frames_since_recog, int interval) {
    if (frames_since_recog >= interval || dets.size() != cached.size()) return true;
    for (size_t di = 0; di < dets.size(); ++di) {
        const float cx = (dets[di].x1 + dets[di].x2) * 0.5f;
        const float cy = (dets[di].y1 + dets[di].y2) * 0.5f;
        float  best_dist = std::numeric_limits<float>::max();
        size_t best_pi   = di;
        for (size_t pi = 0; pi < cached.size(); ++pi) {
            const float px = (cached[pi].x1 + cached[pi].x2) * 0.5f;
            const float py = (cached[pi].y1 + cached[pi].y2) * 0.5f;
            const float d  = std::abs(cx - px) + std::abs(cy - py);
            if (d < best_dist) { best_dist = d; best_pi = pi; }
        }
        const float half_w = (cached[best_pi].x2 - cached[best_pi].x1) * 0.5f;
        if (best_pi != di || best_dist > half_w) return true;
    }
    return false;
}

static const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                              const std::string& label) {
    if (sample.stream_label == label) return &sample;
    for (const auto& field : sample.fields)
        if (const auto* hit = find_field(field, label)) return hit;
    return nullptr;
}

static const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                                const std::string& label, size_t bundle_index) {
    if (const auto* field = find_field(sample, label)) return *field;
    if (sample.kind == simaai::neat::SampleKind::Bundle && sample.fields.size() > bundle_index)
        return sample.fields[bundle_index];
    throw std::runtime_error("joined output missing " + label + " field");
}

// H.264 to Insight. The VideoSender default profile "baseline" forces CAVLC and forbids
// B-frames, which smears motion; "high" is supported by neatencoder. Bitrate targets
// ~0.2 bits/pixel, clamped to the H.264 level 4.0 range.
static simaai::neat::nodes::groups::VideoSenderOptions insight_video_options(
        const AppConfig& cfg, int w, int h, int fps) {
    auto vsopt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(w, h, fps);
    vsopt.host            = cfg.insight_host;
    vsopt.video_port_base = cfg.insight_video_port;
    vsopt.channel         = 0;
    vsopt.encoder.profile = "high";
    const long long pixel_rate = (long long)w * h * fps;
    vsopt.encoder.bitrate_kbps = (int)std::clamp(pixel_rate * 20 / 100 / 1000, 4000LL, 20000LL);
    return vsopt;
}

static std::unique_ptr<simaai::neat::MetadataSender> make_metadata_sender(const AppConfig& cfg) {
    if (cfg.insight_host.empty()) return nullptr;
    simaai::neat::MetadataSenderOptions meta_opt;
    meta_opt.host               = cfg.insight_host;
    meta_opt.channel            = 0;
    meta_opt.metadata_port_base = cfg.insight_metadata_port;
    std::string meta_err;
    auto sender = std::make_unique<simaai::neat::MetadataSender>(meta_opt, &meta_err);
    if (!sender->ok())
        throw std::runtime_error("[stream] MetadataSender init failed: " + meta_err);
    return sender;
}

// ── Pipeline: one graph, same topology as the Python app ──────────────────────
//
//   source ─┬─ video ─→ VideoSender (H.264 → Insight)          [output.insight.host set]
//           ├─ model ─→ SCRFD (EV74 preproc) ─→ detections ─┐
//           └─ frame ──────────────────────────────────────┴─ ByFrame join → app
//
// The source is RtspDecodedInput for rtsp:// URLs and VideoInputGroup for H.264 video
// files (offline: no Insight); both decode on the hardware decoder. Decode, preproc and SCRFD run ahead
// inside the graph; the app pulls joined {frame, detections} samples and copies a
// frame to the CPU only when ArcFace runs or --output records it. Insight draws the
// boxes and labels from the per-frame metadata.
//
// A USB camera sends JPEG frames. They go to Insight unchanged (RTP/JPEG, no re-encode)
// and are decoded on the hardware decoder only for SCRFD and the frame tap, in the
// camera's own sampling: most webcams send 4:2:2 MJPEG, which decodes to NV16.
//
//   camera ─┬─ video ─→ VideoSender (MJPEG passthrough → Insight)
//           └─ decode ─→ SimaDecode ─┬─ model ─→ SCRFD ─→ detections ─┐
//                                    └─ frame ────────────────────────┴─ ByFrame join → app
static constexpr const char* kJoinedOutput = "face_output";

static simaai::neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg) {
    simaai::neat::UsbCameraInputOptions opt;
    opt.camera_name   = cfg.usb_camera;
    opt.width         = static_cast<std::uint32_t>(cfg.usb_width);
    opt.height        = static_cast<std::uint32_t>(cfg.usb_height);
    opt.framerate_num = static_cast<std::uint32_t>(cfg.usb_fps);
    return opt;
}

static simaai::neat::SimaDecodeOptions usb_decode_options(const AppConfig& cfg,
                                                          simaai::neat::FormatTag format) {
    simaai::neat::SimaDecodeOptions opt;
    opt.type       = simaai::neat::SimaDecodeType::MJPEG;
    opt.raw_output = true;
    opt.out_format = format;
    opt.dec_width  = cfg.usb_width;
    opt.dec_height = cfg.usb_height;
    opt.dec_fps    = cfg.usb_fps;
    return opt;
}

static int run_pipeline(const AppConfig& cfg, const face_recog::Gallery& gallery) {
    const bool is_usb      = !cfg.usb_camera.empty();
    const bool is_rtsp     = !is_usb && is_rtsp_uri(cfg.input_uri);
    const bool is_live     = is_usb || is_rtsp;
    const bool use_insight = !cfg.insight_host.empty();
    if (use_insight && !is_live)
        throw std::runtime_error("Insight output needs an RTSP or USB camera input; video files "
                                 "run offline (clear output.insight.host, or record with --output)");

    // Frame size and rate are needed before the graph is built (model input bounds,
    // VideoSender caps, recording writer).
    int frame_w = 0, frame_h = 0, fps = 0;
    // The camera is checked once (connected, free, mode supported, one frame decodes); the
    // same info gives the decoded layout and builds the camera node.
    std::optional<simaai::neat::UsbCameraInfo> usb_info;
    if (is_usb) {
        if (cfg.usb_width <= 0 || cfg.usb_height <= 0 || cfg.usb_fps <= 0)
            throw std::runtime_error("input.usb.width, input.usb.height and input.usb.fps must be > 0");
        // Insight receives the camera's JPEG as RTP/JPEG (RFC 2435): 8-pixel blocks, 2040 at most.
        if (use_insight && (cfg.usb_width % 8 != 0 || cfg.usb_height % 8 != 0 ||
                            cfg.usb_width > 2040 || cfg.usb_height > 2040))
            throw std::runtime_error("input.usb.width and input.usb.height must be multiples of 8 "
                                     "and at most 2040 to send the camera's JPEG to Insight");
        usb_info = simaai::neat::usb_camera_info(usb_camera_options(cfg));
        frame_w = cfg.usb_width;
        frame_h = cfg.usb_height;
        fps     = cfg.usb_fps;
    } else if (is_rtsp) {
        sima_examples::RtspStreamInfo info;
        (void)sima_examples::probe_rtsp_stream_info(cfg.input_uri,
                                                    sima_examples::RtspProbeOptions{}, info);
        frame_w = info.width;
        frame_h = info.height;
        fps     = cfg.rtsp_fps > 0 ? cfg.rtsp_fps : info.fps;
    } else {
        if (!fs::exists(cfg.input_uri))
            throw std::runtime_error("input video not found: " + cfg.input_uri);
        cv::VideoCapture probe(cfg.input_uri);
        if (!probe.isOpened())
            throw std::runtime_error("cannot open input video: " + cfg.input_uri);
        frame_w = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_h = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps     = static_cast<int>(std::lround(probe.get(cv::CAP_PROP_FPS)));
    }
    if (frame_w <= 0 || frame_h <= 0)
        throw std::runtime_error("could not read the input frame size from " + cfg.input_uri);
    if (fps <= 0)
        throw std::runtime_error("could not read the input frame rate; pass --rtsp-fps <n>");
    const bool nv16_frames = is_usb && usb_info->decoded_format == simaai::neat::FormatTag::NV16;
    if (is_usb)
        std::printf("[INPUT] USB camera %s (%s) %dx%d @ %d FPS MJPEG, decoded to %s\n",
                    cfg.usb_camera.c_str(), usb_info->device.c_str(), frame_w, frame_h, fps,
                    nv16_frames ? "NV16" : "NV12");
    else
        std::printf("[INPUT] %s %dx%d @ %d FPS\n", is_rtsp ? "RTSP" : "video file",
                    frame_w, frame_h, fps);
    if (is_live && fps > cfg.supported_fps)
        std::fprintf(stderr, "[WARN] Input %d FPS exceeds runtime.supported_fps (%d); the "
                             "newest frames are kept and older ones dropped.\n",
                     fps, cfg.supported_fps);

    // Build ArcFace first so the live graph does not queue frames while it loads.
    std::cout << "[BUILD] Building ArcFace pipeline...\n";
    auto arcface_run = build_arcface_run(cfg);

    simaai::neat::Model scrfd_model(cfg.scrfd_model,
                                    scrfd_image_options(frame_w, frame_h, nv16_frames));

    simaai::neat::GraphOptions graph_opt;
    graph_opt.advanced_execution.postprocess_target = "EV74";
    simaai::neat::Graph graph(graph_opt);

    simaai::neat::Graph source("source");
    if (is_usb) {
        source.add(simaai::neat::nodes::UsbCameraInput(usb_camera_options(cfg), *usb_info));
        // Without Insight nothing needs the JPEG, so the decoder follows the camera directly.
        if (!use_insight)
            source.add(simaai::neat::nodes::SimaDecode(
                usb_decode_options(cfg, usb_info->decoded_format)));
    } else if (is_rtsp) {
        simaai::neat::nodes::groups::RtspDecodedInputOptions src_opt;
        src_opt.url                   = cfg.input_uri;
        src_opt.latency_ms            = 50;
        src_opt.tcp                   = true;
        src_opt.payload_type          = 96;
        src_opt.insert_queue          = true;
        src_opt.out_format            = simaai::neat::FormatTag::NV12;
        src_opt.decoder_name          = "decoder";
        src_opt.decoder_raw_output    = true;
        src_opt.auto_caps_from_stream = true;
        src_opt.fallback_h264_width   = frame_w;
        src_opt.fallback_h264_height  = frame_h;
        // A fixed fps that mismatches the stream fails caps negotiation; only force it on request.
        src_opt.fallback_h264_fps     = cfg.rtsp_fps > 0 ? cfg.rtsp_fps : -1;
        source = simaai::neat::nodes::groups::RtspDecodedInput(src_opt);
    } else {
        simaai::neat::nodes::groups::VideoInputGroupOptions src_opt;
        src_opt.path = cfg.input_uri;
        source = simaai::neat::nodes::groups::VideoInputGroup(src_opt);
    }

    // RTSP and file frames feed Insight from the decoded branch; a USB camera's JPEG is
    // split off before decode instead.
    const bool usb_split = is_usb && use_insight;
    std::vector<std::string> branch_outputs{"model", "frame"};
    if (use_insight && !is_usb) branch_outputs.insert(branch_outputs.begin(), "video");
    auto branch = simaai::neat::graphs::Branch(usb_split ? "decoded" : "source", branch_outputs);

    simaai::neat::Graph video_graph("video");
    if (use_insight) {
        auto vsopt = is_usb ? simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
                                  simaai::neat::nodes::groups::RtspCodec::MJPEG)
                            : insight_video_options(cfg, frame_w, frame_h, fps);
        vsopt.host            = cfg.insight_host;
        vsopt.video_port_base = cfg.insight_video_port;
        vsopt.channel         = 0;
        video_graph.connect(simaai::neat::nodes::Input("video"),
                            simaai::neat::nodes::groups::VideoSender(vsopt));
    }

    if (usb_split) {
        simaai::neat::Graph decode_graph("decode");
        decode_graph.connect(simaai::neat::nodes::Input("decode"),
                             simaai::neat::nodes::SimaDecode(
                                 usb_decode_options(cfg, usb_info->decoded_format)));
        simaai::neat::GraphLinkOptions video_link;
        video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        auto split = simaai::neat::graphs::Branch("source", {"video", "decode"});
        graph.connect(source, split);
        graph.connect(split, video_graph, video_link);
        graph.connect(split, decode_graph);
        graph.connect(decode_graph, branch);
    } else {
        graph.connect(source, branch);
        if (use_insight) graph.connect(branch, video_graph);
    }

    simaai::neat::Graph model_graph("model");
    model_graph.connect(simaai::neat::nodes::Input("model"), scrfd_model);
    simaai::neat::Graph det_graph("detections");
    det_graph.add(simaai::neat::nodes::Output("detections",
                                              simaai::neat::OutputOptions::EveryFrame(4)));
    simaai::neat::Graph frame_graph("frame");
    frame_graph.add(simaai::neat::nodes::Output("frame",
                                                simaai::neat::OutputOptions::EveryFrame(4)));
    auto joined = simaai::neat::graphs::Combine({"frame", "detections"}, kJoinedOutput,
                                                simaai::neat::CombinePolicy::ByFrame);
    graph.connect(branch, model_graph);
    graph.connect(branch, frame_graph);
    graph.connect(model_graph, det_graph);
    graph.connect(frame_graph, joined);
    graph.connect(det_graph, joined);

    // Live sources keep the newest frame; files are processed frame by frame.
    simaai::neat::RunOptions run_opt;
    if (is_live) run_opt.preset = simaai::neat::RunPreset::Realtime;
    run_opt.queue_depth     = 3;
    run_opt.overflow_policy = is_live ? simaai::neat::OverflowPolicy::KeepLatest
                                      : simaai::neat::OverflowPolicy::Block;
    run_opt.output_memory   = simaai::neat::OutputMemory::ZeroCopy;
    std::cout << "[BUILD] Building " << (is_usb ? "USB camera" : is_rtsp ? "RTSP" : "video-file")
              << " → SCRFD graph (EV74 CVU preproc)...\n";
    auto run = graph.build(run_opt);

    auto metadata_sender = make_metadata_sender(cfg);
    if (use_insight)
        std::cout << "[stream] " << (is_usb ? "MJPEG" : "H.264") << " -> udp://"
                  << cfg.insight_host << ":"
                  << cfg.insight_video_port << ", metadata -> udp://" << cfg.insight_host
                  << ":" << metadata_sender->metadata_port() << " channel=0  [Insight]\n";

    // --output: annotated recording (video) or a single annotated image (first frame).
    std::string out_ext = fs::path(cfg.output_sink).extension().string();
    for (char& c : out_ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool write_image = out_ext == ".jpg" || out_ext == ".jpeg" || out_ext == ".png";
    const bool write_video = !cfg.output_sink.empty() && !write_image;
    cv::VideoWriter writer;
    if (write_video &&
        !writer.open(cfg.output_sink, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                     cv::Size(frame_w, frame_h)))
        throw std::runtime_error("failed to open output: " + cfg.output_sink);

    std::optional<simaai::neat::MeasureScope> scrfd_scope;
    std::optional<simaai::neat::MeasureScope> arcface_scope;
    if (cfg.test_mode) {
        simaai::neat::MeasureOptions measure_opt;
        measure_opt.title = "face-recognizer-scrfd";
        scrfd_scope = run.start_measurement(measure_opt);
        simaai::neat::MeasureOptions arc_measure_opt;
        arc_measure_opt.title = "face-recognizer-arcface";
        arcface_scope = arcface_run.start_measurement(arc_measure_opt);
    }

    const auto pad_meta = face_recog::compute_pad_meta_only(
        frame_w, frame_h, cfg.scrfd.infer_w, cfg.scrfd.infer_h);

    Timings  timings;
    DetStats det_stats;
    std::vector<face_recog::MatchResult> cached_matches;
    std::vector<face_recog::Detection>   cached_detections;
    int last_recog_frame = -cfg.recog_interval;
    int frame_count      = 0;
    std::optional<Clock::time_point> loop_start;  // first joined sample; excludes stream startup

    while (!g_stop && (cfg.max_frames <= 0 || frame_count < cfg.max_frames)) {
        const auto t0 = Clock::now();
        simaai::neat::Sample sample;
        simaai::neat::PullError pull_err;
        const auto status = run.pull(kJoinedOutput, cfg.timeout_ms, sample, &pull_err);
        if (status == simaai::neat::PullStatus::Closed ||
            (status == simaai::neat::PullStatus::Error && pull_err.code == "runtime.pull")) {
            std::cerr << "[INPUT] End of stream.\n";
            break;
        }
        if (status != simaai::neat::PullStatus::Ok)
            throw std::runtime_error("SCRFD pull failed: " + pull_err.code + " " +
                                     pull_err.message);
        const auto t1 = Clock::now();
        if (!loop_start) loop_start = t1;

        const auto detections = face_recog::decode_scrfd(
            face_recog::collect_tensors(joined_field(sample, "detections", 1)),
            cfg.scrfd, pad_meta);
        const auto t2 = Clock::now();

        const bool recog_due = recognition_due(detections, cached_detections,
                                               frame_count - last_recog_frame,
                                               cfg.recog_interval);
        // The decoded frame reaches the CPU only when ArcFace or the recorder needs it.
        // NV12 stays NV12 so alignment converts only the face region; an NV16 frame (4:2:2
        // USB camera) converts to BGR once.
        std::vector<uint8_t> nv12;
        int nv12_w = 0, nv12_h = 0;
        cv::Mat frame_bgr;
        if ((recog_due && !detections.empty()) || !cfg.output_sink.empty()) {
            const auto frames = simaai::neat::tensors_from_sample(
                joined_field(sample, "frame", 0), true);
            if (frames.empty()) throw std::runtime_error("joined frame field has no tensor");
            if (frames.front().image_format() == simaai::neat::ImageSpec::PixelFormat::NV16) {
                std::string err;
                if (!sima_examples::nv16_to_bgr(frames.front(), frame_bgr, err))
                    throw std::runtime_error("NV16 frame: " + err);
            } else {
                nv12   = frames.front().copy_nv12_contiguous();
                nv12_w = frames.front().width();
                nv12_h = frames.front().height();
            }
        }

        if (recog_due) {
            cached_matches.clear();
            cached_detections = detections;
            for (const auto& det : detections) {
                const auto ta = Clock::now();
                const cv::Mat crop = frame_bgr.empty()
                    ? face_recog::align_face_nv12(nv12.data(), nv12_w, nv12_h, det.landmarks)
                    : face_recog::align_face(frame_bgr, det.landmarks);
                const auto tb = Clock::now();
                if (crop.empty()) {
                    cached_matches.push_back({cfg.match.unknown_label, -1.f, -1});
                    continue;
                }
                const auto emb = run_arcface(arcface_run, crop, cfg.timeout_ms);
                const auto tc  = Clock::now();
                cached_matches.push_back(face_recog::match_embedding(emb, gallery, cfg.match));
                if (cfg.test_mode) {
                    timings.align_ms.push_back(Ms(tb - ta).count());
                    timings.arcface_ms.push_back(Ms(tc - tb).count());
                    timings.match_ms.push_back(Ms(Clock::now() - tc).count());
                }
            }
            last_recog_frame = frame_count;
        }

        if (metadata_sender) {
            nlohmann::json objects = nlohmann::json::array();
            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& d   = detections[i];
                const bool have = i < cached_matches.size();
                const float x1 = std::clamp(d.x1, 0.f, (float)frame_w);
                const float y1 = std::clamp(d.y1, 0.f, (float)frame_h);
                const float x2 = std::clamp(d.x2, x1, (float)frame_w);
                const float y2 = std::clamp(d.y2, y1, (float)frame_h);
                objects.push_back({
                    {"id",         "face_" + std::to_string(i + 1)},
                    {"label",      have ? cached_matches[i].name : cfg.match.unknown_label},
                    {"confidence", d.score},
                    {"similarity", have ? cached_matches[i].score : -1.f},
                    {"bbox",       {x1, y1, x2 - x1, y2 - y1}},
                });
            }
            const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
            const std::string frame_id = std::to_string(
                sample.frame_id >= 0 ? sample.frame_id : (int64_t)frame_count);
            std::string meta_err;
            if (!metadata_sender->send_metadata("object-detection",
                                                nlohmann::json{{"objects", objects}}.dump(),
                                                ts_ms, frame_id, &meta_err) &&
                frame_count % 100 == 0)
                std::cerr << "[stream] metadata send failed: " << meta_err << "\n";
        }

        if (!cfg.output_sink.empty()) {
            const auto tr = Clock::now();
            cv::Mat bgr = frame_bgr;
            if (bgr.empty())
                cv::cvtColor(cv::Mat(nv12_h * 3 / 2, nv12_w, CV_8UC1, nv12.data()), bgr,
                             cv::COLOR_YUV2BGR_NV12);
            face_recog::draw_overlay(bgr, detections, cached_matches, cfg.overlay);
            if (write_video) writer.write(bgr);
            else if (frame_count == 0 && !cv::imwrite(cfg.output_sink, bgr))
                throw std::runtime_error("failed to write image output: " + cfg.output_sink);
            if (cfg.test_mode) timings.record_ms.push_back(Ms(Clock::now() - tr).count());
        }

        if (cfg.test_mode) {
            for (size_t i = 0; i < cached_matches.size(); ++i)
                printf("  face[%zu] → %-20s  similarity=%.4f\n",
                       i, cached_matches[i].name.c_str(), cached_matches[i].score);
            timings.pull_ms.push_back(Ms(t1 - t0).count());
            timings.decode_ms.push_back(Ms(t2 - t1).count());
            timings.e2e_ms.push_back(Ms(Clock::now() - t0).count());
        }
        det_stats.add(detections, cached_matches);
        ++frame_count;

        if (frame_count % 100 == 0) {
            std::cout << "[" << frame_count << "] faces=" << detections.size();
            for (size_t i = 0; i < cached_matches.size(); ++i)
                std::cout << "  face[" << i << "]=" << cached_matches[i].name
                          << cv::format("(%.3f)", cached_matches[i].score);
            std::cout << "\n";
        }
    }

    const double total_s = loop_start
        ? std::chrono::duration<double>(Clock::now() - *loop_start).count() : 0.0;

    if (cfg.test_mode && scrfd_scope && arcface_scope) {
        std::cout << "\n[SDK SCRFD]\n"  << scrfd_scope->stop().to_text()   << "\n";
        std::cout << "[SDK ArcFace]\n" << arcface_scope->stop().to_text() << "\n";
        print_timings(timings, frame_count, total_s > 0 ? total_s : 1.0);
    }

    run.close();
    arcface_run.close();
    if (write_video) writer.release();
    det_stats.print();

    std::cout << "Done: " << frame_count << " frames in "
              << cv::format("%.2f", total_s) << " s ("
              << cv::format("%.1f", frame_count / std::max(total_s, 0.001)) << " FPS)\n";
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // --enroll as the first argument dispatches to enrollment, so one installed binary
    // covers both gallery enrollment and recognition.
    if (argc > 1 && std::string(argv[1]) == "--enroll")
        return run_enrollment_mode(argc, argv);

    std::signal(SIGINT,  [](int){ g_stop = true; });
    std::signal(SIGTERM, [](int){ g_stop = true; });
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    try {
        const AppConfig cfg = parse_args(argc, argv);
        if (cfg.input_uri.empty() && cfg.usb_camera.empty())
            throw std::runtime_error("no input: set input.uri or input.usb.camera in config.yaml, "
                                     "or pass --input <rtsp-url|video.mp4>");
        if (!cfg.input_uri.empty() && !cfg.usb_camera.empty())
            throw std::runtime_error("set only one input: input.uri or input.usb.camera");
        if (!fs::exists(cfg.scrfd_model))
            throw std::runtime_error("SCRFD model not found: " + cfg.scrfd_model);
        if (!fs::exists(cfg.arcface_model))
            throw std::runtime_error("ArcFace model not found: " + cfg.arcface_model);

        face_recog::Gallery gallery;
        if (fs::exists(cfg.gallery_path)) {
            gallery = face_recog::load_gallery(cfg.gallery_path);
            std::cout << "Gallery: " << gallery.entries.size() << " identities from "
                      << cfg.gallery_path << "\n";
        } else {
            std::cerr << "Warning: gallery not found at " << cfg.gallery_path
                      << " — all faces will be labelled Unknown\n";
        }
        return run_pipeline(cfg, gallery);
    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 1;
    }
}
