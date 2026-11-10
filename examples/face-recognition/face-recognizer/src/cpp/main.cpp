// Face recognizer: SCRFD finds the faces in a frame, ArcFace turns each face into an
// embedding, and the embedding is matched against a gallery of enrolled people.
//
// Recognition reads an RTSP stream, a USB camera or an H.264 video file and reports each
// face's box and name to Insight. --enroll builds the gallery from images or a video.
#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/io/MetadataSender.h>
#include <nlohmann/json.hpp>

#include "align.h"
#include "gallery.h"
#include "match.h"
#include "scrfd_decode.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs   = std::filesystem;
namespace neat = simaai::neat;

// ── configuration ─────────────────────────────────────────────────────────────

struct AppConfig {
    std::string scrfd_model;
    std::string arcface_model;
    std::string gallery_path = "gallery.bin";
    std::string input_uri;   // RTSP URL or H.264 video file
    std::string usb_camera;  // a usb_camera_ids() ID; used when input_uri is empty
    int usb_width = 1280, usb_height = 720, usb_fps = 30;
    std::string insight_host;  // empty = no Insight output
    int insight_video_port    = 9000;
    int insight_metadata_port = 9100;
    int timeout_ms            = 20000;
    face_recog::ScrfdConfig scrfd;
    face_recog::MatchConfig match;

    // Command line only.
    int  max_frames = 0;   // 0 = unlimited
    int  rtsp_fps   = -1;  // -1 = read the rate from the stream
    bool test_mode  = false;
    bool enroll     = false;
    std::string enroll_images, enroll_video, enroll_name;
    int   sample_every   = 5;
    float min_score      = 0.75f;
    int   max_per_person = 0;  // 0 = unlimited
};

static AppConfig load_config(const fs::path& path) {
    const auto raw = sima_examples::ScalarConfig::load(path);
    // Relative paths in config.yaml are relative to the example directory, which holds
    // src/common/config.yaml.
    const fs::path example_dir =
        fs::absolute(path).lexically_normal().parent_path().parent_path().parent_path();
    auto resolve = [&](const std::string& p) -> std::string {
        if (p.empty() || fs::path(p).is_absolute() || p.find("://") != std::string::npos) return p;
        return (example_dir / p).lexically_normal().string();
    };

    AppConfig cfg;
    cfg.scrfd_model   = resolve(raw.string_or("scrfd.model", ""));
    cfg.arcface_model = resolve(raw.string_or("arcface.model", ""));
    cfg.gallery_path  = resolve(raw.string_or("gallery.path", cfg.gallery_path));
    cfg.input_uri     = resolve(raw.string_or("input.uri", ""));
    cfg.usb_camera    = raw.string_or("input.usb.camera", "");
    cfg.usb_width     = raw.int_or("input.usb.width", cfg.usb_width);
    cfg.usb_height    = raw.int_or("input.usb.height", cfg.usb_height);
    cfg.usb_fps       = raw.int_or("input.usb.fps", cfg.usb_fps);
    cfg.insight_host  = raw.string_or("output.insight.host", "");
    cfg.insight_video_port    = raw.int_or("output.insight.video_port", cfg.insight_video_port);
    cfg.insight_metadata_port =
        raw.int_or("output.insight.metadata_port", cfg.insight_metadata_port);
    cfg.timeout_ms            = raw.int_or("runtime.timeout_ms", cfg.timeout_ms);
    cfg.scrfd.conf_threshold =
        static_cast<float>(raw.double_or("scrfd.conf_threshold", cfg.scrfd.conf_threshold));
    cfg.scrfd.nms_iou    = static_cast<float>(raw.double_or("scrfd.nms_iou", cfg.scrfd.nms_iou));
    cfg.scrfd.top_k      = raw.int_or("scrfd.top_k", cfg.scrfd.top_k);
    cfg.scrfd.keep_top_k = raw.int_or("scrfd.keep_top_k", cfg.scrfd.keep_top_k);
    cfg.match.threshold =
        static_cast<float>(raw.double_or("match.threshold", cfg.match.threshold));
    cfg.match.margin     = static_cast<float>(raw.double_or("match.margin", cfg.match.margin));
    cfg.match.unknown_label = raw.string_or("match.unknown_label", cfg.match.unknown_label);
    return cfg;
}

static void print_usage() {
    std::cout
        << "Usage:\n"
        << "  face-recognizer [--config <path>] [--input <rtsp-url|video.mp4>] [--gallery <path>]\n"
        << "                  [--scrfd-model <path>] [--arcface-model <path>]\n"
        << "                  [--max-frames <n>] [--rtsp-fps <n>] [--test]\n"
        << "  face-recognizer --enroll --images <dir> [--max-per-person <n>]\n"
        << "  face-recognizer --enroll --video <file> --name <person> [--sample-every <n>]\n"
        << "\n"
        << "Recognition reads input.uri or input.usb.camera from config.yaml and, when\n"
        << "output.insight.host is set, sends the video and each face's box and name to Insight.\n"
        << "Enrollment adds people to the gallery; --images expects one folder per person.\n"
        << "Both enrollment modes accept --gallery, --config and --min-score <0-1>.\n";
}

// config.yaml is loaded first; command-line flags then override it.
static AppConfig parse_args(int argc, char** argv) {
    fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--config") config_path = argv[i + 1];
    AppConfig cfg = load_config(config_path);

    auto value = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string(argv[i]) + " requires a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--config") {
            value(i);
        } else if (arg == "--input") {
            cfg.input_uri = value(i);
            cfg.usb_camera.clear();
        } else if (arg == "--gallery") {
            cfg.gallery_path = value(i);
        } else if (arg == "--scrfd-model") {
            cfg.scrfd_model = value(i);
        } else if (arg == "--arcface-model") {
            cfg.arcface_model = value(i);
        } else if (arg == "--max-frames") {
            cfg.max_frames = std::stoi(value(i));
        } else if (arg == "--rtsp-fps") {
            cfg.rtsp_fps = std::stoi(value(i));
        } else if (arg == "--test") {
            cfg.test_mode = true;
        } else if (arg == "--enroll") {
            cfg.enroll = true;
        } else if (arg == "--images") {
            cfg.enroll_images = value(i);
        } else if (arg == "--video") {
            cfg.enroll_video = value(i);
        } else if (arg == "--name") {
            cfg.enroll_name = value(i);
        } else if (arg == "--sample-every") {
            cfg.sample_every = std::stoi(value(i));
        } else if (arg == "--min-score") {
            cfg.min_score = std::stof(value(i));
        } else if (arg == "--max-per-person") {
            cfg.max_per_person = std::stoi(value(i));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    // --test prints every result to the console instead of sending it to Insight.
    if (cfg.test_mode) cfg.insight_host.clear();
    return cfg;
}

// ── models ────────────────────────────────────────────────────────────────────

// A model fed one float tensor at a time: ArcFace (a 112x112 aligned face) and, for the
// still images of enrollment, SCRFD (a 640x640 letterboxed image).
static neat::Run build_tensor_run(const std::string& model_path, int width, int height) {
    neat::Model::Options opt;
    opt.preprocess.kind             = neat::InputKind::Tensor;
    opt.preprocess.input_max_width  = width;
    opt.preprocess.input_max_height = height;
    opt.preprocess.input_max_depth  = 3;
    neat::Model model(model_path, opt);

    neat::Graph graph;
    graph.add(neat::nodes::Input(model.input_appsrc_options(true)));
    graph.add(model);
    graph.add(neat::nodes::Output());

    const cv::Mat seed(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    neat::RunOptions run_opt;
    run_opt.output_memory = neat::OutputMemory::Owned;
    return graph.build(neat::TensorList{face_recog::tensor_from_hwc_f32(seed)}, run_opt);
}

static std::vector<neat::Tensor> run_tensor(neat::Run& run, const neat::Tensor& input,
                                            int timeout_ms, const std::string& what) {
    if (!run.push(neat::TensorList{input}))
        throw std::runtime_error(what + " push failed: " + run.last_error());
    const auto sample = run.pull(timeout_ms);
    // A result still in flight would come back for the next input, so a timeout is fatal.
    if (!sample) throw std::runtime_error(what + " timed out");
    return face_recog::collect_tensors(*sample);
}

// The one path from a detected face to its embedding, shared by enrollment and recognition.
static face_recog::Embedding embed_face(neat::Run& arcface, const cv::Mat& bgr,
                                        const face_recog::Landmarks& landmarks, int timeout_ms) {
    const cv::Mat crop = face_recog::align_face(bgr, landmarks);
    const auto out = run_tensor(
        arcface, face_recog::tensor_from_hwc_f32(face_recog::preprocess_arcface_crop(crop)),
        timeout_ms, "ArcFace");
    if (out.empty()) throw std::runtime_error("ArcFace returned no tensors");
    return face_recog::tensor_to_f32(out[0]);
}

// ── enrollment ────────────────────────────────────────────────────────────────

static bool is_image(const fs::path& p) {
    std::string ext = p.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static int run_enrollment(const AppConfig& cfg) {
    const bool have_images = !cfg.enroll_images.empty();
    const bool have_video  = !cfg.enroll_video.empty();
    if (!have_images && !have_video)
        throw std::runtime_error("--enroll needs --images <dir> or --video <file> --name <person>");
    if (have_video && cfg.enroll_name.empty())
        throw std::runtime_error("--video needs --name <person>");
    if (!have_video && !cfg.enroll_name.empty())
        throw std::runtime_error("--name applies to --video; --images names people by folder");
    if (have_images && !fs::is_directory(cfg.enroll_images))
        throw std::runtime_error("not a directory: " + cfg.enroll_images);
    if (cfg.sample_every <= 0) throw std::runtime_error("--sample-every must be positive");

    std::cout << "[BUILD] Building SCRFD and ArcFace pipelines...\n";
    auto scrfd   = build_tensor_run(cfg.scrfd_model, face_recog::kScrfdInputW,
                                    face_recog::kScrfdInputH);
    auto arcface =
        build_tensor_run(cfg.arcface_model, face_recog::kArcFaceW, face_recog::kArcFaceH);

    // New samples are added to the people already in the gallery. An existing gallery
    // that cannot be read stops enrollment rather than being replaced.
    face_recog::GalleryBuilder builder;
    if (fs::exists(cfg.gallery_path))
        for (const auto& entry : face_recog::load_gallery(cfg.gallery_path).entries)
            builder.add(entry.name, entry.raw_mean, entry.sample_count);

    int enrolled = 0, skipped = 0;
    // Enrolls the highest-scoring face of one image under `name`.
    auto enroll_image = [&](const std::string& name, const cv::Mat& bgr, const std::string& what) {
        auto [tensor, letterbox] = face_recog::preprocess_scrfd(bgr);
        const auto faces = face_recog::decode_scrfd(
            run_tensor(scrfd, tensor, cfg.timeout_ms, "SCRFD"), cfg.scrfd, letterbox);
        if (faces.empty() || faces.front().score < cfg.min_score) {
            std::cout << "  [skip] " << what << ": no face scoring at least " << cfg.min_score
                      << "\n";
            ++skipped;
            return;
        }
        auto embedding = embed_face(arcface, bgr, faces.front().landmarks, cfg.timeout_ms);
        face_recog::l2_normalize(embedding);
        builder.add(name, embedding);
        ++enrolled;
        std::cout << "  [" << name << "] " << what << " (score " << faces.front().score << ")\n";
    };

    if (have_images) {
        // Sorted, so the same folder always produces the same gallery.
        std::vector<fs::path> people;
        for (const auto& entry : fs::directory_iterator(cfg.enroll_images))
            if (entry.is_directory()) people.push_back(entry.path());
        std::sort(people.begin(), people.end());
        for (const auto& person : people) {
            std::vector<fs::path> images;
            for (const auto& file : fs::directory_iterator(person))
                if (file.is_regular_file() && is_image(file.path())) images.push_back(file.path());
            std::sort(images.begin(), images.end());
            if (cfg.max_per_person > 0 && images.size() > static_cast<size_t>(cfg.max_per_person))
                images.resize(cfg.max_per_person);
            for (const auto& image : images) {
                const cv::Mat bgr = cv::imread(image.string(), cv::IMREAD_COLOR);
                if (bgr.empty()) {
                    std::cout << "  [skip] cannot read " << image.filename().string() << "\n";
                    ++skipped;
                    continue;
                }
                enroll_image(person.filename().string(), bgr, image.filename().string());
            }
        }
    }
    if (have_video) {
        cv::VideoCapture video(cfg.enroll_video);
        if (!video.isOpened()) throw std::runtime_error("cannot open video: " + cfg.enroll_video);
        cv::Mat bgr;
        for (int frame = 1; video.read(bgr); ++frame)
            if (frame % cfg.sample_every == 0)
                enroll_image(cfg.enroll_name, bgr, "frame " + std::to_string(frame));
    }
    scrfd.close();
    arcface.close();
    if (enrolled == 0)
        throw std::runtime_error("no faces were enrolled; check that the input shows clear, "
                                 "forward-facing faces");

    // Write beside the gallery, then rename, so a failed save leaves the old gallery intact.
    const auto gallery = builder.finish();
    const fs::path tmp = cfg.gallery_path + ".tmp";
    face_recog::save_gallery(gallery, tmp);
    fs::rename(tmp, cfg.gallery_path);
    std::cout << "Enrolled " << enrolled << " face(s), skipped " << skipped << "; "
              << gallery.entries.size() << " people in " << cfg.gallery_path << "\n";
    return 0;
}

// ── recognition ───────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};
static constexpr const char* kJoinedOutput = "face_output";

// SCRFD fed decoded video frames: the EV74 letterboxes, converts and normalizes each frame.
static neat::Model::Options scrfd_frame_options(int max_w, int max_h, bool nv16_frames) {
    neat::Model::Options opt;
    opt.preprocess.kind   = neat::InputKind::Image;
    opt.preprocess.enable = neat::AutoFlag::On;
    opt.preprocess.color_convert.input_format =
        nv16_frames ? neat::PreprocessColorFormat::NV16 : neat::PreprocessColorFormat::NV12;
    opt.preprocess.input_max_width  = max_w;
    opt.preprocess.input_max_height = max_h;
    opt.preprocess.input_max_depth  = 3;
    // Both settings must match how SCRFD was calibrated; neither is inferred from the model
    // package. Left on Auto, normalization stays off (a 255x input-scale error) and the
    // letterbox bars are grey (114) instead of black. On a 1000-frame clip with one face,
    // frames reporting extra faces: 50.8% with both wrong, 11.5% with both set.
    opt.preprocess.resize.enable    = neat::AutoFlag::On;
    opt.preprocess.resize.mode      = neat::ResizeMode::Letterbox;
    opt.preprocess.resize.pad_value = 0;
    opt.preprocess.normalize.enable = neat::AutoFlag::On;
    opt.preprocess.normalize.mean   = {0.0f, 0.0f, 0.0f};
    opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
    return opt;
}

static neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg) {
    neat::UsbCameraInputOptions opt;
    opt.camera_name   = cfg.usb_camera;
    opt.width         = static_cast<std::uint32_t>(cfg.usb_width);
    opt.height        = static_cast<std::uint32_t>(cfg.usb_height);
    opt.framerate_num = static_cast<std::uint32_t>(cfg.usb_fps);
    return opt;
}

static const neat::Sample* find_field(const neat::Sample& sample, const std::string& label) {
    if (sample.stream_label == label) return &sample;
    for (const auto& field : sample.fields)
        if (const auto* hit = find_field(field, label)) return hit;
    return nullptr;
}

static const neat::Sample& joined_field(const neat::Sample& sample, const std::string& label,
                                        size_t bundle_index) {
    if (const auto* field = find_field(sample, label)) return *field;
    if (sample.kind == neat::SampleKind::Bundle && sample.fields.size() > bundle_index)
        return sample.fields[bundle_index];
    throw std::runtime_error("joined output missing " + label + " field");
}

// A decoded frame as BGR. H.264 decodes to NV12; a 4:2:2 USB camera decodes to NV16.
static cv::Mat frame_to_bgr(const neat::Sample& frame_sample) {
    const auto frames = neat::tensors_from_sample(frame_sample, true);
    if (frames.empty()) throw std::runtime_error("joined frame field has no tensor");
    cv::Mat bgr;
    std::string err;
    const bool nv16 = frames.front().image_format() == neat::ImageSpec::PixelFormat::NV16;
    const bool ok   = nv16 ? sima_examples::nv16_to_bgr(frames.front(), bgr, err)
                           : sima_examples::nv12_to_bgr(frames.front(), bgr, err);
    if (!ok) throw std::runtime_error("decoded frame: " + err);
    return bgr;
}

static void send_faces(neat::MetadataSender& sender, const neat::Sample& sample, int frame_index,
                       const std::vector<face_recog::Detection>& detections,
                       const std::vector<face_recog::MatchResult>& matches, int frame_w,
                       int frame_h) {
    nlohmann::json objects = nlohmann::json::array();
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d  = detections[i];
        const float x1 = std::clamp(d.x1, 0.f, static_cast<float>(frame_w));
        const float y1 = std::clamp(d.y1, 0.f, static_cast<float>(frame_h));
        const float x2 = std::clamp(d.x2, x1, static_cast<float>(frame_w));
        const float y2 = std::clamp(d.y2, y1, static_cast<float>(frame_h));
        objects.push_back({
            {"id", "face_" + std::to_string(i + 1)},
            {"label", matches[i].name},
            {"confidence", d.score},
            {"similarity", matches[i].score},
            {"bbox", {x1, y1, x2 - x1, y2 - y1}},
        });
    }
    // Insight pairs this message with the video frame that carries the same timestamp.
    const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
    const std::string frame_id =
        std::to_string(sample.frame_id >= 0 ? sample.frame_id : static_cast<int64_t>(frame_index));
    std::string err;
    if (!sender.send_metadata("object-detection", nlohmann::json{{"objects", objects}}.dump(),
                              timestamp_ms, frame_id, &err) &&
        frame_index % 100 == 0)
        std::cerr << "[stream] metadata send failed: " << err << "\n";
}

// One graph for every source:
//
//   RTSP / USB camera ─┬─→ VideoSender (the source's own stream → Insight)
//                      └─→ SimaDecode ─┬─ model ─→ SCRFD ─→ detections ─┐
//   video file (decodes itself) ───────┴─ frame ────────────────────────┴─ ByFrame join → app
//
// Insight receives the stream as the source sent it, with no re-encode. The join pairs
// every frame with the detections found in it, so each face is cropped from its own frame.
static int run_recognition(const AppConfig& cfg, const face_recog::Gallery& gallery) {
    const bool is_usb      = !cfg.usb_camera.empty();
    const bool is_rtsp     = !is_usb && (cfg.input_uri.rfind("rtsp://", 0) == 0 ||
                                         cfg.input_uri.rfind("rtsps://", 0) == 0);
    const bool is_live     = is_usb || is_rtsp;
    const bool use_insight = !cfg.insight_host.empty();
    if (use_insight && !is_live)
        throw std::runtime_error("Insight output needs an RTSP or USB camera input; a video "
                                 "file runs offline (clear output.insight.host)");

    // The frame size and rate are needed before the graph is built.
    int frame_w = 0, frame_h = 0, fps = 0;
    std::optional<neat::UsbCameraInfo> usb_info;
    if (is_usb) {
        if (cfg.usb_width <= 0 || cfg.usb_height <= 0 || cfg.usb_fps <= 0)
            throw std::runtime_error("input.usb.width, height and fps must be positive");
        // RTP/JPEG (RFC 2435) carries 8-pixel blocks and at most 2040 pixels per side.
        if (use_insight && (cfg.usb_width % 8 != 0 || cfg.usb_height % 8 != 0 ||
                            cfg.usb_width > 2040 || cfg.usb_height > 2040))
            throw std::runtime_error("input.usb.width and height must be multiples of 8 and at "
                                     "most 2040 to send the camera's JPEG to Insight");
        // Checks the camera once: connected, free, mode supported, one frame decodes.
        usb_info = neat::usb_camera_info(usb_camera_options(cfg));
        frame_w  = cfg.usb_width;
        frame_h  = cfg.usb_height;
        fps      = cfg.usb_fps;
    } else if (is_rtsp) {
        sima_examples::RtspStreamInfo info;
        (void)sima_examples::probe_rtsp_stream_info(cfg.input_uri,
                                                    sima_examples::RtspProbeOptions{}, info);
        frame_w = info.width;
        frame_h = info.height;
        fps     = cfg.rtsp_fps > 0 ? cfg.rtsp_fps : info.fps;
    } else {
        cv::VideoCapture probe(cfg.input_uri);
        if (!probe.isOpened())
            throw std::runtime_error("cannot open input video: " + cfg.input_uri);
        frame_w = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
        frame_h = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps     = static_cast<int>(std::lround(probe.get(cv::CAP_PROP_FPS)));
    }
    if (frame_w <= 0 || frame_h <= 0)
        throw std::runtime_error("could not read the input frame size");
    if (fps <= 0)
        throw std::runtime_error("could not read the input frame rate; pass --rtsp-fps <n>");
    const bool nv16_frames = is_usb && usb_info->decoded_format == neat::FormatTag::NV16;
    std::printf("[INPUT] %s %dx%d @ %d FPS\n",
                is_usb ? "USB camera" : is_rtsp ? "RTSP" : "video file", frame_w, frame_h, fps);

    // ArcFace first, so the live graph does not queue frames while it loads.
    std::cout << "[BUILD] Building ArcFace pipeline...\n";
    auto arcface =
        build_tensor_run(cfg.arcface_model, face_recog::kArcFaceW, face_recog::kArcFaceH);

    std::cout << "[BUILD] Building the source and SCRFD graph...\n";
    neat::Model scrfd(cfg.scrfd_model, scrfd_frame_options(frame_w, frame_h, nv16_frames));
    neat::GraphOptions graph_opt;
    graph_opt.advanced_execution.postprocess_target = "EV74";
    neat::Graph graph(graph_opt);
    auto decoded = neat::graphs::Branch("decoded", {"model", "frame"});

    if (!is_live) {
        neat::nodes::groups::VideoInputGroupOptions file_opt;
        file_opt.path = cfg.input_uri;
        neat::Graph file_source = neat::nodes::groups::VideoInputGroup(file_opt);
        graph.connect(file_source, decoded);
    } else {
        neat::Graph source("source");
        neat::SimaDecodeOptions decode_opt;
        decode_opt.raw_output = true;
        decode_opt.dec_width  = frame_w;
        decode_opt.dec_height = frame_h;
        decode_opt.dec_fps    = fps;
        if (is_usb) {
            source.add(neat::nodes::UsbCameraInput(usb_camera_options(cfg), *usb_info));
            decode_opt.type       = neat::SimaDecodeType::MJPEG;
            decode_opt.out_format = usb_info->decoded_format;
        } else {
            neat::nodes::groups::RtspEncodedInputOptions rtsp_opt;
            rtsp_opt.url                  = cfg.input_uri;
            rtsp_opt.latency_ms           = 50;
            rtsp_opt.source_fps           = fps;
            rtsp_opt.fallback_h264_width  = frame_w;
            rtsp_opt.fallback_h264_height = frame_h;
            source.add(neat::nodes::groups::RtspEncodedInput(rtsp_opt));
            decode_opt.type       = neat::SimaDecodeType::H264;
            decode_opt.out_format = neat::FormatTag::NV12;
        }
        neat::Graph decoder("decoder");
        decoder.add(neat::nodes::SimaDecode(decode_opt));
        graph.connect(source, decoder);
        graph.connect(decoder, decoded);

        if (use_insight) {
            using neat::nodes::groups::RtspCodec;
            auto video_opt = neat::nodes::groups::VideoSenderOptions::Passthrough(
                is_usb ? RtspCodec::MJPEG : RtspCodec::H264);
            video_opt.host            = cfg.insight_host;
            video_opt.video_port_base = cfg.insight_video_port;
            video_opt.channel         = 0;
            neat::Graph video("video");
            video.add(neat::nodes::groups::VideoSender(video_opt));
            // The viewer gets the newest frame; a slow model never holds the video back.
            neat::GraphLinkOptions latest;
            latest.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
            graph.connect(source, video, latest);
        }
    }

    neat::Graph model_graph("model");
    model_graph.connect(neat::nodes::Input("model"), scrfd);
    neat::Graph detections_graph("detections");
    detections_graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));
    neat::Graph frame_graph("frame");
    frame_graph.add(neat::nodes::Output("frame", neat::OutputOptions::EveryFrame(4)));
    auto joined = neat::graphs::Combine({"frame", "detections"}, kJoinedOutput,
                                        neat::CombinePolicy::ByFrame);
    graph.connect(decoded, model_graph);
    graph.connect(decoded, frame_graph);
    graph.connect(model_graph, detections_graph);
    graph.connect(frame_graph, joined);
    graph.connect(detections_graph, joined);

    // A live source keeps its newest frame; a file is processed frame by frame.
    neat::RunOptions run_opt;
    if (is_live) run_opt.preset = neat::RunPreset::Realtime;
    run_opt.queue_depth     = 3;
    run_opt.overflow_policy = is_live ? neat::OverflowPolicy::KeepLatest
                                      : neat::OverflowPolicy::Block;
    run_opt.output_memory   = neat::OutputMemory::ZeroCopy;
    auto run = graph.build(run_opt);

    std::unique_ptr<neat::MetadataSender> metadata_sender;
    if (use_insight) {
        neat::MetadataSenderOptions meta_opt;
        meta_opt.host               = cfg.insight_host;
        meta_opt.channel            = 0;
        meta_opt.metadata_port_base = cfg.insight_metadata_port;
        std::string err;
        metadata_sender = std::make_unique<neat::MetadataSender>(meta_opt, &err);
        if (!metadata_sender->ok())
            throw std::runtime_error("MetadataSender init failed: " + err);
        std::cout << "[stream] video -> udp://" << cfg.insight_host << ":" << cfg.insight_video_port
                  << ", metadata -> udp://" << cfg.insight_host << ":"
                  << metadata_sender->metadata_port() << " (Insight channel 0)\n";
    }

    const auto letterbox = face_recog::letterbox_for(frame_w, frame_h);
    int frame_count = 0;
    std::optional<std::chrono::steady_clock::time_point> first_frame;

    while (!g_stop && (cfg.max_frames <= 0 || frame_count < cfg.max_frames)) {
        neat::Sample sample;
        neat::PullError pull_err;
        const auto status = run.pull(kJoinedOutput, cfg.timeout_ms, sample, &pull_err);
        if (status == neat::PullStatus::Closed ||
            (status == neat::PullStatus::Error && pull_err.code == "runtime.pull")) {
            std::cerr << "[INPUT] End of stream.\n";
            break;
        }
        if (status != neat::PullStatus::Ok)
            throw std::runtime_error("pull failed: " + pull_err.code + " " + pull_err.message);
        if (!first_frame) first_frame = std::chrono::steady_clock::now();

        // Detect, then align, embed and match every face of this frame.
        const auto detections = face_recog::decode_scrfd(
            face_recog::collect_tensors(joined_field(sample, "detections", 1)), cfg.scrfd,
            letterbox);
        std::vector<face_recog::MatchResult> matches;
        if (!detections.empty()) {
            const cv::Mat bgr = frame_to_bgr(joined_field(sample, "frame", 0));
            for (const auto& det : detections)
                matches.push_back(face_recog::match_embedding(
                    embed_face(arcface, bgr, det.landmarks, cfg.timeout_ms), gallery, cfg.match));
        }

        if (metadata_sender)
            send_faces(*metadata_sender, sample, frame_count, detections, matches, frame_w,
                       frame_h);

        ++frame_count;
        if (cfg.test_mode || frame_count % 100 == 0) {
            if (!cfg.test_mode)
                std::cout << "[" << frame_count << "] faces=" << matches.size() << "\n";
            for (size_t i = 0; i < matches.size(); ++i)
                std::printf("  face[%zu] \xe2\x86\x92 %s  similarity=%.4f\n", i,
                            matches[i].name.c_str(), matches[i].score);
        }
    }

    const double seconds = first_frame
        ? std::chrono::duration<double>(std::chrono::steady_clock::now() - *first_frame).count()
        : 0.0;
    run.close();
    arcface.close();
    std::printf("Done: %d frames in %.2f s (%.1f FPS)\n", frame_count, seconds,
                frame_count / std::max(seconds, 0.001));
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    std::cout.setf(std::ios::unitbuf);

    try {
        const AppConfig cfg = parse_args(argc, argv);
        if (!fs::exists(cfg.scrfd_model))
            throw std::runtime_error("SCRFD model not found: " + cfg.scrfd_model);
        if (!fs::exists(cfg.arcface_model))
            throw std::runtime_error("ArcFace model not found: " + cfg.arcface_model);
        if (cfg.enroll) return run_enrollment(cfg);

        if (cfg.input_uri.empty() && cfg.usb_camera.empty())
            throw std::runtime_error("no input: set input.uri or input.usb.camera in config.yaml, "
                                     "or pass --input <rtsp-url|video.mp4>");
        if (!cfg.input_uri.empty() && !cfg.usb_camera.empty())
            throw std::runtime_error("set only one input: input.uri or input.usb.camera");

        face_recog::Gallery gallery;
        if (fs::exists(cfg.gallery_path)) {
            gallery = face_recog::load_gallery(cfg.gallery_path);
            std::cout << "Gallery: " << gallery.entries.size() << " people from "
                      << cfg.gallery_path << "\n";
        } else {
            std::cerr << "Warning: no gallery at " << cfg.gallery_path
                      << "; every face will be labelled " << cfg.match.unknown_label << "\n";
        }
        return run_recognition(cfg, gallery);
    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 1;
    }
}
