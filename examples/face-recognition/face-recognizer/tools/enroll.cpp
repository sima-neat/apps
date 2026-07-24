/**
 * @file enroll.cpp
 * Offline enrollment tool: scan a labeled image folder, detect + align + embed
 * each face with SCRFD + ArcFace, then write a gallery file.
 *
 * Expected folder layout:
 *   gallery_images/
 *     Alice/
 *       photo1.jpg
 *       photo2.png
 *     Bob/
 *       ...
 *
 * Usage:
 *   face-enroll --images <dir> --gallery <out.bin>
 *               [--config <path>] [--max-per-person <N>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include "scrfd_decode.h"
#include "align.h"
#include "gallery.h"
#include "match.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

static bool is_image(const fs::path& p) {
    std::string ext = p.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static simaai::neat::Run build_scrfd_run(const std::string& model_path, int infer_w, int infer_h) {
    simaai::neat::Model::Options opt;
    opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    opt.preprocess.input_max_width  = infer_w;
    opt.preprocess.input_max_height = infer_h;
    opt.preprocess.input_max_depth  = 3;
    simaai::neat::Model model(model_path, opt);

    simaai::neat::Graph g;
    g.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    g.add(model);
    g.add(simaai::neat::nodes::Output());

    cv::Mat dummy(infer_h, infer_w, CV_32FC3, cv::Scalar(0, 0, 0));
    simaai::neat::RunOptions ropt;
    ropt.output_memory = simaai::neat::OutputMemory::Owned;
    return g.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(dummy)}, ropt);
}

static simaai::neat::Run build_arcface_run(const std::string& model_path) {
    simaai::neat::Model::Options opt;
    opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    opt.preprocess.input_max_width  = face_recog::kArcFaceW;
    opt.preprocess.input_max_height = face_recog::kArcFaceH;
    opt.preprocess.input_max_depth  = 3;
    simaai::neat::Model model(model_path, opt);

    simaai::neat::Graph g;
    g.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    g.add(model);
    g.add(simaai::neat::nodes::Output());

    cv::Mat dummy(face_recog::kArcFaceH, face_recog::kArcFaceW, CV_32FC3, cv::Scalar(0, 0, 0));
    simaai::neat::RunOptions ropt;
    ropt.output_memory = simaai::neat::OutputMemory::Owned;
    return g.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(dummy)}, ropt);
}

} // namespace

static int enroll_from_video(
    const std::string& video_path,
    const std::string& name,
    simaai::neat::Run& scrfd_run,
    simaai::neat::Run& arcface_run,
    const face_recog::ScrfdConfig& scrfd_cfg,
    int timeout_ms,
    float min_score,
    int sample_every,
    face_recog::GalleryBuilder& builder)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot open video: " + video_path);

    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps       = cap.get(cv::CAP_PROP_FPS);
    std::cout << "[VIDEO] " << video_path << " — "
              << total_frames << " frames @ " << fps << " fps, "
              << "sampling every " << sample_every << " frames, "
              << "min_score=" << min_score << "\n";

    int frame_idx = 0, enrolled = 0, skipped = 0;
    cv::Mat bgr;
    while (cap.read(bgr)) {
        ++frame_idx;
        if ((frame_idx % sample_every) != 0) continue;

        auto [tensor, pad_meta] = face_recog::preprocess_scrfd(bgr);
        if (!scrfd_run.push(simaai::neat::TensorList{tensor}))
            throw std::runtime_error("SCRFD push failed: " + scrfd_run.last_error());
        const auto sample = scrfd_run.pull(timeout_ms);
        if (!sample) { std::cerr << "  [skip] SCRFD timeout at frame " << frame_idx << "\n"; ++skipped; continue; }

        const auto dets = face_recog::decode_scrfd(
            face_recog::collect_tensors(*sample), scrfd_cfg, pad_meta);

        if (dets.empty()) { ++skipped; continue; }

        const auto& best = *std::max_element(
            dets.begin(), dets.end(),
            [](const face_recog::Detection& a, const face_recog::Detection& b) {
                return a.score < b.score;
            });

        if (best.score < min_score) {
            std::cout << "  [skip] frame " << frame_idx << " score=" << best.score << " < " << min_score << "\n";
            ++skipped; continue;
        }

        const cv::Mat crop = face_recog::align_face(bgr, best.landmarks);
        const cv::Mat f32  = face_recog::preprocess_arcface_crop(crop);
        const auto arc_t   = face_recog::tensor_from_hwc_f32(f32);

        if (!arcface_run.push(simaai::neat::TensorList{arc_t}))
            throw std::runtime_error("ArcFace push failed: " + arcface_run.last_error());
        const auto arc_sample = arcface_run.pull(timeout_ms);
        if (!arc_sample) { std::cerr << "  [skip] ArcFace timeout at frame " << frame_idx << "\n"; ++skipped; continue; }

        const auto emb_tensors = face_recog::collect_tensors(*arc_sample);
        auto emb = face_recog::tensor_to_f32(emb_tensors[0]);
        face_recog::l2_normalize(emb);

        builder.add(name, emb);
        ++enrolled;
        std::cout << "  [" << name << "] frame " << frame_idx
                  << " → enrolled (score=" << best.score << ", total=" << enrolled << ")\n";
    }

    std::cout << "[VIDEO] Done: " << enrolled << " enrolled, " << skipped << " skipped\n";
    return enrolled;
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    std::string images_dir;
    std::string video_path;
    std::string video_name;
    std::string gallery_out = "gallery.bin";
    std::string config_path_str = sima_examples::default_config_path(
        SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).string();
    int   max_per_person = 0;   // 0 = unlimited
    int   sample_every   = 5;   // sample every Nth frame from video
    float min_score      = 0.75f; // minimum SCRFD confidence for video frames

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--images"       && i+1 < argc) images_dir    = argv[++i];
        else if (a == "--video"        && i+1 < argc) video_path    = argv[++i];
        else if (a == "--name"         && i+1 < argc) video_name    = argv[++i];
        else if (a == "--gallery"      && i+1 < argc) gallery_out   = argv[++i];
        else if (a == "--config"       && i+1 < argc) config_path_str = argv[++i];
        else if (a == "--max-per-person" && i+1 < argc) max_per_person = std::stoi(argv[++i]);
        else if (a == "--sample-every" && i+1 < argc) sample_every  = std::stoi(argv[++i]);
        else if (a == "--min-score"    && i+1 < argc) min_score     = std::stof(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage:\n"
                "  face-enroll --images <dir> --gallery <out.bin>\n"
                "              [--config <path>] [--max-per-person <N>]\n"
                "\n"
                "  face-enroll --video <file.mp4> --name <person> --gallery <out.bin>\n"
                "              [--sample-every <N>]   (default 5 — every 5th frame)\n"
                "              [--min-score <0-1>]    (default 0.75 — SCRFD confidence)\n"
                "              [--config <path>]\n"
                "\n"
                "Modes can be combined: --images + --video both contribute to the same gallery.\n";
            return 0;
        }
    }

    const bool have_images = !images_dir.empty();
    const bool have_video  = !video_path.empty();

    if (!have_images && !have_video) {
        std::cerr << "Error: provide --images <dir> and/or --video <file> --name <person>\n";
        return 1;
    }
    if (have_video && video_name.empty()) {
        std::cerr << "Error: --name <person> is required with --video\n";
        return 1;
    }
    if (have_images && !fs::is_directory(images_dir)) {
        std::cerr << "Error: not a directory: " << images_dir << "\n";
        return 1;
    }

    const auto raw = sima_examples::ScalarConfig::load(config_path_str);
    const std::string scrfd_model  = raw.string_or("scrfd.model",   "assets/models/scrfd_2.5g_model.tar.gz");
    const std::string arcface_model= raw.string_or("arcface.model", "assets/models/arcface_mbf_model.tar.gz");
    const int timeout_ms           = raw.int_or("runtime.timeout_ms", 20000);

    face_recog::ScrfdConfig scrfd_cfg;
    scrfd_cfg.conf_threshold = static_cast<float>(raw.double_or("scrfd.conf_threshold", 0.5));
    scrfd_cfg.nms_iou        = static_cast<float>(raw.double_or("scrfd.nms_iou", 0.4));
    scrfd_cfg.top_k          = raw.int_or("scrfd.top_k", 100);
    scrfd_cfg.keep_top_k     = raw.int_or("scrfd.keep_top_k", 5);
    scrfd_cfg.cls_per_anchor = raw.int_or("scrfd.cls_per_anchor", 1);
    scrfd_cfg.num_anchors    = raw.int_or("scrfd.num_anchors", 2);

    try {
        std::cout << "[BUILD] SCRFD pipeline...\n";
        auto scrfd_run   = build_scrfd_run(scrfd_model, scrfd_cfg.infer_w, scrfd_cfg.infer_h);
        std::cout << "[BUILD] ArcFace pipeline...\n";
        auto arcface_run = build_arcface_run(arcface_model);
        std::cout << "[BUILD] Done.\n";

        face_recog::GalleryBuilder builder;

        // Load existing gallery so new enrollments append rather than overwrite.
        if (fs::exists(gallery_out)) {
            try {
                const auto existing = face_recog::load_gallery(gallery_out);
                for (const auto& e : existing.entries)
                    builder.add(e.name, e.embedding);
                std::cout << "[GALLERY] Loaded " << existing.entries.size()
                          << " existing identit" << (existing.entries.size() == 1 ? "y" : "ies")
                          << " from " << gallery_out << "\n";
            } catch (const std::exception& ex) {
                std::cerr << "[GALLERY] Could not load existing gallery (will overwrite): "
                          << ex.what() << "\n";
            }
        }

        int total_images = 0, total_faces = 0, skipped = 0;

        if (have_images) {
            for (const auto& person_entry : fs::directory_iterator(images_dir)) {
                if (!person_entry.is_directory()) continue;
                const std::string name = person_entry.path().filename().string();

                std::vector<fs::path> images;
                for (const auto& img_e : fs::directory_iterator(person_entry.path()))
                    if (img_e.is_regular_file() && is_image(img_e.path()))
                        images.push_back(img_e.path());
                std::sort(images.begin(), images.end());

                if (max_per_person > 0 && images.size() > static_cast<size_t>(max_per_person))
                    images.resize(max_per_person);

                int person_faces = 0;
                for (const auto& img_path : images) {
                    cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
                    if (bgr.empty()) {
                        std::cerr << "  [skip] cannot read: " << img_path.filename() << "\n";
                        ++skipped; continue;
                    }

                    auto [tensor, pad_meta] = face_recog::preprocess_scrfd(bgr);
                    if (!scrfd_run.push(simaai::neat::TensorList{tensor}))
                        throw std::runtime_error("SCRFD push failed: " + scrfd_run.last_error());
                    const auto sample = scrfd_run.pull(timeout_ms);
                    if (!sample) { std::cerr << "  [skip] SCRFD timeout\n"; ++skipped; continue; }

                    const auto dets = face_recog::decode_scrfd(
                        face_recog::collect_tensors(*sample), scrfd_cfg, pad_meta);

                    if (dets.empty()) {
                        std::cerr << "  [skip] no face in: " << img_path.filename() << "\n";
                        ++skipped; continue;
                    }

                    const auto& best = *std::max_element(
                        dets.begin(), dets.end(),
                        [](const face_recog::Detection& a, const face_recog::Detection& b) {
                            return a.score < b.score;
                        });

                    const cv::Mat crop = face_recog::align_face(bgr, best.landmarks);
                    const cv::Mat f32  = face_recog::preprocess_arcface_crop(crop);
                    const auto arc_t   = face_recog::tensor_from_hwc_f32(f32);

                    if (!arcface_run.push(simaai::neat::TensorList{arc_t}))
                        throw std::runtime_error("ArcFace push failed: " + arcface_run.last_error());
                    const auto arc_sample = arcface_run.pull(timeout_ms);
                    if (!arc_sample) { std::cerr << "  [skip] ArcFace timeout\n"; ++skipped; continue; }

                    const auto emb_tensors = face_recog::collect_tensors(*arc_sample);
                    auto emb = face_recog::tensor_to_f32(emb_tensors[0]);
                    face_recog::l2_normalize(emb);

                    builder.add(name, emb);
                    ++person_faces;
                    ++total_faces;
                    ++total_images;

                    std::cout << "  [" << name << "] " << img_path.filename().string()
                              << " → face enrolled (score=" << best.score << ")\n";
                }
                std::cout << "[" << name << "] " << person_faces << " face(s) enrolled\n";
            }
        }

        if (have_video) {
            const int n = enroll_from_video(video_path, video_name,
                                            scrfd_run, arcface_run,
                                            scrfd_cfg, timeout_ms,
                                            min_score, sample_every,
                                            builder);
            total_faces  += n;
            total_images += n;
        }

        const auto gallery = builder.finish();
        face_recog::save_gallery(gallery, gallery_out);

        std::cout << "\nEnrollment complete:\n"
                  << "  Frames/images processed : " << total_images << "\n"
                  << "  Faces enrolled          : " << total_faces  << "\n"
                  << "  Skipped                 : " << skipped      << "\n"
                  << "  Identities              : " << gallery.entries.size() << "\n"
                  << "  Gallery saved           : " << gallery_out << "\n";

        scrfd_run.close();
        arcface_run.close();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
