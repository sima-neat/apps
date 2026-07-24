#include "neat.h"
#include "support/runtime/config_utils.h"

#include "scrfd_decode.h"
#include "align.h"
#include "gallery.h"
#include "match.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── helpers ───────────────────────────────────────────────────────────────────

static void print_tensor_info(const simaai::neat::Tensor& t, int idx) {
    printf("  [%d] dtype=%-3d  shape=[", idx, static_cast<int>(t.dtype));
    for (size_t d = 0; d < t.shape.size(); ++d)
        printf("%lld%s", static_cast<long long>(t.shape[d]), d+1 < t.shape.size() ? "," : "");
    printf("]\n");
    if (t.dtype == simaai::neat::TensorDType::Float32) {
        const auto vals = face_recog::tensor_to_f32(t);
        if (!vals.empty()) {
            float mn = vals[0], mx = vals[0], sum = 0;
            for (float v : vals) { mn = std::min(mn,v); mx = std::max(mx,v); sum += v; }
            printf("       min=%.4f  max=%.4f  mean=%.4f  numel=%zu\n",
                   mn, mx, sum/vals.size(), vals.size());
        }
    }
}

static void validate_cosine(const face_recog::Embedding& sima_emb,
                             const fs::path& ref_path) {
    std::ifstream f(ref_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open reference: " + ref_path.string());
    const auto bytes = fs::file_size(ref_path);
    const size_t n = bytes / sizeof(float);
    std::vector<float> ref(n);
    f.read(reinterpret_cast<char*>(ref.data()), bytes);

    face_recog::Embedding sima = sima_emb;
    face_recog::l2_normalize(sima);
    face_recog::l2_normalize(ref);

    const float cosim = face_recog::cosine_similarity(sima, ref);
    printf("Cosine similarity (SiMa vs ONNX reference): %.6f\n", cosim);
    if (cosim >= 0.95f)
        printf("PASS ✓  (threshold ≥0.95)\n");
    else
        printf("FAIL ✗  (below 0.95 threshold — check quantization or preprocessing)\n");
}

// ── SCRFD test ────────────────────────────────────────────────────────────────

static void test_scrfd(const std::string& model_path, const cv::Mat& bgr_u8,
                       int timeout_ms, const face_recog::ScrfdConfig& scrfd_cfg) {
    printf("\n═══ SCRFD Detection Model ═══\n");
    printf("  Model: %s\n", model_path.c_str());
    printf("  Image: %dx%d\n", bgr_u8.cols, bgr_u8.rows);

    simaai::neat::Model::Options opt;
    opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    opt.preprocess.input_max_width  = scrfd_cfg.infer_w;
    opt.preprocess.input_max_height = scrfd_cfg.infer_h;
    opt.preprocess.input_max_depth  = 3;
    simaai::neat::Model model(model_path, opt);

    simaai::neat::Graph g;
    g.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    g.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
    g.add(simaai::neat::nodes::groups::MLA(model));
    g.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
    g.add(simaai::neat::nodes::Output());

    cv::Mat dummy(scrfd_cfg.infer_h, scrfd_cfg.infer_w, CV_32FC3, cv::Scalar(0,0,0));
    auto run = g.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(dummy)});

    auto [tensor, pad_meta] = face_recog::preprocess_scrfd(bgr_u8);
    printf("  Pad: top=%d left=%d padded=%dx%d\n",
           pad_meta.pad_top, pad_meta.pad_left, pad_meta.pad_w, pad_meta.pad_h);

    if (!run.push(simaai::neat::TensorList{tensor}))
        throw std::runtime_error("SCRFD push failed: " + run.last_error());

    const auto sample = run.pull(timeout_ms);
    if (!sample) throw std::runtime_error("SCRFD pull timeout");

    const auto tensors = face_recog::collect_tensors(*sample);
    printf("  Output tensors: %zu\n", tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
        print_tensor_info(tensors[i], static_cast<int>(i));

    const auto dets = face_recog::decode_scrfd(tensors, scrfd_cfg, pad_meta);
    printf("\n  Detected faces: %zu\n", dets.size());
    for (size_t i = 0; i < std::min<size_t>(dets.size(), 10); ++i) {
        const auto& d = dets[i];
        printf("  [%zu] score=%.3f  box=[%.1f,%.1f,%.1f,%.1f]  "
               "lm=[%.1f,%.1f,%.1f,%.1f,...]\n",
               i, d.score, d.x1, d.y1, d.x2, d.y2,
               d.landmarks[0], d.landmarks[1], d.landmarks[2], d.landmarks[3]);
    }

    run.close();
}

// ── ArcFace test ──────────────────────────────────────────────────────────────

static void test_arcface(const std::string& scrfd_path, const std::string& arcface_path,
                         const cv::Mat& bgr_u8, int timeout_ms,
                         const face_recog::ScrfdConfig& scrfd_cfg,
                         const fs::path& ref_emb_path) {
    printf("\n═══ ArcFace Embedding Model ═══\n");
    printf("  Model: %s\n", arcface_path.c_str());

    simaai::neat::Model::Options scrfd_opt;
    scrfd_opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    scrfd_opt.preprocess.input_max_width  = scrfd_cfg.infer_w;
    scrfd_opt.preprocess.input_max_height = scrfd_cfg.infer_h;
    scrfd_opt.preprocess.input_max_depth  = 3;
    simaai::neat::Model scrfd_model(scrfd_path, scrfd_opt);
    simaai::neat::Graph sg;
    sg.add(simaai::neat::nodes::Input(scrfd_model.input_appsrc_options(true)));
    sg.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(scrfd_model)));
    sg.add(simaai::neat::nodes::groups::MLA(scrfd_model));
    sg.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(scrfd_model)));
    sg.add(simaai::neat::nodes::Output());
    cv::Mat sdummy(scrfd_cfg.infer_h, scrfd_cfg.infer_w, CV_32FC3, cv::Scalar(0,0,0));
    auto scrfd_run = sg.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(sdummy)});

    auto [st, pad_meta] = face_recog::preprocess_scrfd(bgr_u8);
    if (!scrfd_run.push(simaai::neat::TensorList{st}))
        throw std::runtime_error("SCRFD push failed: " + scrfd_run.last_error());
    const auto ss = scrfd_run.pull(timeout_ms);
    if (!ss) throw std::runtime_error("SCRFD pull timeout");
    const auto dets = face_recog::decode_scrfd(face_recog::collect_tensors(*ss), scrfd_cfg, pad_meta);
    scrfd_run.close();

    if (dets.empty()) throw std::runtime_error("No faces detected in test image");
    printf("  Using detection[0]: score=%.3f\n", dets[0].score);

    const cv::Mat crop = face_recog::align_face(bgr_u8, dets[0].landmarks);
    const cv::Mat f32  = face_recog::preprocess_arcface_crop(crop);
    printf("  Aligned crop: %dx%d (BGR→RGB, normalized to [-1,1])\n",
           crop.cols, crop.rows);

    simaai::neat::Model::Options arc_opt;
    arc_opt.preprocess.kind             = simaai::neat::InputKind::Tensor;
    arc_opt.preprocess.input_max_width  = face_recog::kArcFaceW;
    arc_opt.preprocess.input_max_height = face_recog::kArcFaceH;
    arc_opt.preprocess.input_max_depth  = 3;
    simaai::neat::Model arc_model(arcface_path, arc_opt);
    simaai::neat::Graph ag;
    ag.add(simaai::neat::nodes::Input(arc_model.input_appsrc_options(true)));
    ag.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(arc_model)));
    ag.add(simaai::neat::nodes::groups::MLA(arc_model));
    ag.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(arc_model)));
    ag.add(simaai::neat::nodes::Output());
    cv::Mat adummy(face_recog::kArcFaceH, face_recog::kArcFaceW, CV_32FC3, cv::Scalar(0,0,0));
    auto arc_run = ag.build(simaai::neat::TensorList{face_recog::tensor_from_hwc_f32(adummy)});

    const auto arc_tensor = face_recog::tensor_from_hwc_f32(f32);
    if (!arc_run.push(simaai::neat::TensorList{arc_tensor}))
        throw std::runtime_error("ArcFace push failed: " + arc_run.last_error());
    const auto arc_sample = arc_run.pull(timeout_ms);
    if (!arc_sample) throw std::runtime_error("ArcFace pull timeout");

    const auto out_tensors = face_recog::collect_tensors(*arc_sample);
    printf("  Output tensors: %zu\n", out_tensors.size());
    for (size_t i = 0; i < out_tensors.size(); ++i)
        print_tensor_info(out_tensors[i], static_cast<int>(i));

    const auto emb = face_recog::tensor_to_f32(out_tensors[0]);
    printf("  Embedding size: %zu\n", emb.size());

    if (!ref_emb_path.empty() && fs::exists(ref_emb_path))
        validate_cosine(emb, ref_emb_path);
    else
        printf("  (No reference embedding provided — skipping cosine validation)\n");

    arc_run.close();
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    std::string which     = "both";
    std::string image_path;
    std::string config_str = sima_examples::default_config_path(
        SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).string();
    fs::path ref_emb_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"   && i+1 < argc) which        = argv[++i];
        else if (a == "--image"   && i+1 < argc) image_path   = argv[++i];
        else if (a == "--config"  && i+1 < argc) config_str   = argv[++i];
        else if (a == "--ref-emb" && i+1 < argc) ref_emb_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            printf("Usage: face-model-test --model [scrfd|arcface|both] --image <path>\n"
                   "       [--config <yaml>] [--ref-emb <raw-float32-file>]\n");
            return 0;
        }
    }

    if (image_path.empty()) { std::cerr << "Error: --image required\n"; return 1; }
    const cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) { std::cerr << "Error: cannot read: " << image_path << "\n"; return 1; }

    const auto raw = sima_examples::ScalarConfig::load(config_str);
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
        if (which == "scrfd" || which == "both")
            test_scrfd(scrfd_model, bgr, timeout_ms, scrfd_cfg);
        if (which == "arcface" || which == "both")
            test_arcface(scrfd_model, arcface_model, bgr, timeout_ms, scrfd_cfg, ref_emb_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
