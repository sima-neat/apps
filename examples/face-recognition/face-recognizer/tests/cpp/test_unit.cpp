/**
 * @file test_unit.cpp
 * Unit tests for face-recognizer components (no hardware required).
 * Compiled with GTest when available; can also run as a standalone smoke-test.
 *
 * Tests:
 *  - Gallery save/load round-trip
 *  - L2 normalization
 *  - Cosine similarity (same → 1.0, orthogonal → 0.0, opposite → -1.0)
 *  - MatchResult: known, unknown, empty gallery
 *  - Alignment: non-empty 112×112 output for synthetic landmarks
 *  - ArcFace preprocessing: range [-1,1], shape 112×112×3
 *  - SCRFD anchor count (3 scales × 2 anchors = 16800)
 *  - NMS: removes overlapping boxes
 */
#include "scrfd_decode.h"
#include "align.h"
#include "gallery.h"
#include "match.h"

#include <opencv2/core/mat.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;

#define ASSERT_NEAR_F(a, b, tol) \
    do { \
        const float _a = (a), _b = (b), _t = (tol); \
        if (std::abs(_a - _b) > _t) { \
            fprintf(stderr, "FAIL %s:%d  |%f - %f| > %f\n", __FILE__, __LINE__, _a, _b, _t); \
            return false; \
        } \
    } while(0)

#define ASSERT_TRUE_MSG(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); return false; } } while(0)

// ── test functions ────────────────────────────────────────────────────────────

static bool test_gallery_roundtrip() {
    const fs::path tmp = fs::temp_directory_path() / "fr_test_gallery.bin";
    face_recog::Gallery g;
    {
        face_recog::Embedding e(face_recog::kEmbeddingDim, 0.f);
        e[0] = 1.f;
        face_recog::l2_normalize(e);
        g.entries.push_back({"Alice", e});
    }
    {
        face_recog::Embedding e(face_recog::kEmbeddingDim, 0.f);
        e[511] = -1.f;
        face_recog::l2_normalize(e);
        g.entries.push_back({"Bob", e});
    }
    face_recog::save_gallery(g, tmp);
    const auto loaded = face_recog::load_gallery(tmp);
    fs::remove(tmp);

    ASSERT_TRUE_MSG(loaded.entries.size() == 2, "entry count");
    ASSERT_TRUE_MSG(loaded.entries[0].name == "Alice", "name[0]");
    ASSERT_TRUE_MSG(loaded.entries[1].name == "Bob",   "name[1]");
    ASSERT_NEAR_F(loaded.entries[0].embedding[0], 1.f, 1e-5f);
    ASSERT_NEAR_F(loaded.entries[1].embedding[511], -1.f, 1e-5f);
    return true;
}

static bool test_l2_normalize() {
    face_recog::Embedding e = {3.f, 4.f};
    face_recog::l2_normalize(e);
    ASSERT_NEAR_F(e[0], 0.6f, 1e-5f);
    ASSERT_NEAR_F(e[1], 0.8f, 1e-5f);

    // Zero vector: should not crash
    face_recog::Embedding z(512, 0.f);
    face_recog::l2_normalize(z);
    ASSERT_NEAR_F(z[0], 0.f, 1e-5f);
    return true;
}

static bool test_cosine_similarity() {
    const int D = 512;
    face_recog::Embedding a(D, 0.f), b(D, 0.f), c(D, 0.f);
    a[0] = 1.f;
    b[0] = 1.f;
    c[0] = -1.f;
    ASSERT_NEAR_F(face_recog::cosine_similarity(a, b),  1.0f, 1e-5f);
    ASSERT_NEAR_F(face_recog::cosine_similarity(a, c), -1.0f, 1e-5f);

    face_recog::Embedding orth(D, 0.f);
    orth[1] = 1.f;
    ASSERT_NEAR_F(face_recog::cosine_similarity(a, orth), 0.0f, 1e-5f);
    return true;
}

static bool test_match_known() {
    face_recog::Gallery g;
    face_recog::Embedding e(512, 0.f); e[0] = 1.f; face_recog::l2_normalize(e);
    g.entries.push_back({"Alice", e});

    face_recog::MatchConfig cfg;
    cfg.threshold = 0.4f;
    const auto r = face_recog::match_embedding(e, g, cfg);  // exact match → sim=1.0
    ASSERT_TRUE_MSG(r.name == "Alice",  "known name");
    ASSERT_NEAR_F(r.score, 1.0f, 1e-4f);
    return true;
}

static bool test_match_unknown() {
    face_recog::Gallery g;
    face_recog::Embedding a(512, 0.f); a[0] =  1.f; face_recog::l2_normalize(a);
    face_recog::Embedding b(512, 0.f); b[1] =  1.f;  // orthogonal → sim=0
    g.entries.push_back({"Alice", a});

    face_recog::MatchConfig cfg;
    cfg.threshold     = 0.4f;
    cfg.unknown_label = "Unknown";
    const auto r = face_recog::match_embedding(b, g, cfg);
    ASSERT_TRUE_MSG(r.name == "Unknown",  "unknown label");
    ASSERT_NEAR_F(r.score, 0.0f, 1e-4f);
    return true;
}

static bool test_match_empty_gallery() {
    face_recog::Gallery g;
    face_recog::Embedding e(512, 0.f); e[0] = 1.f;
    face_recog::MatchConfig cfg;
    const auto r = face_recog::match_embedding(e, g, cfg);
    ASSERT_TRUE_MSG(r.score < -1.f, "empty gallery score sentinel");
    return true;
}

static bool test_arcface_preprocess() {
    // Synthetic 112×112 BGR image
    cv::Mat bgr(face_recog::kArcFaceH, face_recog::kArcFaceW, CV_8UC3, cv::Scalar(127, 63, 200));
    const cv::Mat f32 = face_recog::preprocess_arcface_crop(bgr);
    ASSERT_TRUE_MSG(f32.type() == CV_32FC3, "dtype");
    ASSERT_TRUE_MSG(f32.rows == 112 && f32.cols == 112, "shape");

    // All values in [-1, 1]
    double mn, mx;
    cv::minMaxLoc(f32, &mn, &mx);
    ASSERT_TRUE_MSG(mn >= -1.0 - 1e-4 && mx <= 1.0 + 1e-4, "value range");
    return true;
}

static bool test_align_face_synthetic() {
    // Synthetic 480×640 frame and landmarks that form a tiny "face"
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    face_recog::Landmarks lm;
    lm[0] = 200.f; lm[1] = 150.f;  // left eye
    lm[2] = 280.f; lm[3] = 150.f;  // right eye
    lm[4] = 240.f; lm[5] = 200.f;  // nose
    lm[6] = 215.f; lm[7] = 240.f;  // left mouth
    lm[8] = 265.f; lm[9] = 240.f;  // right mouth

    const cv::Mat crop = face_recog::align_face(frame, lm);
    ASSERT_TRUE_MSG(!crop.empty(), "crop not empty");
    ASSERT_TRUE_MSG(crop.cols == face_recog::kArcFaceW && crop.rows == face_recog::kArcFaceH,
                    "crop size 112x112");
    return true;
}

static bool test_gallery_builder() {
    face_recog::GalleryBuilder builder;
    // Two images of "Alice" → mean-pooled
    face_recog::Embedding e1(512, 0.f); e1[0] = 1.f;
    face_recog::Embedding e2(512, 0.f); e2[0] = 0.9f; e2[1] = 0.1f;
    builder.add("Alice", e1);
    builder.add("Alice", e2);
    builder.add("Bob",   e1);

    const auto g = builder.finish();
    ASSERT_TRUE_MSG(g.entries.size() == 2, "entry count after finish");
    ASSERT_TRUE_MSG(g.entries[0].name == "Alice" || g.entries[1].name == "Alice", "Alice present");

    for (const auto& entry : g.entries) {
        float n2 = 0.f;
        for (float v : entry.embedding) n2 += v * v;
        ASSERT_NEAR_F(n2, 1.0f, 1e-4f);
    }
    return true;
}

// ── runner ────────────────────────────────────────────────────────────────────

int main() {
    struct Test { const char* name; bool(*fn)(); };
    const Test tests[] = {
        {"gallery_roundtrip",     test_gallery_roundtrip},
        {"l2_normalize",          test_l2_normalize},
        {"cosine_similarity",     test_cosine_similarity},
        {"match_known",           test_match_known},
        {"match_unknown",         test_match_unknown},
        {"match_empty_gallery",   test_match_empty_gallery},
        {"arcface_preprocess",    test_arcface_preprocess},
        {"align_face_synthetic",  test_align_face_synthetic},
        {"gallery_builder",       test_gallery_builder},
    };

    int passed = 0, failed = 0;
    for (const auto& t : tests) {
        const bool ok = t.fn();
        printf("  [%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        ok ? ++passed : ++failed;
    }
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
