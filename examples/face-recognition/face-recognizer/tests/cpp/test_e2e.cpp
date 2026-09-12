// E2E test for face-recognizer.
//
// Launches face-recognizer with real SCRFD + ArcFace models and checks the
// pipeline exits cleanly.  Recognition accuracy is verified when a test gallery
// is present.
//
// Required env vars (any missing → hard fail):
//   SIMANEAT_APPS_TEST_MODELS_DIR    directory holding model tar.gz files
//   SIMANEAT_TEST_RTSP_H264_URL      RTSP stream  OR
//   SIMANEAT_APPS_TEST_INPUT_VIDEO   path to a face-containing MP4/H.264 file
//
// Optional:
//   SIMANEAT_APPS_TEST_GALLERY_BIN   path to a pre-enrolled gallery.bin;
//                                    when set, at least one recognised identity
//                                    must appear in stdout (non-Unknown match).
//   SIMANEAT_APPS_TEST_TIMEOUT_MS    per-run timeout (default 60 000 ms)
//
// Face detection assertion:
//   Asserted only when SIMANEAT_APPS_TEST_INPUT_VIDEO is set, because a generic
//   shared RTSP stream may not contain detectable faces in any 60-frame window.
//   When only RTSP is available the test verifies exit 0 and prints a warning.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

static const char* kScrfdFile   = "scrfd_2.5g_bnkps.mla_mpk.tar.gz";
static const char* kArcFaceFile = "w600k_r50.surgery_mpk.tar.gz";

static std::string find_in_dir(const std::string& dir, const char* filename) {
    const fs::path direct = fs::path(dir) / filename;
    if (fs::exists(direct)) return direct.string();
    for (auto& e : fs::recursive_directory_iterator(dir)) {
        if (e.path().filename() == filename) return e.path().string();
    }
    return {};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "[ERR] usage: " << argv[0] << " <face-recognizer-binary>\n";
        return 2;
    }
    const std::string binary = argv[1];

    // ── Models ────────────────────────────────────────────────────────────────
    const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
    if (!models_dir_raw) {
        std::cerr << "[FAIL] SIMANEAT_APPS_TEST_MODELS_DIR not set\n";
        return 1;
    }
    const std::string models_dir = models_dir_raw;

    const std::string scrfd_path   = find_in_dir(models_dir, kScrfdFile);
    const std::string arcface_path = find_in_dir(models_dir, kArcFaceFile);
    if (scrfd_path.empty() || arcface_path.empty()) {
        std::cerr << "[FAIL] SCRFD or ArcFace model not found under SIMANEAT_APPS_TEST_MODELS_DIR\n";
        return 1;
    }

    // ── Input source ──────────────────────────────────────────────────────────
    // Track whether the input is a known face-containing video so the detection
    // assertion can be gated appropriately (a shared generic RTSP stream may not
    // contain detectable faces in a 60-frame window).
    bool input_is_face_video = false;
    std::string input;
    if (const char* v = env_or_null("SIMANEAT_APPS_TEST_INPUT_VIDEO")) {
        input = v;
        input_is_face_video = true;
        if (!fs::exists(input)) {
            std::cerr << "[FAIL] SIMANEAT_APPS_TEST_INPUT_VIDEO file not found: " << input << "\n";
            return 1;
        }
    } else {
        const auto rtsp_urls = rtsp_h264_urls_from_env();
        if (rtsp_urls.empty()) {
            std::cerr << "[FAIL] No input source: set SIMANEAT_APPS_TEST_INPUT_VIDEO (video file)"
                         " or SIMANEAT_TEST_RTSP_H264_URL (RTSP stream)\n";
            return 1;
        }
        input = rtsp_urls.front();
    }

    // ── Gallery (optional — recognition check only when present) ──────────────
    std::string gallery_path;
    if (const char* g = env_or_null("SIMANEAT_APPS_TEST_GALLERY_BIN")) {
        gallery_path = g;
        if (!fs::exists(gallery_path)) {
            throw std::runtime_error(
                "SIMANEAT_APPS_TEST_GALLERY_BIN set but not found: " + gallery_path);
        }
    }

    // ── Write test config ─────────────────────────────────────────────────────
    const fs::path config_dir = fs::temp_directory_path() / "face_recognizer_e2e";
    fs::create_directories(config_dir);
    const fs::path config_path = config_dir / "config.yaml";

    ConfigScalars overrides = {
        {"scrfd.model",          scrfd_path},
        {"arcface.model",        arcface_path},
        {"input.uri",            input},
        {"output.sink",          ""},            // headless
        {"output.insight.host",  "127.0.0.1"},  // exercise encoder + MetadataSender;
                                                 // UDP drops silently with no listener
    };
    if (!gallery_path.empty()) {
        overrides["gallery.path"] = gallery_path;
    }
    write_e2e_config("face-recognizer", config_path, overrides);

    // ── Run ───────────────────────────────────────────────────────────────────
    const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 60000);
    const std::vector<std::string> args = {
        "--config", config_path.string(),
        "--test", "--max-frames", "60",
    };

    std::cout << "[RUN] " << binary << " --config " << config_path
              << " --test --max-frames 60\n";
    const ProcessResult r = spawn_and_wait(binary, args, timeout);
    fs::remove_all(config_dir);

    if (r.exit_code != 0) {
        std::cerr << "[FAIL] exit code " << r.exit_code << "\n"
                  << "stderr:\n" << r.stderr_text << "\n";
        return 1;
    }

    // ── Mandatory detection check (gallery-independent) ───────────────────────
    // Parse the shutdown "Detection quality" summary that main.cpp always prints.
    // "0 faces : <N>" tells us how many frames had zero detections; if ALL frames
    // had zero detections SCRFD is broken regardless of gallery presence.
    // The progress block that prints "det[0]=" only fires at frame 100, so it is
    // never visible in a 60-frame run — parse the summary instead.
    {
        // Find the "0 faces :" line and extract the count.
        int zero_face_frames = -1;  // -1 = summary block not found
        int total_frames     = -1;
        std::istringstream ds(r.stdout_text);
        std::string dl;
        while (std::getline(ds, dl)) {
            // "═══ Detection quality (N frames) ═══"
            if (dl.find("Detection quality") != std::string::npos) {
                const auto lp = dl.find('(');
                const auto rp = dl.find(' ', lp + 1);
                if (lp != std::string::npos && rp != std::string::npos)
                    total_frames = std::stoi(dl.substr(lp + 1, rp - lp - 1));
            }
            // "  0 faces :     N (X.X%)   <- misses"
            if (dl.find("0 faces :") != std::string::npos) {
                std::istringstream ls(dl);
                std::string tok;
                int col = 0;
                while (ls >> tok) { if (col++ == 3) { zero_face_frames = std::stoi(tok); break; } }
            }
        }
        if (total_frames <= 0) {
            std::cerr << "[FAIL] Detection quality summary not found in output.\n"
                      << "stdout:\n" << r.stdout_text << "\n";
            return 1;
        }
        if (zero_face_frames == total_frames) {
            if (input_is_face_video) {
                std::cerr << "[FAIL] SCRFD detected zero faces in all " << total_frames << " frames — "
                             "ensure SIMANEAT_APPS_TEST_INPUT_VIDEO contains at least one visible face.\n";
                return 1;
            }
            std::cout << "[WARN] SCRFD detected zero faces in " << total_frames << " frames using a "
                         "generic RTSP stream; face detection not verified.\n"
                         "       Set SIMANEAT_APPS_TEST_INPUT_VIDEO to a face-containing video "
                         "to enable this check.\n";
        }
    }

    // ── Optional recognition check ────────────────────────────────────────────
    if (!gallery_path.empty()) {
        // test mode prints: "  face[N] → <name>  similarity=X.XXXX"
        // A non-Unknown match means the arrow line exists and its name is not "Unknown".
        bool has_known_match = false;
        std::istringstream ss(r.stdout_text);
        std::string line;
        while (std::getline(ss, line)) {
            const auto arrow = line.find("\xe2\x86\x92");  // UTF-8 for →
            if (arrow == std::string::npos) continue;
            const std::string after = line.substr(arrow + 3);  // skip → (3 bytes)
            const auto name_start = after.find_first_not_of(' ');
            if (name_start == std::string::npos) continue;
            const std::string name = after.substr(name_start,
                after.find_first_of(' ', name_start) - name_start);
            if (name != "Unknown" && !name.empty()) { has_known_match = true; break; }
        }
        if (!has_known_match) {
            std::cerr << "[FAIL] SIMANEAT_APPS_TEST_GALLERY_BIN is set but no "
                         "non-Unknown match found in 60 frames.\n"
                         "  Ensure the test clip contains enrolled faces, or "
                         "unset SIMANEAT_APPS_TEST_GALLERY_BIN to skip this check.\n";
            return 1;
        }
        std::cout << "[OK] recognition match confirmed\n";
    }

    std::cout << "[OK] face-recognizer processed 60 frames (exit 0)\n";

    // ── Enrollment E2E (when face video is available) ─────────────────────────
    // Run --enroll mode using the same video input.  This exercises the enrollment
    // path (SCRFD → ArcFace → GalleryBuilder → gallery.bin write) without needing
    // a separate binary or a second test executable that would confuse the CI
    // runner's binary-name derivation heuristic.
    if (input_is_face_video) {
        const fs::path enroll_dir     = fs::temp_directory_path() / "face_recognizer_enroll_e2e";
        const fs::path enroll_gallery = enroll_dir / "gallery.bin";
        const fs::path enroll_config  = enroll_dir / "config.yaml";
        fs::create_directories(enroll_dir);

        ConfigScalars enroll_overrides = {
            {"scrfd.model",   scrfd_path},
            {"arcface.model", arcface_path},
            {"input.uri",     ""},
            {"output.sink",   ""},
        };
        write_e2e_config("face-recognizer", enroll_config, enroll_overrides);

        const std::vector<std::string> enroll_args = {
            "--enroll",
            "--config",       enroll_config.string(),
            "--video",        input,
            "--name",         "TestIdentity",
            "--gallery",      enroll_gallery.string(),
            "--sample-every", "10",
        };

        std::cout << "[RUN] " << binary << " --enroll --video " << input
                  << " --name TestIdentity --gallery " << enroll_gallery << "\n";

        const ProcessResult er = spawn_and_wait(binary, enroll_args, timeout);
        const bool gallery_written = fs::exists(enroll_gallery) && fs::file_size(enroll_gallery) > 0;
        fs::remove_all(enroll_dir);

        if (er.exit_code != 0) {
            std::cerr << "[FAIL] --enroll exit code " << er.exit_code << "\n"
                      << "stderr:\n" << er.stderr_text << "\n";
            return 1;
        }
        if (!gallery_written) {
            std::cerr << "[FAIL] --enroll exited 0 but gallery.bin was not written\n";
            return 1;
        }
        std::cout << "[OK] enrollment completed and gallery.bin written\n";
    }

    return 0;
}
