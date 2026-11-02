// E2E test for face-recognizer.
//
// Runs the binary with real SCRFD + ArcFace models:
//   1. Bundled face clip (tests/assets/face_clip.mp4) with the bundled gallery: every
//      frame must contain a detected face and every frame must be recognized as the
//      enrolled identity.
//   2. RTSP stream: the live-stream graph must process 60 frames and exit 0. The shared
//      CI stream content is not controlled, so detections are not asserted here.
//   3. --output recording of the bundled clip must produce a non-empty video.
//   4. Enrollment from the bundled image folder and from the bundled clip must write
//      a gallery.
//
// Required env vars (missing → fail):
//   SIMANEAT_APPS_TEST_MODELS_DIR    directory holding the model tar.gz files
//   SIMANEAT_TEST_RTSP_H264_URL      live H.264 RTSP stream
// Optional:
//   SIMANEAT_APPS_TEST_TIMEOUT_MS    per-run timeout (default 60 000 ms)
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
// Tests run from the apps root (CMake WORKING_DIRECTORY).
static const fs::path kAssets = "examples/face-recognition/face-recognizer/tests/assets";
static const char* kEnrolledName = "Paresh";
static constexpr int kFrames = 60;

static std::string find_in_dir(const std::string& dir, const char* filename) {
    const fs::path direct = fs::path(dir) / filename;
    if (fs::exists(direct)) return direct.string();
    for (auto& e : fs::recursive_directory_iterator(dir)) {
        if (e.path().filename() == filename) return e.path().string();
    }
    return {};
}

struct Summary {
    int frames = -1;      // "═══ Detection quality (N frames) ═══"
    int zero_faces = -1;  // "  0 faces :     N (...)"
    int recognized = -1;  // "  recognized/unknown: R / U"
};

static Summary parse_summary(const std::string& out) {
    Summary s;
    std::istringstream in(out);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("Detection quality") != std::string::npos) {
            const auto lp = line.find('(');
            if (lp != std::string::npos) s.frames = std::stoi(line.substr(lp + 1));
        } else if (line.find("0 faces :") != std::string::npos) {
            s.zero_faces = std::stoi(line.substr(line.find(':') + 1));
        } else if (line.find("recognized/unknown:") != std::string::npos) {
            s.recognized = std::stoi(line.substr(line.find(':') + 1));
        }
    }
    return s;
}

// Every "  face[N] → <name>  similarity=X" line must name the enrolled identity.
static int count_face_lines(const std::string& out, const std::string& name, int& others) {
    int matches = 0;
    others = 0;
    std::istringstream in(out);
    std::string line;
    while (std::getline(in, line)) {
        const auto arrow = line.find("\xe2\x86\x92");  // UTF-8 →
        if (line.find("face[") == std::string::npos || arrow == std::string::npos) continue;
        std::istringstream rest(line.substr(arrow + 3));
        std::string got;
        rest >> got;
        if (got == name) ++matches; else ++others;
    }
    return matches;
}

static bool run_ok(const std::string& what, const ProcessResult& r) {
    if (r.exit_code == 0) return true;
    std::cerr << "[FAIL] " << what << " exit code " << r.exit_code << "\n"
              << "stdout:\n" << r.stdout_text << "\nstderr:\n" << r.stderr_text << "\n";
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "[ERR] usage: " << argv[0] << " <face-recognizer-binary>\n";
        return 2;
    }
    const std::string binary = argv[1];

    const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
    if (!models_dir_raw) {
        std::cerr << "[FAIL] SIMANEAT_APPS_TEST_MODELS_DIR not set\n";
        return 1;
    }
    const std::string scrfd_path   = find_in_dir(models_dir_raw, kScrfdFile);
    const std::string arcface_path = find_in_dir(models_dir_raw, kArcFaceFile);
    if (scrfd_path.empty() || arcface_path.empty()) {
        std::cerr << "[FAIL] SCRFD or ArcFace model not found under SIMANEAT_APPS_TEST_MODELS_DIR\n";
        return 1;
    }
    const auto rtsp_urls = rtsp_h264_urls_from_env();
    if (rtsp_urls.empty()) {
        std::cerr << "[FAIL] SIMANEAT_TEST_RTSP_H264_URL not set\n";
        return 1;
    }
    const fs::path clip    = fs::absolute(kAssets / "face_clip.mp4");
    const fs::path gallery = fs::absolute(kAssets / "gallery.bin");
    const fs::path images  = fs::absolute(kAssets / "gallery_images");
    for (const auto& p : {clip, gallery, images}) {
        if (!fs::exists(p)) {
            std::cerr << "[FAIL] bundled test asset missing: " << p << "\n";
            return 1;
        }
    }

    const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 60000);
    const fs::path work = fs::temp_directory_path() / "face_recognizer_e2e";
    fs::remove_all(work);
    fs::create_directories(work);
    const fs::path config_path = work / "config.yaml";
    write_e2e_config("face-recognizer", config_path, {
        {"scrfd.model",         scrfd_path},
        {"arcface.model",       arcface_path},
        {"gallery.path",        gallery.string()},
        {"input.uri",           clip.string()},
        {"output.sink",         ""},
        {"output.insight.host", ""},
    });
    const std::vector<std::string> test_args = {
        "--config", config_path.string(), "--test", "--max-frames", std::to_string(kFrames)};

    // ── 1. Bundled clip: detection + recognition ──────────────────────────────
    {
        std::cout << "[RUN] " << binary << " --input " << clip << " --test\n";
        auto args = test_args;
        args.insert(args.end(), {"--input", clip.string()});
        const ProcessResult r = spawn_and_wait(binary, args, timeout);
        if (!run_ok("bundled clip run", r)) return 1;
        const Summary s = parse_summary(r.stdout_text);
        int others = 0;
        const int named = count_face_lines(r.stdout_text, kEnrolledName, others);
        if (s.frames != kFrames || s.zero_faces != 0 || s.recognized != kFrames ||
            named < kFrames || others != 0) {
            std::cerr << "[FAIL] bundled clip: expected " << kFrames << " frames, each with a face "
                      << "recognized as " << kEnrolledName << "; got frames=" << s.frames
                      << " zero_face_frames=" << s.zero_faces << " recognized=" << s.recognized
                      << " " << kEnrolledName << "_lines=" << named << " other_lines=" << others
                      << "\nstdout:\n" << r.stdout_text << "\n";
            return 1;
        }
        std::cout << "[OK] bundled clip: " << kFrames << "/" << kFrames << " frames recognized as "
                  << kEnrolledName << "\n";
    }

    // ── 2. RTSP: live-stream graph runs ───────────────────────────────────────
    {
        std::cout << "[RUN] " << binary << " --input <rtsp> --test\n";
        auto args = test_args;
        args.insert(args.end(), {"--input", rtsp_urls.front()});
        const ProcessResult r = spawn_and_wait(binary, args, timeout);
        if (!run_ok("RTSP run", r)) return 1;
        if (parse_summary(r.stdout_text).frames != kFrames) {
            std::cerr << "[FAIL] RTSP run did not process " << kFrames << " frames\n"
                      << "stdout:\n" << r.stdout_text << "\n";
            return 1;
        }
        std::cout << "[OK] RTSP: processed " << kFrames << " frames\n";
    }

    // ── 3. --output recording ─────────────────────────────────────────────────
    {
        const fs::path recording = work / "recording.mp4";
        std::cout << "[RUN] " << binary << " --input " << clip << " --output " << recording << "\n";
        const ProcessResult r = spawn_and_wait(binary, {
            "--config", config_path.string(), "--input", clip.string(),
            "--output", recording.string()}, timeout);
        if (!run_ok("--output run", r)) return 1;
        if (!fs::exists(recording) || fs::file_size(recording) == 0) {
            std::cerr << "[FAIL] --output exited 0 but " << recording << " is missing or empty\n";
            return 1;
        }
        std::cout << "[OK] --output wrote " << fs::file_size(recording) << " bytes\n";
    }

    // ── 4. Enrollment from images and from video ──────────────────────────────
    const std::vector<std::pair<std::string, std::vector<std::string>>> enrollments = {
        {"--images", {"--images", images.string()}},
        {"--video",  {"--video", clip.string(), "--name", "TestIdentity", "--sample-every", "10"}},
    };
    for (const auto& [mode, extra] : enrollments) {
        const fs::path out = work / ("gallery" + mode + ".bin");
        std::vector<std::string> args = {"--enroll", "--config", config_path.string(),
                                         "--gallery", out.string()};
        args.insert(args.end(), extra.begin(), extra.end());
        std::cout << "[RUN] " << binary << " --enroll " << mode << "\n";
        const ProcessResult r = spawn_and_wait(binary, args, timeout);
        if (!run_ok("--enroll " + mode, r)) return 1;
        if (!fs::exists(out) || fs::file_size(out) == 0) {
            std::cerr << "[FAIL] --enroll " << mode << " exited 0 but gallery.bin was not written\n";
            return 1;
        }
        std::cout << "[OK] --enroll " << mode << " wrote gallery.bin\n";
    }

    fs::remove_all(work);
    return 0;
}
