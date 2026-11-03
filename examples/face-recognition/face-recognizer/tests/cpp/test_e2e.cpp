// E2E test for face-recognizer.
//
// Runs the binary with real SCRFD + ArcFace models:
//   1. Bundled face clip (tests/assets/face_clip.mp4) with the bundled gallery: every
//      frame must contain both enrolled identities, PersonA and PersonB, and no other
//      name.
//   2. RTSP stream: the live-stream graph must process 60 frames and exit 0. The shared
//      CI stream content is not controlled, so detections are not asserted here.
//   3. --output recording of the bundled clip must produce a non-empty video.
//   4. Enrollment from a PersonA/PersonB folder of the shared COCO images and from the
//      bundled clip must write a gallery.
//   5. USB camera (only when SIMANEAT_TEST_USB_CAMERA names one; CI has no camera): the
//      headless graph must process 60 frames, and with Insight on loopback it must also
//      forward the camera's JPEG as RTP/JPEG (payload type 26).
//
// Required env vars (missing → fail):
//   SIMANEAT_APPS_TEST_MODELS_DIR    directory holding the model tar.gz files
//   SIMANEAT_TEST_RTSP_H264_URL      live H.264 RTSP stream
// Optional:
//   SIMANEAT_APPS_TEST_TIMEOUT_MS    per-run timeout (default 60 000 ms)
//   SIMANEAT_TEST_USB_CAMERA         USB camera ID from usb_camera_ids()
//   SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT / _METADATA_PORT  loopback Insight ports
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

static const char* kScrfdFile   = "scrfd_2.5g_bnkps.mla_mpk.tar.gz";
static const char* kArcFaceFile = "w600k_r50.surgery_mpk.tar.gz";
// Tests run from the apps root (CMake WORKING_DIRECTORY).
static const fs::path kAssets = "examples/face-recognition/face-recognizer/tests/assets";
static const fs::path kCoco   = "assets/datasets/coco";
// The bundled gallery's identities and the shared COCO image each was enrolled from.
static const std::vector<std::pair<std::string, std::string>> kIdentities = {
    {"PersonA", "000000493864.jpg"},
    {"PersonB", "000000116439.jpg"},
};
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

// Counts "  face[N] → <name>  similarity=X" lines that name `name`; `others` gets the
// lines that name anyone outside kIdentities.
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
        if (got == name) {
            ++matches;
        } else if (std::none_of(kIdentities.begin(), kIdentities.end(),
                                [&](const auto& id) { return id.first == got; })) {
            ++others;
        }
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
    for (const auto& p : {clip, gallery}) {
        if (!fs::exists(p)) {
            std::cerr << "[FAIL] bundled test asset missing: " << p << "\n";
            return 1;
        }
    }

    const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 60000);
    const fs::path work = fs::temp_directory_path() / "face_recognizer_e2e";
    fs::remove_all(work);
    fs::create_directories(work);
    // --enroll --images reads one subdirectory per identity.
    const fs::path images = work / "gallery_images";
    for (const auto& [name, file] : kIdentities) {
        const fs::path src = fs::absolute(kCoco / file);
        if (!fs::exists(src)) {
            std::cerr << "[FAIL] shared test image missing: " << src << "\n";
            return 1;
        }
        fs::create_directories(images / name);
        fs::copy_file(src, images / name / file);
    }
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
        bool ok = s.frames == kFrames && s.zero_faces == 0 && s.recognized == kFrames;
        std::ostringstream got;
        got << "frames=" << s.frames << " zero_face_frames=" << s.zero_faces
            << " recognized=" << s.recognized;
        for (const auto& [name, file] : kIdentities) {
            int others = 0;
            const int named = count_face_lines(r.stdout_text, name, others);
            ok = ok && named == kFrames && others == 0;
            got << " " << name << "_lines=" << named << " other_lines=" << others;
        }
        if (!ok) {
            std::cerr << "[FAIL] bundled clip: expected " << kFrames << " frames, each with "
                      << "PersonA and PersonB recognized; got " << got.str()
                      << "\nstdout:\n" << r.stdout_text << "\n";
            return 1;
        }
        std::cout << "[OK] bundled clip: PersonA and PersonB recognized in " << kFrames << "/"
                  << kFrames << " frames\n";
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

    // ── 5. USB camera ─────────────────────────────────────────────────────────
    if (const char* camera = env_or_null("SIMANEAT_TEST_USB_CAMERA")) {
        const fs::path usb_config = work / "config_usb.yaml";
        const int video_port    = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
        const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
        write_e2e_config("face-recognizer", usb_config, {
            {"scrfd.model",                  scrfd_path},
            {"arcface.model",                arcface_path},
            {"gallery.path",                 gallery.string()},
            {"input.uri",                    ""},
            {"input.usb.camera",             camera},
            {"output.sink",                  ""},
            // This test is the Insight video receiver, so it publishes to its own loopback.
            {"output.insight.host",          "127.0.0.1"},
            {"output.insight.video_port",    std::to_string(video_port)},
            {"output.insight.metadata_port", std::to_string(metadata_port)},
        });
        const std::vector<std::string> usb_args = {
            "--config", usb_config.string(), "--max-frames", std::to_string(kFrames)};

        // Headless: --test clears the Insight host, so the decoder follows the camera.
        {
            std::cout << "[RUN] " << binary << " input.usb.camera=" << camera << " --test\n";
            auto args = usb_args;
            args.push_back("--test");
            const ProcessResult r = spawn_and_wait(binary, args, timeout);
            if (!run_ok("USB camera run", r)) return 1;
            if (parse_summary(r.stdout_text).frames != kFrames) {
                std::cerr << "[FAIL] USB camera run did not process " << kFrames << " frames\n"
                          << "stdout:\n" << r.stdout_text << "\n";
                return 1;
            }
            std::cout << "[OK] USB camera: processed " << kFrames << " frames\n";
        }

        // Insight: the camera's JPEG is split off before decode and sent unchanged.
        {
            const int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
            sockaddr_in addr{};
            addr.sin_family      = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port        = htons(static_cast<uint16_t>(video_port));
            if (sock < 0 || ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
                if (sock >= 0) ::close(sock);
                std::cerr << "[FAIL] could not bind the Insight video port " << video_port << "\n";
                return 1;
            }
            std::cout << "[RUN] " << binary << " input.usb.camera=" << camera
                      << " output.insight.host=127.0.0.1\n";
            const ProcessResult r = spawn_and_wait(binary, usb_args, timeout);
            std::array<unsigned char, 65536> packet{};
            const auto bytes = ::recv(sock, packet.data(), packet.size(), MSG_DONTWAIT);
            ::close(sock);
            if (!run_ok("USB camera Insight run", r)) return 1;
            if (parse_summary(r.stdout_text).frames != kFrames) {
                std::cerr << "[FAIL] USB camera Insight run did not process " << kFrames
                          << " frames\nstdout:\n" << r.stdout_text << "\n";
                return 1;
            }
            if (bytes <= 12 || (packet[0] >> 6) != 2 || (packet[1] & 0x7F) != 26) {
                std::cerr << "[FAIL] USB camera Insight run did not publish RTP/JPEG video\n";
                return 1;
            }
            std::cout << "[OK] USB camera: processed " << kFrames
                      << " frames and sent RTP/JPEG to Insight\n";
        }
    } else {
        std::cerr << "[SKIP] set SIMANEAT_TEST_USB_CAMERA to a USB camera ID to run the USB "
                     "camera checks\n";
    }

    fs::remove_all(work);
    return 0;
}
