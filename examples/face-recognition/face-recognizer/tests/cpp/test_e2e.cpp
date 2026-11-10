// E2E test for face-recognizer.
//
// Runs the binary with real SCRFD + ArcFace models:
//   1. Enrollment from a PersonA/PersonB folder of the shared COCO images, and from the
//      bundled clip, must each write a gallery.
//   2. Bundled face clip (tests/assets/face_clip.mp4), recognized with the gallery that
//      step 1 enrolled from the images: every frame must name both PersonA and PersonB,
//      and no other name.
//   3. RTSP stream: the live-stream graph must process 60 frames and exit 0. The shared
//      CI stream content is not controlled, so detections are not asserted here.
//   4. RTSP stream with Insight on loopback: this test listens where Insight would. The
//      stream must arrive as RTP/H.264 and face-metadata messages must carry the
//      timestamp of a video frame, which is how Insight pairs a result with its frame.
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
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <set>
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
// The identities this test enrolls and the shared COCO image each is enrolled from.
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

// Frames processed, from the final "Done: N frames in ..." line; -1 when absent.
static int frames_done(const std::string& out) {
    const auto at = out.rfind("Done: ");
    return at == std::string::npos ? -1 : std::stoi(out.substr(at + 6));
}

// A UDP socket bound where Insight would listen for video; -1 on failure.
static int bind_video_port(int port) {
    const int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    if (sock >= 0 && ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
        return sock;
    if (sock >= 0) ::close(sock);
    return -1;
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
    const fs::path clip = fs::absolute(kAssets / "face_clip.mp4");
    if (!fs::exists(clip)) {
        std::cerr << "[FAIL] bundled test asset missing: " << clip << "\n";
        return 1;
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
    // Step 1 enrolls this gallery from the images; every later run recognizes with it.
    const fs::path gallery     = work / "gallery--images.bin";
    const fs::path config_path = work / "config.yaml";
    write_e2e_config("face-recognizer", config_path, {
        {"scrfd.model",         scrfd_path},
        {"arcface.model",       arcface_path},
        {"gallery.path",        gallery.string()},
        {"input.uri",           clip.string()},
        {"output.insight.host", ""},
    });
    const std::vector<std::string> test_args = {
        "--config", config_path.string(), "--test", "--max-frames", std::to_string(kFrames)};

    // ── 1. Enrollment from images and from video ──────────────────────────────
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

    // ── 2. Bundled clip, recognized with the gallery enrolled above ───────────
    {
        std::cout << "[RUN] " << binary << " --input " << clip << " --test\n";
        auto args = test_args;
        args.insert(args.end(), {"--input", clip.string()});
        const ProcessResult r = spawn_and_wait(binary, args, timeout);
        if (!run_ok("bundled clip run", r)) return 1;
        const int frames = frames_done(r.stdout_text);
        bool ok = frames == kFrames;
        std::ostringstream got;
        got << "frames=" << frames;
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

    // ── 3. RTSP: live-stream graph runs ───────────────────────────────────────
    {
        std::cout << "[RUN] " << binary << " --input <rtsp> --test\n";
        auto args = test_args;
        args.insert(args.end(), {"--input", rtsp_urls.front()});
        const ProcessResult r = spawn_and_wait(binary, args, timeout);
        if (!run_ok("RTSP run", r)) return 1;
        if (frames_done(r.stdout_text) != kFrames) {
            std::cerr << "[FAIL] RTSP run did not process " << kFrames << " frames\n"
                      << "stdout:\n" << r.stdout_text << "\n";
            return 1;
        }
        std::cout << "[OK] RTSP: processed " << kFrames << " frames\n";
    }

    // ── 4. RTSP with Insight on loopback: video and paired metadata ───────────
    {
        const int video_port    = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
        const int metadata_port =
            env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
        const fs::path insight_config = work / "config_insight.yaml";
        write_e2e_config("face-recognizer", insight_config, {
            {"scrfd.model",                  scrfd_path},
            {"arcface.model",                arcface_path},
            {"gallery.path",                 gallery.string()},
            {"input.uri",                    rtsp_urls.front()},
            {"output.insight.host",          "127.0.0.1"},
            {"output.insight.video_port",    std::to_string(video_port)},
            {"output.insight.metadata_port", std::to_string(metadata_port)},
        });
        MetadataJsonListenerOptions metadata_options;
        metadata_options.base_port = metadata_port;
        MetadataJsonListener metadata_listener(metadata_options);
        const int video_sock = bind_video_port(video_port);
        if (!metadata_listener.ok() || video_sock < 0) {
            if (video_sock >= 0) ::close(video_sock);
            std::cerr << "[FAIL] could not listen on the Insight ports " << video_port << " and "
                      << metadata_port << "\n";
            return 1;
        }

        // Insight pairs a metadata message with the video frame whose RTP timestamp
        // (90 kHz) equals the message's millisecond timestamp, within one millisecond.
        std::set<uint32_t> video_timestamps;
        MetadataJsonListenerResult metadata;
        auto paired_messages = [&] {
            return std::count_if(metadata.messages.begin(), metadata.messages.end(),
                                 [&](const MetadataJsonMessage& m) {
                const uint32_t want = static_cast<uint32_t>(m.timestamp_ms * 90);
                const auto next = video_timestamps.lower_bound(want >= 90 ? want - 90 : 0);
                return m.timestamp_ms >= 0 && next != video_timestamps.end() && *next <= want + 90;
            });
        };
        std::cout << "[RUN] " << binary << " input.uri=<rtsp> output.insight.host=127.0.0.1\n";
        // The application runs until stopped; stop it once ten results are paired.
        const ProcessResult r = spawn_until(binary, {"--config", insight_config.string()}, [&] {
            std::array<unsigned char, 2048> packet{};
            for (;;) {
                const auto bytes = ::recv(video_sock, packet.data(), packet.size(), MSG_DONTWAIT);
                if (bytes <= 0) break;
                // RTP version 2, payload type 96 (H.264); bytes 4-7 are the timestamp.
                if (bytes > 12 && (packet[0] >> 6) == 2 && (packet[1] & 0x7F) == 96)
                    video_timestamps.insert(
                        (uint32_t{packet[4]} << 24) | (uint32_t{packet[5]} << 16) |
                        (uint32_t{packet[6]} << 8) | uint32_t{packet[7]});
            }
            (void)metadata_listener.poll_messages(metadata, 100);
            return !metadata.error.empty() || paired_messages() >= 10;
        }, timeout);
        ::close(video_sock);
        const std::string problem = exit_problem(r);
        if (!problem.empty() || !metadata.error.empty() || paired_messages() < 10) {
            std::cerr << "[FAIL] RTSP Insight run: " << problem << " " << metadata.error
                      << " video frames=" << video_timestamps.size()
                      << " metadata messages=" << metadata.messages.size()
                      << " paired=" << paired_messages() << "\n"
                      << "stdout:\n" << r.stdout_text << "\nstderr:\n" << r.stderr_text << "\n";
            return 1;
        }
        std::cout << "[OK] RTSP Insight: RTP/H.264 video received and " << paired_messages()
                  << " metadata messages paired with their video frames\n";
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
            // This test is the Insight video receiver, so it publishes to its own loopback.
            {"output.insight.host",          "127.0.0.1"},
            {"output.insight.video_port",    std::to_string(video_port)},
            {"output.insight.metadata_port", std::to_string(metadata_port)},
        });
        const std::vector<std::string> usb_args = {
            "--config", usb_config.string(), "--max-frames", std::to_string(kFrames)};

        // Headless: --test clears the Insight host.
        {
            std::cout << "[RUN] " << binary << " input.usb.camera=" << camera << " --test\n";
            auto args = usb_args;
            args.push_back("--test");
            const ProcessResult r = spawn_and_wait(binary, args, timeout);
            if (!run_ok("USB camera run", r)) return 1;
            if (frames_done(r.stdout_text) != kFrames) {
                std::cerr << "[FAIL] USB camera run did not process " << kFrames << " frames\n"
                          << "stdout:\n" << r.stdout_text << "\n";
                return 1;
            }
            std::cout << "[OK] USB camera: processed " << kFrames << " frames\n";
        }

        // Insight: the camera's JPEG is sent unchanged.
        {
            const int sock = bind_video_port(video_port);
            if (sock < 0) {
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
            if (frames_done(r.stdout_text) != kFrames) {
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
