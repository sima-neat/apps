// Unit test for single-stream-object-detector: validates CLI arg handling.
#include "../../src/cpp/ffprobe_command.h"
#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  using sima_examples::single_stream_object_detector::build_ffprobe_geometry_command;
  using sima_examples::single_stream_object_detector::FfprobeCommandOptions;

  {
    FfprobeCommandOptions options;
    options.rtsp_source = true;
    options.tcp = true;
    options.tls_verify = false;
    const std::string command = build_ffprobe_geometry_command("rtsp://camera/live", options);
    if (command.find("-rtsp_transport tcp") == std::string::npos ||
        command.find("-rtsp_transport tcp") != command.rfind("-rtsp_transport tcp") ||
        command.find("-tls_verify 0") == std::string::npos ||
        command.find("-rtsp_transport tcp") > command.find("'rtsp://camera/live'")) {
      std::cerr
          << "[FAIL] TCP RTSP probe should select TCP before the URL and preserve TLS options\n";
      ++failures;
    }
  }

  {
    FfprobeCommandOptions options;
    options.rtsp_source = true;
    const std::string command = build_ffprobe_geometry_command("rtsp://camera/live", options);
    if (command.find("-rtsp_transport") != std::string::npos) {
      std::cerr << "[FAIL] default RTSP probe should not force TCP\n";
      ++failures;
    }
  }

  {
    FfprobeCommandOptions options;
    options.tcp = true;
    options.tls_verify = false;
    const std::string command = build_ffprobe_geometry_command("https://camera/live", options);
    if (command.find("-rtsp_transport") != std::string::npos ||
        command.find("-tls_verify 0") == std::string::npos) {
      std::cerr << "[FAIL] HTTP probe should omit RTSP transport and preserve TLS options\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0 || r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help should pass and print usage\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent/single-rtsp-config.yaml"}, 20000);
    if (r.exit_code == 0 || r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config should fail and mention config\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0 || r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag should fail and mention unknown argument\n";
      ++failures;
    }
  }

  return failures > 0 ? 1 : 0;
}
