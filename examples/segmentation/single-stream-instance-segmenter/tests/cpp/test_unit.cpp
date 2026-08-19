// Unit test for single-stream-instance-segmenter: validates CLI arg handling.
#include "../../src/cpp/ffprobe_command.h"
#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  using sima_examples::single_stream_instance_segmenter::build_ffprobe_geometry_command;
  using sima_examples::single_stream_instance_segmenter::FfprobeCommandOptions;

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

  // Test 1: --help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] --help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] --help printed usage\n";
    }
  }

  // Test 2: unknown flag is rejected.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --bogus: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag rejected\n";
    }
  }

  // Test 3: bad config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad config: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
