// Copyright 2026 SiMa Technologies, Inc.

#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/config_api.cpp"
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/workers_api.cpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  const fs::path default_config = multi_stream_people_tracker::default_config_path();
  std::cout << "Multi-stream people tracking with Insight output.\n\n";
  std::cout << "Usage: " << argv0 << " [--config <path>]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config <path>  Path to YAML configuration. Default: " << default_config << "\n";
  std::cout << "  --help           Show this help message.\n";
}

} // namespace

int main(int argc, char** argv) {
  fs::path config_path = multi_stream_people_tracker::default_config_path();

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    }
    if (arg == "--config") {
      if (index + 1 >= argc) {
        std::cerr << "Error: --config requires a path\n";
        return 2;
      }
      config_path = argv[++index];
      continue;
    }
    std::cerr << "Error: unknown argument: " << arg << "\n";
    return 2;
  }

  try {
    const auto cfg = multi_stream_people_tracker::load_app_config(config_path);
    return multi_stream_people_tracker::run_app(cfg);
  } catch (const std::exception& ex) {
    if (!fs::exists(config_path)) {
      std::cerr << "Error: config file not found: " << config_path << "\n";
    } else {
      std::cerr << "Error: failed to load config " << config_path << ": " << ex.what() << "\n";
    }
    return 2;
  }
}
