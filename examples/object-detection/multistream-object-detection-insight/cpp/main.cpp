// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/config_api.cpp"
#include "utils/workers_api.cpp"

#include <iostream>
#include <string>

namespace multistream_object_detection_insight {
namespace {

void print_help(const char* argv0) {
  std::cout << "Multistream YOLOv8 object detection with Insight output.\n\n";
  std::cout << "Usage: " << argv0 << " [--config <path>]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config <path>  Path to YAML configuration. Default: " << default_config_path()
            << "\n";
  std::cout << "  --validate-config-only  Validate config and exit without opening RTSP streams.\n";
  std::cout << "  --help           Show this help message.\n";
}

} // namespace
} // namespace multistream_object_detection_insight

int main(int argc, char** argv) {
  using namespace multistream_object_detection_insight;

  std::filesystem::path config_path = default_config_path();
  bool validate_config_only = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    }
    if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --config requires a path\n";
        return 2;
      }
      config_path = argv[++i];
      continue;
    }
    if (arg == "--validate-config-only") {
      validate_config_only = true;
      continue;
    }
    std::cerr << "Error: unknown argument: " << arg << "\n";
    return 2;
  }

  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Error: config file not found: " << config_path << "\n";
    return 2;
  }

  try {
    const AppConfig cfg = load_app_config(config_path);
    const ModelFamily family = resolve_model_family(cfg.model.path);
    if (validate_config_only) {
      std::cout << "Config validated: " << config_path << " (family=" << to_string(family)
                << ", workers=" << cfg.worker_count << ", streams=" << cfg.rtsp_urls.size()
                << ")\n";
      return 0;
    }
    return run_app(cfg, family);
  } catch (const std::exception& ex) {
    std::cerr << "Error: failed to load config " << config_path << ": " << ex.what() << "\n";
    return 2;
  }
}
