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

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace pcie_high_density {

struct AppConfig {
  std::string profile_name;
  int stream_count = 0;

  std::filesystem::path model_path;
  std::filesystem::path labels_path;
  std::string decode_type = "yolo26";

  int card_id = 0;
  int queue = 0;

  std::vector<std::string> rtsp_urls;
  bool rtsp_tcp = true;
  bool rtsp_drop_on_latency = false;
  int latency_ms = 100;
  int startup_stagger_ms = 50;
  int input_width = 1280;
  int input_height = 720;
  int input_fps = 25;
  int decoder_buffers = 8;
  int decoder_input_buffers = 8;
  std::string decoder_tuning = "auto";

  int pcie_queue_size = 256;
  int pcie_buffer_size = 4 * 1024 * 1024;
  int pcie_pool_size = 4;
  int max_inflight_per_stream = 1;
  int max_inflight_total = 8;
  int result_timeout_ms = 30000;
  int correlation_cache_size = 256;

  double min_score = 0.30;
  double nms_iou = 0.60;
  int max_detections = 50;
  int inference_queue_depth = 16;
  int inference_internal_queue_depth = 1;
  int inference_max_inflight_per_stream = 1;
  int inference_max_inflight_total = 8;

  bool profile = false;
  int warmup_frames = 30;
  int stall_timeout_ms = 0;
  int teardown_timeout_ms = 10000;

  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
};

AppConfig load_config(const std::filesystem::path& path);
std::vector<std::string> load_labels(const std::filesystem::path& path);
std::string config_summary(const AppConfig& config);

} // namespace pcie_high_density
