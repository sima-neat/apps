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

#include "../common/app_config.h"
#include "stream_demux.h"

#include "neat.h"
#include "neat/models.h"
#include "neat/nodes.h"

#include <builder/OutputSpec.h>
#include <gst/GstInit.h>
#include <nodes/sima/SimaDecode.h>

#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
  bool dump_backend = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  const fs::path adjacent = fs::path(argv[0]).parent_path() / "config.yaml";
  options.config_path = fs::exists(adjacent)
                            ? adjacent
                            : fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR) / "../common/config.yaml";
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--config") {
      if (++index >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[index];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--dump-backend") {
      options.dump_backend = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--config <path>] [--validate-config-only] [--dump-backend]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }
  return options;
}

std::shared_ptr<simaai::neat::PCIeSrc>
make_pcie_source(const pcie_high_density::AppConfig& config) {
  simaai::neat::PCIeSrcOptions options;
  options.queue = config.queue;
  options.buffer_size = config.pcie_buffer_size;
  options.pool_size = config.pcie_pool_size;
  return std::make_shared<simaai::neat::PCIeSrc>(std::move(options));
}

simaai::neat::Graph make_decoder(const pcie_high_density::AppConfig& config, int stream) {
  simaai::neat::SimaDecodeOptions decode;
  decode.type = simaai::neat::SimaDecodeType::H264;
  decode.out_format = simaai::neat::FormatTag::NV12;
  decode.decoder_name = "pcie_decoder_" + std::to_string(stream);
  decode.raw_output = true;
  decode.next_element = "CVU";
  decode.dec_width = config.input_width;
  decode.dec_height = config.input_height;
  decode.dec_fps = config.input_fps;
  decode.num_buffers = config.decoder_buffers;
  decode.input_buffers = config.decoder_input_buffers;
  decode.decoder_tuning = config.decoder_tuning;
  decode.memory_opt =
      config.decoder_tuning == "low-memory" || config.decoder_tuning == "throughput-low-latency";

  simaai::neat::Graph graph("pcie-decoder-" + std::to_string(stream));
  graph.add(simaai::neat::nodes::SimaDecode(std::move(decode)));
  graph.add(simaai::neat::nodes::CapsRaw("NV12", config.input_width, config.input_height,
                                         config.input_fps));
  graph.add(simaai::neat::nodes::Output("detector_frame"));
  return graph;
}

class PcieMultiStreamResultSink final : public simaai::neat::Node {
public:
  PcieMultiStreamResultSink(const int stream_count, const int queue)
      : stream_count_(stream_count), queue_(queue) {}

  std::string kind() const override {
    return "PcieMultiStreamResultSink";
  }

  std::string user_label() const override {
    return "pcie-multi-stream-result-sink";
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

  simaai::neat::MemoryContract memory_contract() const override {
    return simaai::neat::MemoryContract::AllowEitherButReport;
  }

  std::string backend_fragment(int node_index) const override {
    const std::string prefix = "n" + std::to_string(node_index) + "_pcie_result";
    const std::string demux_name = prefix + "_demux";
    const std::string sink_name = prefix + "_sink";

    std::ostringstream fragment;
    fragment << "neatappstreamdemux name=" << demux_name;
    for (int stream = 0; stream < stream_count_; ++stream) {
      fragment << ' ' << demux_name << ".src_" << stream << " ! queue name=" << prefix << "_queue_"
               << stream << " max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! ";
      if (stream == 0) {
        fragment << "neatpciesink name=" << sink_name << " queue=" << queue_;
      } else {
        fragment << sink_name << ".sink_" << stream;
      }
    }
    return fragment.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    const std::string prefix = "n" + std::to_string(node_index) + "_pcie_result";
    std::vector<std::string> names{prefix + "_demux", prefix + "_sink"};
    names.reserve(static_cast<std::size_t>(2 + stream_count_));
    for (int stream = 0; stream < stream_count_; ++stream) {
      names.push_back(prefix + "_queue_" + std::to_string(stream));
    }
    return names;
  }

private:
  int stream_count_ = 0;
  int queue_ = 0;
};

simaai::neat::BoxDecodeType decode_type(const std::string& value) {
  if (value == "yolo26") {
    return simaai::neat::BoxDecodeType::YoloV26;
  }
  if (value == "yolov8") {
    return simaai::neat::BoxDecodeType::YoloV8;
  }
  throw std::runtime_error("unsupported model.decode_type: " + value);
}

std::unique_ptr<simaai::neat::Model> make_model(const pcie_high_density::AppConfig& config) {
  simaai::neat::Model::Options options;
  options.verbose = simaai::neat::VerboseOptions::quiet();
  options.preprocess.kind = simaai::neat::InputKind::Image;
  options.preprocess.enable = simaai::neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  options.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  options.decode_type = decode_type(config.decode_type);
  options.score_threshold = config.min_score;
  options.nms_iou_threshold = config.nms_iou;
  options.top_k = config.max_detections;
  return std::make_unique<simaai::neat::Model>(config.model_path.string(), options);
}

simaai::neat::Graph make_graph(const pcie_high_density::AppConfig& config,
                               std::unique_ptr<simaai::neat::Model>& model) {
  model = make_model(config);

  const auto source = make_pcie_source(config);
  simaai::neat::Graph detector = model->graph();
  simaai::neat::Graph results;
  results.add(std::make_shared<PcieMultiStreamResultSink>(config.stream_count, config.queue));

  simaai::neat::GraphOptions graph_options;
  graph_options.advanced_execution.internal_queue_depth = config.inference_internal_queue_depth;
  graph_options.advanced_execution.inference_async = true;
  simaai::neat::Graph graph(graph_options);
  for (int stream = 0; stream < config.stream_count; ++stream) {
    simaai::neat::Graph decoder = make_decoder(config, stream);
    graph.connect(source, "src_" + std::to_string(stream), decoder);

    simaai::neat::GraphLinkOptions detector_link;
    detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    detector_link.queue_depth = config.inference_queue_depth;
    detector_link.stream_id = std::to_string(stream);
    detector_link.max_inflight_per_stream = config.inference_max_inflight_per_stream;
    detector_link.max_inflight_total = config.inference_max_inflight_total;
    graph.connect(decoder, detector, detector_link);
  }
  graph.connect(detector, results);
  return graph;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    const auto config = pcie_high_density::load_config(cli.config_path);
    std::cout << "[card] " << pcie_high_density::config_summary(config) << "\n";
    if (cli.validate_config_only) {
      return 0;
    }

    simaai::neat::gst_init_once();
    if (!pcie_high_density::register_stream_demux()) {
      throw std::runtime_error("failed to register the stream-aware BBOX demultiplexer");
    }

    std::unique_ptr<simaai::neat::Model> model;
    simaai::neat::Graph graph = make_graph(config, model);
    if (cli.dump_backend) {
      std::cout << graph.describe_backend() << "\n";
      return 0;
    }

    simaai::neat::RunOptions run_options;
    run_options.preset = simaai::neat::RunPreset::Realtime;
    run_options.queue_depth = config.inference_queue_depth;
    run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
    run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;

    const auto previous_sigint = std::signal(SIGINT, request_stop);
    const auto previous_sigterm = std::signal(SIGTERM, request_stop);
    simaai::neat::Run run = graph.build(run_options);
    std::cout << "[card] PCIe source, shared detector, and result sink are running\n";
    while (!g_stop_requested && run.running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    const bool stopped_unexpectedly = !g_stop_requested && !run.running();
    run.close();
    std::signal(SIGINT, previous_sigint);
    std::signal(SIGTERM, previous_sigterm);
    if (stopped_unexpectedly) {
      throw std::runtime_error("card pipeline stopped unexpectedly");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pcie-high-density card: " << error.what() << "\n";
    return 2;
  }
}
