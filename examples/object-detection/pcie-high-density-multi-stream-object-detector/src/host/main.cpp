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

#include "app_config.h"
#include "support/object_detection/detection_egress.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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
  bool dump_pipeline = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  const fs::path adjacent = fs::path(argv[0]).parent_path() / "config.yaml";
  options.config_path = fs::exists(adjacent)
                            ? adjacent
                            : fs::path(PCIE_HIGH_DENSITY_SOURCE_DIR) / "../common/config.yaml";

  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--config") {
      if (++index >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[index];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--dump-pipeline") {
      options.dump_pipeline = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--config <path>] [--validate-config-only] [--dump-pipeline]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }
  return options;
}

std::string gst_quote(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2U);
  out.push_back('"');
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::int64_t time_ns(GstClockTime value) {
  return GST_CLOCK_TIME_IS_VALID(value) ? static_cast<std::int64_t>(value) : -1;
}

std::uint32_t rtp_timestamp_from_pts(std::int64_t pts_ns) {
  if (pts_ns < 0) {
    return 0;
  }
  const auto ticks = (static_cast<__int128>(pts_ns) * 90000) / GST_SECOND;
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(ticks));
}

struct Box {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float score = 0.0F;
  int class_id = -1;
};

struct RawBox {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t w = 0;
  std::int32_t h = 0;
  float score = 0.0F;
  std::int32_t class_id = -1;
};

static_assert(sizeof(RawBox) == 24U);

std::vector<Box> parse_bbox_payload(const std::uint8_t* data, std::size_t size, int width,
                                    int height, int top_k) {
  if (!data || size < sizeof(std::uint32_t)) {
    throw std::runtime_error("BBOX payload is smaller than its count header");
  }

  std::uint32_t count = 0;
  std::memcpy(&count, data, sizeof(count));
  const std::size_t capacity = (size - sizeof(count)) / sizeof(RawBox);
  if (count > capacity) {
    throw std::runtime_error("BBOX count exceeds payload capacity");
  }
  if (top_k > 0 && count > static_cast<std::uint32_t>(top_k)) {
    throw std::runtime_error("BBOX count exceeds inference.max_detections");
  }

  std::vector<Box> boxes;
  boxes.reserve(count);
  const std::uint8_t* records = data + sizeof(count);
  for (std::size_t index = 0; index < count; ++index) {
    RawBox raw{};
    std::memcpy(&raw, records + index * sizeof(RawBox), sizeof(raw));
    const float x1 = std::clamp(static_cast<float>(raw.x), 0.0F, static_cast<float>(width));
    const float y1 = std::clamp(static_cast<float>(raw.y), 0.0F, static_cast<float>(height));
    const float x2 = std::clamp(static_cast<float>(raw.x + raw.w), 0.0F, static_cast<float>(width));
    const float y2 =
        std::clamp(static_cast<float>(raw.y + raw.h), 0.0F, static_cast<float>(height));
    boxes.push_back(Box{x1, y1, x2, y2, raw.score, raw.class_id});
  }
  return boxes;
}

class UdpEndpoint {
public:
  UdpEndpoint(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    addrinfo* result = nullptr;
    const std::string service = std::to_string(port);
    const int status = getaddrinfo(host.c_str(), service.c_str(), &hints, &result);
    if (status != 0) {
      throw std::runtime_error("failed to resolve Insight host " + host + ": " +
                               gai_strerror(status));
    }

    for (addrinfo* current = result; current; current = current->ai_next) {
      const int fd = socket(current->ai_family, current->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC,
                            current->ai_protocol);
      if (fd < 0) {
        continue;
      }
      socket_ = fd;
      std::memcpy(&address_, current->ai_addr, current->ai_addrlen);
      address_length_ = static_cast<socklen_t>(current->ai_addrlen);
      break;
    }
    freeaddrinfo(result);
    if (socket_ < 0) {
      throw std::runtime_error("failed to create Insight metadata UDP socket");
    }
  }

  ~UdpEndpoint() {
    if (socket_ >= 0) {
      close(socket_);
    }
  }

  UdpEndpoint(const UdpEndpoint&) = delete;
  UdpEndpoint& operator=(const UdpEndpoint&) = delete;

  bool send(std::string_view payload) const {
    const ssize_t sent = sendto(socket_, payload.data(), payload.size(), MSG_DONTWAIT,
                                reinterpret_cast<const sockaddr*>(&address_), address_length_);
    return sent == static_cast<ssize_t>(payload.size());
  }

private:
  int socket_ = -1;
  sockaddr_storage address_{};
  socklen_t address_length_ = 0;
};

struct FrameContext {
  std::uint64_t identifier = 0;
  std::uint32_t frame_id = 0;
  std::int64_t pts_ns = -1;
  std::int64_t dts_ns = -1;
  std::int64_t duration_ns = -1;
  std::chrono::steady_clock::time_point admitted_at;
};

struct SharedState {
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error;

  void fail(std::string message) {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true)) {
      std::lock_guard lock(error_mutex);
      error = std::move(message);
    }
  }
};

struct StreamRuntime {
  int index = 0;
  const pcie_high_density::AppConfig* config = nullptr;
  const std::vector<std::string>* labels = nullptr;
  SharedState* shared = nullptr;
  std::unique_ptr<UdpEndpoint> metadata_endpoint;

  std::atomic<std::uint32_t> next_frame_id{0};
  std::atomic<bool> waiting_for_random_access{true};
  std::mutex frame_mutex;
  std::unordered_map<std::uint32_t, FrameContext> frames;

  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> admitted{0};
  std::atomic<std::uint64_t> dropped_before_admission{0};
  std::atomic<std::uint64_t> result_timeouts{0};
  std::atomic<std::uint64_t> correlation_misses{0};
  std::atomic<std::uint64_t> returned{0};
  std::atomic<std::uint64_t> metadata_sent{0};
  std::atomic<std::uint64_t> metadata_dropped{0};
};

GstPadProbeReturn stamp_input(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* stream = static_cast<StreamRuntime*>(user_data);
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  ++stream->received;

  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) {
    ++stream->dropped_before_admission;
    return GST_PAD_PROBE_DROP;
  }

  if (stream->waiting_for_random_access &&
      GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT)) {
    ++stream->dropped_before_admission;
    return GST_PAD_PROBE_DROP;
  }

  std::unique_lock frame_lock(stream->frame_mutex);
  if (stream->shared->failed || g_stop_requested) {
    ++stream->dropped_before_admission;
    return GST_PAD_PROBE_DROP;
  }
  if (!gst_buffer_is_writable(buffer)) {
    buffer = gst_buffer_make_writable(buffer);
    if (!buffer) {
      stream->shared->fail("stream " + std::to_string(stream->index) +
                           " could not make its PCIe input writable");
      return GST_PAD_PROBE_DROP;
    }
    GST_PAD_PROBE_INFO_DATA(info) = buffer;
  }

  const std::uint32_t frame_id = stream->next_frame_id.fetch_add(1);
  const std::uint64_t identifier = frame_id;
  GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaHostMeta");
  GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!structure) {
    stream->shared->fail("stream " + std::to_string(stream->index) +
                         " could not attach GstSimaHostMeta");
    return GST_PAD_PROBE_DROP;
  }
  gst_structure_set(structure, "frame-identifier", G_TYPE_UINT64, identifier, "stream-id",
                    G_TYPE_UINT, static_cast<guint>(stream->index), "frame-id", G_TYPE_UINT,
                    frame_id, nullptr);

  FrameContext frame;
  frame.identifier = identifier;
  frame.frame_id = frame_id;
  frame.pts_ns = time_ns(GST_BUFFER_PTS(buffer));
  frame.dts_ns = time_ns(GST_BUFFER_DTS(buffer));
  frame.duration_ns = time_ns(GST_BUFFER_DURATION(buffer));
  frame.admitted_at = std::chrono::steady_clock::now();
  const auto insertion = stream->frames.emplace(frame_id, frame);
  if (!insertion.second) {
    stream->shared->fail("stream " + std::to_string(stream->index) +
                         " generated a duplicate frame identifier");
    return GST_PAD_PROBE_DROP;
  }
  const auto capacity = static_cast<std::size_t>(stream->config->correlation_cache_size);
  while (stream->frames.size() > capacity) {
    const auto oldest = std::min_element(stream->frames.begin(), stream->frames.end(),
                                         [](const auto& lhs, const auto& rhs) {
                                           return lhs.second.admitted_at < rhs.second.admitted_at;
                                         });
    if (oldest == stream->frames.end()) {
      break;
    }
    stream->frames.erase(oldest);
  }
  stream->waiting_for_random_access = false;
  ++stream->admitted;
  return GST_PAD_PROBE_OK;
}

void input_queue_overrun(GstElement*, gpointer user_data) {
  auto* stream = static_cast<StreamRuntime*>(user_data);
  stream->waiting_for_random_access = true;
  ++stream->dropped_before_admission;
}

GstFlowReturn consume_result(GstAppSink* sink, gpointer user_data) {
  auto* stream = static_cast<StreamRuntime*>(user_data);
  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (!sample) {
    return GST_FLOW_EOS;
  }

  try {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
      throw std::runtime_error("result sample has no buffer");
    }
    GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaHostMeta");
    const GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    guint64 identifier = 0;
    guint metadata_stream = 0;
    guint metadata_frame = 0;
    if (!structure || !gst_structure_get_uint64(structure, "frame-identifier", &identifier) ||
        !gst_structure_get_uint(structure, "stream-id", &metadata_stream) ||
        !gst_structure_get_uint(structure, "frame-id", &metadata_frame)) {
      throw std::runtime_error("result is missing PCIe correlation metadata");
    }
    if (metadata_stream != static_cast<guint>(stream->index)) {
      throw std::runtime_error("result stream-id does not match neatpciehost.src_N");
    }

    FrameContext frame;
    {
      std::lock_guard lock(stream->frame_mutex);
      const auto found = stream->frames.find(metadata_frame);
      if (found == stream->frames.end() || found->second.identifier != identifier) {
        ++stream->correlation_misses;
        gst_sample_unref(sample);
        return GST_FLOW_OK;
      }
      frame = found->second;
      stream->frames.erase(found);
    }

    GstMapInfo mapping{};
    if (!gst_buffer_map(buffer, &mapping, GST_MAP_READ)) {
      throw std::runtime_error("failed to map returned BBOX buffer");
    }
    std::vector<Box> boxes;
    try {
      boxes = parse_bbox_payload(mapping.data, mapping.size, stream->config->input_width,
                                 stream->config->input_height, stream->config->max_detections);
    } catch (...) {
      gst_buffer_unmap(buffer, &mapping);
      throw;
    }
    gst_buffer_unmap(buffer, &mapping);
    const std::uint64_t completed = ++stream->returned;

    if (completed > static_cast<std::uint64_t>(stream->config->warmup_frames)) {
      sima_examples::detection_egress::FrameMetadata output;
      output.stream_index = stream->index;
      const std::string stream_id = "stream" + std::to_string(stream->index);
      output.stream_id = stream_id;
      output.frame_id = frame.frame_id;
      output.pts_ns = frame.pts_ns;
      output.dts_ns = frame.dts_ns;
      output.duration_ns = frame.duration_ns;
      output.input_seq = static_cast<std::int64_t>(frame.identifier);
      output.orig_input_seq = static_cast<std::int64_t>(frame.identifier);
      if (frame.pts_ns >= 0) {
        output.rtp_timestamp = rtp_timestamp_from_pts(frame.pts_ns);
      }
      const std::string payload = sima_examples::detection_egress::serialize(
          boxes, *stream->labels, stream->config->input_width, stream->config->input_height,
          output);
      if (stream->metadata_endpoint->send(payload)) {
        ++stream->metadata_sent;
      } else {
        ++stream->metadata_dropped;
      }
    }
  } catch (const std::exception& error) {
    stream->shared->fail("stream " + std::to_string(stream->index) +
                         " result handling failed: " + error.what());
  }

  gst_sample_unref(sample);
  return stream->shared->failed ? GST_FLOW_ERROR : GST_FLOW_OK;
}

std::string make_pipeline(const pcie_high_density::AppConfig& config) {
  std::ostringstream pipeline;
  pipeline << "neatpciehost name=pcie" << " queue=" << config.queue
           << " queuesize=" << config.pcie_queue_size << " buffersize=" << config.pcie_buffer_size
           << " queuedepth=" << config.max_inflight_total
           << " request-timeout=" << std::max(1, config.result_timeout_ms / 1000)
           << " card-number=" << config.card_id;

  for (int stream = 0; stream < config.stream_count; ++stream) {
    pipeline << " rtspsrc name=rtsp_" << stream
             << " location=" << gst_quote(config.rtsp_urls[static_cast<std::size_t>(stream)])
             << " latency=" << config.latency_ms
             << " drop-on-latency=" << (config.rtsp_drop_on_latency ? "true" : "false")
             << " do-rtsp-keep-alive=true";
    if (config.rtsp_tcp) {
      pipeline << " protocols=tcp";
    }
    pipeline << " rtsp_" << stream << ". ! application/x-rtp,media=video,encoding-name=H264"
             << " ! rtph264depay wait-for-keyframe=true"
             << " ! h264parse disable-passthrough=true config-interval=-1"
             << " ! video/x-h264,parsed=true,stream-format=byte-stream,alignment=au,width="
             << config.input_width << ",height=" << config.input_height
             << ",framerate=" << config.input_fps << "/1" << " ! tee name=encoded_" << stream
             << " encoded_" << stream << ". ! queue name=pcie_input_queue_" << stream
             << " max-size-buffers=" << config.max_inflight_per_stream
             << " max-size-bytes=0 max-size-time=0 leaky=upstream" << " ! identity name=pcie_stamp_"
             << stream << " silent=true" << " ! pcie.sink_" << stream;

    if (config.video_enabled) {
      pipeline << " encoded_" << stream
               << ". ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
               << " ! rtph264pay pt=96 config-interval=1 timestamp-offset=0"
               << " ! udpsink host=" << gst_quote(config.insight_host)
               << " port=" << config.video_port_base + stream << " sync=false async=false";
    }

    pipeline << " pcie.src_" << stream
             << " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0"
             << " ! appsink name=result_" << stream
             << " emit-signals=true sync=false async=false max-buffers=2 drop=false";
  }
  return pipeline.str();
}

void validate_pipeline(const std::string& launch) {
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  if (!pipeline || error) {
    const std::string message = error ? error->message : "unknown parse error";
    g_clear_error(&error);
    if (pipeline) {
      gst_object_unref(pipeline);
    }
    throw std::runtime_error("failed to parse host GStreamer pipeline: " + message);
  }
  gst_object_unref(pipeline);
}

class HostApplication {
public:
  explicit HostApplication(pcie_high_density::AppConfig config)
      : config_(std::move(config)), labels_(pcie_high_density::load_labels(config_.labels_path)) {
    const std::string launch = make_pipeline(config_);
    GError* error = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &error);
    if (!pipeline_) {
      const std::string message = error ? error->message : "unknown parse error";
      g_clear_error(&error);
      throw std::runtime_error("failed to construct host GStreamer pipeline: " + message);
    }
    if (error) {
      const std::string message = error->message;
      g_clear_error(&error);
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      throw std::runtime_error("host GStreamer pipeline is incomplete: " + message);
    }

    streams_.reserve(static_cast<std::size_t>(config_.stream_count));
    for (int index = 0; index < config_.stream_count; ++index) {
      auto stream = std::make_unique<StreamRuntime>();
      stream->index = index;
      stream->config = &config_;
      stream->labels = &labels_;
      stream->shared = &shared_;
      stream->metadata_endpoint =
          std::make_unique<UdpEndpoint>(config_.insight_host, config_.metadata_port_base + index);

      const std::string stamp_name = "pcie_stamp_" + std::to_string(index);
      GstElement* stamp = gst_bin_get_by_name(GST_BIN(pipeline_), stamp_name.c_str());
      if (!stamp) {
        throw std::runtime_error("pipeline is missing " + stamp_name);
      }
      GstPad* stamp_src = gst_element_get_static_pad(stamp, "src");
      gst_object_unref(stamp);
      if (!stamp_src) {
        throw std::runtime_error(stamp_name + " has no src pad");
      }
      gst_pad_add_probe(stamp_src, GST_PAD_PROBE_TYPE_BUFFER, stamp_input, stream.get(), nullptr);
      gst_object_unref(stamp_src);

      const std::string input_queue_name = "pcie_input_queue_" + std::to_string(index);
      GstElement* input_queue = gst_bin_get_by_name(GST_BIN(pipeline_), input_queue_name.c_str());
      if (!input_queue) {
        throw std::runtime_error("pipeline is missing " + input_queue_name);
      }
      g_signal_connect(input_queue, "overrun", G_CALLBACK(input_queue_overrun), stream.get());
      gst_object_unref(input_queue);

      const std::string result_name = "result_" + std::to_string(index);
      GstElement* result = gst_bin_get_by_name(GST_BIN(pipeline_), result_name.c_str());
      if (!result) {
        throw std::runtime_error("pipeline is missing " + result_name);
      }
      g_signal_connect(result, "new-sample", G_CALLBACK(consume_result), stream.get());
      gst_object_unref(result);
      streams_.push_back(std::move(stream));
    }
  }

  ~HostApplication() {
    if (pipeline_) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(pipeline_);
    }
  }

  int run() {
    const GstStateChangeReturn state = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state == GST_STATE_CHANGE_FAILURE) {
      throw std::runtime_error("failed to start host GStreamer pipeline");
    }
    std::cout << "[host] running " << pcie_high_density::config_summary(config_) << "\n";

    GstBus* bus = gst_element_get_bus(pipeline_);
    auto last_report = std::chrono::steady_clock::now();
    while (!g_stop_requested && !shared_.failed) {
      GstMessage* message = gst_bus_timed_pop_filtered(
          bus, 250 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
      if (message) {
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
          GError* error = nullptr;
          gchar* debug = nullptr;
          gst_message_parse_error(message, &error, &debug);
          std::string text = error ? error->message : "unknown GStreamer error";
          if (debug && *debug) {
            text += " (" + std::string(debug) + ")";
          }
          g_clear_error(&error);
          g_free(debug);
          shared_.fail(std::move(text));
        } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
          g_stop_requested = 1;
        }
        gst_message_unref(message);
      }
      expire_results();

      const auto now = std::chrono::steady_clock::now();
      if (config_.profile && now - last_report >= std::chrono::seconds(5)) {
        print_stats(false);
        last_report = now;
      }
    }
    gst_object_unref(bus);

    gst_element_send_event(pipeline_, gst_event_new_eos());
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    print_stats(true);
    if (shared_.failed) {
      std::lock_guard lock(shared_.error_mutex);
      std::cerr << "[host] failed: " << shared_.error << "\n";
      return 1;
    }
    return 0;
  }

  void print_stats(bool final) const {
    std::uint64_t received = 0;
    std::uint64_t admitted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t result_timeouts = 0;
    std::uint64_t correlation_misses = 0;
    std::uint64_t returned = 0;
    std::size_t outstanding = 0;
    std::uint64_t metadata_sent = 0;
    std::uint64_t metadata_dropped = 0;
    for (const auto& stream : streams_) {
      received += stream->received;
      admitted += stream->admitted;
      dropped += stream->dropped_before_admission;
      result_timeouts += stream->result_timeouts;
      correlation_misses += stream->correlation_misses;
      returned += stream->returned;
      {
        std::lock_guard lock(stream->frame_mutex);
        outstanding += stream->frames.size();
      }
      metadata_sent += stream->metadata_sent;
      metadata_dropped += stream->metadata_dropped;
    }
    std::cout << (final ? "[host] final" : "[host] stats") << " received=" << received
              << " admitted=" << admitted << " dropped_before_admission=" << dropped
              << " result_timeouts=" << result_timeouts
              << " correlation_misses=" << correlation_misses << " outstanding=" << outstanding
              << " returned=" << returned << " metadata_sent=" << metadata_sent
              << " metadata_dropped=" << metadata_dropped << "\n";
  }

  void expire_results() {
    const auto now = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(config_.result_timeout_ms);
    for (const auto& stream : streams_) {
      std::lock_guard lock(stream->frame_mutex);
      for (auto it = stream->frames.begin(); it != stream->frames.end();) {
        if (now - it->second.admitted_at > timeout) {
          ++stream->result_timeouts;
          it = stream->frames.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

private:
  pcie_high_density::AppConfig config_;
  std::vector<std::string> labels_;
  SharedState shared_;
  GstElement* pipeline_ = nullptr;
  std::vector<std::unique_ptr<StreamRuntime>> streams_;
};

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    const auto config = pcie_high_density::load_config(cli.config_path);
    std::cout << "[host] " << pcie_high_density::config_summary(config) << "\n";
    if (cli.validate_config_only) {
      return 0;
    }

    gst_init(&argc, &argv);
    GstElementFactory* pcie_factory = gst_element_factory_find("neatpciehost");
    if (!pcie_factory) {
      throw std::runtime_error("GStreamer element neatpciehost is not installed");
    }
    gst_object_unref(pcie_factory);
    const std::string launch = make_pipeline(config);
    if (cli.dump_pipeline) {
      validate_pipeline(launch);
      std::cout << launch << "\n";
      return 0;
    }

    const auto previous_sigint = std::signal(SIGINT, request_stop);
    const auto previous_sigterm = std::signal(SIGTERM, request_stop);
    HostApplication app(config);
    const int result = app.run();
    std::signal(SIGINT, previous_sigint);
    std::signal(SIGTERM, previous_sigterm);
    return result;
  } catch (const std::exception& error) {
    std::cerr << "pcie-high-density host: " << error.what() << "\n";
    return 2;
  }
}
