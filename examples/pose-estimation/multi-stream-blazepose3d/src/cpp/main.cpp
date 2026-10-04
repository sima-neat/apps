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

#include "pose_logic.h"

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxDetections = 100;
constexpr int kMaxInflightPerStream = 4;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct StreamConfig {
  std::string id;
  std::string url;
  neat::nodes::groups::RtspCodec codec = neat::nodes::groups::RtspCodec::H264;
  int insight_channel = -1;
  int width = 0;
  int height = 0;
  int fps = 0;
};

struct AppConfig {
  std::string detector_model_path;
  std::string pose_model_path;
  std::vector<StreamConfig> streams;
  bool tcp = true;
  int latency_ms = 100;
  double detector_min_score = 0.30;
  double detector_nms_iou = 0.60;
  int max_people_per_frame = 4;
  double roi_scale = 1.65;
  double pose_presence_threshold = 0.50;
  bool pose_temporal_filter_enabled = true;
  int frame_limit = 0;
  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
};

struct FrameIdentity {
  std::string stream_id;
  int64_t frame_id = -1;
  int64_t pts_ns = -1;
  std::uint64_t sequence = 0;
};

struct FrameJob {
  std::uint64_t job_id = 0;
  int stream_index = 0;
  neat::Tensor rgb;
  std::vector<blazepose_app::Box> people;
  FrameIdentity identity;
};

struct PoseInputContext {
  std::uint64_t job_id = 0;
  int roi_index = 0;
  blazepose_app::Box box;
  blazepose_app::Affine affine;
};

struct PoseAggregate {
  int stream_index = 0;
  int expected = 0;
  int completed = 0;
  FrameIdentity identity;
  std::vector<blazepose_app::Pose> poses;
};

struct StreamRuntime {
  int index = 0;
  StreamConfig config;
  neat::nodes::groups::RtspDecodedInputOptions source_options;
  neat::Graph source_graph;
  neat::Run source_run;
  int width = 0;
  int height = 0;
  int fps = 0;
  std::unique_ptr<neat::MetadataSender> metadata_sender;
  std::mutex metadata_mutex;
  bool pose_temporal_filter_enabled = true;
  blazepose_app::PoseSmoother pose_smoother;
  std::uint64_t last_published_sequence = 0;
  std::atomic<int> frames_in{0};
  std::atomic<int> frames_out{0};
  // Admitted frames that are still queued or inside a model.
  std::atomic<int> outstanding_frames{0};
  std::atomic<bool> closed{false};
  std::atomic<bool> source_worker_finished{false};
};

struct SharedState {
  std::mutex mutex;
  std::condition_variable cv;
  // Latest-only work per stream; newer frames replace queued ones.
  std::vector<std::optional<FrameJob>> detector_mailboxes;
  std::vector<std::optional<FrameJob>> pose_mailboxes;
  // FIFO context for each input pushed to a shared model, in output order.
  std::deque<FrameJob> pending_detector_outputs;
  std::deque<PoseInputContext> pending_pose_outputs;
  std::unordered_map<std::uint64_t, PoseAggregate> aggregates;
  std::size_t next_detector_stream = 0;
  std::size_t next_pose_stream = 0;
  bool stopping = false;
  std::exception_ptr error;
};

struct AppRuntime {
  neat::Graph detector_graph;
  neat::Run detector_run;
  neat::Graph pose_graph;
  neat::Run pose_run;
  std::unique_ptr<neat::Model> detector_model;
  std::unique_ptr<neat::Model> pose_model;
  std::vector<std::unique_ptr<StreamRuntime>> streams;
  SharedState state;
  std::atomic<std::uint64_t> next_job_id{1};
};

neat::nodes::groups::RtspCodec parse_codec(const std::string& value) {
  if (value == "h264") {
    return neat::nodes::groups::RtspCodec::H264;
  }
  if (value == "h265") {
    return neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("stream codec must be h264 or h265");
}

std::string codec_name(neat::nodes::groups::RtspCodec codec) {
  return codec == neat::nodes::groups::RtspCodec::H265 ? "h265" : "h264";
}

int parse_yaml_integer(const std::string& value, const std::string& key);

void apply_stream_field(StreamConfig& stream, const std::string& key, const std::string& value) {
  const auto integer = [&]() { return parse_yaml_integer(value, "stream " + key); };
  if (key == "id") {
    stream.id = value;
  } else if (key == "url") {
    stream.url = value;
  } else if (key == "codec") {
    stream.codec = parse_codec(value);
  } else if (key == "insight_channel") {
    stream.insight_channel = integer();
  } else if (key == "width") {
    stream.width = integer();
  } else if (key == "height") {
    stream.height = integer();
  } else if (key == "fps") {
    stream.fps = integer();
  }
}

std::string strip_yaml_comment(const std::string& text) {
  const std::size_t colon = text.find(':');
  const std::size_t value_start =
      colon == std::string::npos ? std::string::npos : text.find_first_not_of(" \t", colon + 1);
  const char quote =
      value_start != std::string::npos && (text[value_start] == '\'' || text[value_start] == '"')
          ? text[value_start]
          : '\0';
  bool in_quote = quote != '\0';
  for (std::size_t index = 0; index < text.size(); ++index) {
    const char character = text[index];
    if (in_quote && quote == '"' && character == '\\' && index + 1 < text.size()) {
      ++index;
    } else if (in_quote && character == quote) {
      if (quote == '\'' && index + 1 < text.size() && text[index + 1] == quote) {
        ++index;
      } else if (index != value_start) {
        in_quote = false;
      }
    } else if (character == '#' && !in_quote &&
               (index == 0 || std::isspace(static_cast<unsigned char>(text[index - 1])))) {
      return text.substr(0, index);
    }
  }
  return text;
}

void append_utf8(std::string& output, std::uint32_t code_point) {
  if (code_point > 0x10FFFF || (code_point >= 0xD800 && code_point <= 0xDFFF)) {
    throw std::runtime_error("invalid Unicode escape in YAML scalar");
  }
  if (code_point <= 0x7F) {
    output.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7FF) {
    output.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point <= 0xFFFF) {
    output.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    output.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    output.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    output.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

std::uint32_t parse_hex_escape(const std::string& value, std::size_t& index, std::size_t digits) {
  if (index + digits >= value.size()) {
    throw std::runtime_error("incomplete YAML escape in YAML scalar");
  }
  std::uint32_t code_point = 0;
  for (std::size_t offset = 1; offset <= digits; ++offset) {
    const char digit = value[index + offset];
    code_point <<= 4;
    if (digit >= '0' && digit <= '9') {
      code_point |= static_cast<std::uint32_t>(digit - '0');
    } else if (digit >= 'a' && digit <= 'f') {
      code_point |= static_cast<std::uint32_t>(digit - 'a' + 10);
    } else if (digit >= 'A' && digit <= 'F') {
      code_point |= static_cast<std::uint32_t>(digit - 'A' + 10);
    } else {
      throw std::runtime_error("invalid hexadecimal YAML escape in YAML scalar");
    }
  }
  index += digits;
  return code_point;
}

std::string decode_yaml_scalar(const std::string& value) {
  if (value.size() < 2 || (value.front() != '\'' && value.front() != '"')) {
    return value;
  }
  if (value.back() != value.front()) {
    throw std::runtime_error("unterminated quoted YAML scalar");
  }
  std::string decoded;
  decoded.reserve(value.size() - 2);
  if (value.front() == '\'') {
    for (std::size_t index = 1; index + 1 < value.size(); ++index) {
      if (value[index] == '\'') {
        if (index + 2 >= value.size() || value[index + 1] != '\'') {
          throw std::runtime_error("invalid single-quoted YAML scalar");
        }
        ++index;
      }
      decoded.push_back(value[index]);
    }
    return decoded;
  }

  for (std::size_t index = 1; index + 1 < value.size(); ++index) {
    if (value[index] != '\\') {
      decoded.push_back(value[index]);
      continue;
    }
    if (++index + 1 >= value.size()) {
      throw std::runtime_error("incomplete YAML escape in YAML scalar");
    }
    const char escaped = value[index];
    switch (escaped) {
    case '0':
      decoded.push_back('\0');
      break;
    case 'a':
      decoded.push_back('\a');
      break;
    case 'b':
      decoded.push_back('\b');
      break;
    case 't':
      decoded.push_back('\t');
      break;
    case 'n':
      decoded.push_back('\n');
      break;
    case 'v':
      decoded.push_back('\v');
      break;
    case 'f':
      decoded.push_back('\f');
      break;
    case 'r':
      decoded.push_back('\r');
      break;
    case 'e':
      decoded.push_back('\x1B');
      break;
    case ' ':
      decoded.push_back(' ');
      break;
    case '"':
      decoded.push_back('"');
      break;
    case '/':
      decoded.push_back('/');
      break;
    case '\\':
      decoded.push_back('\\');
      break;
    case 'N':
      append_utf8(decoded, 0x85);
      break;
    case '_':
      append_utf8(decoded, 0xA0);
      break;
    case 'L':
      append_utf8(decoded, 0x2028);
      break;
    case 'P':
      append_utf8(decoded, 0x2029);
      break;
    case 'x':
      append_utf8(decoded, parse_hex_escape(value, index, 2));
      break;
    case 'u':
      append_utf8(decoded, parse_hex_escape(value, index, 4));
      break;
    case 'U':
      append_utf8(decoded, parse_hex_escape(value, index, 8));
      break;
    default:
      throw std::runtime_error("unsupported YAML escape in YAML scalar");
    }
  }
  return decoded;
}

bool is_plain_yaml_null(std::string value) {
  value = sima_examples::trim_copy(value);
  if (!value.empty() && (value.front() == '\'' || value.front() == '"')) {
    return false;
  }
  if (value.empty() || value == "~") {
    return true;
  }
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "null";
}

bool is_yaml_integer(std::string value) {
  value.erase(std::remove(value.begin(), value.end(), '_'), value.end());
  if (!value.empty() && (value.front() == '+' || value.front() == '-')) {
    value.erase(0, 1);
  }
  if (value.empty()) {
    return false;
  }
  const auto all_digits = [](const std::string& digits, int base) {
    return !digits.empty() && std::all_of(digits.begin(), digits.end(), [base](unsigned char c) {
      return std::isdigit(c) != 0 ? c - '0' < base
                                  : base == 16 && std::tolower(c) >= 'a' && std::tolower(c) <= 'f';
    });
  };
  if (value.size() > 2 && value.rfind("0b", 0) == 0) {
    return all_digits(value.substr(2), 2);
  }
  if (value.size() > 2 && value.rfind("0x", 0) == 0) {
    return all_digits(value.substr(2), 16);
  }
  if (value.find(':') != std::string::npos) {
    std::istringstream segments(value);
    std::string segment;
    bool first = true;
    while (std::getline(segments, segment, ':')) {
      if (!all_digits(segment, 10) || (first && segment.front() == '0') ||
          (!first && (segment.size() > 2 || (segment.size() == 2 &&
                                             (segment[0] - '0') * 10 + (segment[1] - '0') > 59)))) {
        return false;
      }
      first = false;
    }
    return value.back() != ':';
  }
  if (value.size() > 1 && value.front() == '0') {
    return all_digits(value.substr(1), 8);
  }
  return all_digits(value, 10);
}

int parse_yaml_integer(const std::string& value, const std::string& key) {
  if (!is_yaml_integer(value)) {
    throw std::runtime_error(key + " must be an integer");
  }
  std::string scalar = value;
  scalar.erase(std::remove(scalar.begin(), scalar.end(), '_'), scalar.end());
  const bool negative = scalar.front() == '-';
  if (scalar.front() == '+' || scalar.front() == '-') {
    scalar.erase(0, 1);
  }
  const std::uint64_t limit =
      negative ? static_cast<std::uint64_t>(INT_MAX) + 1U : static_cast<std::uint64_t>(INT_MAX);
  std::uint64_t magnitude = 0;
  const auto append = [&](std::uint64_t part, std::uint64_t base) {
    if (magnitude > (limit - part) / base) {
      throw std::runtime_error(key + " is outside the supported integer range");
    }
    magnitude = magnitude * base + part;
  };
  if (scalar.find(':') != std::string::npos) {
    std::istringstream segments(scalar);
    std::string segment;
    while (std::getline(segments, segment, ':')) {
      std::uint64_t part = 0;
      for (const char digit : segment) {
        if (part > (limit - static_cast<std::uint64_t>(digit - '0')) / 10U) {
          throw std::runtime_error(key + " is outside the supported integer range");
        }
        part = part * 10U + static_cast<std::uint64_t>(digit - '0');
      }
      append(part, 60U);
    }
  } else {
    int base = 10;
    std::size_t start = 0;
    if (scalar.size() > 2 && scalar.rfind("0b", 0) == 0) {
      base = 2;
      start = 2;
    } else if (scalar.size() > 2 && scalar.rfind("0x", 0) == 0) {
      base = 16;
      start = 2;
    } else if (scalar.size() > 1 && scalar.front() == '0') {
      base = 8;
      start = 1;
    }
    for (std::size_t index = start; index < scalar.size(); ++index) {
      const unsigned char character = static_cast<unsigned char>(scalar[index]);
      const int digit =
          std::isdigit(character) != 0 ? character - '0' : std::tolower(character) - 'a' + 10;
      append(static_cast<std::uint64_t>(digit), static_cast<std::uint64_t>(base));
    }
  }
  if (negative && magnitude == static_cast<std::uint64_t>(INT_MAX) + 1U) {
    return INT_MIN;
  }
  const int parsed = static_cast<int>(magnitude);
  return negative ? -parsed : parsed;
}

bool is_plain_yaml_string(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "true" || lowered == "false" || lowered == "yes" || lowered == "no" ||
      lowered == "on" || lowered == "off" || value.empty() || value.front() == '[' ||
      value.front() == '{' || value.front() == '&' || value.front() == '*' ||
      value.front() == '!' || is_yaml_integer(value)) {
    return false;
  }
  static const std::regex number_pattern(
      R"(^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$)");
  static const std::regex timestamp_pattern(
      R"(^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$)");
  return !std::regex_match(value, number_pattern) && !std::regex_match(value, timestamp_pattern);
}

bool is_yaml_block_scalar_header(const std::string& value) {
  static const std::regex pattern(R"(^[|>](?:[1-9][+-]?|[+-][1-9]?)?$)");
  return std::regex_match(value, pattern);
}

// ScalarConfig skips YAML lists, so the stream entries are read here: a
// "- key: value" line starts an entry and deeper "key: value" lines continue it.
std::vector<StreamConfig> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  std::vector<StreamConfig> streams;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string text = strip_yaml_comment(raw_line);
    std::string line = sima_examples::trim_copy(text);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const int indent = static_cast<int>(text.find_first_not_of(" \t"));
    if (streams_indent < 0) {
      if (line == "streams:") {
        streams_indent = indent;
      }
      continue;
    }
    const bool entry = line == "-" || line.rfind("- ", 0) == 0;
    if (indent <= streams_indent && !entry) {
      break;
    }
    if (entry) {
      streams.emplace_back();
      line = line == "-" ? "" : sima_examples::trim_copy(line.substr(2));
      if (line.empty()) {
        continue;
      }
    }
    const std::size_t colon = line.find(':');
    if (streams.empty() || colon == std::string::npos) {
      throw std::runtime_error("streams must be a list of 'key: value' mappings");
    }
    const std::string key = sima_examples::trim_copy(line.substr(0, colon));
    const std::string raw_value = sima_examples::trim_copy(line.substr(colon + 1));
    const bool quoted = raw_value.size() >= 2 &&
                        (raw_value.front() == '"' || raw_value.front() == '\'') &&
                        raw_value.back() == raw_value.front();
    if (quoted && (key == "insight_channel" || key == "width" || key == "height" || key == "fps")) {
      throw std::runtime_error("stream " + key + " must be an integer");
    }
    if (!quoted && (key == "id" || key == "url")) {
      if (is_plain_yaml_null(raw_value)) {
        apply_stream_field(streams.back(), key, "");
        continue;
      }
      if (!is_plain_yaml_string(raw_value)) {
        throw std::runtime_error("stream " + key + " must be a string");
      }
    }
    if (key != "codec" || !is_plain_yaml_null(raw_value)) {
      apply_stream_field(streams.back(), key, decode_yaml_scalar(raw_value));
    }
  }
  return streams;
}

std::unordered_map<std::string, std::string> load_raw_scalars(const fs::path& config_path) {
  std::unordered_map<std::string, std::string> scalars;
  std::ifstream input(config_path);
  std::vector<std::pair<int, std::string>> stack;
  int list_block_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string text = strip_yaml_comment(raw_line);
    const std::string line = sima_examples::trim_copy(text);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::string content = line;
    if (content == "-" || content.rfind("- ", 0) == 0) {
      content = sima_examples::trim_copy(content.substr(1));
    }
    if (!content.empty() && (content.front() == '\'' || content.front() == '"')) {
      throw std::runtime_error("quoted YAML mapping keys are not supported; use plain keys: " +
                               line);
    }
    const std::size_t flow_colon = content.find(':');
    const std::string flow_value = flow_colon == std::string::npos
                                       ? content
                                       : sima_examples::trim_copy(content.substr(flow_colon + 1));
    if (is_yaml_block_scalar_header(flow_value)) {
      throw std::runtime_error("YAML block scalar values are not supported; use a quoted string: " +
                               line);
    }
    for (const std::string& part : {content, flow_value}) {
      if (!part.empty() && (part.front() == '[' || part.front() == '{') && part != "{}" &&
          part != "[]") {
        throw std::runtime_error(
            "flow-style YAML collections are not supported; use block style: " + line);
      }
    }
    const int indent = static_cast<int>(text.find_first_not_of(" \t"));
    if (list_block_indent >= 0) {
      if (indent > list_block_indent) {
        continue;
      }
      list_block_indent = -1;
    }
    if (line == "-" || line.rfind("- ", 0) == 0) {
      list_block_indent = indent;
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    while (!stack.empty() && indent <= stack.back().first) {
      stack.pop_back();
    }
    const std::string key = sima_examples::trim_copy(line.substr(0, colon));
    std::string full_key;
    for (const auto& [parent_indent, parent_key] : stack) {
      static_cast<void>(parent_indent);
      full_key += (full_key.empty() ? "" : ".") + parent_key;
    }
    full_key += (full_key.empty() ? "" : ".") + key;
    const std::string value = sima_examples::trim_copy(line.substr(colon + 1));
    scalars[full_key] = value;
    if (value.empty() || value == "{}") {
      stack.emplace_back(indent, key);
    }
  }
  return scalars;
}

bool is_yaml_null(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
      value.back() == value.front()) {
    value = sima_examples::trim_copy(value.substr(1, value.size() - 2));
  }
  if (value == "~") {
    return true;
  }
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "null";
}

// ScalarConfig removes scalar quotes and maps explicit `null` to a missing
// optional value. Preserve Python's typed-field behavior by rejecting those
// representations before ScalarConfig applies conversions or defaults.
void reject_invalid_typed_fields(const std::unordered_map<std::string, std::string>& raw_scalars) {
  static const std::unordered_map<std::string, std::string> errors = {
      {"input.tcp", " must be true or false"},
      {"input.latency_ms", " must be an integer"},
      {"detector.min_score", " must be numeric"},
      {"detector.nms_iou", " must be numeric"},
      {"pose.max_people_per_frame", " must be an integer"},
      {"pose.roi_scale", " must be numeric"},
      {"pose.presence_threshold", " must be numeric"},
      {"pose.temporal_filter_enabled", " must be true or false"},
      {"runtime.frames", " must be an integer"},
      {"output.insight.video_port_base", " must be an integer"},
      {"output.insight.metadata_port_base", " must be an integer"},
  };
  for (const auto& [full_key, value] : raw_scalars) {
    const auto error = errors.find(full_key);
    const bool quoted = value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
                        value.back() == value.front();
    if (error != errors.end() && (value.empty() || value == "{}" || quoted ||
                                  value.find('#') != std::string::npos || is_yaml_null(value))) {
      throw std::runtime_error(full_key + error->second);
    }
  }
}

std::string decoded_string_or(const std::unordered_map<std::string, std::string>& raw_scalars,
                              const std::string& key, const std::string& default_value) {
  const auto value = raw_scalars.find(key);
  if (value == raw_scalars.end() || is_plain_yaml_null(value->second)) {
    return default_value;
  }
  const bool quoted = value->second.size() >= 2 &&
                      (value->second.front() == '"' || value->second.front() == '\'') &&
                      value->second.back() == value->second.front();
  if (!quoted && !is_plain_yaml_string(value->second)) {
    throw std::runtime_error(key + " must be a string");
  }
  return decode_yaml_scalar(value->second);
}

int yaml_int_or(const std::unordered_map<std::string, std::string>& raw_scalars,
                const std::string& key, int default_value) {
  const auto value = raw_scalars.find(key);
  return value == raw_scalars.end() ? default_value : parse_yaml_integer(value->second, key);
}

bool app_bool_or(const sima_examples::ScalarConfig& config,
                 const std::unordered_map<std::string, std::string>& raw_scalars,
                 const std::string& key, bool default_value) {
  const auto raw = raw_scalars.find(key);
  if (raw == raw_scalars.end()) {
    return default_value;
  }
  std::string value = sima_examples::trim_copy(raw->second);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "yes" || value == "on") {
    return true;
  }
  if (value == "no" || value == "off") {
    return false;
  }
  return config.bool_or(key, default_value);
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.detector_model_path.empty(), "models.detector_path must be set");
  sima_examples::require(!cfg.pose_model_path.empty(), "models.pose_path must be set");
  sima_examples::require(!cfg.streams.empty() && cfg.streams.size() <= 4,
                         "streams must contain between 1 and 4 entries");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.detector_min_score >= 0.0 && cfg.detector_min_score <= 1.0,
                         "detector.min_score must be between 0 and 1");
  sima_examples::require(cfg.detector_nms_iou >= 0.0 && cfg.detector_nms_iou <= 1.0,
                         "detector.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_people_per_frame > 0 && cfg.max_people_per_frame <= 10,
                         "pose.max_people_per_frame must be between 1 and 10");
  sima_examples::require(std::isfinite(cfg.roi_scale) && cfg.roi_scale > 0.0,
                         "pose.roi_scale must be finite and > 0");
  sima_examples::require(cfg.pose_presence_threshold >= 0.0 && cfg.pose_presence_threshold <= 1.0,
                         "pose.presence_threshold must be between 0 and 1");
  sima_examples::require(cfg.frame_limit >= 0, "runtime.frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0 && cfg.video_port_base <= 65535 &&
                             cfg.metadata_port_base > 0 && cfg.metadata_port_base <= 65535,
                         "Insight port bases must be between 1 and 65535");

  std::set<std::string> ids;
  std::set<int> video_ports;
  std::set<int> metadata_ports;
  for (const StreamConfig& stream : cfg.streams) {
    sima_examples::require(!stream.id.empty(), "stream id must be set");
    sima_examples::require(!stream.url.empty(), "stream url must be set");
    sima_examples::require(stream.insight_channel >= 0, "stream insight_channel must be >= 0");
    sima_examples::require(stream.width > 0 && stream.height > 0 && stream.fps > 0,
                           "stream width, height, and fps must all be > 0");
    sima_examples::require(stream.insight_channel <= 65535 - cfg.video_port_base,
                           "stream video port must be <= 65535");
    sima_examples::require(stream.insight_channel <= 65535 - cfg.metadata_port_base,
                           "stream metadata port must be <= 65535");
    sima_examples::require(ids.insert(stream.id).second, "stream ids must be unique");
    sima_examples::require(video_ports.insert(cfg.video_port_base + stream.insight_channel).second,
                           "stream insight channels must be unique");
    metadata_ports.insert(cfg.metadata_port_base + stream.insight_channel);
  }
  std::vector<int> overlapping_ports;
  std::set_intersection(video_ports.begin(), video_ports.end(), metadata_ports.begin(),
                        metadata_ports.end(), std::back_inserter(overlapping_ports));
  sima_examples::require(overlapping_ports.empty(),
                         "Insight video and metadata ports must not overlap");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw_scalars = load_raw_scalars(config_path);
  reject_invalid_typed_fields(raw_scalars);
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  AppConfig cfg;
  cfg.detector_model_path = decoded_string_or(raw_scalars, "models.detector_path", "");
  cfg.pose_model_path = decoded_string_or(raw_scalars, "models.pose_path", "");
  cfg.streams = parse_streams(config_path);
  cfg.tcp = app_bool_or(raw, raw_scalars, "input.tcp", true);
  cfg.latency_ms = yaml_int_or(raw_scalars, "input.latency_ms", 100);
  cfg.detector_min_score = raw.double_or("detector.min_score", 0.30);
  cfg.detector_nms_iou = raw.double_or("detector.nms_iou", 0.60);
  cfg.max_people_per_frame = yaml_int_or(raw_scalars, "pose.max_people_per_frame", 4);
  cfg.roi_scale = raw.double_or("pose.roi_scale", 1.65);
  cfg.pose_presence_threshold = raw.double_or("pose.presence_threshold", 0.50);
  cfg.pose_temporal_filter_enabled =
      app_bool_or(raw, raw_scalars, "pose.temporal_filter_enabled", true);
  cfg.frame_limit = yaml_int_or(raw_scalars, "runtime.frames", 0);
  cfg.insight_host = decoded_string_or(raw_scalars, "output.insight.host", "");
  cfg.video_port_base = yaml_int_or(raw_scalars, "output.insight.video_port_base", 9000);
  cfg.metadata_port_base = yaml_int_or(raw_scalars, "output.insight.metadata_port_base", 9100);
  validate_config(cfg);
  return cfg;
}

fs::path parse_args(int argc, char** argv) {
  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--config") {
      if (index + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      config_path = argv[++index];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return config_path;
}

neat::InputOptions encoded_input_options(neat::nodes::groups::RtspCodec codec,
                                         neat::InputMemoryPolicy memory) {
  neat::InputOptions options;
  options.payload_type = neat::PayloadType::Encoded;
  options.format =
      codec == neat::nodes::groups::RtspCodec::H265 ? neat::FormatTag::H265 : neat::FormatTag::H264;
  options.memory_policy = memory;
  return options;
}

neat::nodes::groups::RtspDecodedInputOptions make_source_options(const AppConfig& cfg,
                                                                 StreamRuntime& runtime) {
  runtime.width = runtime.config.width;
  runtime.height = runtime.config.height;
  runtime.fps = runtime.config.fps;
  neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = runtime.config.url;
  options.codec = runtime.config.codec;
  options.latency_ms = cfg.latency_ms;
  options.tcp = cfg.tcp;
  options.payload_type = 96;
  // The configured integer FPS is only the decoder's rate hint. Pinning it into
  // caps would reject NTSC-rate cameras: a 29.97 fps
  // stream negotiates 30000/1001, which a 30/1 caps filter cannot accept.
  options.dec_fps = runtime.fps;
  options.insert_queue = true;
  options.decoder_name = "decoder_" + runtime.config.id;
  options.decoder_raw_output = true;
  options.auto_caps_from_stream = true;
  options.dec_width = runtime.width;
  options.dec_height = runtime.height;
  if (runtime.config.codec == neat::nodes::groups::RtspCodec::H264) {
    options.fallback_h264_width = runtime.width;
    options.fallback_h264_height = runtime.height;
    options.fallback_h264_fps = runtime.fps;
  }
  return options;
}

neat::Graph make_encoded_source(const neat::nodes::groups::RtspDecodedInputOptions& options) {
  neat::nodes::groups::RtspEncodedInputOptions encoded;
  encoded.url = options.url;
  encoded.codec = options.codec;
  encoded.latency_ms = options.latency_ms;
  encoded.tcp = options.tcp;
  encoded.source_fps = options.source_fps;
  encoded.payload_type = options.payload_type;
  encoded.insert_queue = options.insert_queue;
  encoded.auto_caps_from_stream = options.auto_caps_from_stream;
  encoded.fallback_h264_width = options.fallback_h264_width;
  encoded.fallback_h264_height = options.fallback_h264_height;
  encoded.fallback_h264_fps = options.fallback_h264_fps;
  return neat::nodes::groups::RtspEncodedInput(encoded);
}

neat::Graph make_decoder(const neat::nodes::groups::RtspDecodedInputOptions& options) {
  neat::SimaDecodeOptions decode;
  decode.type = options.codec == neat::nodes::groups::RtspCodec::H265 ? neat::SimaDecodeType::H265
                                                                      : neat::SimaDecodeType::H264;
  decode.sima_allocator_type = options.sima_allocator_type;
  decode.out_format = "NV12";
  decode.decoder_name = options.decoder_name;
  decode.raw_output = options.decoder_raw_output;
  decode.next_element = options.decoder_next_element;
  decode.dec_width = options.dec_width;
  decode.dec_height = options.dec_height;
  decode.dec_fps = options.dec_fps;
  decode.num_buffers = options.num_buffers;
  decode.input_buffers = options.decoder_input_buffers;
  decode.decoder_tuning = options.decoder_tuning;
  decode.memory_opt = options.decoder_memory_opt;

  neat::Graph graph("decoder_" + options.decoder_name);
  graph.connect(neat::nodes::Input(
                    "encoded", encoded_input_options(options.codec, neat::InputMemoryPolicy::Ev74)),
                neat::nodes::SimaDecode(decode));
  graph.add(neat::nodes::CapsRaw("NV12", options.dec_width, options.dec_height,
                                 options.output_caps.fps, neat::CapsMemory::Any));
  graph.add(neat::nodes::Output("analytics_frame"));
  return graph;
}

neat::Graph make_video_sender(const AppConfig& cfg, const StreamRuntime& stream) {
  auto options = neat::nodes::groups::VideoSenderOptions::Passthrough(stream.config.codec);
  options.host = cfg.insight_host;
  options.channel = stream.config.insight_channel;
  options.video_port_base = cfg.video_port_base;
  options.async = false;
  neat::Graph graph("video_" + stream.config.id);
  graph.connect(
      neat::nodes::Input("encoded", encoded_input_options(stream.config.codec,
                                                          neat::InputMemoryPolicy::SystemMemory)),
      neat::nodes::groups::VideoSender(options));
  return graph;
}

std::unique_ptr<neat::Model> make_detector_model(const AppConfig& cfg, int max_width,
                                                 int max_height) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.input_max_width = max_width;
  options.preprocess.input_max_height = max_height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = cfg.detector_min_score;
  options.nms_iou_threshold = cfg.detector_nms_iou;
  options.top_k = kMaxDetections;
  return std::make_unique<neat::Model>(cfg.detector_model_path, options);
}

std::unique_ptr<neat::Model> make_pose_model(const AppConfig& cfg) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.width = 256;
  options.preprocess.resize.height = 256;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  return std::make_unique<neat::Model>(cfg.pose_model_path, options);
}

void validate_detector_contract(const neat::Model& model) {
  const auto inputs = model.input_specs();
  const auto outputs = model.output_specs();
  sima_examples::require(inputs.size() == 1, "YOLO26 must have one input");
  sima_examples::require(inputs[0].shape == std::vector<int64_t>({-1, -1, 3}),
                         "YOLO26 public input must be dynamic HWC RGB");
  sima_examples::require(outputs.size() == 1, "YOLO26 must expose one decoded BBOX output");
}

void validate_pose_contract(const neat::Model& model) {
  const auto inputs = model.input_specs();
  const auto outputs = model.output_specs();
  sima_examples::require(inputs.size() == 1, "BlazePose must have one input");
  sima_examples::require(inputs[0].shape == std::vector<int64_t>({-1, -1, 3}),
                         "BlazePose public input must be dynamic HWC RGB");
  sima_examples::require(outputs.size() == 3, "BlazePose must have three outputs");
  sima_examples::require(outputs[0].shape == std::vector<int64_t>({1, 195}),
                         "BlazePose output 0 must be [1,195]");
  sima_examples::require(outputs[1].shape == std::vector<int64_t>({1, 1}),
                         "BlazePose output 1 must be [1,1]");
  sima_examples::require(outputs[2].shape == std::vector<int64_t>({1, 117}),
                         "BlazePose output 2 must be [1,117]");
}

void copy_identity(const FrameIdentity& identity, neat::Sample& sample) {
  sample.stream_id = identity.stream_id;
  sample.frame_id = identity.frame_id;
  sample.pts_ns = identity.pts_ns;
}

neat::RunOptions reliable_model_run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Reliable;
  options.overflow_policy = neat::OverflowPolicy::Block;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  options.input_timeout_ms = 30000;
  options.startup_preflight = false;
  return options;
}

neat::Sample pose_input_sample(const neat::Tensor& tensor, const FrameIdentity* identity) {
  neat::Sample sample = neat::make_tensor_sample("pose_input", tensor);
  sample.payload_type = neat::PayloadType::Tensor;
  sample.media_type = "application/vnd.simaai.tensor";
  if (tensor.semantic.tess.has_value()) {
    sample.format = tensor.semantic.tess->format;
    sample.payload_tag = sample.format;
  }
  if (identity != nullptr) {
    copy_identity(*identity, sample);
  }
  return sample;
}

void build_pose_run(AppRuntime& app) {
  validate_pose_contract(*app.pose_model);
  cv::Mat seed_image = cv::Mat::zeros(256, 256, CV_8UC3);
  const std::vector<neat::PreprocessRoi> seed_rois = {{0, 0, 0, 256, 256}};
  const neat::TensorList seed_tensors =
      neat::stages::Preproc(std::vector<cv::Mat>{seed_image}, *app.pose_model, seed_rois);
  sima_examples::require(seed_tensors.size() == 1, "BlazePose seed Preproc returned no tensor");
  const neat::Sample seed = pose_input_sample(seed_tensors.front(), nullptr);

  app.pose_graph = neat::Graph("blazepose_runner");
  app.pose_graph.add(neat::nodes::Input("pose_input"));
  app.pose_graph.add(app.pose_model->inference());
  app.pose_graph.add(app.pose_model->postprocess());
  app.pose_graph.add(neat::nodes::Output("pose_output", neat::OutputOptions::EveryFrame(4)));
  app.pose_run = app.pose_graph.build(seed, reliable_model_run_options());
}

neat::GraphLinkOptions realtime_link(const StreamRuntime& stream) {
  neat::GraphLinkOptions options;
  options.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  options.stream_id = stream.config.id;
  options.max_inflight_per_stream = kMaxInflightPerStream;
  options.max_inflight_total = kMaxInflightPerStream;
  return options;
}

std::string frame_output_name(int stream_index) {
  return "frame_" + std::to_string(stream_index);
}

neat::Graph make_rgb_output(const StreamRuntime& stream) {
  neat::Graph graph("rgb_" + std::to_string(stream.index));
  graph.add(neat::nodes::Input("analytics_frame"));
  graph.add(neat::nodes::VideoConvert());
  graph.add(neat::nodes::CapsRaw("RGB", stream.width, stream.height));
  graph.add(neat::nodes::Output(frame_output_name(stream.index), neat::OutputOptions::Latest()));
  return graph;
}

void initialize_streams(AppRuntime& app, const AppConfig& cfg) {
  int max_width = 0;
  int max_height = 0;
  for (std::size_t index = 0; index < cfg.streams.size(); ++index) {
    auto stream = std::make_unique<StreamRuntime>();
    stream->index = static_cast<int>(index);
    stream->config = cfg.streams[index];
    stream->source_options = make_source_options(cfg, *stream);
    stream->pose_temporal_filter_enabled = cfg.pose_temporal_filter_enabled;
    max_width = std::max(max_width, stream->width);
    max_height = std::max(max_height, stream->height);

    neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = stream->config.insight_channel;
    metadata_options.metadata_port_base = cfg.metadata_port_base;
    neat::MetadataSenderSendOptions send_options;
    send_options.nonblocking = true;
    std::string error;
    stream->metadata_sender =
        std::make_unique<neat::MetadataSender>(metadata_options, send_options, &error);
    sima_examples::require(stream->metadata_sender->ok(), error);
    app.streams.push_back(std::move(stream));
  }
  app.detector_model = make_detector_model(cfg, max_width, max_height);
}

void prepare_source_graphs(AppRuntime& app, const AppConfig& cfg) {
  for (const auto& stream_ptr : app.streams) {
    StreamRuntime& stream = *stream_ptr;
    stream.source_graph = neat::Graph("blazepose3d_source_" + std::to_string(stream.index));
    const neat::Graph source = make_encoded_source(stream.source_options);
    const neat::Graph decoder = make_decoder(stream.source_options);
    stream.source_graph.connect(source, decoder);
    stream.source_graph.connect(decoder, make_rgb_output(stream), realtime_link(stream));
    stream.source_graph.connect(source, make_video_sender(cfg, stream), realtime_link(stream));
    std::cout << "[stream " << stream.config.id << "] codec=" << codec_name(stream.config.codec)
              << " source=" << stream.width << "x" << stream.height << "@" << stream.fps
              << " channel=" << stream.config.insight_channel
              << " video=" << cfg.video_port_base + stream.config.insight_channel
              << " metadata=" << stream.metadata_sender->metadata_port() << "\n";
  }
}

neat::Sample image_input_sample(const std::string& name, const neat::Tensor& tensor,
                                const FrameIdentity* identity) {
  neat::Sample sample = neat::make_tensor_sample(name, tensor);
  sample.payload_type = neat::PayloadType::Image;
  sample.media_type = "video/x-raw";
  sample.format = "RGB";
  sample.payload_tag = sample.format;
  if (identity != nullptr) {
    copy_identity(*identity, sample);
  }
  return sample;
}

void build_detector_run(AppRuntime& app) {
  validate_detector_contract(*app.detector_model);
  int seed_width = 0;
  int seed_height = 0;
  for (const auto& stream : app.streams) {
    seed_width = std::max(seed_width, stream->width);
    seed_height = std::max(seed_height, stream->height);
  }
  cv::Mat seed_image = cv::Mat::zeros(seed_height, seed_width, CV_8UC3);
  const neat::Tensor seed_tensor = neat::Tensor::from_cv_mat(
      seed_image, neat::ImageSpec::PixelFormat::RGB, neat::TensorMemory::EV74);
  const neat::Sample seed = image_input_sample("detector_input", seed_tensor, nullptr);

  app.detector_graph = neat::Graph("yolo26_runner");
  auto input_options = app.detector_model->input_appsrc_options(false);
  input_options.block = true;
  neat::Graph input_graph;
  input_graph.add(neat::nodes::Input("detector_input", input_options));
  const neat::Graph model_graph = app.detector_model->graph();
  neat::Graph output_graph;
  output_graph.add(neat::nodes::Output("detector_output", neat::OutputOptions::EveryFrame(4)));
  app.detector_graph.connect(input_graph, model_graph);
  app.detector_graph.connect(model_graph, output_graph);
  app.detector_run = app.detector_graph.build(seed, reliable_model_run_options());
}

bool extract_bbox_payload(const neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& error) {
  if (sample.kind == neat::SampleKind::Bundle) {
    for (const neat::Sample& field : sample.fields) {
      if (extract_bbox_payload(field, payload, error)) {
        return true;
      }
    }
    error = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    neat::Sample tensor_sample = sample;
    tensor_sample.kind = neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, error);
  }
  return objdet::extract_bbox_payload(sample, payload, error);
}

neat::Tensor require_rgb_tensor(const neat::Sample& sample) {
  const neat::TensorList tensors = neat::tensors_from_sample(sample, false);
  if (tensors.size() != 1) {
    throw std::runtime_error("RGB frame output must contain one tensor");
  }
  const auto format = tensors.front().image_format();
  if (!format.has_value() || *format != neat::ImageSpec::PixelFormat::RGB) {
    throw std::runtime_error("VideoConvert output is not RGB");
  }
  return tensors.front();
}

std::vector<blazepose_app::Box> select_people(const neat::Sample& detections, int width, int height,
                                              const AppConfig& cfg) {
  std::vector<std::uint8_t> payload;
  std::string error;
  if (!extract_bbox_payload(detections, payload, error)) {
    throw std::runtime_error("failed to read detector BBOX output: " + error);
  }
  const std::vector<objdet::Box> boxes =
      objdet::parse_boxes_strict(payload, width, height, kMaxDetections, false);
  std::vector<blazepose_app::Box> people;
  for (const objdet::Box& box : boxes) {
    const blazepose_app::Box person{box.x1, box.y1, box.x2, box.y2, box.score, box.class_id};
    if (blazepose_app::is_finite_person_box(person)) {
      people.push_back(person);
    }
  }
  std::sort(people.begin(), people.end(),
            [](const auto& left, const auto& right) { return left.score > right.score; });
  if (people.size() > static_cast<std::size_t>(cfg.max_people_per_frame)) {
    people.resize(static_cast<std::size_t>(cfg.max_people_per_frame));
  }
  return people;
}

// Marks one admitted frame as finished: published or dropped for newer work.
void finish_frame(AppRuntime& app, StreamRuntime& stream) {
  std::lock_guard<std::mutex> lock(app.state.mutex);
  --stream.outstanding_frames;
  app.state.cv.notify_all();
}

// Publishes one frame's 2D and 3D metadata. The per-stream lock keeps the two
// messages of one frame from interleaving with another frame's.
void publish_frame(StreamRuntime& stream, const FrameIdentity& identity,
                   std::vector<blazepose_app::Pose> poses) {
  std::lock_guard<std::mutex> lock(stream.metadata_mutex);
  if (!blazepose_app::claim_newer_frame(identity.sequence, stream.last_published_sequence)) {
    return;
  }
  if (stream.pose_temporal_filter_enabled) {
    poses = stream.pose_smoother.filter(std::move(poses));
  }
  nlohmann::json overlay = blazepose_app::poses_data_json(std::move(poses), identity.stream_id);
  const std::string overlay_data = overlay.dump();
  const std::string auxiliary_data =
      blazepose_app::world_pose_auxiliary_from_overlay(std::move(overlay)).dump();
  const int64_t timestamp_ms = identity.pts_ns >= 0 ? identity.pts_ns / 1'000'000 : -1;
  const std::string frame_id = std::to_string(identity.frame_id);
  const auto send = [&](const char* type) {
    const std::string& data =
        std::string_view(type) == "pose-estimation" ? overlay_data : auxiliary_data;
    std::string error;
    if (stream.metadata_sender->send_metadata(type, data, timestamp_ms, frame_id, &error)) {
      return true;
    }
    std::cerr << "[warn] stream " << stream.config.id << " " << type
              << " metadata send failed: " << error << "\n";
    return false;
  };
  if (blazepose_app::send_metadata_pair(send)) {
    ++stream.frames_out;
  }
}

std::vector<float> tensor_floats(const neat::Tensor& tensor, std::size_t expected) {
  const std::vector<std::uint8_t> bytes = tensor.copy_payload_bytes();
  if (bytes.size() != expected * sizeof(float)) {
    throw std::runtime_error("unexpected BlazePose output byte count");
  }
  std::vector<float> values(expected);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::optional<blazepose_app::Pose> parse_pose_output(const neat::Sample& sample,
                                                     const PoseInputContext& context,
                                                     const AppConfig& cfg) {
  const neat::TensorList tensors = neat::tensors_from_sample(sample, false);
  if (tensors.size() != 3) {
    throw std::runtime_error("BlazePose output must contain three tensors");
  }
  const std::vector<float> presence = tensor_floats(tensors[1], 1);
  if (!std::isfinite(presence[0])) {
    return std::nullopt;
  }
  const float presence_probability = blazepose_app::sigmoid(presence[0]);
  if (presence_probability < cfg.pose_presence_threshold) {
    return std::nullopt;
  }
  const std::vector<float> landmarks = tensor_floats(tensors[0], 195);
  const std::vector<float> world_landmarks = tensor_floats(tensors[2], 117);
  return blazepose_app::decode_finite_pose(landmarks, world_landmarks, context.affine, context.box,
                                           presence_probability, context.roi_index);
}

void record_error(AppRuntime& app) {
  std::lock_guard<std::mutex> lock(app.state.mutex);
  if (!app.state.error) {
    app.state.error = std::current_exception();
  }
  app.state.stopping = true;
  app.state.cv.notify_all();
}

// Waits for queued work and takes it round-robin across streams.
std::optional<FrameJob> take_next_job(AppRuntime& app,
                                      std::vector<std::optional<FrameJob>>& mailboxes,
                                      std::size_t& next_stream) {
  std::unique_lock<std::mutex> lock(app.state.mutex);
  app.state.cv.wait(lock, [&]() {
    return app.state.stopping || std::any_of(mailboxes.begin(), mailboxes.end(),
                                             [](const auto& item) { return item.has_value(); });
  });
  if (app.state.stopping) {
    return std::nullopt;
  }
  for (std::size_t offset = 0; offset < mailboxes.size(); ++offset) {
    const std::size_t index = (next_stream + offset) % mailboxes.size();
    if (mailboxes[index].has_value()) {
      FrameJob job = std::move(*mailboxes[index]);
      mailboxes[index].reset();
      next_stream = (index + 1) % mailboxes.size();
      return job;
    }
  }
  return std::nullopt;
}

blazepose_app::Affine affine_from_tensor(const neat::Tensor& tensor) {
  if (!tensor.semantic.preprocess.has_value()) {
    throw std::runtime_error("BlazePose Preproc output is missing affine metadata");
  }
  const auto& meta = *tensor.semantic.preprocess;
  return {meta.affine_m00, meta.affine_m01, meta.affine_m02,
          meta.affine_m10, meta.affine_m11, meta.affine_m12};
}

void close_source_stream(AppRuntime& app, StreamRuntime& stream, const std::string& reason) {
  if (!stream.closed.exchange(true)) {
    std::cerr << "[warn] stream " << stream.config.id << " stopped: " << reason << "\n";
  }
  std::lock_guard<std::mutex> lock(app.state.mutex);
  app.state.cv.notify_all();
}

void pull_source_frames(AppRuntime& app, const AppConfig& cfg, StreamRuntime& stream) {
  const std::string output = frame_output_name(stream.index);
  while (true) {
    {
      std::lock_guard<std::mutex> lock(app.state.mutex);
      if (app.state.stopping) {
        return;
      }
    }
    if (cfg.frame_limit > 0 && stream.frames_in.load() >= cfg.frame_limit) {
      close_source_stream(app, stream, "runtime frame limit reached");
      return;
    }
    try {
      neat::Sample sample;
      neat::PullError error;
      const auto status = stream.source_run.pull(output, 50, sample, &error);
      if (status == neat::PullStatus::Timeout) {
        continue;
      }
      if (status == neat::PullStatus::Closed) {
        close_source_stream(app, stream, "source reached end of stream");
        return;
      }
      if (status != neat::PullStatus::Ok) {
        close_source_stream(app, stream, "failed to pull RGB source frame: " + error.message);
        return;
      }
      FrameJob job;
      job.job_id = app.next_job_id.fetch_add(1);
      job.stream_index = stream.index;
      job.rgb = require_rgb_tensor(sample);
      job.identity = {stream.config.id, sample.frame_id, sample.pts_ns,
                      static_cast<std::uint64_t>(stream.frames_in.load() + 1)};
      bool dropped = false;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping) {
          return;
        }
        ++stream.frames_in;
        dropped = blazepose_app::keep_latest(
            app.state.detector_mailboxes[static_cast<std::size_t>(stream.index)], std::move(job));
        if (!dropped) {
          ++stream.outstanding_frames;
        }
        app.state.cv.notify_all();
      }
    } catch (const std::exception& error) {
      close_source_stream(app, stream, error.what());
      return;
    }
  }
}

void run_source_stream(AppRuntime& app, const AppConfig& cfg, StreamRuntime& stream) {
  struct Completion {
    AppRuntime& app;
    StreamRuntime& stream;
    ~Completion() {
      stream.source_worker_finished = true;
      std::lock_guard<std::mutex> lock(app.state.mutex);
      app.state.cv.notify_all();
    }
  } completion{app, stream};
  try {
    neat::RunOptions options;
    options.preset = neat::RunPreset::Realtime;
    options.output_memory = neat::OutputMemory::ZeroCopy;
    neat::Run source_run = stream.source_graph.build(options);
    bool stopping = false;
    {
      std::lock_guard<std::mutex> lock(app.state.mutex);
      stopping = app.state.stopping;
      if (stopping) {
        stream.closed = true;
      } else {
        stream.source_run = std::move(source_run);
      }
    }
    if (stopping) {
      source_run.close();
      return;
    }
    pull_source_frames(app, cfg, stream);
  } catch (const std::exception& error) {
    close_source_stream(app, stream, error.what());
  }
}

// Pushes one model input without the blocking push API. Its FIFO context is
// queued first so the output puller can always correlate the result. Returns
// false when the application is stopping.
template <typename Context>
bool push_with_context(AppRuntime& app, neat::Run& run, std::string_view input_name,
                       const neat::Sample& input, std::deque<Context>& pending,
                       const Context& context, const std::string& rejection_message) {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(app.state.mutex);
      if (app.state.stopping) {
        return false;
      }
      pending.push_back(context);
    }
    if (run.try_push(input_name, input)) {
      return true;
    }
    std::unique_lock<std::mutex> lock(app.state.mutex);
    if (app.state.stopping) {
      return false;
    }
    pending.pop_back();
    if (!run.can_push()) {
      throw std::runtime_error(rejection_message);
    }
    app.state.cv.wait_for(lock, std::chrono::milliseconds(1), [&]() { return app.state.stopping; });
  }
}

void dispatch_detector_jobs(AppRuntime& app, const AppConfig& /*cfg*/) {
  try {
    while (true) {
      const std::optional<FrameJob> job =
          take_next_job(app, app.state.detector_mailboxes, app.state.next_detector_stream);
      if (!job.has_value()) {
        return;
      }
      const neat::Tensor detector_frame = job->rgb.cvu();
      const neat::Sample input =
          image_input_sample("detector_input", detector_frame, &job->identity);
      if (!push_with_context(app, app.detector_run, "detector_input", input,
                             app.state.pending_detector_outputs, *job,
                             "YOLO26 Run rejected a frame input")) {
        return;
      }
    }
  } catch (...) {
    record_error(app);
  }
}

// Pulls one output of a shared model. Returns false when the application is
// stopping, and throws when the Run closed or failed on its own, or stalled.
template <typename Pending>
bool pull_model_output(AppRuntime& app, neat::Run& run, const char* output, const char* model,
                       const Pending& pending, neat::Sample& sample) {
  std::optional<Clock::time_point> waiting_since;
  while (true) {
    neat::PullError error;
    const auto status = run.pull(output, 20, sample, &error);
    if (status == neat::PullStatus::Ok) {
      return true;
    }
    bool input_pending = false;
    {
      std::lock_guard<std::mutex> lock(app.state.mutex);
      if (app.state.stopping) {
        return false;
      }
      input_pending = !pending.empty();
    }
    if (status == neat::PullStatus::Closed) {
      const std::string detail = run.last_error();
      throw std::runtime_error(std::string(model) + " output closed unexpectedly" +
                               (detail.empty() ? std::string{} : ": " + detail));
    }
    if (status != neat::PullStatus::Timeout) {
      throw std::runtime_error(std::string("failed to pull ") + model +
                               " output: " + error.message);
    }
    if (blazepose_app::inference_stalled(waiting_since, input_pending, Clock::now())) {
      throw std::runtime_error(std::string(model) + " inference stalled: no output for " +
                               std::to_string(blazepose_app::kInferenceStallTimeout.count()) +
                               " s");
    }
  }
}

void pull_detector_outputs(AppRuntime& app, const AppConfig& cfg) {
  try {
    neat::Sample sample;
    while (pull_model_output(app, app.detector_run, "detector_output", "YOLO26",
                             app.state.pending_detector_outputs, sample)) {
      FrameJob job;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.pending_detector_outputs.empty()) {
          if (app.state.stopping) {
            return;
          }
          throw std::runtime_error("YOLO26 output arrived without pending frame context");
        }
        job = std::move(app.state.pending_detector_outputs.front());
        app.state.pending_detector_outputs.pop_front();
      }
      StreamRuntime& stream = *app.streams[static_cast<std::size_t>(job.stream_index)];
      job.people = select_people(sample, stream.width, stream.height, cfg);
      if (job.people.empty()) {
        // An empty pair clears the stream's previous poses in Insight.
        publish_frame(stream, job.identity, {});
        finish_frame(app, stream);
        continue;
      }
      bool dropped = false;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.stopping) {
          return;
        }
        dropped = blazepose_app::keep_latest(
            app.state.pose_mailboxes[static_cast<std::size_t>(job.stream_index)], std::move(job));
        app.state.cv.notify_all();
      }
      if (dropped) {
        finish_frame(app, stream);
      }
    }
  } catch (...) {
    record_error(app);
  }
}

void dispatch_pose_jobs(AppRuntime& app, const AppConfig& cfg) {
  try {
    while (true) {
      std::optional<FrameJob> maybe_job =
          take_next_job(app, app.state.pose_mailboxes, app.state.next_pose_stream);
      if (!maybe_job.has_value()) {
        return;
      }
      FrameJob job = std::move(*maybe_job);
      auto rgb_view = job.rgb.map_cv_mat_view(neat::ImageSpec::PixelFormat::RGB);
      if (!rgb_view.has_value()) {
        throw std::runtime_error("failed to map packed RGB frame without copying");
      }
      std::vector<neat::PreprocessRoi> pose_rois;
      pose_rois.reserve(job.people.size());
      for (const blazepose_app::Box& person : job.people) {
        const blazepose_app::Roi roi = blazepose_app::square_roi(person, cfg.roi_scale);
        pose_rois.push_back({0, roi.x, roi.y, roi.width, roi.height});
      }
      const neat::TensorList output =
          neat::stages::Preproc(std::vector<cv::Mat>{rgb_view->mat}, *app.pose_model, pose_rois);
      if (output.size() != pose_rois.size()) {
        throw std::runtime_error("BlazePose Preproc output count does not match ROI count");
      }
      std::vector<std::pair<PoseInputContext, neat::Tensor>> inputs;
      inputs.reserve(output.size());
      for (std::size_t index = 0; index < output.size(); ++index) {
        // Detached asynchronous Runs may retain their input after push(). Give
        // each ROI independent EV74 storage so Preproc can recycle its pool.
        inputs.emplace_back(PoseInputContext{job.job_id, static_cast<int>(index), job.people[index],
                                             affine_from_tensor(output[index])},
                            output[index].clone().cvu());
      }
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        PoseAggregate aggregate;
        aggregate.stream_index = job.stream_index;
        aggregate.expected = static_cast<int>(inputs.size());
        aggregate.identity = job.identity;
        app.state.aggregates.emplace(job.job_id, std::move(aggregate));
      }
      for (const auto& [context, tensor] : inputs) {
        if (!push_with_context(
                app, app.pose_run, "pose_input", pose_input_sample(tensor, &job.identity),
                app.state.pending_pose_outputs, context, "BlazePose Run rejected an ROI input")) {
          return;
        }
      }
    }
  } catch (...) {
    record_error(app);
  }
}

void pull_pose_outputs(AppRuntime& app, const AppConfig& cfg) {
  try {
    neat::Sample sample;
    while (pull_model_output(app, app.pose_run, "pose_output", "BlazePose",
                             app.state.pending_pose_outputs, sample)) {
      PoseInputContext context;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        if (app.state.pending_pose_outputs.empty()) {
          if (app.state.stopping) {
            return;
          }
          throw std::runtime_error("BlazePose output arrived without pending ROI context");
        }
        context = app.state.pending_pose_outputs.front();
        app.state.pending_pose_outputs.pop_front();
      }
      const auto pose = parse_pose_output(sample, context, cfg);
      std::optional<PoseAggregate> completed;
      {
        std::lock_guard<std::mutex> lock(app.state.mutex);
        const auto found = app.state.aggregates.find(context.job_id);
        if (found == app.state.aggregates.end()) {
          return; // stop_runtime() cleared the in-flight frames.
        }
        if (pose.has_value()) {
          found->second.poses.push_back(*pose);
        }
        if (++found->second.completed == found->second.expected) {
          completed = std::move(found->second);
          app.state.aggregates.erase(found);
        }
      }
      if (completed.has_value()) {
        StreamRuntime& stream = *app.streams[static_cast<std::size_t>(completed->stream_index)];
        publish_frame(stream, completed->identity, std::move(completed->poses));
        finish_frame(app, stream);
      }
    }
  } catch (...) {
    record_error(app);
  }
}

bool all_streams_done(const AppRuntime& app) {
  return std::all_of(app.streams.begin(), app.streams.end(), [](const auto& stream) {
    return stream->closed.load() && stream->outstanding_frames.load() == 0;
  });
}

void require_successful_completion(const AppRuntime& app, int frame_limit) {
  if (frame_limit == 0) {
    throw std::runtime_error("all source streams stopped");
  }
  std::string incomplete;
  for (const auto& stream : app.streams) {
    if (stream->frames_in.load() < frame_limit) {
      incomplete += (incomplete.empty() ? "" : ", ") + stream->config.id;
    }
  }
  if (!incomplete.empty()) {
    throw std::runtime_error("source streams stopped before reaching runtime.frames=" +
                             std::to_string(frame_limit) + ": " + incomplete);
  }
}

void stop_runtime(AppRuntime& app) {
  std::vector<neat::Run*> source_runs;
  {
    std::lock_guard<std::mutex> lock(app.state.mutex);
    app.state.stopping = true;
    for (auto& mailbox : app.state.detector_mailboxes) {
      mailbox.reset();
    }
    for (auto& mailbox : app.state.pose_mailboxes) {
      mailbox.reset();
    }
    app.state.pending_detector_outputs.clear();
    app.state.pending_pose_outputs.clear();
    app.state.aggregates.clear();
    for (auto& stream : app.streams) {
      if (stream->source_run) {
        source_runs.push_back(&stream->source_run);
      }
    }
    app.state.cv.notify_all();
  }
  for (neat::Run* source_run : source_runs) {
    source_run->close();
  }
  app.detector_run.close();
  app.pose_run.close();
}

void run_app(const AppConfig& cfg) {
  if (!fs::exists(cfg.detector_model_path)) {
    throw std::runtime_error("detector model not found: " + cfg.detector_model_path);
  }
  if (!fs::exists(cfg.pose_model_path)) {
    throw std::runtime_error("pose model not found: " + cfg.pose_model_path);
  }

  auto app = std::make_shared<AppRuntime>();
  app->streams.reserve(cfg.streams.size());
  app->state.detector_mailboxes.resize(cfg.streams.size());
  app->state.pose_mailboxes.resize(cfg.streams.size());
  initialize_streams(*app, cfg);
  prepare_source_graphs(*app, cfg);
  build_detector_run(*app);
  app->pose_model = make_pose_model(cfg);
  build_pose_run(*app);

  g_stop_requested = 0;
  auto previous_signal = std::signal(SIGINT, request_stop);
  // Each source run starts and pulls on its own thread, so an offline source
  // cannot delay the others; the shared models have dedicated workers.
  std::vector<std::thread> source_pullers;
  source_pullers.reserve(app->streams.size());
  for (const auto& stream : app->streams) {
    StreamRuntime* runtime = stream.get();
    source_pullers.emplace_back([app, cfg, runtime]() { run_source_stream(*app, cfg, *runtime); });
  }
  std::vector<std::thread> model_workers;
  for (auto* worker :
       {dispatch_detector_jobs, pull_detector_outputs, dispatch_pose_jobs, pull_pose_outputs}) {
    model_workers.emplace_back([app, &cfg, worker]() { worker(*app, cfg); });
  }

  try {
    std::unique_lock<std::mutex> lock(app->state.mutex);
    while (g_stop_requested == 0 && !app->state.error && !all_streams_done(*app)) {
      app->state.cv.wait_for(lock, std::chrono::milliseconds(50));
    }
    if (app->state.error) {
      std::rethrow_exception(app->state.error);
    }
    if (g_stop_requested == 0) {
      require_successful_completion(*app, cfg.frame_limit);
    }
  } catch (...) {
    record_error(*app);
  }

  stop_runtime(*app);
  const auto source_shutdown_deadline = Clock::now() + std::chrono::milliseconds(500);
  for (std::size_t index = 0; index < source_pullers.size(); ++index) {
    StreamRuntime& stream = *app->streams[index];
    {
      std::unique_lock<std::mutex> lock(app->state.mutex);
      app->state.cv.wait_until(lock, source_shutdown_deadline,
                               [&stream]() { return stream.source_worker_finished.load(); });
    }
    if (stream.source_worker_finished.load()) {
      source_pullers[index].join();
    } else {
      std::cerr << "[warn] stream " << stream.config.id
                << " is still starting; not waiting for its startup timeout during shutdown\n";
      source_pullers[index].detach();
    }
  }
  for (std::thread& worker : model_workers) {
    worker.join();
  }
  std::signal(SIGINT, previous_signal);
  for (const auto& stream : app->streams) {
    std::cout << "[summary stream=" << stream->config.id
              << "] frames_in=" << stream->frames_in.load()
              << " frames_out=" << stream->frames_out.load() << "\n";
  }
  if (app->state.error) {
    std::rethrow_exception(app->state.error);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path config_path = parse_args(argc, argv);
    if (!fs::exists(config_path)) {
      std::cerr << "Error: config file not found: " << config_path << "\n";
      return 2;
    }
    run_app(load_app_config(config_path));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}
