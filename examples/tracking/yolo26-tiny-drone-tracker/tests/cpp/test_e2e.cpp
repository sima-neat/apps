// End-to-end tests: RTSP streams reach Insight as video and tracking metadata, each on its own
// channel. tests/python/test_e2e.py runs the same checks in Python.
#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "yolo26-tiny-drone-tracker";
constexpr int kFrames = 150;
// Above the shipped 0.05, so the tracks show that the example applies inference.min_score.
constexpr double kMinScore = 0.1;
// Each stream's results per second, about 80% of this test in Vulcan CI run 45586507970 on
// 1f70fac (Modalix devkit-8, firmware 3.0.1_master_B2798, Core develop/0420557512d1, 25 fps
// sources, 4 frames in flight for all streams). With one or two streams, each stream gave 24.7 to
// 25.2, the source's rate. With four, each gave 12.7 to 12.9, about 51 in all, which is what the
// model gives.
constexpr double kMinResultsPerS = 20.0;
constexpr double kMinFourStreamResultsPerS = 10.0;

// Each video socket's receive buffer. A keyframe arrives as a burst of packets, and a full buffer
// drops the rest of the burst. In Vulcan CI run 45612521061 on 35bfc9c, 30 of 150 messages of the
// Python h264-four-streams case had no video frame, in runs of adjacent frames. Linux caps the size
// at net.core.rmem_max (212992 bytes by default), and getsockopt reports twice what it set, so the
// report prints what the socket got.
constexpr int kVideoReceiveBuffer = 8 << 20;

using Clock = std::chrono::steady_clock;

// RTP payload types: H.264 96, H.265 98.
int payload_type(const std::string& codec) {
  return codec == "h264" ? 96 : 98;
}

std::string regex_escape(const std::string& text) {
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(text, special, R"(\$&)");
}

using RtpHeader = std::array<unsigned char, 12>;

// Receives one stream's video and keeps each packet's RTP header until `stop` is set. A copy of
// single-stream's helper: #823 moves it into support/cpp/sima_examples/testing/rtp_video.
class VideoDrain {
public:
  VideoDrain(int port, const std::atomic<bool>& stop) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{0, 200000};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &kVideoReceiveBuffer, sizeof(kVideoReceiveBuffer));
    socklen_t size = sizeof(receive_buffer_);
    ::getsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &receive_buffer_, &size);
    ok_ = fd_ >= 0 && ::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (ok_) {
      // The loop only reads, into one buffer, so it empties the socket as fast as it can.
      thread_ = std::thread([this, &stop] {
        std::vector<unsigned char> packet(65536);
        for (;;) {
          const auto size = ::recv(fd_, packet.data(), packet.size(), 0);
          if (size >= 12) {
            RtpHeader header{};
            std::copy_n(packet.begin(), header.size(), header.begin());
            headers_.push_back(header);
          } else if (stop) {
            return;
          }
        }
      });
    }
  }
  ~VideoDrain() {
    if (thread_.joinable()) {
      thread_.join();
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  bool ok() const {
    return ok_;
  }
  int receive_buffer() const {
    return receive_buffer_;
  }
  // Call after `stop` is set.
  const std::vector<RtpHeader>& headers() {
    if (thread_.joinable()) {
      thread_.join();
    }
    return headers_;
  }

private:
  int fd_ = -1;
  bool ok_ = false;
  int receive_buffer_ = 0;
  std::vector<RtpHeader> headers_;
  std::thread thread_;
};

// The host's UDP counters, from the Udp: lines of /proc/net/snmp.
std::map<std::string, long long> udp_counters() {
  std::ifstream file("/proc/net/snmp");
  std::vector<std::vector<std::string>> rows;
  for (std::string line; std::getline(file, line) && rows.size() < 2;) {
    if (line.rfind("Udp:", 0) == 0) {
      std::istringstream words(line);
      rows.emplace_back(std::istream_iterator<std::string>(words),
                        std::istream_iterator<std::string>());
    }
  }
  std::map<std::string, long long> counters;
  for (std::size_t i = 1; rows.size() == 2 && i < rows[0].size() && i < rows[1].size(); ++i) {
    counters[rows[0][i]] = std::stoll(rows[1][i]);
  }
  return counters;
}

// Datagrams the kernel dropped for the socket bound to 127.0.0.1:`port`, the last column of
// /proc/net/udp, or -1 when the socket is not listed.
long long socket_drops(int port) {
  char local[16];
  std::snprintf(local, sizeof(local), "0100007F:%04X", port);
  std::ifstream file("/proc/net/udp");
  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    std::istringstream words(line);
    const std::vector<std::string> fields{std::istream_iterator<std::string>(words),
                                          std::istream_iterator<std::string>()};
    if (fields.size() > 2 && fields[1] == local) {
      return std::stoll(fields.back());
    }
  }
  return -1;
}

std::uint32_t rtp_timestamp(const RtpHeader& header) {
  return static_cast<std::uint32_t>(header[4]) << 24 | static_cast<std::uint32_t>(header[5]) << 16 |
         static_cast<std::uint32_t>(header[6]) << 8 | header[7];
}

std::uint16_t rtp_sequence(const RtpHeader& header) {
  return static_cast<std::uint16_t>(header[2] << 8 | header[3]);
}

// A gap in the RTP sequence: the packets missing, and the RTP timestamps of the packets before and
// after it. A frame wholly in a gap has a timestamp between those two.
struct SequenceGap {
  long long missing = 0;
  std::uint32_t before = 0;
  std::uint32_t after = 0;
};

std::vector<SequenceGap> sequence_gaps(const std::vector<RtpHeader>& headers) {
  std::vector<SequenceGap> gaps;
  for (std::size_t i = 1; i < headers.size(); ++i) {
    const auto step =
        static_cast<std::uint16_t>(rtp_sequence(headers[i]) - rtp_sequence(headers[i - 1]));
    if (step > 1 && step < 0x8000) {
      gaps.push_back({step - 1, rtp_timestamp(headers[i - 1]), rtp_timestamp(headers[i])});
    }
  }
  return gaps;
}

// Whether the frame at `timestamp_ms` may be one of the packets a gap is missing.
bool in_gap(std::int64_t timestamp_ms, const std::vector<SequenceGap>& gaps) {
  const auto ticks = static_cast<std::uint32_t>(timestamp_ms * 90);
  return std::any_of(gaps.begin(), gaps.end(), [&](const SequenceGap& gap) {
    return static_cast<std::uint32_t>(ticks - gap.before + 90) <=
           static_cast<std::uint32_t>(gap.after - gap.before + 180);
  });
}

// What one video socket got: its packets, the packets missing from the RTP sequence, and the
// frames with a timestamp in the span the metadata covers against the frames the source has in
// that span.
std::string video_report(const std::vector<RtpHeader>& headers,
                         const std::vector<std::int64_t>& timestamps_ms, int fps) {
  long long missing = 0;
  for (const auto& gap : sequence_gaps(headers)) {
    missing += gap.missing;
  }
  std::set<std::uint32_t> rtp_timestamps;
  for (const auto& header : headers) {
    rtp_timestamps.insert(rtp_timestamp(header));
  }
  std::string report = std::to_string(headers.size()) + " RTP packets, " + std::to_string(missing) +
                       " missing from the RTP sequence, " + std::to_string(rtp_timestamps.size()) +
                       " frame timestamps";
  if (timestamps_ms.empty()) {
    return report;
  }
  const auto [first, last] = std::minmax_element(timestamps_ms.begin(), timestamps_ms.end());
  const auto start = static_cast<std::uint32_t>(*first * 90 - 90);
  const auto span = static_cast<std::uint32_t>((*last - *first) * 90 + 180);
  const auto in_span = std::count_if(rtp_timestamps.begin(), rtp_timestamps.end(),
                                     [&](std::uint32_t rtp) { return rtp - start <= span; });
  const auto expected = std::llround(static_cast<double>(*last - *first) * fps / 1000.0) + 1;
  return report + ", " + std::to_string(in_span) + " of the source's " + std::to_string(expected) +
         " frames from " + std::to_string(*first) + " to " + std::to_string(*last) + " ms";
}

// Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a video
// frame's RTP timestamp.
bool pairs(std::int64_t timestamp_ms, const std::set<std::uint32_t>& rtp_timestamps) {
  const auto ticks = static_cast<std::uint32_t>(timestamp_ms * 90);
  return std::any_of(rtp_timestamps.begin(), rtp_timestamps.end(), [&](std::uint32_t rtp) {
    return std::llabs(static_cast<long long>(ticks) - static_cast<long long>(rtp)) <= 90;
  });
}

int run_case(const std::string& binary, const std::string& model_path,
             const MultiStreamSourceCase& source_case, int streams, double min_results_per_s,
             bool first_stream_tracks, const std::string& label) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_streams_reach_their_insight_channels_as_video_and_tracks_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path config_dir = fs::path(output_dir).parent_path();
  const fs::path config_path = config_dir / "config.yaml";
  const std::string relative_model =
      fs::relative(fs::absolute(model_path), fs::absolute(config_dir)).string();
  // The app runs from the Apps root, so only a path joined to the config file finds the model.
  if (fs::exists(relative_model)) {
    std::cerr << "[FAIL] " << label << " the working directory also finds " << relative_model
              << "\n";
    return 1;
  }
  // Four streams of the first URL, the most the README lists, share 4 frames in flight.
  const std::vector<std::string> urls =
      streams == 4
          ? std::vector<std::string>(4, source_case.urls.front())
          : std::vector<std::string>(source_case.urls.begin(), source_case.urls.begin() + streams);
  write_e2e_config(kExampleName, config_path,
                   {// Relative to the generated config, so the test runs the app's path join.
                    {"model.path", relative_model},
                    {"input.codec", source_case.codec},
                    {"inference.frames", std::to_string(kFrames)},
                    {"inference.min_score", std::to_string(kMinScore)},
                    // This test is the Insight receiver, so the example sends to loopback.
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port_base", std::to_string(video_port)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port)}},
                   {{"input.source", urls}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = "127.0.0.1";
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = streams;
  metadata_options.metadata_type = "tracking";
  metadata_options.data_array_key = "tracks";
  MetadataJsonListener listener(metadata_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] " << label << " cannot bind the metadata ports: " << listener.error()
              << "\n";
    return 1;
  }
  std::atomic<bool> stop{false};
  std::vector<std::unique_ptr<VideoDrain>> videos;
  for (int index = 0; index < streams; ++index) {
    videos.push_back(std::make_unique<VideoDrain>(video_port + index, stop));
    if (!videos.back()->ok()) {
      std::cerr << "[FAIL] " << label << " cannot bind video port " << video_port + index << "\n";
      stop = true;
      return 1;
    }
  }

  // Read during the run and at exit, so no message waits long enough to overflow the socket
  // buffer.
  MetadataJsonListenerResult metadata;
  std::map<int, std::vector<Clock::time_point>> arrivals;
  const auto read_metadata = [&] {
    const std::size_t before = metadata.messages.size();
    // Once a message has arrived, poll_messages returns after each datagram, so read until the
    // sockets are empty.
    listener.poll_messages(metadata, 100);
    while (listener.poll_messages(metadata, 0)) {
    }
    for (std::size_t i = before; i < metadata.messages.size(); ++i) {
      arrivals[metadata.messages[i].port - metadata_port].push_back(Clock::now());
    }
    return false; // The application stops by itself after inference.frames.
  };
  const auto udp_before = udp_counters();
  const ProcessResult result =
      spawn_until(binary, {"--config", config_path.string()}, read_metadata, timeout_ms);
  const auto rest = listener.drain_pending();
  metadata.messages.insert(metadata.messages.end(), rest.messages.begin(), rest.messages.end());
  stop = true;
  for (const auto& video : videos) {
    video->headers();
  }
  const auto udp_after = udp_counters();
  // The host's UDP counters over the run: RcvbufErrors counts datagrams dropped because a
  // receiving socket's buffer was full.
  std::string udp_change;
  for (const std::string name : {"RcvbufErrors", "SndbufErrors"}) {
    if (udp_before.count(name) != 0 && udp_after.count(name) != 0) {
      udp_change += (udp_change.empty() ? "" : ", ") + name + " +" +
                    std::to_string(udp_after.at(name) - udp_before.at(name));
    }
  }

  std::vector<std::string> problems;
  if (const std::string problem = exit_problem(result); !problem.empty()) {
    problems.push_back(problem + "\nstdout:\n" + result.stdout_text + "\nstderr:\n" +
                       result.stderr_text);
  }
  for (const std::string& error : {metadata.error, rest.error}) {
    if (!error.empty()) {
      problems.push_back("metadata listener: " + error);
    }
  }
  for (int index = 0; index < streams; ++index) {
    const std::string stream = "stream " + std::to_string(index);
    const std::regex start_line("(^|\n)\\[" + stream +
                                "\\] rtsp=" + regex_escape(urls[static_cast<std::size_t>(index)]) +
                                " stream=\\d+x\\d+@(\\d+) insight=127\\.0\\.0\\.1 video=" +
                                std::to_string(video_port + index) +
                                " metadata=" + std::to_string(metadata_port + index) + "\n");
    std::smatch start;
    int fps = 0;
    if (std::regex_search(result.stdout_text, start, start_line)) {
      fps = std::stoi(start[2]);
    } else {
      problems.push_back(stream + " did not print its source and ports");
    }
    std::smatch summary;
    const std::regex summary_line("(^|\n)\\[" + stream + "\\] processed=" +
                                  std::to_string(kFrames) + " tracks_started=(\\d+)\n");
    int tracks_started = 0;
    if (std::regex_search(result.stdout_text, summary, summary_line)) {
      tracks_started = std::stoi(summary[2]);
    } else {
      problems.push_back(stream + " did not report processed=" + std::to_string(kFrames) +
                         " tracks_started=<count>");
    }

    std::set<int> payload_types;
    std::set<std::uint32_t> rtp_timestamps;
    for (const auto& header : videos[static_cast<std::size_t>(index)]->headers()) {
      payload_types.insert(header[1] & 0x7F);
      rtp_timestamps.insert(rtp_timestamp(header));
    }
    if (payload_types != std::set<int>{payload_type(source_case.codec)}) {
      problems.push_back(stream + " video did not arrive with RTP payload type " +
                         std::to_string(payload_type(source_case.codec)));
    }

    int messages = 0;
    int tracks = 0;
    std::map<std::string, int> messages_per_id;
    long long previous_frame_id = -1;
    std::vector<std::int64_t> unpaired;
    std::vector<std::int64_t> timestamps_ms;
    for (const auto& message : metadata.messages) {
      if (message.port != metadata_port + index) {
        continue;
      }
      ++messages;
      timestamps_ms.push_back(message.timestamp_ms);
      if (message.frame_id.empty()) {
        problems.push_back(stream + ": a message lacks frame_id");
        continue;
      }
      const long long frame_id = std::stoll(message.frame_id);
      if (frame_id <= previous_frame_id) {
        problems.push_back(stream + ": frame IDs out of order: " + message.frame_id + " after " +
                           std::to_string(previous_frame_id));
      }
      previous_frame_id = frame_id;
      // The per-track checks cover only the tracks a stream sent. In run 45586507970, stream 1 of
      // the h264 and h265 cases, CI's second URLs, sent none.
      // Named, because before C++23 a range-for over a member of a temporary reads it after the
      // temporary is destroyed.
      const auto payload = nlohmann::json::parse(message.payload);
      std::set<std::string> ids;
      for (const auto& track : payload.at("data").at("tracks")) {
        const double confidence = track.at("confidence").get<double>();
        const std::string id_text = track.at("id").get<std::string>();
        const int id = std::stoi(id_text);
        // The model has one class, so a class ID other than 0 would arrive as "unknown".
        if (track.at("label") != "drone" || confidence < kMinScore || confidence > 1.0 || id < 1 ||
            id > tracks_started) {
          problems.push_back(stream + " frame " + message.frame_id + ": " + track.dump());
        }
        ids.insert(id_text);
        ++tracks;
      }
      for (const auto& id : ids) {
        ++messages_per_id[id];
      }
      // Insight draws each message's tracks on the video frame its timestamp names.
      if (!pairs(message.timestamp_ms, rtp_timestamps)) {
        unpaired.push_back(message.timestamp_ms);
      }
    }
    if (index == 0 && first_stream_tracks) {
      if (tracks == 0) {
        problems.push_back(stream + " sent no tracks");
      }
      // The tracker keeps an ID across frames: some ID is in two or more messages.
      if (std::none_of(messages_per_id.begin(), messages_per_id.end(),
                       [](const auto& entry) { return entry.second >= 2; })) {
        problems.push_back(stream + ": no track ID is in two messages");
      }
    }
    // UDP on loopback may drop a message under load, so allow a few to go missing.
    if (messages < kFrames * 95 / 100) {
      problems.push_back(stream + ": " + std::to_string(messages) + " of " +
                         std::to_string(kFrames) + " messages arrived");
    }
    // A frame in a gap of the RTP sequence was lost between the sender and this test, so its
    // message counts against the frames received, not against the app's timestamps.
    VideoDrain& drain = *videos[static_cast<std::size_t>(index)];
    const auto gaps = sequence_gaps(drain.headers());
    const auto lost = std::count_if(unpaired.begin(), unpaired.end(), [&](std::int64_t timestamp) {
      return in_gap(timestamp, gaps);
    });
    unpaired.erase(std::remove_if(unpaired.begin(), unpaired.end(),
                                  [&](std::int64_t timestamp) { return in_gap(timestamp, gaps); }),
                   unpaired.end());
    const std::string video =
        "video port " + std::to_string(video_port + index) + ": " +
        video_report(drain.headers(), timestamps_ms, fps) + ", " +
        std::to_string(socket_drops(video_port + index)) +
        " dropped by the socket, receive buffer " + std::to_string(drain.receive_buffer()) +
        " bytes from getsockopt for " + std::to_string(kVideoReceiveBuffer) + " asked, " +
        std::to_string(lost) +
        " messages in a sequence gap. Host UDP: " + (udp_change.empty() ? "not read" : udp_change);
    // The video link sends every frame, so a message stays unpaired only when its frame was lost
    // after the sender. Allow 5%, as for the count.
    if (static_cast<int>(unpaired.size()) > messages * 5 / 100) {
      problems.push_back(stream + ": " + std::to_string(unpaired.size()) +
                         " timestamps with no video frame, first " +
                         std::to_string(unpaired.front()) + ". " + video);
    }
    const auto& times = arrivals[index];
    const std::size_t first = times.size() / 10;
    if (times.size() - first <= 10) {
      problems.push_back(stream + ": " + std::to_string(times.size() - first) +
                         " messages arrived during the run");
      continue;
    }
    const double seconds = std::chrono::duration<double>(times.back() - times[first]).count();
    const double results_per_s = static_cast<double>(times.size() - first - 1) / seconds;
    std::cout << label << " " << stream << ": " << messages << " messages, " << results_per_s
              << " results/s, " << tracks << " tracks in " << tracks_started << " track IDs, "
              << unpaired.size() << " unpaired. " << video << "\n";
    if (results_per_s < min_results_per_s) {
      problems.push_back(stream + ": " + std::to_string(results_per_s) + " results/s");
    }
  }
  remove_dir(output_dir);
  for (const auto& problem : problems) {
    std::cerr << "[FAIL] " << label << " " << problem << "\n";
  }
  if (problems.empty()) {
    std::cout << "[OK] " << label << " reached its Insight channels as video and tracks\n";
  }
  return problems.empty() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured tracker model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }
  const MultiStreamSourceCase h264{"h264", rtsp_h264_urls_from_env()};
  const MultiStreamSourceCase h265{"h265", rtsp_h265_urls_from_env()};
  // The bool: whether stream 0 must send tracks. CI's first H.264 and H.265 URLs show objects the
  // model tracks: in run 45586507970 on 1f70fac, stream 0 sent 273 tracks in 10 track IDs for H.264
  // and 185 in 10 for H.265. With one source, Core names the samples after the decoder instead of
  // the stream.
  const std::vector<std::tuple<MultiStreamSourceCase, int, double, bool, std::string>> cases = {
      {h264, 2, kMinResultsPerS, true, "h264"},
      {h265, 2, kMinResultsPerS, true, "h265"},
      {h264, 1, kMinResultsPerS, false, "h264-one-stream"},
      {h264, 4, kMinFourStreamResultsPerS, false, "h264-four-streams"}};
  int cases_run = 0;
  for (const auto& [source_case, streams, min_results_per_s, first_stream_tracks, label] : cases) {
    const int needed = streams == 4 ? 1 : streams;
    if (source_case.urls.size() < static_cast<std::size_t>(needed)) {
      record_unavailable_case("SIMANEAT_TEST_RTSP_" +
                                  std::string(source_case.codec == "h264" ? "H264" : "H265") +
                                  "_URLS needs " + std::to_string(needed) + " URL(s) for " + label,
                              "not enough " + source_case.codec + " URLs for " + label, rc);
      continue;
    }
    ++cases_run;
    if (run_case(binary, model_paths.front(), source_case, streams, min_results_per_s,
                 first_stream_tracks, label) != 0) {
      rc = 1;
    }
  }
  if (cases_run == 0) {
    return rc != 0 ? rc : skip_or_fail("no RTSP URLs configured for " + std::string(kExampleName));
  }
  return rc;
}
