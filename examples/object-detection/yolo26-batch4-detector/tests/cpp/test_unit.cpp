// Unit tests for yolo26-batch4-detector (C++).
// Cover the parts that need no hardware: CLI handling and config validation.
#include "support/testing/test_process.h"
#include "support/testing/metadata_json_listener.h"
#define YOLO26_BATCH4_CONTRACT_ONLY
#include "examples/object-detection/yolo26-batch4-detector/src/cpp/main.cpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kModelLine = "  path: assets/models/yolo26m-det-int8-b4.tar.gz\n";

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir("yolo26-batch4-detector", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_accepts_four_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_four_streams",
                                            std::string("model:\n") + kModelLine +
                                                "streams:\n"
                                                "  - rtsp://127.0.0.1:8554/src1\n"
                                                "  - rtsp://127.0.0.1:8554/src2\n"
                                                "  - rtsp://127.0.0.1:8554/src3\n"
                                                "  - rtsp://127.0.0.1:8554/src4\n"
                                                "inference:\n"
                                                "  max_detections: 42\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "max_detections=42",
                      "validate output reports the detection cap");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_wrong_stream_count(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_wrong_stream_count",
                                            std::string("model:\n") + kModelLine +
                                                "streams:\n"
                                                "  - rtsp://127.0.0.1:8554/src1\n"
                                                "  - rtsp://127.0.0.1:8554/src2\n"
                                                "  - rtsp://127.0.0.1:8554/src3\n"
                                                "  - rtsp://127.0.0.1:8554/src4\n"
                                                "  - rtsp://127.0.0.1:8554/src5\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "exactly 4 streams",
                                  "wrong-stream-count error names the batch contract");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_empty_streams",
                                            std::string("model:\n") + kModelLine +
                                                "streams: []\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "empty stream config is rejected") &&
      expect_contains(result.stderr_text, "streams", "empty-stream error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_out_of_range_threshold(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_out_of_range_threshold",
                   std::string("model:\n") + kModelLine +
                       "streams:\n"
                       "  - rtsp://127.0.0.1:8554/src1\n"
                       "  - rtsp://127.0.0.1:8554/src2\n"
                       "  - rtsp://127.0.0.1:8554/src3\n"
                       "  - rtsp://127.0.0.1:8554/src4\n"
                       "inference:\n"
                       "  score_threshold: 1.5\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "out-of-range score threshold is rejected") &&
                  expect_contains(result.stderr_text, "score_threshold must be between 0 and 1",
                                  "threshold error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_placeholder_stream(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_placeholder_stream",
                                            std::string("model:\n") + kModelLine +
                                                "streams:\n"
                                                "  - <rtsp-url-1>\n"
                                                "  - rtsp://127.0.0.1:8554/src2\n"
                                                "  - rtsp://127.0.0.1:8554/src3\n"
                                                "  - rtsp://127.0.0.1:8554/src4\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "placeholder stream URL is rejected") &&
      expect_contains(result.stderr_text, "placeholder", "placeholder error says so explicitly");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_placeholder_model(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_placeholder_model",
                                            "model:\n"
                                            "  path: <model-path>\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "placeholder model path is rejected") &&
                  expect_contains(result.stderr_text, "placeholder",
                                  "placeholder model error says so explicitly");
  remove_dir(config_path.parent_path().string());
  return ok;
}

std::vector<std::vector<std::int64_t>> valid_output_shapes(std::int64_t classes = 6) {
  return {{4, 80, 80, 4},       {4, 40, 40, 4},       {4, 20, 20, 4},
          {4, 80, 80, classes}, {4, 40, 40, classes}, {4, 20, 20, classes}};
}

bool test_model_contract_accepts_reordered_non80_heads() {
  auto outputs = valid_output_shapes();
  outputs = {outputs[4], outputs[0], outputs[5], outputs[2], outputs[3], outputs[1]};
  const auto contract = yolo26_batch4::validate_model_contract({{4, 640, 640, 3}}, outputs, 6);
  return expect_true(contract.net == 640 && contract.class_count == 6,
                     "model contract accepts reordered non-80 heads") &&
         expect_true(contract.grids[0].height == 80 && contract.grids[2].width == 20,
                     "model contract orders matched grid pairs") &&
         expect_true(contract.bbox_indices == std::array<std::size_t, 3>{1, 5, 3} &&
                         contract.class_indices == std::array<std::size_t, 3>{4, 0, 2},
                     "model contract resolves shuffled output indices once");
}

bool rejects_contract(const std::vector<std::vector<std::int64_t>>& inputs,
                      const std::vector<std::vector<std::int64_t>>& outputs, std::size_t labels) {
  try {
    (void)yolo26_batch4::validate_model_contract(inputs, outputs, labels);
    return false;
  } catch (const std::runtime_error&) {
    return true;
  }
}

bool test_model_contract_rejects_invalid_shapes() {
  auto wrong_grid = valid_output_shapes();
  wrong_grid.back() = {4, 10, 10, 6};
  auto wrong_classes = valid_output_shapes();
  wrong_classes.back()[3] = 7;
  auto rectangular = valid_output_shapes();
  rectangular.front() = {4, 80, 40, 4};
  return expect_true(rejects_contract({{2, 640, 640, 3}}, valid_output_shapes(), 6),
                     "model contract rejects non-batch-4 input") &&
         expect_true(rejects_contract({{4, 640, 320, 3}}, valid_output_shapes(), 6),
                     "model contract rejects non-square input") &&
         expect_true(rejects_contract({{4, 640, 640, 3}}, rectangular, 6),
                     "model contract rejects rectangular output grids") &&
         expect_true(rejects_contract({{4, 640, 640, 3}}, wrong_grid, 6),
                     "model contract rejects mismatched grid pairs") &&
         expect_true(rejects_contract({{4, 640, 640, 3}}, wrong_classes, 6),
                     "model contract rejects inconsistent class counts") &&
         expect_true(rejects_contract({{4, 640, 640, 3}}, valid_output_shapes(), 5),
                     "model contract rejects mismatched labels") &&
         expect_true(rejects_contract({{4, 640, 640, 3}}, valid_output_shapes(4), 4),
                     "model contract rejects ambiguous four-class heads");
}

bool test_head_index_uses_width() {
  return expect_true(yolo26_batch4::head_cell_offset(2, 3, 7, 6) == 102U,
                     "head cell indexing uses grid width");
}

int available_udp_port() {
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (fd < 0 || ::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
    if (fd >= 0)
      ::close(fd);
    return -1;
  }
  socklen_t size = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
    ::close(fd);
    return -1;
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

std::string metadata_payload(const std::string& frame_id, std::size_t padding) {
  return "{\"type\":\"object-detection\",\"timestamp\":42,\"frame_id\":\"" + frame_id +
         "\",\"data\":{\"objects\":[{\"label\":\"person\",\"padding\":\"" +
         std::string(padding, 'x') + "\"}]}}";
}

std::vector<std::string> chunk_payload(const std::string& payload, std::uint64_t message_id) {
  constexpr std::size_t chunk_bytes = 1188;
  const std::size_t count = (payload.size() + chunk_bytes - 1U) / chunk_bytes;
  std::vector<std::string> chunks;
  for (std::size_t index = 0; index < count; ++index) {
    std::string chunk(12, '\0');
    chunk[0] = 0x4e;
    chunk[1] = 0x01;
    for (std::size_t byte = 0; byte < 8; ++byte) {
      chunk[2 + byte] = static_cast<char>(message_id >> ((7U - byte) * 8U));
    }
    chunk[10] = static_cast<char>(index);
    chunk[11] = static_cast<char>(count);
    chunk.append(payload, index * chunk_bytes,
                 std::min(chunk_bytes, payload.size() - index * chunk_bytes));
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

bool send_datagram(int fd, int port, const std::string& payload) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  return ::sendto(fd, payload.data(), payload.size(), 0,
                  reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) == static_cast<ssize_t>(payload.size());
}

bool test_metadata_listener_reassembles_chunks() {
  const int port = available_udp_port();
  sima_examples::testing::MetadataJsonListenerOptions options;
  options.host = "127.0.0.1";
  options.base_port = port;
  options.timeout_ms = 500;
  sima_examples::testing::MetadataJsonListener listener(options);
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  auto chunks = chunk_payload(metadata_payload("chunked", 3000), 7);
  bool sent = send_datagram(fd, port, chunks[1]) && send_datagram(fd, port, chunks[1]) &&
              send_datagram(fd, port, chunks[0]);
  for (std::size_t index = 2; index < chunks.size(); ++index) {
    sent = send_datagram(fd, port, chunks[index]) && sent;
  }
  const auto result = listener.wait_for_messages();
  ::close(fd);
  return expect_true(port > 0 && listener.ok() && sent && result.success,
                     "metadata listener reassembles out-of-order chunks and duplicates") &&
         expect_true(result.messages.front().frame_id == "chunked",
                     "metadata listener parses only the reconstructed JSON");
}

bool test_metadata_listener_accepts_raw_json() {
  const int port = available_udp_port();
  sima_examples::testing::MetadataJsonListenerOptions options;
  options.host = "127.0.0.1";
  options.base_port = port;
  options.timeout_ms = 200;
  sima_examples::testing::MetadataJsonListener listener(options);
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  const bool sent = send_datagram(fd, port, metadata_payload("raw", 0));
  const auto result = listener.wait_for_messages();
  ::close(fd);
  return expect_true(listener.ok() && sent && result.success &&
                         result.messages.front().frame_id == "raw",
                     "metadata listener preserves raw JSON datagrams");
}

bool test_metadata_listener_rejects_conflicting_duplicate() {
  const int conflict_port = available_udp_port();
  sima_examples::testing::MetadataJsonListenerOptions conflict_options;
  conflict_options.host = "127.0.0.1";
  conflict_options.base_port = conflict_port;
  conflict_options.timeout_ms = 100;
  sima_examples::testing::MetadataJsonListener conflict_listener(conflict_options);
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  auto chunks = chunk_payload(metadata_payload("conflict", 1500), 8);
  auto conflicting = chunks[0];
  conflicting.back() ^= 1;
  (void)send_datagram(fd, conflict_port, chunks[0]);
  (void)send_datagram(fd, conflict_port, conflicting);
  (void)send_datagram(fd, conflict_port, chunks[1]);
  const auto conflict = conflict_listener.wait_for_messages();

  ::close(fd);
  return expect_true(!conflict.success && conflict.error.find("metadata chunk contents changed") !=
                                              std::string::npos,
                     "metadata listener rejects conflicting duplicates");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_accepts_four_streams(binary);
  ok &= test_validate_config_only_rejects_wrong_stream_count(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_out_of_range_threshold(binary);
  ok &= test_validate_config_only_rejects_placeholder_stream(binary);
  ok &= test_validate_config_only_rejects_placeholder_model(binary);
  ok &= test_model_contract_accepts_reordered_non80_heads();
  ok &= test_model_contract_rejects_invalid_shapes();
  ok &= test_head_index_uses_width();
  ok &= test_metadata_listener_reassembles_chunks();
  ok &= test_metadata_listener_accepts_raw_json();
  ok &= test_metadata_listener_rejects_conflicting_duplicate();
  return ok ? 0 : 1;
}
