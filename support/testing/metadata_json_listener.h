#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sima_examples::testing {

// Test-only UDP listener for metadata JSON outputs.
struct MetadataJsonListenerOptions {
  std::string host = "127.0.0.1";
  int base_port = 9100;
  int num_ports = 1;
  int timeout_ms = 20000;
  std::string metadata_type = "object-detection";
  std::string data_array_key = "objects";
  // When false, any valid JSON message on any configured port is enough.
  // When true, each configured port must receive at least one valid message.
  bool require_all_ports = false;
  // Ignore otherwise valid messages until the data array contains at least this many objects.
  int min_object_count = 0;
};

struct MetadataJsonMessage {
  int port = -1;
  std::string payload;
  std::string frame_id;
  int64_t timestamp_ms = 0;
  int object_count = 0;
};

struct MetadataJsonListenerResult {
  bool success = false;
  std::vector<int> ports_with_valid_json;
  std::vector<MetadataJsonMessage> messages;
  std::string error;
};

class MetadataJsonListener {
public:
  explicit MetadataJsonListener(const MetadataJsonListenerOptions& opt);
  ~MetadataJsonListener();

  MetadataJsonListener(const MetadataJsonListener&) = delete;
  MetadataJsonListener& operator=(const MetadataJsonListener&) = delete;

  bool ok() const;
  const std::string& error() const;

  // Wait until timeout for valid metadata matching MetadataJsonListenerOptions.
  MetadataJsonListenerResult wait_for_messages();

private:
  struct SocketState;

  bool bind_ports();
  bool handle_datagram(SocketState& sock, MetadataJsonListenerResult& result);
  bool success_reached(const MetadataJsonListenerResult& result) const;

  MetadataJsonListenerOptions opt_;
  std::string err_;
  std::vector<SocketState> sockets_;
};

} // namespace sima_examples::testing
