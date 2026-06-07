#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sima_examples::testing {

// Test-only UDP listener for metadata JSON outputs.
//
// This utility is intended for end-to-end example validation under CTest:
// start an example, listen on one or more metadata JSON ports, and return
// success once valid object-detection metadata is observed. It is not part of the
// runtime/sample data path used by the examples themselves.
struct MetadataJsonListenerOptions {
  std::string host = "127.0.0.1";
  int base_port = 9100;
  int num_ports = 1;
  int timeout_ms = 20000;
  // When false, any valid JSON message on any configured port is enough.
  // When true, each configured port must receive at least one valid message.
  bool require_all_ports = false;
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

  // Wait until timeout for valid object-detection metadata. This is designed
  // around the typical e2e test contract: pass if the example emits at least
  // one well-formed detection payload, or fail after the timeout expires.
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
