#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sima_examples::testing {

// Test-only UDP listener for OptiView JSON outputs.
//
// This utility is intended for end-to-end example validation under CTest:
// start an example, listen on one or more OptiView JSON ports, and return
// success once valid detection JSON is observed. It is not part of the
// runtime/sample data path used by the examples themselves.
struct OptiViewJsonListenerOptions {
  std::string host = "127.0.0.1";
  int base_port = 9100;
  int num_ports = 1;
  int timeout_ms = 10000;
  // When false, any valid JSON message on any configured port is enough.
  // When true, each configured port must receive at least one valid message.
  bool require_all_ports = false;
};

struct OptiViewJsonMessage {
  int port = -1;
  std::string payload;
  std::string frame_id;
  int64_t timestamp_ms = 0;
  int object_count = 0;
};

struct OptiViewJsonListenerResult {
  bool success = false;
  std::vector<int> ports_with_valid_json;
  std::vector<OptiViewJsonMessage> messages;
  std::string error;
};

class OptiViewJsonListener {
public:
  explicit OptiViewJsonListener(const OptiViewJsonListenerOptions& opt);
  ~OptiViewJsonListener();

  OptiViewJsonListener(const OptiViewJsonListener&) = delete;
  OptiViewJsonListener& operator=(const OptiViewJsonListener&) = delete;

  bool ok() const;
  const std::string& error() const;

  // Wait until timeout for valid OptiView detection JSON. This is designed
  // around the typical e2e test contract: pass if the example emits at least
  // one well-formed detection payload, or fail after the timeout expires.
  OptiViewJsonListenerResult wait_for_messages();

private:
  struct SocketState;

  bool bind_ports();
  bool handle_datagram(SocketState& sock, OptiViewJsonListenerResult& result);
  bool success_reached(const OptiViewJsonListenerResult& result) const;

  OptiViewJsonListenerOptions opt_;
  std::string err_;
  std::vector<SocketState> sockets_;
};

} // namespace sima_examples::testing
