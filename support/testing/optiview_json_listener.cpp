#include "support/testing/optiview_json_listener.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <poll.h>
#include <set>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace sima_examples::testing {
namespace {

using json = nlohmann::json;

// Resolve the bind address for a local UDP listener used by tests.
bool resolve_bind_addr(const std::string& host, int port, sockaddr_storage& out, socklen_t& out_len,
                       std::string& err) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;
  hints.ai_flags = host.empty() ? AI_PASSIVE : 0;

  addrinfo* result = nullptr;
  const std::string port_str = std::to_string(port);
  const char* bind_host = host.empty() ? nullptr : host.c_str();
  const int rc = ::getaddrinfo(bind_host, port_str.c_str(), &hints, &result);
  if (rc != 0 || !result) {
    err = "getaddrinfo failed for " + (host.empty() ? std::string("*") : host) + ":" + port_str +
          " (" + gai_strerror(rc) + ")";
    return false;
  }

  bool ok = false;
  for (addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
    if (!ai->ai_addr || ai->ai_addrlen == 0)
      continue;
    std::memset(&out, 0, sizeof(out));
    std::memcpy(&out, ai->ai_addr, ai->ai_addrlen);
    out_len = static_cast<socklen_t>(ai->ai_addrlen);
    ok = true;
    break;
  }
  ::freeaddrinfo(result);

  if (!ok) {
    err = "failed to resolve bind address for " + (host.empty() ? std::string("*") : host) + ":" +
          port_str;
  }
  return ok;
}

// Keep validation intentionally narrow and stable: the goal is to verify that
// an example emitted structurally valid OptiView detection JSON, not to enforce
// model-specific semantics such as object count or label names.
bool is_valid_optiview_detection_json(const std::string& payload, OptiViewJsonMessage& out,
                                      std::string& err) {
  json parsed;
  try {
    parsed = json::parse(payload);
  } catch (const std::exception& ex) {
    err = std::string("json parse failed: ") + ex.what();
    return false;
  }

  if (!parsed.is_object()) {
    err = "json root is not an object";
    return false;
  }
  if (!parsed.contains("type") || parsed["type"] != "object-detection") {
    err = "missing or invalid type";
    return false;
  }
  if (!parsed.contains("timestamp") || !parsed["timestamp"].is_number_integer()) {
    err = "missing or invalid timestamp";
    return false;
  }
  if (!parsed.contains("frame_id") || !parsed["frame_id"].is_string()) {
    err = "missing or invalid frame_id";
    return false;
  }
  if (!parsed.contains("data") || !parsed["data"].is_object()) {
    err = "missing or invalid data";
    return false;
  }
  if (!parsed["data"].contains("objects") || !parsed["data"]["objects"].is_array()) {
    err = "missing or invalid data.objects";
    return false;
  }

  out.payload = payload;
  out.frame_id = parsed["frame_id"].get<std::string>();
  out.timestamp_ms = parsed["timestamp"].get<int64_t>();
  out.object_count = static_cast<int>(parsed["data"]["objects"].size());
  return true;
}

} // namespace

struct OptiViewJsonListener::SocketState {
  int fd = -1;
  int port = -1;
};

OptiViewJsonListener::OptiViewJsonListener(const OptiViewJsonListenerOptions& opt) : opt_(opt) {
  if (opt_.num_ports <= 0) {
    err_ = "num_ports must be > 0";
    return;
  }
  if (opt_.base_port <= 0) {
    err_ = "base_port must be > 0";
    return;
  }
  if (!bind_ports() && err_.empty()) {
    err_ = "failed to bind listener ports";
  }
}

OptiViewJsonListener::~OptiViewJsonListener() {
  for (auto& sock : sockets_) {
    if (sock.fd >= 0) {
      ::close(sock.fd);
    }
  }
}

bool OptiViewJsonListener::ok() const {
  return err_.empty();
}

const std::string& OptiViewJsonListener::error() const {
  return err_;
}

bool OptiViewJsonListener::bind_ports() {
  sockets_.clear();
  sockets_.reserve(static_cast<size_t>(opt_.num_ports));

  // Tests may listen on one port for single-camera examples or many ports for
  // multi-stream examples. Bind the entire requested range up front so the
  // listener is ready before the example under test starts sending packets.
  for (int i = 0; i < opt_.num_ports; ++i) {
    const int port = opt_.base_port + i;
    sockaddr_storage addr{};
    socklen_t addr_len = 0;
    std::string resolve_err;
    if (!resolve_bind_addr(opt_.host, port, addr, addr_len, resolve_err)) {
      err_ = resolve_err;
      return false;
    }

    const int fd = ::socket(addr.ss_family, SOCK_DGRAM, 0);
    if (fd < 0) {
      err_ = "socket failed for port " + std::to_string(port) + ": " + std::strerror(errno);
      return false;
    }

    const int reuse = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr), addr_len) < 0) {
      err_ = "bind failed for port " + std::to_string(port) + ": " + std::strerror(errno);
      ::close(fd);
      return false;
    }

    sockets_.push_back(SocketState{fd, port});
  }

  return true;
}

bool OptiViewJsonListener::handle_datagram(SocketState& sock, OptiViewJsonListenerResult& result) {
  char buf[65536];
  const ssize_t n = ::recv(sock.fd, buf, sizeof(buf), 0);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }
    result.error = "recv failed on port " + std::to_string(sock.port) + ": " + std::strerror(errno);
    return false;
  }

  OptiViewJsonMessage msg;
  msg.port = sock.port;
  std::string parse_err;
  const std::string payload(buf, static_cast<size_t>(n));
  if (!is_valid_optiview_detection_json(payload, msg, parse_err)) {
    if (result.error.empty()) {
      result.error = "invalid json on port " + std::to_string(sock.port) + ": " + parse_err;
    }
    return false;
  }

  result.messages.push_back(std::move(msg));
  if (std::find(result.ports_with_valid_json.begin(), result.ports_with_valid_json.end(), sock.port) ==
      result.ports_with_valid_json.end()) {
    result.ports_with_valid_json.push_back(sock.port);
  }
  return true;
}

bool OptiViewJsonListener::success_reached(const OptiViewJsonListenerResult& result) const {
  if (opt_.require_all_ports) {
    return static_cast<int>(result.ports_with_valid_json.size()) >= opt_.num_ports;
  }
  return !result.ports_with_valid_json.empty();
}

OptiViewJsonListenerResult OptiViewJsonListener::wait_for_messages() {
  OptiViewJsonListenerResult result;
  if (!ok()) {
    result.error = err_;
    return result;
  }

  std::vector<pollfd> pfds;
  pfds.reserve(sockets_.size());
  for (const auto& sock : sockets_) {
    pfds.push_back(pollfd{sock.fd, POLLIN, 0});
  }

  const int64_t deadline_ms = static_cast<int64_t>(opt_.timeout_ms);
  int64_t waited_ms = 0;
  // Poll all sockets from a single loop so the same utility scales from
  // single-port smoke tests to larger multi-port OptiView e2e checks.
  while (waited_ms < deadline_ms) {
    const int poll_ms = static_cast<int>(std::min<int64_t>(250, deadline_ms - waited_ms));
    const int rc = ::poll(pfds.data(), pfds.size(), poll_ms);
    if (rc < 0) {
      result.error = std::string("poll failed: ") + std::strerror(errno);
      return result;
    }
    waited_ms += poll_ms;
    if (rc == 0) {
      continue;
    }

    for (size_t i = 0; i < pfds.size(); ++i) {
      if ((pfds[i].revents & POLLIN) == 0)
        continue;
      (void)handle_datagram(sockets_[i], result);
      if (success_reached(result)) {
        result.success = true;
        return result;
      }
    }
  }

  if (result.error.empty()) {
    result.error = opt_.require_all_ports
                       ? "timed out waiting for valid json on all configured ports"
                       : "timed out waiting for valid json on any configured port";
  }
  return result;
}

} // namespace sima_examples::testing
