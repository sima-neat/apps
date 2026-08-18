#include "support/testing/metadata_json_listener.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <netdb.h>
#include <poll.h>
#include <set>
#include <string>
#include <sys/socket.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace sima_examples::testing {
namespace {

using json = nlohmann::json;

constexpr size_t kMaxDatagramPayload = 1200;
constexpr size_t kChunkHeaderSize = 12;
constexpr size_t kMaxChunkPayload = kMaxDatagramPayload - kChunkHeaderSize;
constexpr size_t kMaxLogicalPayload = 65507;
constexpr uint8_t kMaxChunkCount = 56;

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

bool is_valid_metadata_json(const std::string& payload, const std::string& metadata_type,
                            const std::string& data_array_key, MetadataJsonMessage& out,
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
  if (!parsed.contains("type") || parsed["type"] != metadata_type) {
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
  if (!parsed["data"].contains(data_array_key) || !parsed["data"][data_array_key].is_array()) {
    err = "missing or invalid data." + data_array_key;
    return false;
  }

  out.payload = payload;
  out.frame_id = parsed["frame_id"].get<std::string>();
  out.timestamp_ms = parsed["timestamp"].get<int64_t>();
  out.object_count = static_cast<int>(parsed["data"][data_array_key].size());
  return true;
}

} // namespace

struct MetadataJsonListener::SocketState {
  struct PartialMessage {
    uint8_t chunk_count = 0;
    std::vector<std::string> chunks;
    std::vector<bool> present;
    size_t size = 0;
    std::chrono::steady_clock::time_point updated_at;
  };

  int fd = -1;
  int port = -1;
  std::unordered_map<std::string, PartialMessage> partial;
};

MetadataJsonListener::MetadataJsonListener(const MetadataJsonListenerOptions& opt) : opt_(opt) {
  if (opt_.num_ports <= 0) {
    err_ = "num_ports must be > 0";
    return;
  }
  if (opt_.base_port <= 0) {
    err_ = "base_port must be > 0";
    return;
  }
  if (opt_.chunk_expiry_ms <= 0 || opt_.max_inflight_messages <= 0 ||
      opt_.min_data_items_per_port < 0) {
    err_ = "chunk reassembly limits must be > 0";
    return;
  }
  if (!bind_ports() && err_.empty()) {
    err_ = "failed to bind listener ports";
  }
}

MetadataJsonListener::~MetadataJsonListener() {
  for (auto& sock : sockets_) {
    if (sock.fd >= 0) {
      ::close(sock.fd);
    }
  }
}

bool MetadataJsonListener::ok() const {
  return err_.empty();
}

const std::string& MetadataJsonListener::error() const {
  return err_;
}

bool MetadataJsonListener::bind_ports() {
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

bool MetadataJsonListener::handle_datagram(SocketState& sock, MetadataJsonListenerResult& result) {
  std::array<char, 65536> buf{};
  sockaddr_storage sender{};
  socklen_t sender_len = sizeof(sender);
  const ssize_t n = ::recvfrom(sock.fd, buf.data(), buf.size(), 0,
                               reinterpret_cast<sockaddr*>(&sender), &sender_len);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }
    result.error = "recv failed on port " + std::to_string(sock.port) + ": " + std::strerror(errno);
    return false;
  }

  const auto now = std::chrono::steady_clock::now();
  for (auto it = sock.partial.begin(); it != sock.partial.end();) {
    const auto age =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.updated_at);
    if (age.count() >= opt_.chunk_expiry_ms) {
      it = sock.partial.erase(it);
    } else {
      ++it;
    }
  }

  std::string payload;
  const auto size = static_cast<size_t>(n);
  const auto* bytes = reinterpret_cast<const uint8_t*>(buf.data());
  if (size == 0 || bytes[0] != 0x4e) {
    payload.assign(buf.data(), size);
  } else {
    auto reject = [&](const std::string& reason) {
      if (result.error.empty()) {
        result.error = "invalid chunk on port " + std::to_string(sock.port) + ": " + reason;
      }
      return false;
    };
    if (size < kChunkHeaderSize) {
      return reject("header is too short");
    }
    if (bytes[1] != 0x01) {
      return reject("unsupported version");
    }
    uint64_t message_id = 0;
    for (size_t byte = 0; byte < sizeof(message_id); ++byte) {
      message_id = (message_id << 8U) | bytes[2 + byte];
    }
    const uint8_t chunk_index = bytes[10];
    const uint8_t chunk_count = bytes[11];
    const size_t chunk_size = size - kChunkHeaderSize;
    if (chunk_count < 2 || chunk_count > kMaxChunkCount || chunk_index >= chunk_count) {
      return reject("index or count is out of range");
    }
    if (chunk_index + 1U < chunk_count && chunk_size != kMaxChunkPayload) {
      return reject("non-final payload has the wrong size");
    }
    if (chunk_index + 1U == chunk_count && (chunk_size == 0 || chunk_size > kMaxChunkPayload)) {
      return reject("final payload has the wrong size");
    }

    char host[NI_MAXHOST]{};
    char service[NI_MAXSERV]{};
    if (::getnameinfo(reinterpret_cast<const sockaddr*>(&sender), sender_len, host, sizeof(host),
                      service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
      return reject("could not identify sender");
    }
    const std::string key = std::string(host) + ":" + service + "/" + std::to_string(message_id);
    auto it = sock.partial.find(key);
    if (it == sock.partial.end()) {
      if (static_cast<int>(sock.partial.size()) >= opt_.max_inflight_messages) {
        return reject("reassembly state limit reached");
      }
      SocketState::PartialMessage state;
      state.chunk_count = chunk_count;
      state.chunks.resize(chunk_count);
      state.present.assign(chunk_count, false);
      state.updated_at = now;
      it = sock.partial.emplace(key, std::move(state)).first;
    } else if (it->second.chunk_count != chunk_count) {
      sock.partial.erase(it);
      return reject("inconsistent chunk count");
    }

    auto& state = it->second;
    const std::string chunk(buf.data() + kChunkHeaderSize, chunk_size);
    if (state.present[chunk_index]) {
      if (state.chunks[chunk_index] != chunk) {
        sock.partial.erase(it);
        return reject("conflicting duplicate");
      }
      state.updated_at = now;
      return false;
    }
    if (state.size + chunk.size() > kMaxLogicalPayload) {
      sock.partial.erase(it);
      return reject("logical payload limit exceeded");
    }
    state.chunks[chunk_index] = chunk;
    state.present[chunk_index] = true;
    state.size += chunk.size();
    state.updated_at = now;
    if (!std::all_of(state.present.begin(), state.present.end(),
                     [](bool present) { return present; })) {
      return false;
    }
    payload.reserve(state.size);
    for (const auto& part : state.chunks) {
      payload += part;
    }
    sock.partial.erase(it);
    if (payload.size() <= kMaxDatagramPayload || payload.size() > kMaxLogicalPayload) {
      return reject("logical payload has an invalid size");
    }
  }

  MetadataJsonMessage msg;
  msg.port = sock.port;
  std::string parse_err;
  if (!is_valid_metadata_json(payload, opt_.metadata_type, opt_.data_array_key, msg, parse_err)) {
    if (result.error.empty()) {
      result.error = "invalid json on port " + std::to_string(sock.port) + ": " + parse_err;
    }
    return false;
  }

  const bool meets_data_requirement = msg.object_count >= opt_.min_data_items_per_port;
  result.messages.push_back(std::move(msg));
  if (meets_data_requirement &&
      std::find(result.ports_with_valid_json.begin(), result.ports_with_valid_json.end(),
                sock.port) == result.ports_with_valid_json.end()) {
    result.ports_with_valid_json.push_back(sock.port);
  }
  return true;
}

bool MetadataJsonListener::success_reached(const MetadataJsonListenerResult& result) const {
  if (opt_.require_all_ports) {
    return static_cast<int>(result.ports_with_valid_json.size()) >= opt_.num_ports;
  }
  return !result.ports_with_valid_json.empty();
}

MetadataJsonListenerResult MetadataJsonListener::wait_for_messages() {
  MetadataJsonListenerResult result;
  if (!ok()) {
    result.error = err_;
    return result;
  }

  std::vector<pollfd> pfds;
  pfds.reserve(sockets_.size());
  for (const auto& sock : sockets_) {
    pfds.push_back(pollfd{sock.fd, POLLIN, 0});
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(opt_.timeout_ms);
  // Poll all sockets from a single loop so the same utility scales from
  // single-port smoke tests to larger multi-port metadata e2e checks.
  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    const int poll_ms =
        static_cast<int>(std::min<int64_t>(250, std::max<int64_t>(1, remaining.count())));
    const int rc = ::poll(pfds.data(), pfds.size(), poll_ms);
    if (rc < 0) {
      result.error = std::string("poll failed: ") + std::strerror(errno);
      return result;
    }
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
