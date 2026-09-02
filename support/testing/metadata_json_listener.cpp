#include "support/testing/metadata_json_listener.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <map>
#include <netdb.h>
#include <optional>
#include <poll.h>
#include <set>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace sima_examples::testing {
namespace {

using json = nlohmann::json;

constexpr uint8_t kMetadataChunkMagic = 0x4e;
constexpr uint8_t kMetadataChunkVersion = 0x01;
constexpr size_t kMetadataChunkHeaderSize = 12;
constexpr size_t kMetadataMaxDatagramSize = 1200;
constexpr size_t kMetadataMaxLogicalMessageSize = 65507;
constexpr size_t kMetadataMaxChunkCount = 56;
constexpr size_t kMetadataReassemblyCapacity = 4;
constexpr auto kMetadataReassemblyMaxAge = std::chrono::milliseconds(250);

struct MetadataReassemblyResult {
  std::string payload;
  bool complete = false;
  std::string error;
};

class MetadataReassembler {
public:
  MetadataReassemblyResult accept(std::string_view datagram) {
    const auto now = std::chrono::steady_clock::now();
    drop_expired(now);
    if (datagram.empty() || static_cast<uint8_t>(datagram[0]) != kMetadataChunkMagic) {
      return {std::string(datagram), true, {}};
    }
    if (datagram.size() < kMetadataChunkHeaderSize || datagram.size() > kMetadataMaxDatagramSize ||
        static_cast<uint8_t>(datagram[1]) != kMetadataChunkVersion) {
      return {{}, false, "invalid metadata chunk header"};
    }

    uint64_t message_id = 0;
    for (size_t i = 2; i < 10; ++i) {
      message_id = (message_id << 8) | static_cast<uint8_t>(datagram[i]);
    }
    const size_t index = static_cast<uint8_t>(datagram[10]);
    const size_t count = static_cast<uint8_t>(datagram[11]);
    if (count == 0 || count > kMetadataMaxChunkCount || index >= count ||
        datagram.size() == kMetadataChunkHeaderSize) {
      return {{}, false, "invalid metadata chunk fields"};
    }

    auto it = assemblies_.find(message_id);
    if (it == assemblies_.end()) {
      if (assemblies_.size() == kMetadataReassemblyCapacity) {
        drop_oldest();
      }
      Assembly assembly;
      assembly.chunks.resize(count);
      assembly.started = now;
      it = assemblies_.emplace(message_id, std::move(assembly)).first;
    }

    Assembly& assembly = it->second;
    if (assembly.chunks.size() != count) {
      assemblies_.erase(it);
      return {{}, false, "metadata chunk count changed"};
    }

    const std::string fragment(datagram.substr(kMetadataChunkHeaderSize));
    if (!assembly.chunks[index].has_value()) {
      if (assembly.size + fragment.size() > kMetadataMaxLogicalMessageSize) {
        assemblies_.erase(it);
        return {{}, false, "metadata message exceeds maximum size"};
      }
      assembly.chunks[index] = fragment;
      assembly.size += fragment.size();
      ++assembly.received;
    } else if (*assembly.chunks[index] != fragment) {
      assemblies_.erase(it);
      return {{}, false, "metadata chunk contents changed"};
    }

    if (assembly.received != assembly.chunks.size()) {
      return {};
    }

    std::string payload;
    payload.reserve(assembly.size);
    for (const auto& chunk : assembly.chunks) {
      payload += *chunk;
    }
    assemblies_.erase(it);
    return {std::move(payload), true, {}};
  }

private:
  using TimePoint = std::chrono::steady_clock::time_point;

  struct Assembly {
    std::vector<std::optional<std::string>> chunks;
    size_t received = 0;
    size_t size = 0;
    TimePoint started;
  };

  void drop_expired(TimePoint now) {
    for (auto it = assemblies_.begin(); it != assemblies_.end();) {
      if (now - it->second.started > kMetadataReassemblyMaxAge) {
        it = assemblies_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void drop_oldest() {
    const auto oldest = std::min_element(
        assemblies_.begin(), assemblies_.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second.started < rhs.second.started; });
    if (oldest != assemblies_.end()) {
      assemblies_.erase(oldest);
    }
  }

  std::map<uint64_t, Assembly> assemblies_;
};

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
  int fd = -1;
  int port = -1;
  MetadataReassembler reassembler;
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
  if (opt_.min_object_count < 0) {
    err_ = "min_object_count must be >= 0";
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

    sockets_.push_back(SocketState{fd, port, {}});
  }

  return true;
}

bool MetadataJsonListener::handle_datagram(SocketState& sock, MetadataJsonListenerResult& result) {
  char buf[65536];
  const ssize_t n = ::recv(sock.fd, buf, sizeof(buf), 0);
  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }
    result.error = "recv failed on port " + std::to_string(sock.port) + ": " + std::strerror(errno);
    return false;
  }

  const auto reassembled = sock.reassembler.accept(std::string_view(buf, static_cast<size_t>(n)));
  if (!reassembled.error.empty()) {
    if (result.error.empty()) {
      result.error =
          "invalid metadata on port " + std::to_string(sock.port) + ": " + reassembled.error;
    }
    return false;
  }
  if (!reassembled.complete) {
    return false;
  }

  MetadataJsonMessage msg;
  msg.port = sock.port;
  std::string parse_err;
  if (!is_valid_metadata_json(reassembled.payload, opt_.metadata_type, opt_.data_array_key, msg,
                              parse_err)) {
    if (result.error.empty()) {
      result.error = "invalid json on port " + std::to_string(sock.port) + ": " + parse_err;
    }
    return false;
  }

  result.messages.push_back(std::move(msg));
  if (result.messages.back().object_count < opt_.min_object_count) {
    result.error =
        "data." + opt_.data_array_key + " contains " +
        std::to_string(result.messages.back().object_count) + " objects; expected at least " +
        std::to_string(opt_.min_object_count);
    return true;
  }
  if (std::find(result.ports_with_valid_json.begin(), result.ports_with_valid_json.end(),
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
  while (true) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      break;
    }
    const int poll_ms = static_cast<int>(std::min<int64_t>(250, remaining.count()));
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
