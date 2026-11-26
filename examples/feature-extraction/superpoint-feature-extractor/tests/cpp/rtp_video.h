#pragma once

// Receive the example's Insight video as RTP and check that its metadata pairs with the frames.
// Mirrors tests/python/rtp_video.py, so the C++ and Python end-to-end tests check video the same
// way.

#include <sys/socket.h>
#include <sys/time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <set>
#include <vector>

namespace superpoint::testing {

using RtpHeader = std::array<unsigned char, 12>;

// Each video packet's RTP header, kept until `stop` is set and the socket is empty.
inline std::vector<RtpHeader> drain_video(int socket, const std::atomic<bool>& stop) {
  const timeval timeout{0, 200000};
  ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  std::vector<RtpHeader> headers;
  std::array<unsigned char, 65536> packet{};
  while (true) {
    if (::recv(socket, packet.data(), packet.size(), 0) >= 12) {
      headers.emplace_back();
      std::copy_n(packet.begin(), 12, headers.back().begin());
    } else if (stop) {
      return headers;
    }
  }
}

// The RTP timestamp of each kept header.
inline std::set<uint32_t> rtp_timestamps(const std::vector<RtpHeader>& headers) {
  std::set<uint32_t> timestamps;
  for (const auto& header : headers) {
    timestamps.insert(static_cast<uint32_t>(header[4]) << 24 | header[5] << 16 | header[6] << 8 |
                      header[7]);
  }
  return timestamps;
}

// Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a video
// frame's RTP timestamp. Both wrap at 2^32.
inline bool pairs(int64_t timestamp_ms, const std::set<uint32_t>& timestamps) {
  const auto ticks = static_cast<uint32_t>(timestamp_ms * 90);
  return std::any_of(timestamps.begin(), timestamps.end(), [ticks](uint32_t rtp) {
    // Unsigned differences wrap at 2^32, as the RTP clock does.
    return std::min(ticks - rtp, rtp - ticks) <= 90U;
  });
}

} // namespace superpoint::testing
