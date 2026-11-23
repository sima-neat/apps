#include "sima_examples/testing/rtp_video.h"

#include <sys/socket.h>
#include <sys/time.h>

#include <algorithm>

namespace sima_examples::testing {

std::vector<RtpHeader> drain_video(int socket, const std::atomic<bool>& stop) {
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

std::set<uint32_t> rtp_timestamps(const std::vector<RtpHeader>& headers) {
  std::set<uint32_t> timestamps;
  for (const auto& header : headers) {
    timestamps.insert(static_cast<uint32_t>(header[4]) << 24 | header[5] << 16 | header[6] << 8 |
                      header[7]);
  }
  return timestamps;
}

bool pairs(int64_t timestamp_ms, const std::set<uint32_t>& timestamps) {
  const auto ticks = static_cast<uint32_t>(timestamp_ms * 90);
  return std::any_of(timestamps.begin(), timestamps.end(), [ticks](uint32_t rtp) {
    // Unsigned differences wrap at 2^32, as the RTP clock does.
    return std::min(ticks - rtp, rtp - ticks) <= 90U;
  });
}

} // namespace sima_examples::testing
