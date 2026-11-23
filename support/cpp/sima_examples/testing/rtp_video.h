#pragma once

// Receive an example's Insight video as RTP and check that its metadata pairs with the frames.
// Mirrors tests/utils/rtp_video.py, so the C++ and Python end-to-end tests check video the same
// way.

#include <array>
#include <atomic>
#include <cstdint>
#include <set>
#include <vector>

namespace sima_examples::testing {

using RtpHeader = std::array<unsigned char, 12>;

// Each video packet's RTP header, kept until `stop` is set and the socket is empty.
std::vector<RtpHeader> drain_video(int socket, const std::atomic<bool>& stop);

// The RTP timestamp of each kept header.
std::set<uint32_t> rtp_timestamps(const std::vector<RtpHeader>& headers);

// Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a video
// frame's RTP timestamp. Both wrap at 2^32.
bool pairs(int64_t timestamp_ms, const std::set<uint32_t>& timestamps);

} // namespace sima_examples::testing
