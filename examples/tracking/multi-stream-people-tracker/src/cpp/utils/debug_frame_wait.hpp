// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>

namespace multi_stream_people_tracker {

inline constexpr auto kRealtimeSideFrameWait = std::chrono::milliseconds(50);
inline constexpr auto kBenchmarkSideFrameDeadline = std::chrono::seconds(30);

template <typename Predicate>
bool wait_for_matching_side_frame(std::condition_variable& ready,
                                  std::unique_lock<std::mutex>& lock, bool benchmark_mode,
                                  Predicate&& has_matching_frame) {
  const auto timeout = benchmark_mode ? kBenchmarkSideFrameDeadline : kRealtimeSideFrameWait;
  return ready.wait_for(lock, timeout, std::forward<Predicate>(has_matching_frame));
}

} // namespace multi_stream_people_tracker
