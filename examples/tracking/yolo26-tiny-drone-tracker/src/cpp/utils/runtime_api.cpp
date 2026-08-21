#pragma once

#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace yolo26_tiny_drone_tracker {

inline int configure_output_fps(simaai::neat::nodes::groups::RtspDecodedInputOptions& options,
                                int source_fps, int requested_fps) {
  const int output_fps = requested_fps > 0 ? requested_fps : source_fps;
  options.use_videorate = output_fps != source_fps;
  options.video_rate_fps = options.use_videorate ? output_fps : -1;
  options.output_caps.fps = output_fps;
  return output_fps;
}

inline bool
output_caps_enabled(const simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps& caps) {
  return caps.enable || caps.width > 0 || caps.height > 0 || caps.fps > 0;
}

inline void
append_decode_output_nodes(simaai::neat::Graph& graph,
                           const simaai::neat::nodes::groups::RtspDecodedInputOptions& options) {
  if (options.use_videoconvert) {
    graph.add(simaai::neat::nodes::VideoConvert());
  }
  if (options.use_videorate) {
    graph.add(simaai::neat::nodes::VideoRate());
  }
  if (options.use_videoscale) {
    graph.add(simaai::neat::nodes::VideoScale());
  }
  if (output_caps_enabled(options.output_caps)) {
    const auto& caps = options.output_caps;
    graph.add(
        simaai::neat::nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }
  if (!options.extra_fragment.empty()) {
    graph.add(simaai::neat::nodes::Custom(options.extra_fragment));
  }
}

inline bool pull_status_has_sample(simaai::neat::PullStatus status, const std::string& output_name,
                                   const simaai::neat::PullError& pull_error,
                                   const std::string& run_error) {
  if (status == simaai::neat::PullStatus::Timeout) {
    return false;
  }
  if (status == simaai::neat::PullStatus::Closed) {
    throw std::runtime_error(output_name + " output closed unexpectedly" +
                             (run_error.empty() ? std::string{} : ": " + run_error));
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
  }
  return true;
}

inline bool sample_identities_correlate(int64_t first_frame_id, int64_t first_pts_ns,
                                        int64_t second_frame_id, int64_t second_pts_ns) {
  if (first_frame_id >= 0 || second_frame_id >= 0) {
    return first_frame_id >= 0 && second_frame_id >= 0 && first_frame_id == second_frame_id;
  }
  return first_pts_ns >= 0 && second_pts_ns >= 0 && first_pts_ns == second_pts_ns;
}

inline bool samples_correlate(const simaai::neat::Sample& first,
                              const simaai::neat::Sample& second) {
  return sample_identities_correlate(first.frame_id, first.pts_ns, second.frame_id, second.pts_ns);
}

} // namespace yolo26_tiny_drone_tracker
