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

#include <string>

namespace sima_examples::single_stream_object_detector {

struct FfprobeCommandOptions {
  bool rtsp_source = false;
  bool tcp = false;
  bool tls_verify = true;
};

inline std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    out += c == '\'' ? "'\\''" : std::string(1, c);
  }
  out += "'";
  return out;
}

inline std::string build_ffprobe_geometry_command(const std::string& url,
                                                  const FfprobeCommandOptions& options) {
  std::string command =
      "ffprobe -v error -rw_timeout 5000000 -select_streams v:0 "
      "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 ";
  if (options.rtsp_source && options.tcp) {
    command += "-rtsp_transport tcp ";
  }
  if (!options.tls_verify) {
    command += "-tls_verify 0 ";
  }
  return command + shell_quote(url) + " 2>/dev/null";
}

} // namespace sima_examples::single_stream_object_detector
