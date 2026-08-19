#pragma once

#include <string>

namespace sima_examples {

inline std::string build_ffprobe_rtsp_stream_info_command(const std::string& url,
                                                          bool rtsp_tcp) {
  std::string quoted_url = "'";
  for (const char c : url) {
    quoted_url += c == '\'' ? "'\\''" : std::string(1, c);
  }
  quoted_url += "'";

  return "ffprobe -v error " + std::string(rtsp_tcp ? "-rtsp_transport tcp " : "") +
         "-rw_timeout 5000000 -select_streams v:0 "
         "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 " +
         quoted_url + " 2>/dev/null";
}

} // namespace sima_examples
