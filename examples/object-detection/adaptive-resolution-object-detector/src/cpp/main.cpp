// Multi-stream RTSP YOLO26 object detection - one entry point, two topologies.
//
//   --mode adaptive  (default) ONE GRAPH PER STREAM. A stream is built or torn
//                    down while the others keep running, and each stream's
//                    delivered resolution comes from a shared output bandwidth
//                    budget. Per-stream bridges cap reliable metadata at about
//                    six streams.
//
//   --mode fused     ONE GRAPH for every stream, fanning into a single shared
//                    detector, with the source H.264 passed through to Insight
//                    untouched (no re-encode). Boxes stay correct at higher
//                    stream counts, but adding a stream rebuilds the graph.
//
// The two modes take DIFFERENT config schemas: `adaptive` reads
// `streams.sources` plus the `adaptive:` policy sections, `fused` reads a bare
// `streams:` list. Handing one mode the other's config fails validation instead
// of running with silently wrong settings.
//
// Mirrors src/python/main.py, which takes the same flags - that symmetry is what
// lets the pipelines chooser toggle language without changing anything else.
//
// Everything except --mode is forwarded unchanged to the selected mode.

#include "adaptive_app.h"
#include "fused_app.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage =
    "usage: adaptive-resolution-object-detector [--mode adaptive|fused] "
    "--config CONFIG [--validate-config-only]\n";

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "adaptive";
  bool want_help = false;
  std::vector<char*> forwarded;
  forwarded.reserve(static_cast<std::size_t>(argc));
  forwarded.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mode") {
      if (i + 1 >= argc) {
        std::cerr << "[ERR] --mode needs a value (adaptive|fused)\n" << kUsage;
        return 2;
      }
      mode = argv[++i];
      continue;
    }
    if (arg.rfind("--mode=", 0) == 0) {
      mode = arg.substr(std::strlen("--mode="));
      continue;
    }
    if (arg == "-h" || arg == "--help") {
      want_help = true;
      continue;
    }
    forwarded.push_back(argv[i]);
  }

  // Validate the mode BEFORE honouring --help, so `--mode nonsense --help` is an
  // error rather than a usage dump. src/python/main.py behaves the same way
  // (argparse rejects an invalid choice first) and the two must not drift: the
  // pipelines chooser assumes both entry points take identical flags.
  if (mode != "adaptive" && mode != "fused") {
    std::cerr << "[ERR] unknown --mode '" << mode << "' (adaptive|fused)\n" << kUsage;
    return 2;
  }

  if (want_help) {
    std::cout << kUsage;
    return 0;
  }

  const int forwarded_argc = static_cast<int>(forwarded.size());
  char** forwarded_argv = forwarded.data();

  return mode == "fused" ? fused_app::run(forwarded_argc, forwarded_argv)
                         : adaptive_app::run(forwarded_argc, forwarded_argv);
}
