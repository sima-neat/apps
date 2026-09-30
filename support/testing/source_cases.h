#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace sima_examples::testing {

// One streaming source a single-stream e2e suite exercises: the environment variable that
// carries its URL and the config values that select it.
struct StreamSourceCase {
  std::string name;
  std::string env_key;
  std::string type;
  std::string codec;
  int fps = 0;
  bool ssl_strict = true;
};

// One multistream source: a codec and the URLs its plural environment variable lists.
struct MultiStreamSourceCase {
  std::string codec;
  std::vector<std::string> urls;
};

// Reports a case that cannot run: a failure under strict e2e mode, otherwise a skip.
void record_unavailable_case(const std::string& fail_reason, const std::string& skip_reason,
                             int& rc);

// Runs `run_case` for every case whose URL is set, reports the ones that are not, and turns
// "nothing ran" into the suite's skip-or-fail exit; `suite_label` names the suite in that
// message. Returns the exit code for main.
int run_single_stream_source_cases(
    const std::string& suite_label, const std::vector<StreamSourceCase>& cases,
    const std::function<int(const StreamSourceCase&, const std::string& url)>& run_case);

// The multistream twin: a case runs when it has at least `min_urls` URLs.
int run_multistream_source_cases(const std::string& suite_label,
                                 const std::vector<MultiStreamSourceCase>& cases,
                                 std::size_t min_urls,
                                 const std::function<int(const MultiStreamSourceCase&)>& run_case);

} // namespace sima_examples::testing
