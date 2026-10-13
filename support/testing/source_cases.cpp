#include "support/testing/source_cases.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>

namespace sima_examples::testing {

void record_unavailable_case(const std::string& fail_reason, const std::string& skip_reason,
                             int& rc) {
  if (require_e2e_mode()) {
    std::cerr << "[FAIL] " << fail_reason << "\n";
    rc = 1;
  } else {
    std::cerr << "[SKIP] " << skip_reason << "\n";
  }
}

std::vector<std::string> available_model_paths(const std::vector<std::string>& model_paths,
                                               int& rc) {
  std::vector<std::string> available;
  for (const std::string& model_path : model_paths) {
    if (std::filesystem::exists(model_path)) {
      available.push_back(model_path);
      continue;
    }
    const std::string model_name = std::filesystem::path(model_path).filename().string();
    record_unavailable_case(model_name + " is scoped for e2e but missing under " +
                                std::filesystem::path(model_path).parent_path().string(),
                            "download " + model_name + " to run its e2e cases", rc);
  }
  return available;
}

namespace {

int finish(const std::string& nothing_ran_reason, int cases_run, int rc) {
  if (cases_run == 0) {
    return skip_or_fail(nothing_ran_reason);
  }
  return rc;
}

} // namespace

int run_single_stream_source_cases(
    const std::string& suite_label, const std::vector<StreamSourceCase>& cases,
    const std::function<int(const StreamSourceCase&, const std::string& url)>& run_case) {
  int cases_run = 0;
  int rc = 0;
  for (const StreamSourceCase& source_case : cases) {
    const char* source_url = env_or_null(source_case.env_key.c_str());
    if (!source_url) {
      record_unavailable_case(source_case.env_key + " is required for " + source_case.name + " e2e",
                              "set " + source_case.env_key + " to run " + source_case.name + " e2e",
                              rc);
      continue;
    }
    ++cases_run;
    if (run_case(source_case, source_url) != 0) {
      rc = 1;
    }
  }
  return finish("no " + suite_label + " source e2e URLs configured", cases_run, rc);
}

int run_multistream_source_cases(const std::string& suite_label,
                                 const std::vector<MultiStreamSourceCase>& cases,
                                 std::size_t min_urls,
                                 const std::function<int(const MultiStreamSourceCase&)>& run_case) {
  int cases_run = 0;
  int rc = 0;
  const std::string count = std::to_string(min_urls);
  for (const MultiStreamSourceCase& source_case : cases) {
    if (source_case.urls.size() < min_urls) {
      record_unavailable_case("need at least " + count + " RTSP " + source_case.codec +
                                  " URLs for multistream e2e",
                              "set at least " + count + " RTSP " + source_case.codec +
                                  " URLs to run " + source_case.codec + " multistream e2e",
                              rc);
      continue;
    }
    ++cases_run;
    if (run_case(source_case) != 0) {
      rc = 1;
    }
  }
  return finish("no " + suite_label + " RTSP e2e URLs configured", cases_run, rc);
}

} // namespace sima_examples::testing
