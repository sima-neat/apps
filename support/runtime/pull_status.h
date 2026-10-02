#pragma once

#include <neat.h>

#include <stdexcept>
#include <string>

namespace sima_examples {

// Interprets one pull from a run output. True means `status` delivered a sample; false
// means the pull timed out and the loop should try again. A closed output and a pull
// error are terminal: they throw, with the run's own error when it has one, because a
// loop that treats a closed output like a timeout waits forever on a dead source, and a
// loop that leaves quietly reports a dead source as a completed run.
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

} // namespace sima_examples
