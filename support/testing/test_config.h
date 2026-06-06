#pragma once

#include <string>

namespace sima_examples::testing {

// Read a required scalar from tests/configs/e2e.yaml under:
// e2e.<example_name>.<section>.<key>
double e2e_double(const std::string& example_name, const std::string& section,
                  const std::string& key);

int e2e_int(const std::string& example_name, const std::string& section, const std::string& key);

} // namespace sima_examples::testing
