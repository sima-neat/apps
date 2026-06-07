#include "support/testing/test_config.h"

#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace sima_examples::testing {
namespace {

namespace fs = std::filesystem;

std::map<std::string, ScalarConfig>& config_cache() {
  static std::map<std::string, ScalarConfig> cache;
  return cache;
}

std::string full_key(const std::string& example_name, const std::string& section,
                     const std::string& key) {
  (void)example_name;
  return section + "." + key;
}

void require_present(const ScalarConfig& config, const std::string& key) {
  if (!config.string_value(key).has_value()) {
    throw std::runtime_error("common/config.yaml missing required key: " + key);
  }
}

std::string example_key_for_name(const std::string& example_name) {
  const fs::path common_config = example_common_config_path(example_name);
  const fs::path example_dir = common_config.parent_path().parent_path();
  return example_dir.parent_path().filename().string() + "/" + example_dir.filename().string();
}

std::string scoped_model_file_from_env(const std::string& example_name) {
  const char* raw = std::getenv("SIMANEAT_APPS_TEST_MODEL_FILES");
  if (!raw || !*raw) {
    return {};
  }

  const std::string target_example = example_key_for_name(example_name);
  std::istringstream lines(raw);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string example_key = line.substr(0, separator);
    if (example_key == target_example) {
      return sima_examples::trim_copy(line.substr(separator + 1));
    }
  }
  return {};
}

} // namespace

std::filesystem::path example_common_config_path(const std::string& example_name) {
  const char* apps_root = std::getenv("APPS_ROOT");
  const fs::path root =
      (apps_root && *apps_root) ? fs::path(apps_root) / "examples" : fs::path("examples");
  if (!fs::exists(root)) {
    throw std::runtime_error("examples directory not found: " + root.string());
  }
  std::vector<fs::path> matches;
  for (const auto& category : fs::directory_iterator(root)) {
    if (!category.is_directory()) {
      continue;
    }
    const fs::path candidate = category.path() / example_name / "common" / "config.yaml";
    if (fs::exists(candidate)) {
      matches.push_back(candidate);
    }
  }
  if (matches.size() != 1) {
    throw std::runtime_error("expected one common/config.yaml for example '" + example_name +
                             "', found " + std::to_string(matches.size()));
  }
  return matches.front();
}

const ScalarConfig& example_common_config(const std::string& example_name) {
  auto& cache = config_cache();
  auto it = cache.find(example_name);
  if (it == cache.end()) {
    it = cache.emplace(example_name, ScalarConfig::load(example_common_config_path(example_name)))
             .first;
  }
  return it->second;
}

std::string configured_model_path(const std::string& example_name, const std::string& models_dir) {
  if (const std::string scoped = scoped_model_file_from_env(example_name); !scoped.empty()) {
    return (fs::path(models_dir) / fs::path(scoped).filename()).string();
  }

  const ScalarConfig& config = example_common_config(example_name);
  std::string configured = config.string_or("model.path", "");
  if (configured.empty()) {
    configured = config.string_or("model", "");
  }
  if (configured.empty()) {
    return {};
  }
  return (fs::path(models_dir) / fs::path(configured).filename()).string();
}

double e2e_double(const std::string& example_name, const std::string& section,
                  const std::string& key) {
  const std::string path = full_key(example_name, section, key);
  const ScalarConfig& config = example_common_config(example_name);
  require_present(config, path);
  return config.double_or(path, 0.0);
}

int e2e_int(const std::string& example_name, const std::string& section, const std::string& key) {
  const std::string path = full_key(example_name, section, key);
  const ScalarConfig& config = example_common_config(example_name);
  require_present(config, path);
  return config.int_or(path, 0);
}

bool e2e_bool(const std::string& example_name, const std::string& section, const std::string& key,
              bool default_value) {
  return example_common_config(example_name)
      .bool_or(full_key(example_name, section, key), default_value);
}

} // namespace sima_examples::testing
