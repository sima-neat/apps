#include "support/testing/test_config.h"

#include <cstdlib>
#include <fstream>
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
    throw std::runtime_error("src/common/config.yaml missing required key: " + key);
  }
}

std::string example_key_for_name(const std::string& example_name) {
  const fs::path common_config = example_common_config_path(example_name);
  const fs::path example_dir = common_config.parent_path().parent_path().parent_path();
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

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::vector<std::string> split_dot_key(const std::string& key) {
  std::vector<std::string> parts;
  std::istringstream stream(key);
  std::string part;
  while (std::getline(stream, part, '.')) {
    if (part.empty()) {
      throw std::runtime_error("invalid empty config key segment in: " + key);
    }
    parts.push_back(part);
  }
  return parts;
}

std::string quote_scalar(const std::string& value) {
  std::string quoted = "'";
  for (const char c : value) {
    if (c == '\'') {
      quoted += "''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted += "'";
  return quoted;
}

struct ConfigNode {
  std::map<std::string, ConfigNode> children;
  std::vector<std::string> list;
  std::string value;
  bool has_value = false;
};

ConfigNode& node_for_key(ConfigNode& root, const std::string& key) {
  ConfigNode* node = &root;
  for (const auto& part : split_dot_key(key)) {
    node = &node->children[part];
  }
  return *node;
}

void write_node(std::ostream& out, const ConfigNode& node, int indent) {
  const std::string spaces(static_cast<std::size_t>(indent), ' ');
  for (const auto& [key, child] : node.children) {
    if (!child.list.empty()) {
      out << spaces << key << ":\n";
      for (const auto& value : child.list) {
        out << spaces << "  - " << quote_scalar(value) << "\n";
      }
    } else if (child.has_value) {
      out << spaces << key << ": " << quote_scalar(child.value) << "\n";
    } else {
      out << spaces << key << ":\n";
      write_node(out, child, indent + 2);
    }
  }
}

std::map<std::string, std::string> runtime_scalars_for_e2e(const ScalarConfig& common) {
  std::map<std::string, std::string> values = common.scalars();
  std::map<std::string, std::string> e2e_overrides;

  for (const auto& [key, value] : values) {
    constexpr const char* prefix = "testing.e2e.";
    if (!starts_with(key, prefix)) {
      continue;
    }

    const std::string runtime_key = key.substr(std::string(prefix).size());
    if (runtime_key == "output.total_saved_frames") {
      continue;
    }
    e2e_overrides[runtime_key] = value;
  }

  for (auto it = values.begin(); it != values.end();) {
    if (starts_with(it->first, "testing.")) {
      it = values.erase(it);
    } else {
      ++it;
    }
  }

  for (const auto& [key, value] : e2e_overrides) {
    values[key] = value;
  }
  return values;
}

void write_nested_config(std::ostream& out, const ConfigScalars& values,
                         const ConfigLists& list_values) {
  ConfigNode root;
  for (const auto& [key, value] : values) {
    ConfigNode& node = node_for_key(root, key);
    node.value = value;
    node.has_value = true;
    node.children.clear();
    node.list.clear();
  }
  for (const auto& [key, values] : list_values) {
    ConfigNode& node = node_for_key(root, key);
    node.list = values;
    node.has_value = false;
    node.children.clear();
  }
  write_node(out, root, 0);
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
    const fs::path candidate = category.path() / example_name / "src" / "common" / "config.yaml";
    if (fs::exists(candidate)) {
      matches.push_back(candidate);
    }
  }
  if (matches.size() != 1) {
    throw std::runtime_error("expected one src/common/config.yaml for example '" + example_name +
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

std::filesystem::path write_e2e_config(const std::string& example_name,
                                       const std::filesystem::path& config_path,
                                       const ConfigScalars& overrides,
                                       const ConfigLists& list_overrides) {
  std::map<std::string, std::string> values =
      runtime_scalars_for_e2e(example_common_config(example_name));
  for (const auto& [key, value] : overrides) {
    values[key] = value;
  }
  for (const auto& [key, list] : list_overrides) {
    static_cast<void>(list);
    values.erase(key);
  }

  std::filesystem::create_directories(config_path.parent_path());
  std::ofstream out(config_path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to write e2e config: " + config_path.string());
  }
  write_nested_config(out, values, list_overrides);
  return config_path;
}

} // namespace sima_examples::testing
