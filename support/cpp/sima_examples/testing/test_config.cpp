#include "sima_examples/testing/test_config.h"

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <map>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace sima_examples::testing {
namespace {

namespace fs = std::filesystem;

std::map<std::string, Config>& config_cache() {
  static std::map<std::string, Config> cache;
  return cache;
}

std::string full_key(const std::string& example_name, const std::string& section,
                     const std::string& key) {
  (void)example_name;
  return section + "." + key;
}

void require_present(const Config& config, const std::string& key) {
  if (!config.string_value(key).has_value()) {
    throw std::runtime_error("src/common/config.yaml missing required key: " + key);
  }
}

std::string example_key_for_name(const std::string& example_name) {
  const fs::path common_config = example_common_config_path(example_name);
  const fs::path example_dir = common_config.parent_path().parent_path().parent_path();
  return example_dir.parent_path().filename().string() + "/" + example_dir.filename().string();
}

// Every model file the scope lists for this example, in scope order: the suite's model
// first, then any variants the run was asked to cover.
std::vector<std::string> scoped_model_files_from_env(const std::string& example_name) {
  std::vector<std::string> files;
  const char* raw = std::getenv("SIMANEAT_APPS_TEST_MODEL_FILES");
  if (!raw || !*raw) {
    return files;
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
      const std::string file = sima_examples::trim_copy(line.substr(separator + 1));
      if (!file.empty()) {
        files.push_back(file);
      }
    }
  }
  return files;
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
  std::vector<ConfigScalars> entries;
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

void write_node(std::ostream& out, const ConfigNode& node, int indent);

// One "- field: value" item per entry, its fields under the dash.
void write_entries(std::ostream& out, const std::vector<ConfigScalars>& entries, int indent) {
  for (const auto& entry : entries) {
    ConfigNode fields;
    for (const auto& [key, value] : entry) {
      ConfigNode& field = node_for_key(fields, key);
      field.value = value;
      field.has_value = true;
    }
    std::ostringstream text;
    write_node(text, fields, indent + 2);
    std::string lines = text.str();
    if (lines.size() > static_cast<std::size_t>(indent)) {
      lines[static_cast<std::size_t>(indent)] = '-';
    }
    out << lines;
  }
}

void write_node(std::ostream& out, const ConfigNode& node, int indent) {
  const std::string spaces(static_cast<std::size_t>(indent), ' ');
  for (const auto& [key, child] : node.children) {
    if (!child.entries.empty()) {
      out << spaces << key << ":\n";
      write_entries(out, child.entries, indent + 2);
    } else if (!child.list.empty()) {
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

std::map<std::string, std::string> runtime_scalars_for_e2e(const Config& common) {
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
                         const ConfigLists& list_values, const ConfigEntries& entry_values) {
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
  for (const auto& [key, entries] : entry_values) {
    ConfigNode& node = node_for_key(root, key);
    node.entries = entries;
    node.list.clear();
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

const Config& example_common_config(const std::string& example_name) {
  auto& cache = config_cache();
  auto it = cache.find(example_name);
  if (it == cache.end()) {
    it = cache.emplace(example_name, Config::load(example_common_config_path(example_name))).first;
  }
  return it->second;
}

std::vector<std::string> configured_model_paths(const std::string& example_name,
                                                const std::string& models_dir) {
  std::vector<std::string> paths;
  for (const std::string& scoped : scoped_model_files_from_env(example_name)) {
    paths.push_back((fs::path(models_dir) / fs::path(scoped).filename()).string());
  }
  if (!paths.empty()) {
    return paths;
  }

  const Config& config = example_common_config(example_name);
  std::string configured = config.string_or("model.path", "");
  if (configured.empty()) {
    configured = config.string_or("model", "");
  }
  if (!configured.empty()) {
    paths.push_back((fs::path(models_dir) / fs::path(configured).filename()).string());
  }
  return paths;
}

std::string configured_model_path(const std::string& example_name, const std::string& models_dir) {
  const std::vector<std::string> paths = configured_model_paths(example_name, models_dir);
  return paths.empty() ? std::string{} : paths.front();
}

std::string model_case_label(const std::string& case_name, const std::string& model_path,
                             std::size_t model_count) {
  if (model_count < 2) {
    return case_name;
  }
  std::string model_id = fs::path(model_path).filename().string();
  const std::string archive = ".tar.gz";
  if (model_id.size() > archive.size() &&
      model_id.compare(model_id.size() - archive.size(), archive.size(), archive) == 0) {
    model_id.erase(model_id.size() - archive.size());
  }
  return case_name + "_" + model_id;
}

int e2e_int(const std::string& example_name, const std::string& section, const std::string& key) {
  const std::string path = full_key(example_name, section, key);
  const Config& config = example_common_config(example_name);
  require_present(config, path);
  return config.int_or(path, 0);
}

std::filesystem::path write_e2e_config(const std::string& example_name,
                                       const std::filesystem::path& config_path,
                                       const ConfigScalars& overrides,
                                       const ConfigLists& list_overrides,
                                       const ConfigEntries& entry_overrides) {
  std::map<std::string, std::string> values =
      runtime_scalars_for_e2e(example_common_config(example_name));
  for (const auto& [key, value] : overrides) {
    values[key] = value;
  }
  for (const auto& [key, list] : list_overrides) {
    static_cast<void>(list);
    values.erase(key);
  }
  for (const auto& [key, entries] : entry_overrides) {
    static_cast<void>(entries);
    // The entries replace a mapping at the same key, such as a single model.path.
    for (auto it = values.begin(); it != values.end();) {
      it = it->first == key || starts_with(it->first, key + ".") ? values.erase(it) : std::next(it);
    }
  }

  std::filesystem::create_directories(config_path.parent_path());
  std::ofstream out(config_path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to write e2e config: " + config_path.string());
  }
  write_nested_config(out, values, list_overrides, entry_overrides);
  return config_path;
}

} // namespace sima_examples::testing
