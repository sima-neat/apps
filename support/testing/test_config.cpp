#include "support/testing/test_config.h"

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <map>
#include <stdexcept>
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

int leading_spaces(const std::string& line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) &&
         (line[static_cast<std::size_t>(indent)] == ' ' ||
          line[static_cast<std::size_t>(indent)] == '\t')) {
    ++indent;
  }
  return indent;
}

std::string strip_comment(const std::string& line) {
  const std::size_t comment = line.find('#');
  return comment == std::string::npos ? line : line.substr(0, comment);
}

std::string scalar_after_colon(const std::string& line) {
  const std::size_t colon = line.find(':');
  if (colon == std::string::npos) {
    return {};
  }
  return sima_examples::trim_copy(line.substr(colon + 1));
}

fs::path default_scope_file() {
  if (const char* scope_file = std::getenv("SIMANEAT_APPS_TEST_SCOPE_FILE");
      scope_file && *scope_file) {
    return scope_file;
  }
  if (const char* apps_root = std::getenv("APPS_ROOT"); apps_root && *apps_root) {
    return fs::path(apps_root) / "tests" / "configs" / "test-scope.yaml";
  }
  return fs::path("tests") / "configs" / "test-scope.yaml";
}

std::string example_key_for_name(const std::string& example_name) {
  const fs::path common_config = example_common_config_path(example_name);
  const fs::path example_dir = common_config.parent_path().parent_path();
  return example_dir.parent_path().filename().string() + "/" + example_dir.filename().string();
}

std::string scoped_model_file(const std::string& example_name) {
  const fs::path scope_file = default_scope_file();
  std::ifstream input(scope_file);
  if (!input.is_open()) {
    return {};
  }

  const std::string target_example = example_key_for_name(example_name) + ":";
  bool in_example = false;
  bool in_models = false;
  bool in_e2e = false;
  bool in_cpp = false;
  bool in_cpp_models = false;
  std::string current_model;
  std::string selected_model;
  std::map<std::string, std::string> model_files;

  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const int indent = leading_spaces(raw_line);
    const std::string line = sima_examples::trim_copy(strip_comment(raw_line));
    if (line.empty()) {
      continue;
    }

    if (!in_example) {
      if (indent == 2 && line == target_example) {
        in_example = true;
      }
      continue;
    }
    if (indent <= 2 && line != target_example && !line.empty() && line.back() == ':') {
      break;
    }

    if (indent == 4) {
      in_models = line == "models:";
      in_e2e = line == "e2e:";
      in_cpp = false;
      in_cpp_models = false;
      continue;
    }

    if (in_models && indent == 6 && line.back() == ':') {
      current_model = line.substr(0, line.size() - 1);
      continue;
    }
    if (in_models && indent == 8 && line.rfind("file:", 0) == 0 && !current_model.empty()) {
      model_files[current_model] = scalar_after_colon(line);
      continue;
    }

    if (in_e2e && indent == 6) {
      in_cpp = line == "cpp:";
      in_cpp_models = false;
      continue;
    }
    if (in_cpp && indent == 8) {
      in_cpp_models = line == "models:";
      continue;
    }
    if (in_cpp_models && indent == 10 && line.rfind("- ", 0) == 0 && selected_model.empty()) {
      selected_model = sima_examples::trim_copy(line.substr(2));
    }
  }

  if (!selected_model.empty()) {
    const auto selected = model_files.find(selected_model);
    if (selected != model_files.end()) {
      return selected->second;
    }
  }
  return model_files.size() == 1 ? model_files.begin()->second : "";
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
  if (const std::string scoped = scoped_model_file(example_name); !scoped.empty()) {
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
