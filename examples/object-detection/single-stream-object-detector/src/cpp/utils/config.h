// Copied from support/templates/cpp/config.h;
// edit the template and run scripts/sync_utils.py.
#pragma once

// Read config.yaml and the data files it names.
//
// The SDK sysroot ships the yaml-cpp runtime library but not its headers, so ScalarConfig is a
// small reader for the YAML these configs use: nested mappings of scalars, with comments.
// Lists are skipped. Each key is its dotted path, such as "source.rtsp.url".

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {

class ScalarConfig {
public:
  static ScalarConfig load(const std::filesystem::path& path);

  // The value, or nullopt when the key is missing or null.
  [[nodiscard]] std::optional<std::string> string_value(const std::string& key) const;
  // Each *_or returns the default when the key is missing or null, and throws naming the key
  // when the value has the wrong type.
  [[nodiscard]] std::string string_or(const std::string& key,
                                      const std::string& default_value) const;
  [[nodiscard]] int int_or(const std::string& key, int default_value) const;
  [[nodiscard]] double double_or(const std::string& key, double default_value) const;
  [[nodiscard]] bool bool_or(const std::string& key, bool default_value) const;

private:
  std::unordered_map<std::string, std::string> scalars_;
};

// <source_dir>/../common/config.yaml, where source_dir is the example's src/cpp folder.
std::filesystem::path default_config_path(const char* source_dir);

// One class name per line; empty lines are skipped.
std::vector<std::string> load_labels(const std::filesystem::path& labels_path);

} // namespace utils
