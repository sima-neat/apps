#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace sima_examples {

// The values of one config.yaml, read by dotted key such as "input.rtsp.tcp". Scalars and
// lists of strings are read; string_list() rejects a list holding anything else.
class Config {
public:
  static Config load(const std::filesystem::path& path);

  [[nodiscard]] std::optional<std::string> string_value(const std::string& key) const;
  [[nodiscard]] std::string string_or(const std::string& key,
                                      const std::string& default_value) const;
  [[nodiscard]] int int_or(const std::string& key, int default_value) const;
  [[nodiscard]] double double_or(const std::string& key, double default_value) const;
  [[nodiscard]] bool bool_or(const std::string& key, bool default_value) const;
  [[nodiscard]] std::map<std::string, std::string> scalars() const;

  // Required values: each throws when the key is missing. Mirrors sima_examples.config.Config
  // in support/python.
  [[nodiscard]] std::string string(const std::string& key) const;
  [[nodiscard]] int integer(const std::string& key) const;
  [[nodiscard]] double number(const std::string& key) const;
  [[nodiscard]] bool boolean(const std::string& key) const;
  // A single value reads as a list of one.
  [[nodiscard]] std::vector<std::string> string_list(const std::string& key) const;

  // Throws on keys no getter read, such as a renamed or misspelled key.
  void reject_unread_keys() const;

private:
  std::string require_key(const std::string& key) const;

  std::filesystem::path path_;
  std::unordered_map<std::string, std::string> scalars_;
  std::unordered_map<std::string, std::vector<std::string>> lists_;
  std::set<std::string> lists_with_non_strings_;
  // Filled by the getters, so a const reader can still report what it never read.
  mutable std::set<std::string> read_keys_;
};

std::string trim_copy(const std::string& value);
std::filesystem::path default_config_path(const char* source_dir);

} // namespace sima_examples
