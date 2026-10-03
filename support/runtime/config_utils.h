#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>

namespace sima_examples {

class ScalarConfig {
public:
  static ScalarConfig load(const std::filesystem::path& path);

  [[nodiscard]] std::optional<std::string> string_value(const std::string& key) const;
  [[nodiscard]] std::string string_or(const std::string& key,
                                      const std::string& default_value) const;
  [[nodiscard]] int int_or(const std::string& key, int default_value) const;
  [[nodiscard]] double double_or(const std::string& key, double default_value) const;
  [[nodiscard]] bool bool_or(const std::string& key, bool default_value) const;
  [[nodiscard]] std::map<std::string, std::string> scalars() const;

private:
  std::unordered_map<std::string, std::string> scalars_;
};

std::string trim_copy(const std::string& value);
std::filesystem::path default_config_path(const char* source_dir);

} // namespace sima_examples
