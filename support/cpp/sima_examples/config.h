#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sima_examples {

// table.at(value); a value the table lacks throws with key, the value, and the choices. Use it
// for a value main derives from the config; Config::choice reads the key itself. Mirrors
// sima_examples.config.lookup in support/python.
template <typename T>
const T& lookup(const std::string& key, const std::string& value,
                const std::map<std::string, T>& table) {
  const auto it = table.find(value);
  if (it == table.end()) {
    std::string expected;
    for (const auto& [name, entry] : table) {
      expected += (expected.empty() ? "" : ", ") + name;
    }
    throw std::runtime_error(key + ": unknown value '" + value + "' (expected " + expected + ")");
  }
  return it->second;
}

// The values of one config.yaml, read by dotted key such as "input.rtsp.tcp". A mapping_list()
// entry is a Config too, whose getters take a field and whose errors name the entry.
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
  // The table entry the string at key names, such as a codec name's Neat enum.
  template <typename T>
  [[nodiscard]] const T& choice(const std::string& key,
                                const std::map<std::string, T>& table) const {
    return lookup(prefix_ + key, string(key), table);
  }
  // A non-empty list of mappings, one Config per entry, such as model[1].
  [[nodiscard]] std::vector<Config> mapping_list(const std::string& key) const;

private:
  friend struct ConfigLoader;

  std::string require_key(const std::string& key) const;

  std::filesystem::path path_;
  // "model[1]." for an entry of mapping_list("model"), "" for the file itself.
  std::string prefix_;
  std::unordered_map<std::string, std::string> scalars_;
  std::unordered_map<std::string, std::vector<std::string>> lists_;
  std::set<std::string> lists_with_non_strings_;
  // Each list that holds a mapping, kept apart from scalars_ so scalars() lists none of them.
  std::map<std::string, std::vector<Config>> mapping_lists_;
  // The error mapping_list() throws for a list whose item i is not a mapping.
  std::map<std::string, std::string> non_mapping_items_;
};

std::string trim_copy(const std::string& value);
std::filesystem::path default_config_path(const char* source_dir);

} // namespace sima_examples
