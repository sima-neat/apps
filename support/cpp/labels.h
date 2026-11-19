#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_examples {

// Reads a class-label file with one label per line. Mirrors sima_examples.labels in
// support/python.
inline std::vector<std::string> load_labels(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("labels file does not exist: " + path.string());
  }
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) {
    const auto start = line.find_first_not_of(" \t\r");
    if (start != std::string::npos) {
      labels.push_back(line.substr(start, line.find_last_not_of(" \t\r") - start + 1));
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + path.string());
  }
  return labels;
}

} // namespace sima_examples
