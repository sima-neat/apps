#pragma once

// List the images in a folder, for the examples that read a folder of image files. Mirrors
// sima_examples.image_folder in support/python.

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_examples {

// The .jpg, .jpeg, .png, and .bmp files directly in folder, in file-name order. A suffix matches
// in any case, such as .PNG. Subfolders and their files are skipped.
inline std::vector<std::filesystem::path> image_paths(const std::filesystem::path& folder) {
  static constexpr std::array<const char*, 4> kImageSuffixes = {".jpg", ".jpeg", ".png", ".bmp"};
  if (!std::filesystem::is_directory(folder)) {
    throw std::runtime_error("image folder does not exist: " + folder.string());
  }
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    std::string suffix = entry.path().extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (entry.is_regular_file() &&
        std::find(kImageSuffixes.begin(), kImageSuffixes.end(), suffix) != kImageSuffixes.end()) {
      paths.push_back(entry.path());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no .jpg, .jpeg, .png, or .bmp files in " + folder.string());
  }
  std::sort(paths.begin(), paths.end(),
            [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  return paths;
}

} // namespace sima_examples
