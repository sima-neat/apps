#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssd_mobilenet {

namespace fs = std::filesystem;

// Output stem keeping the (case-preserved) source extension so frame.jpg/frame.png don't collide.
inline std::string output_stem(const fs::path& image_path) {
  std::string ext = image_path.extension().string();
  if (!ext.empty() && ext.front() == '.') {
    ext.erase(ext.begin());
  }
  const std::string stem = image_path.stem().string();
  return ext.empty() ? stem : stem + "_" + ext;
}

// Remove only the output entries this run will regenerate. symlink_status() deliberately does not
// follow symlinks, so a dangling link is removed by pathname instead of being followed by imwrite.
inline int clear_output_images(const fs::path& output_dir,
                               const std::vector<fs::path>& image_paths) {
  int removed = 0;
  for (const fs::path& image_path : image_paths) {
    const fs::path candidate = output_dir / (output_stem(image_path) + ".png");
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(candidate, ec);
    if (ec == std::errc::no_such_file_or_directory || status.type() == fs::file_type::not_found) {
      continue;
    }
    if (ec) {
      throw fs::filesystem_error("failed to inspect stale output", candidate, ec);
    }
    if (fs::is_directory(status)) {
      throw std::runtime_error("refusing to replace output directory: " + candidate.string());
    }
    if (!fs::remove(candidate, ec) && !ec) {
      continue;
    }
    if (ec) {
      throw fs::filesystem_error("failed to remove stale output", candidate, ec);
    }
    ++removed;
  }
  return removed;
}

} // namespace ssd_mobilenet
