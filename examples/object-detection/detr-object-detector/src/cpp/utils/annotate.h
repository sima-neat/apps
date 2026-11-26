// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// The annotated images' file names. Mirrors utils/annotate.py in the Python implementation.

#pragma once

#include <filesystem>
#include <string>

namespace annotate {

// The annotated PNG's file name.
inline std::string output_name(const std::filesystem::path& image_path) {
  return image_path.stem().string() + "_detr.png";
}

} // namespace annotate
