#pragma once

#include "support/runtime/example_utils.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace app::clip {

// Throw (with a re-run hint) unless the features' .prompt.txt sidecar matches the current prompt.
inline void verify_prompt_sidecar(const std::string& features_path, const std::string& prompt) {
  const std::string sidecar = features_path + ".prompt.txt";
  std::ifstream in(sidecar, std::ios::binary);
  sima_examples::require(
      in.good(),
      "text features prompt sidecar not found: " + sidecar +
          "\n  Cannot confirm the features match prompt.text. Regenerate them for the current prompt:"
          "\n    python3 src/tools/precompute_text_features.py");
  std::string saved((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  // sidecar is written verbatim; drop a trailing newline before comparing.
  while (!saved.empty() && (saved.back() == '\n' || saved.back() == '\r')) {
    saved.pop_back();
  }
  sima_examples::require(
      saved == prompt,
      "prompt.text does not match the precomputed text features."
      "\n  config prompt.text : \"" + prompt + "\""
      "\n  features built for : \"" + saved + "\""
      "\n  Regenerate the features for the current prompt:"
      "\n    examples/segmentation/fastsam-multistream/src/tools/precompute_text_features.py");
}

// Read a 2-D [rows,cols] float32 matrix from a .npy file (C-order, '<f4'), one row per vector.
inline std::vector<std::vector<float>> read_npy_f32_matrix(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  sima_examples::require(in.good(), "text features file not found: " + path);

  char magic[6];
  in.read(magic, 6);
  sima_examples::require(in.gcount() == 6 && std::memcmp(magic, "\x93NUMPY", 6) == 0,
                         "not a .npy file: " + path);
  std::uint8_t major = 0;
  std::uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);

  std::uint32_t header_len = 0;
  if (major == 1) {
    std::uint16_t len16 = 0;
    in.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;  // .npy header length is little-endian; assume host little-endian (aarch64)
  } else {
    in.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);

  sima_examples::require(header.find("'<f4'") != std::string::npos ||
                             header.find("\"<f4\"") != std::string::npos,
                         "text features .npy must be little-endian float32 ('<f4'): " + path);
  sima_examples::require(header.find("'fortran_order': True") == std::string::npos,
                         "text features .npy must be C-order: " + path);

  const std::size_t sh = header.find("'shape':");
  const std::size_t open = header.find('(', sh);
  const std::size_t close = header.find(')', open);
  sima_examples::require(sh != std::string::npos && open != std::string::npos &&
                             close != std::string::npos,
                         "could not parse shape in .npy header: " + path);
  const std::string shape = header.substr(open + 1, close - open - 1);
  int rows = 0;
  int cols = 0;
  sima_examples::require(std::sscanf(shape.c_str(), " %d , %d", &rows, &cols) == 2 && rows > 0 &&
                             cols > 0,
                         "text features .npy must be 2-D [M,512]: " + path);

  std::vector<float> flat(static_cast<std::size_t>(rows) * cols);
  in.read(reinterpret_cast<char*>(flat.data()),
          static_cast<std::streamsize>(flat.size() * sizeof(float)));
  sima_examples::require(in.gcount() == static_cast<std::streamsize>(flat.size() * sizeof(float)),
                         "text features .npy truncated: " + path);

  std::vector<std::vector<float>> features(rows);
  for (int r = 0; r < rows; ++r) {
    features[r].assign(flat.begin() + static_cast<std::size_t>(r) * cols,
                       flat.begin() + static_cast<std::size_t>(r + 1) * cols);
  }
  return features;
}

// Load [M,512] float32 prompt features from a .npy
inline std::vector<std::vector<float>> load_text_features(const std::string& path,
                                                          const std::string& prompt) {
  verify_prompt_sidecar(path, prompt);
  return read_npy_f32_matrix(path);
}

}  // namespace app::clip
