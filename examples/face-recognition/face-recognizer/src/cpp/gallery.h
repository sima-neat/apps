#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace face_recog {

constexpr int kEmbeddingDim = 512;

using Embedding = std::vector<float>;

// One enrolled person: the mean of the embeddings enrolled under their name.
struct Person {
  std::string name;
  Embedding mean;
  std::uint32_t samples = 0;
};

struct Gallery {
  std::vector<Person> people;

  // Averages `embedding` into the person called `name`, adding them on first use.
  void add(const std::string& name, const Embedding& embedding);
};

void l2_normalize(Embedding& embedding);

// File layout, little-endian: the 8-byte magic "FRGAL1\n\0", a uint32 person count, then per
// person a uint16 name length, the name, a uint32 sample count and 512 float32 values.
//
// save_gallery writes a temporary file and renames it over `path`, so a failed save leaves
// the previous gallery in place.
void save_gallery(const Gallery& gallery, const std::filesystem::path& path);
Gallery load_gallery(const std::filesystem::path& path);

} // namespace face_recog
