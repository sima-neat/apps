#include "gallery.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace face_recog {

namespace {

constexpr char kMagic[8] = {'F', 'R', 'G', 'A', 'L', '1', '\n', '\0'};

template <typename T> void write_value(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T> T read_value(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

} // namespace

void Gallery::add(const std::string& name, const Embedding& embedding) {
  if (embedding.size() != kEmbeddingDim) {
    throw std::runtime_error("embedding for '" + name + "' does not have 512 values");
  }
  for (auto& person : people) {
    if (person.name == name) {
      const float count = static_cast<float>(person.samples);
      for (std::size_t i = 0; i < embedding.size(); ++i) {
        person.mean[i] = (person.mean[i] * count + embedding[i]) / (count + 1.0f);
      }
      ++person.samples;
      return;
    }
  }
  people.push_back({name, embedding, 1});
}

void l2_normalize(Embedding& embedding) {
  float squared = 0.0f;
  for (const float value : embedding) {
    squared += value * value;
  }
  if (squared <= 0.0f) {
    return;
  }
  const float scale = 1.0f / std::sqrt(squared);
  for (float& value : embedding) {
    value *= scale;
  }
}

void save_gallery(const Gallery& gallery, const std::filesystem::path& path) {
  for (const auto& person : gallery.people) {
    if (person.mean.size() != kEmbeddingDim || person.name.size() > UINT16_MAX) {
      throw std::runtime_error("gallery entry '" + person.name + "' cannot be saved");
    }
  }

  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.write(kMagic, sizeof(kMagic));
  write_value(out, static_cast<std::uint32_t>(gallery.people.size()));
  for (const auto& person : gallery.people) {
    write_value(out, static_cast<std::uint16_t>(person.name.size()));
    out.write(person.name.data(), static_cast<std::streamsize>(person.name.size()));
    write_value(out, person.samples);
    out.write(reinterpret_cast<const char*>(person.mean.data()), kEmbeddingDim * sizeof(float));
  }
  out.close();
  if (!out) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed to write gallery: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

Gallery load_gallery(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[sizeof(kMagic)] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("not a face-recognizer gallery: " + path.string());
  }

  Gallery gallery;
  const auto count = read_value<std::uint32_t>(in);
  for (std::uint32_t i = 0; i < count && in; ++i) {
    Person person;
    person.name.resize(read_value<std::uint16_t>(in));
    in.read(person.name.data(), static_cast<std::streamsize>(person.name.size()));
    person.samples = read_value<std::uint32_t>(in);
    person.mean.resize(kEmbeddingDim);
    in.read(reinterpret_cast<char*>(person.mean.data()), kEmbeddingDim * sizeof(float));
    if (in) {
      gallery.people.push_back(std::move(person));
    }
  }
  if (!in || gallery.people.size() != count) {
    throw std::runtime_error("gallery is truncated: " + path.string());
  }
  return gallery;
}

} // namespace face_recog
