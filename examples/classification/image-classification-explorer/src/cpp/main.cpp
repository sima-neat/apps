/**
 * @example image-classification-explorer.cpp
 * Classify a single image or a directory of images with one or more models
 * and generate a browsable HTML report plus JSON/CSV results.
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <cerrno>
#include <csignal>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const std::vector<std::string> kDefaultExtensions = {".jpg", ".jpeg", ".png", ".bmp"};

// The shipped config references the bundled label map by its in-package path,
// which is relative to the example directory rather than the caller's cwd. Only
// this exact reference falls back to the bundled copy; any other missing
// label_map path is a configuration error.
const char* const kBundledLabelMapRef = "src/common/imagenet_labels.txt";

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

// ScalarConfig unquotes values but not mapping keys, so a profile declared as
// `"resnet_50":` arrives with its quotes attached. Strip them for display and
// metadata while the original spelling stays the lookup key.
// Profile names travel through config keys, report columns, CSV/JSON fields and
// the HTML controls, and ScalarConfig addresses them through dotted, colon-
// separated keys. Restrict them to a portable set so both languages accept
// exactly the same names instead of diverging on exotic YAML keys.
bool is_valid_profile_name(const std::string& name) {
  if (name.empty())
    return false;
  return std::all_of(name.begin(), name.end(),
                     [](unsigned char c) { return std::isalnum(c) != 0 || c == '_' || c == '-'; });
}

// Locate the ':' that separates a mapping key from its value, skipping colons
// inside a quoted key (e.g. `"resnet:50":`) so the whole key is recovered and
// can be reported accurately rather than silently truncated.
std::size_t find_key_colon(const std::string& line) {
  char quote = '\0';
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quote != '\0') {
      if (c == quote)
        quote = '\0';
    } else if (c == '"' || c == '\'') {
      quote = c;
    } else if (c == ':') {
      return i;
    }
  }
  return std::string::npos;
}

// True when a key is an unquoted YAML date (YYYY-MM-DD), which PyYAML resolves
// to a datetime.date and Python therefore rejects as a non-string key. Full
// timestamps carry ':' or spaces and are already refused elsewhere.
bool looks_like_yaml_date(const std::string& key) {
  size_t i = 0;
  const auto take_digits = [&](size_t min_n, size_t max_n) {
    size_t n = 0;
    while (i < key.size() && std::isdigit(static_cast<unsigned char>(key[i])) && n < max_n) {
      ++i;
      ++n;
    }
    return n >= min_n;
  };
  if (!take_digits(4, 4))
    return false;
  if (i >= key.size() || key[i] != '-')
    return false;
  ++i;
  if (!take_digits(1, 2))
    return false;
  if (i >= key.size() || key[i] != '-')
    return false;
  ++i;
  if (!take_digits(1, 2))
    return false;
  return i == key.size();
}

// True when an *unquoted* YAML scalar would be read as something other than a
// string (bool, null, or an integer in any spelling). PyYAML turns those into
// Python objects whose text differs from the raw spelling (`01` -> 1,
// `true` -> True), so accepting them here would let C++ run a profile Python
// rejects. Such names must be quoted, which makes them strings in both.
bool looks_like_yaml_non_string(const std::string& key) {
  if (key.empty() || key == "~")
    return true;
  const std::string lowered = lower_copy(key);
  static const std::set<std::string> kWords = {"true", "false", "yes", "no", "on", "off", "null"};
  if (kWords.count(lowered) != 0)
    return true;
  if (looks_like_yaml_date(key))
    return true;
  // Integer spellings: an optional sign, then decimal digits (with YAML 1.1
  // underscores or leading zeros) or a 0x/0o/0b radix prefix.
  std::string body = lowered;
  if (!body.empty() && (body.front() == '-' || body.front() == '+'))
    body.erase(0, 1);
  if (body.rfind("0x", 0) == 0 || body.rfind("0o", 0) == 0 || body.rfind("0b", 0) == 0)
    return true;
  const bool digits_only = std::all_of(
      body.begin(), body.end(), [](unsigned char c) { return std::isdigit(c) != 0 || c == '_'; });
  return digits_only && std::any_of(body.begin(), body.end(),
                                    [](unsigned char c) { return std::isdigit(c) != 0; });
}

// True when the key carries explicit YAML quotes, which make it a string.
bool is_quoted_yaml_key(const std::string& key) {
  return key.size() >= 2 &&
         ((key.front() == '"' && key.back() == '"') || (key.front() == '\'' && key.back() == '\''));
}

std::string unquote_yaml_key(const std::string& key) {
  if (key.size() >= 2 &&
      ((key.front() == '"' && key.back() == '"') || (key.front() == '\'' && key.back() == '\''))) {
    return key.substr(1, key.size() - 2);
  }
  return key;
}

struct ModelProfile {
  std::string name;       // display name (YAML quotes removed)
  std::string config_key; // key exactly as spelled in the config, for lookups
  std::string path;
  int input_width = 224;
  int input_height = 224;
  std::string preprocess = "imagenet";
  std::string output = "softmax";
  int num_classes = 1000;
  std::string label_map;
  int top_k = 5;
  std::vector<std::string> labels;
};

struct Prediction {
  std::vector<sima_examples::ScoredIndex> top_k;
  double inference_ms = 0.0;
};

struct ImageResult {
  fs::path image_path;
  std::map<std::string, Prediction> predictions;
  std::map<std::string, std::string> errors; // model name -> error message
};

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = sima_examples::trim_copy(item);
    if (!item.empty())
      out.push_back(lower_copy(item));
  }
  return out;
}

// ScalarConfig flattens nested maps into dotted keys (e.g. "models.resnet_50.path") and
// stores them in an unordered/sorted map, so it cannot tell us the order profiles were
// declared in. Recover the *set* of profile names (order not meaningful here) by
// scanning the raw scalar map. The profile name is everything before the final
// ".<field>", so a declared name containing '.' surfaces intact and can be rejected
// with a clear message instead of being silently truncated.
std::vector<std::string> profile_names(const sima_examples::ScalarConfig& raw) {
  std::vector<std::string> names;
  std::set<std::string> seen;
  for (const auto& [key, value] : raw.scalars()) {
    static_cast<void>(value);
    const std::string prefix = "models.";
    if (key.rfind(prefix, 0) != 0)
      continue;
    const std::string rest = key.substr(prefix.size());
    const auto dot = rest.rfind('.');
    const std::string name = dot == std::string::npos ? rest : rest.substr(0, dot);
    if (seen.insert(name).second)
      names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

// Drop a YAML inline comment (`#` at the start of the line or preceded by
// whitespace) and trailing whitespace, so `models:   # profiles` matches `models:`.
std::string strip_yaml_comment(const std::string& line) {
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '#' && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t'))
      return sima_examples::trim_copy(line.substr(0, i));
  }
  return sima_examples::trim_copy(line);
}

// Recover the declaration order of top-level keys under `models:` by scanning the
// config file's own text, so C++ iterates profiles in the same order Python does
// (Python's dict/YAML loader preserves declaration order; ScalarConfig does not).
// Falls back to an empty vector (caller falls back to alphabetical) if the file
// can't be read or doesn't look like the expected shape.
std::vector<std::string> ordered_model_keys(const fs::path& config_path) {
  std::ifstream in(config_path);
  if (!in.is_open())
    return {};

  std::vector<std::string> keys;
  std::string raw_line;
  int models_indent = -1;
  int child_indent = -1;
  bool in_models = false;
  while (std::getline(in, raw_line)) {
    const std::string trimmed = strip_yaml_comment(raw_line);
    if (trimmed.empty())
      continue;
    int indent = 0;
    while (indent < static_cast<int>(raw_line.size()) &&
           (raw_line[static_cast<size_t>(indent)] == ' ' ||
            raw_line[static_cast<size_t>(indent)] == '\t'))
      ++indent;

    if (!in_models) {
      if (trimmed == "models:") {
        models_indent = indent;
        in_models = true;
      }
      continue;
    }

    if (indent <= models_indent)
      break; // left the `models:` block
    if (child_indent == -1)
      child_indent = indent;
    if (indent == child_indent) {
      const auto colon = find_key_colon(trimmed);
      if (colon != std::string::npos)
        keys.push_back(sima_examples::trim_copy(trimmed.substr(0, colon)));
    }
  }
  return keys;
}

std::vector<ModelProfile> load_profiles(const sima_examples::ScalarConfig& raw,
                                        const fs::path& config_path) {
  const auto known_names = profile_names(raw);
  auto ordered = ordered_model_keys(config_path);
  // Validate the declared spelling before reconciliation, so a key ScalarConfig
  // cannot represent (a colon inside the name) is reported by its real name.
  for (const auto& key : ordered) {
    if (!is_quoted_yaml_key(key) && looks_like_yaml_non_string(key)) {
      throw std::runtime_error("models: profile name " + key +
                               " is not a string; quote it in config.yaml");
    }
    const std::string declared = unquote_yaml_key(key);
    if (!is_valid_profile_name(declared)) {
      throw std::runtime_error("models." + declared +
                               ": profile names may only contain letters, digits, '_' and '-'");
    }
  }
  // ScalarConfig has no scalar for an empty mapping, but the text scan does. Keep
  // those extra declared profile names so the required-field validation below can
  // reject them just as Python does. Fall back only when the scan misses a scalar
  // profile or contains duplicates.
  std::set<std::string> ordered_set(ordered.begin(), ordered.end());
  std::set<std::string> known_set(known_names.begin(), known_names.end());
  if (ordered.size() != ordered_set.size() ||
      !std::includes(ordered_set.begin(), ordered_set.end(), known_set.begin(), known_set.end())) {
    ordered = known_names;
  }

  std::vector<ModelProfile> profiles;
  for (const auto& key : ordered) {
    ModelProfile profile;
    profile.config_key = key;
    profile.name = unquote_yaml_key(key);
    const std::string& name = profile.name;
    profile.path = raw.string_or("models." + key + ".path", "");
    profile.input_width = raw.int_or("models." + key + ".input_width", 224);
    profile.input_height = raw.int_or("models." + key + ".input_height", 224);
    profile.preprocess = raw.string_or("models." + key + ".preprocess", "imagenet");
    profile.output = raw.string_or("models." + key + ".output", "softmax");
    profile.num_classes = raw.int_or("models." + key + ".num_classes", 1000);
    profile.label_map = raw.string_or("models." + key + ".label_map", "");
    profile.top_k = raw.int_or("models." + key + ".top_k", 5);
    if (!is_valid_profile_name(name)) {
      throw std::runtime_error("models." + name +
                               ": profile names may only contain letters, digits, '_' and '-'");
    }
    if (profile.path.empty()) {
      throw std::runtime_error("models." + name + ".path is required");
    }
    if (profile.output != "softmax") {
      throw std::runtime_error("models." + name + ".output=" + profile.output +
                               " is not supported; only 'softmax' is implemented (raw "
                               "per-class scores, softmax applied, index i maps to "
                               "label_map[i])");
    }
    if (profile.top_k <= 0) {
      throw std::runtime_error("models." + name + ".top_k must be positive, got " +
                               std::to_string(profile.top_k));
    }
    if (profile.num_classes <= 0) {
      throw std::runtime_error("models." + name + ".num_classes must be positive, got " +
                               std::to_string(profile.num_classes));
    }
    if (profile.input_width <= 0 || profile.input_height <= 0) {
      throw std::runtime_error("models." + name +
                               ".input_width/input_height must be positive, "
                               "got " +
                               std::to_string(profile.input_width) + "x" +
                               std::to_string(profile.input_height));
    }
    profiles.push_back(std::move(profile));
  }
  if (profiles.empty()) {
    throw std::runtime_error("config.yaml must define at least one entry under `models`");
  }
  return profiles;
}

std::vector<std::string> load_label_map(const std::string& path, int num_classes) {
  std::vector<std::string> labels;
  if (path.empty()) {
    for (int i = 0; i < num_classes; ++i)
      labels.push_back(std::to_string(i));
    return labels;
  }

  fs::path label_path = path;
  if (!fs::exists(label_path) && fs::path(path).generic_string() == kBundledLabelMapRef) {
    // Resolve the shipped reference next to this example's source tree, regardless
    // of the caller's cwd (model.path stays cwd-relative: it points at a file the
    // customer downloaded). A missing custom path is NOT redirected here.
    fs::path bundled = fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR) / ".." / "common" /
                       fs::path(kBundledLabelMapRef).filename();
    if (fs::exists(bundled))
      label_path = bundled;
  }

  std::ifstream in(label_path);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open label map: " + label_path.string());
  }
  // Positional: physical line index == class id, so blank lines are never dropped
  // (that would silently shift every later label).
  std::string line;
  while (std::getline(in, line)) {
    labels.push_back(sima_examples::trim_copy(line));
  }
  if (static_cast<int>(labels.size()) < num_classes) {
    throw std::runtime_error("label map " + label_path.string() + " has " +
                             std::to_string(labels.size()) + " entries, expected at least " +
                             std::to_string(num_classes));
  }
  for (int class_id = 0; class_id < num_classes; ++class_id) {
    if (labels[static_cast<size_t>(class_id)].empty()) {
      throw std::runtime_error("label map " + label_path.string() + " line " +
                               std::to_string(class_id + 1) + " is blank; every class id 0.." +
                               std::to_string(num_classes - 1) + " needs a label");
    }
  }
  return labels;
}

// Flush and close an output stream, then report any failure (including one
// surfaced only at close, e.g. a full filesystem) so the caller never claims a
// file was written when it is absent or truncated.
void close_or_throw(std::ofstream& out, const fs::path& path) {
  out.flush();
  out.close();
  if (out.fail()) {
    throw std::runtime_error("failed to write: " + path.string());
  }
}

// Cache path for a fallback URL. The URL is part of the file name, so different
// URLs never share a cache entry and there is no separate "which URL is this?"
// marker that could be left paired with another run's bytes.
fs::path fallback_cache_path(const std::string& url, const fs::path& base) {
  std::ostringstream digest;
  digest << std::hex << std::setw(16) << std::setfill('0') << std::hash<std::string>{}(url);
  const std::string stem = base.stem().string();
  const std::string extension = base.extension().string();
  return base.parent_path() / (stem + "-" + digest.str() + extension);
}

// Download a fallback image into a URL-keyed cache entry. The download lands on
// a process-private temporary file, is decoded before it is published, and is
// then moved into place with a single rename, so concurrent runs cannot observe
// or leave a half-updated cache entry and a non-image payload served with HTTP
// 200 (e.g. a proxy error page) is never cached.
fs::path download_fallback_image(const std::string& url, const fs::path& base) {
  const fs::path dest = fallback_cache_path(url, base);
  if (fs::exists(dest))
    return dest;

  const fs::path temporary = dest.string() + ".tmp-" + std::to_string(::getpid());
  std::error_code ec;
  // download_file intentionally keeps a nonempty destination, so clear any
  // leftover partial download from a previous crash first.
  fs::remove(temporary, ec);
  if (ec) {
    throw std::runtime_error("failed to refresh fallback image: " + temporary.string() + ": " +
                             ec.message());
  }
  if (!sima_examples::download_file(url, temporary)) {
    throw std::runtime_error("failed to download fallback image: " + url);
  }
  if (cv::imread(temporary.string(), cv::IMREAD_COLOR).empty()) {
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to download fallback image: " + url +
                             ": downloaded file is not a decodable image");
  }
  fs::rename(temporary, dest, ec);
  if (ec) {
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to refresh fallback image: " + dest.string() + ": " +
                             ec.message());
  }
  return dest;
}

std::vector<fs::path> discover_images(const std::string& input_path,
                                      const std::vector<std::string>& extensions,
                                      const std::string& fallback_url,
                                      const fs::path& fallback_dest,
                                      std::vector<std::string>& skipped) {
  if (input_path.empty()) {
    return {download_fallback_image(fallback_url, fallback_dest)};
  }

  fs::path path = input_path;
  if (fs::is_regular_file(path)) {
    const std::string ext = lower_copy(path.extension().string());
    if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
      skipped.push_back(path.string() + ": unsupported extension " +
                        (ext.empty() ? "(none)" : ext));
      return {};
    }
    return {path};
  }

  if (!fs::is_directory(path)) {
    throw std::runtime_error("input path does not exist: " + path.string());
  }

  std::vector<fs::path> entries;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.is_regular_file())
      entries.push_back(entry.path());
  }
  std::sort(entries.begin(), entries.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename() < b.filename(); });

  std::vector<fs::path> images;
  for (const auto& entry : entries) {
    const std::string ext = lower_copy(entry.extension().string());
    if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
      skipped.push_back(entry.string() + ": unsupported extension " +
                        (ext.empty() ? "(none)" : ext));
      continue;
    }
    images.push_back(entry);
  }

  if (images.empty() && skipped.empty()) {
    throw std::runtime_error("no image files found under " + path.string());
  }
  return images;
}

simaai::neat::Model build_model(const ModelProfile& profile) {
  if (profile.preprocess != "imagenet") {
    throw std::runtime_error("models." + profile.name + ".preprocess=" + profile.preprocess +
                             " is not supported; only 'imagenet' is implemented");
  }
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = profile.input_width;
  opt.preprocess.input_max_height = profile.input_height;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  return simaai::neat::Model(profile.path, opt);
}

Prediction classify(simaai::neat::Model& model, const ModelProfile& profile,
                    const fs::path& image_path, int timeout_ms) {
  const cv::Mat rgb = sima_examples::load_rgb_resized(image_path.string(), profile.input_width,
                                                      profile.input_height);
  const auto input =
      simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB);

  const auto start = std::chrono::steady_clock::now();
  const auto outputs = model.run(simaai::neat::TensorList{input}, timeout_ms);
  const double inference_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  if (outputs.empty()) {
    throw std::runtime_error("model run returned empty output");
  }
  auto scores = sima_examples::tensor_to_floats(outputs.front());
  if (static_cast<int>(scores.size()) < profile.num_classes) {
    throw std::runtime_error("expected at least " + std::to_string(profile.num_classes) +
                             " scores, got " + std::to_string(scores.size()));
  }
  scores.resize(profile.num_classes);

  Prediction pred;
  pred.top_k = sima_examples::topk_with_softmax(scores, profile.top_k);
  pred.inference_ms = inference_ms;
  return pred;
}

std::vector<ImageResult> run_all(std::vector<ModelProfile>& profiles,
                                 const std::vector<fs::path>& images, int timeout_ms) {
  std::vector<ImageResult> results;
  results.reserve(images.size());
  for (const auto& image : images) {
    results.push_back(ImageResult{image, {}, {}});
  }

  for (auto& profile : profiles) {
    std::cout << "Loading model '" << profile.name << "': " << profile.path << "\n";
    simaai::neat::Model model = build_model(profile);
    for (auto& result : results) {
      try {
        result.predictions[profile.name] = classify(model, profile, result.image_path, timeout_ms);
      } catch (const std::exception& e) {
        std::cerr << "  " << result.image_path << ": " << profile.name << " failed: " << e.what()
                  << "\n";
        result.errors[profile.name] = e.what();
      }
    }
  }
  return results;
}

// True/False only when every named profile has a top-1 result; otherwise
// indeterminate rather than silently agreeing/disagreeing over a partial subset.
// Identity is the class id, not the label: ImageNet has distinct classes that
// share a display label (e.g. 134/517 "crane").
std::optional<bool> agreement(const ImageResult& result,
                              const std::vector<ModelProfile>& profiles) {
  if (profiles.size() < 2)
    return std::nullopt;
  std::vector<int> class_ids;
  for (const auto& profile : profiles) {
    const auto it = result.predictions.find(profile.name);
    if (it == result.predictions.end() || it->second.top_k.empty())
      return std::nullopt;
    class_ids.push_back(it->second.top_k.front().index);
  }
  return std::all_of(class_ids.begin(), class_ids.end(),
                     [&](int id) { return id == class_ids.front(); });
}

std::string label_for(const ModelProfile& profile, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < profile.labels.size())
    return profile.labels[static_cast<size_t>(class_id)];
  return std::to_string(class_id);
}

void write_json_report(const fs::path& path, const std::vector<ImageResult>& results,
                       const std::vector<ModelProfile>& profiles,
                       const std::vector<std::string>& skipped, double total_ms) {
  json payload;
  for (const auto& profile : profiles)
    payload["models"].push_back(profile.name);
  payload["skipped"] = skipped;
  payload["timing"] = {
      {"total_ms", total_ms}, {"image_count", results.size()}, {"model_count", profiles.size()}};

  json class_summary = json::object();
  for (const auto& profile : profiles)
    class_summary[profile.name] = json::object();

  json images = json::array();
  for (const auto& result : results) {
    json entry;
    entry["path"] = result.image_path.string();
    if (!result.predictions.empty()) {
      json predictions = json::object();
      for (const auto& profile : profiles) {
        const auto it = result.predictions.find(profile.name);
        if (it == result.predictions.end())
          continue;
        json top_k = json::array();
        for (const auto& scored : it->second.top_k) {
          top_k.push_back({{"class_id", scored.index},
                           {"label", label_for(profile, scored.index)},
                           {"probability", scored.prob}});
        }
        predictions[profile.name] = {{"top_k", top_k}, {"inference_ms", it->second.inference_ms}};
        if (!it->second.top_k.empty()) {
          // Keyed by class id so distinct classes sharing a label are never merged.
          const int class_id = it->second.top_k.front().index;
          auto& entry = class_summary[profile.name][std::to_string(class_id)];
          if (!entry.is_object())
            entry = {{"label", label_for(profile, class_id)}, {"count", 0}};
          entry["count"] = entry["count"].get<int>() + 1;
        }
      }
      entry["predictions"] = predictions;
      const auto agree = agreement(result, profiles);
      entry["agreement"] = agree.has_value() ? json(*agree) : json(nullptr);
    }
    if (!result.errors.empty()) {
      entry["errors"] = result.errors;
    }
    images.push_back(entry);
  }
  payload["images"] = images;
  payload["class_summary"] = class_summary;

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open for writing: " + path.string());
  }
  out << payload.dump(2);
  close_or_throw(out, path);
}

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char c : value) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += "\"";
  return out;
}

void write_csv_report(const fs::path& path, const std::vector<ImageResult>& results,
                      const std::vector<ModelProfile>& profiles) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open for writing: " + path.string());
  }
  out << "image,model,status,top1_class_id,top1_label,top1_probability,inference_ms,top_k\n";
  for (const auto& result : results) {
    for (const auto& profile : profiles) {
      const auto it = result.predictions.find(profile.name);
      if (it == result.predictions.end() || it->second.top_k.empty()) {
        const auto err_it = result.errors.find(profile.name);
        if (err_it != result.errors.end()) {
          out << csv_escape(result.image_path.string()) << "," << csv_escape(profile.name)
              << ",error,,,,," << csv_escape(err_it->second) << "\n";
        } else {
          out << csv_escape(result.image_path.string()) << "," << csv_escape(profile.name)
              << ",no_result,,,,,\n";
        }
        continue;
      }
      const auto& top1 = it->second.top_k.front();
      std::ostringstream top_k_str;
      bool first = true;
      for (const auto& s : it->second.top_k) {
        if (!first)
          top_k_str << ";";
        first = false;
        top_k_str << label_for(profile, s.index) << ":" << std::fixed << std::setprecision(4)
                  << s.prob;
      }
      out << csv_escape(result.image_path.string()) << "," << csv_escape(profile.name) << ",ok,"
          << top1.index << "," << csv_escape(label_for(profile, top1.index)) << "," << std::fixed
          << std::setprecision(4) << top1.prob << "," << std::fixed << std::setprecision(2)
          << it->second.inference_ms << "," << csv_escape(top_k_str.str()) << "\n";
    }
  }
  close_or_throw(out, path);
}

std::string html_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    default:
      out += c;
    }
  }
  return out;
}

std::optional<std::string> make_thumbnail(const fs::path& image_path, const fs::path& thumb_dir,
                                          int max_side = 160) {
  cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (img.empty())
    return std::nullopt;
  const double scale = static_cast<double>(max_side) / std::max(img.cols, img.rows);
  cv::Mat resized;
  cv::resize(img, resized,
             cv::Size(std::max(1, static_cast<int>(img.cols * scale)),
                      std::max(1, static_cast<int>(img.rows * scale))));
  fs::create_directories(thumb_dir);
  const std::string name = std::to_string(std::hash<std::string>{}(image_path.string())) + ".jpg";
  const fs::path thumb_path = thumb_dir / name;
  if (!cv::imwrite(thumb_path.string(), resized)) {
    throw std::runtime_error("failed to write thumbnail: " + thumb_path.string());
  }
  return "thumbnails/" + name;
}

void write_html_report(const fs::path& path, const std::vector<ImageResult>& results,
                       const std::vector<ModelProfile>& profiles,
                       const std::vector<std::string>& skipped, const fs::path& output_dir) {
  std::ostringstream rows;
  size_t row_idx = 0;
  for (const auto& result : results) {
    const auto thumb = make_thumbnail(result.image_path, output_dir / "thumbnails");
    const std::string img_cell = thumb.has_value() ? "<img src=\"" + *thumb + "\">" : "";

    json top1_obj = json::object();
    std::ostringstream cells;
    for (const auto& profile : profiles) {
      const auto it = result.predictions.find(profile.name);
      const auto err_it = result.errors.find(profile.name);
      if (it != result.predictions.end() && !it->second.top_k.empty()) {
        top1_obj[profile.name] = {{"class_id", it->second.top_k.front().index},
                                  {"label", label_for(profile, it->second.top_k.front().index)},
                                  {"prob", it->second.top_k.front().prob}};
        std::ostringstream top_str;
        bool first = true;
        for (const auto& s : it->second.top_k) {
          if (!first)
            top_str << "<br>";
          first = false;
          top_str << html_escape(label_for(profile, s.index)) << " (" << std::fixed
                  << std::setprecision(1) << (s.prob * 100.0) << "%)";
        }
        cells << "<td data-model-col=\"" << html_escape(profile.name) << "\">" << top_str.str()
              << "<br><span class=\"timing\">" << std::fixed << std::setprecision(1)
              << it->second.inference_ms << " ms</span></td>";
      } else if (err_it != result.errors.end()) {
        cells << "<td data-model-col=\"" << html_escape(profile.name) << "\" class=\"error\">"
              << "error: " << html_escape(err_it->second) << "</td>";
      } else {
        cells << "<td data-model-col=\"" << html_escape(profile.name) << "\">no result</td>";
      }
    }

    const std::string has_error = result.errors.empty() ? "0" : "1";
    rows << "<tr class=\"row\" data-has-error=\"" << has_error << "\" data-idx=\"" << row_idx
         << "\" data-top1=\"" << html_escape(top1_obj.dump()) << "\"><td>" << img_cell
         << "</td><td>" << html_escape(result.image_path.string()) << "</td>" << cells.str()
         << "<td class=\"agree-cell\">&mdash;</td></tr>\n";
    ++row_idx;
  }

  std::ostringstream header_cols;
  for (const auto& profile : profiles)
    header_cols << "<th data-model-col=\"" << html_escape(profile.name) << "\">"
                << html_escape(profile.name) << "</th>";

  std::ostringstream model_checkboxes;
  for (const auto& profile : profiles)
    model_checkboxes << "<label><input type=\"checkbox\" class=\"model-checkbox\" value=\""
                     << html_escape(profile.name) << "\" checked> " << html_escape(profile.name)
                     << "</label>";

  // model -> class id -> count (keyed by id so distinct classes sharing a label
  // are never merged).
  std::map<std::string, std::map<int, int>> class_summary;
  for (const auto& result : results) {
    for (const auto& profile : profiles) {
      const auto it = result.predictions.find(profile.name);
      if (it == result.predictions.end() || it->second.top_k.empty())
        continue;
      class_summary[profile.name][it->second.top_k.front().index]++;
    }
  }
  std::ostringstream summary_rows;
  for (const auto& profile : profiles) {
    const auto model_it = class_summary.find(profile.name);
    if (model_it == class_summary.end())
      continue;
    std::vector<std::pair<int, int>> ordered(model_it->second.begin(), model_it->second.end());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [class_id, count] : ordered) {
      summary_rows << "<tr><td>" << html_escape(profile.name) << "</td><td>" << class_id
                   << "</td><td>" << html_escape(label_for(profile, class_id)) << "</td><td>"
                   << count << "</td></tr>";
    }
  }

  std::ostringstream skipped_rows;
  for (const auto& s : skipped)
    skipped_rows << "<li>" << html_escape(s) << "</li>";

  std::vector<std::string> names;
  for (const auto& p : profiles)
    names.push_back(p.name);
  std::string joined_names;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i)
      joined_names += ", ";
    joined_names += names[i];
  }

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open for writing: " + path.string());
  }
  out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
      << "<title>Image Classification Explorer Report</title>\n<style>\n"
      << "  body { font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #1a1a1a; }\n"
      << "  table { border-collapse: collapse; width: 100%; margin-top: 12px; }\n"
      << "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; "
         "font-size: 13px; }\n"
      << "  th { background: #f4f4f4; position: sticky; top: 0; }\n"
      << "  img { max-width: 100px; max-height: 100px; }\n"
      << "  .timing { color: #888; font-size: 11px; }\n"
      << "  .error { color: #b00020; }\n"
      << "  tr[data-has-error=\"1\"] { background: #fff6e5; }\n"
      << "  #controls { margin: 12px 0; display: flex; gap: 12px; align-items: flex-start; "
         "flex-wrap: wrap; }\n"
      << "  #controls input, #controls select { padding: 4px; }\n"
      << "  .dropdown { position: relative; display: inline-block; }\n"
      << "  .dropdown-btn { padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; "
         "background: #fff; cursor: pointer; font-size: 13px; }\n"
      << "  .dropdown-panel { display: none; position: absolute; top: 100%; left: 0; margin-top: "
         "4px; padding: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; "
         "box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 10; min-width: 160px; max-height: "
         "240px; overflow-y: auto; }\n"
      << "  .dropdown-panel.open { display: block; }\n"
      << "  .dropdown-panel label { display: block; font-size: 13px; padding: 2px 0; "
         "white-space: nowrap; }\n"
      << "  .dropdown-panel .dropdown-actions { margin-top: 6px; padding-top: 6px; border-top: "
         "1px solid #eee; }\n"
      << "  .dropdown-panel .dropdown-actions button { font-size: 12px; padding: 2px 6px; "
         "margin-right: 6px; cursor: pointer; }\n"
      << "</style>\n</head>\n<body>\n"
      << "<h1>Image Classification Explorer Report</h1>\n"
      << "<p>Models: " << html_escape(joined_names) << " &middot; Images: " << results.size()
      << " &middot; Skipped: " << skipped.size() << "</p>\n"
      << "<div id=\"controls\">\n"
      << "  <div class=\"dropdown\" id=\"modelDropdown\">\n"
      << "    <button type=\"button\" class=\"dropdown-btn\" id=\"modelDropdownBtn\">All models "
         "&#9662;</button>\n"
      << "    <div class=\"dropdown-panel\" id=\"modelDropdownPanel\">\n"
      << "      " << model_checkboxes.str() << "\n"
      << "      <div class=\"dropdown-actions\">\n"
      << "        <button type=\"button\" id=\"modelSelectAll\">All</button>\n"
      << "        <button type=\"button\" id=\"modelSelectNone\">None</button>\n"
      << "      </div>\n"
      << "    </div>\n"
      << "  </div>\n"
      << "  <select id=\"filterResult\">\n"
      << "    <option value=\"\">All results</option>\n"
      << "    <option value=\"agree\">Agreement</option>\n"
      << "    <option value=\"disagree\">Disagreement</option>\n"
      << "    <option value=\"error\">Errors</option>\n"
      << "  </select>\n"
      << "  <input id=\"filterClass\" placeholder=\"Filter by predicted class...\">\n"
      << "  <input id=\"minConfidence\" type=\"number\" min=\"0\" max=\"100\" step=\"1\" "
         "placeholder=\"Min confidence %\">\n"
      << "  <select id=\"sortBy\">\n"
      << "    <option value=\"\">Sort: default order</option>\n"
      << "    <option value=\"confidence\">Sort: confidence (high to low)</option>\n"
      << "    <option value=\"class\">Sort: predicted class (A-Z)</option>\n"
      << "    <option value=\"result\">Sort: result</option>\n"
      << "  </select>\n"
      << "</div>\n"
      << "<table id=\"reportTable\">\n<thead><tr><th>Image</th><th>Path</th>" << header_cols.str()
      << "<th>Models Agree?</th></tr></thead>\n<tbody>\n"
      << rows.str() << "</tbody>\n</table>\n"
      << "<h2>Per-class summary</h2>\n<table><thead><tr><th>Model</th><th>Class id</th>"
         "<th>Predicted class</th><th>Count</th></tr></thead><tbody>"
      << summary_rows.str() << "</tbody></table>\n"
      << "<h2>Skipped files</h2>\n<ul>" << (skipped.empty() ? "<li>None</li>" : skipped_rows.str())
      << "</ul>\n"
      << "<script>\n"
      << "  const classInput = document.getElementById('filterClass');\n"
      << "  const resultSelect = document.getElementById('filterResult');\n"
      << "  const confidenceInput = document.getElementById('minConfidence');\n"
      << "  const sortSelect = document.getElementById('sortBy');\n"
      << "  const tbody = document.querySelector('#reportTable tbody');\n"
      << "  const rows = Array.from(document.querySelectorAll('#reportTable tbody tr'));\n"
      << "  const modelBtn = document.getElementById('modelDropdownBtn');\n"
      << "  const modelPanel = document.getElementById('modelDropdownPanel');\n"
      << "  const modelChecks = Array.from(document.querySelectorAll('.model-checkbox'));\n"
      << "\n"
      << "  function selectedModels() {\n"
      << "    return modelChecks.filter((c) => c.checked).map((c) => c.value);\n"
      << "  }\n"
      << "\n"
      << "  function updateModelBtnLabel() {\n"
      << "    const selected = modelChecks.filter((c) => c.checked);\n"
      << "    let label;\n"
      << "    if (selected.length === 0) {\n"
      << "      label = 'No models';\n"
      << "    } else if (selected.length === modelChecks.length) {\n"
      << "      label = 'All models';\n"
      << "    } else {\n"
      << "      label = selected.length + ' model' + (selected.length > 1 ? 's' : '');\n"
      << "    }\n"
      << "    modelBtn.textContent = label + ' \\u25BE';\n"
      << "  }\n"
      << "\n"
      << "  modelBtn.addEventListener('click', (e) => {\n"
      << "    e.stopPropagation();\n"
      << "    modelPanel.classList.toggle('open');\n"
      << "  });\n"
      << "  document.addEventListener('click', () => modelPanel.classList.remove('open'));\n"
      << "  modelPanel.addEventListener('click', (e) => e.stopPropagation());\n"
      << "  document.getElementById('modelSelectAll').addEventListener('click', () => {\n"
      << "    modelChecks.forEach((c) => { c.checked = true; });\n"
      << "    updateModelBtnLabel();\n"
      << "    applyFilters();\n"
      << "  });\n"
      << "  document.getElementById('modelSelectNone').addEventListener('click', () => {\n"
      << "    modelChecks.forEach((c) => { c.checked = false; });\n"
      << "    updateModelBtnLabel();\n"
      << "    applyFilters();\n"
      << "  });\n"
      << "  modelChecks.forEach((c) => c.addEventListener('change', () => {\n"
      << "    updateModelBtnLabel();\n"
      << "    applyFilters();\n"
      << "  }));\n"
      << "\n"
      << "  function rowTop1(row) {\n"
      << "    try { return JSON.parse(row.dataset.top1 || '{}'); } catch (e) { return {}; }\n"
      << "  }\n"
      << "\n"
      << "  function rowAgreement(row, models) {\n"
      << "    // Identity is the class id: distinct classes can share a display label.\n"
      << "    if (models.length < 2) return null;\n"
      << "    const top1 = rowTop1(row);\n"
      << "    const ids = models.map((m) => top1[m] && top1[m].class_id);\n"
      << "    if (ids.some((id) => id === undefined)) return null;\n"
      << "    return ids.every((id) => id === ids[0]);\n"
      << "  }\n"
      << "\n"
      << "  function rowMaxConfidence(row, models) {\n"
      << "    const top1 = rowTop1(row);\n"
      << "    const probs = models.map((m) => top1[m] && top1[m].prob).filter((v) => v !== "
         "undefined);\n"
      << "    return probs.length ? Math.max(...probs) : null;\n"
      << "  }\n"
      << "\n"
      << "  function rowClassText(row, models) {\n"
      << "    const top1 = rowTop1(row);\n"
      << "    return models.map((m) => (top1[m] && top1[m].label) || '').join(' ');\n"
      << "  }\n"
      << "\n"
      << "  function applyFilters() {\n"
      << "    const models = selectedModels();\n"
      << "    const classQuery = classInput.value.trim().toLowerCase();\n"
      << "    const resultQuery = resultSelect.value;\n"
      << "    const minConfidence = confidenceInput.value === '' ? null : "
         "parseFloat(confidenceInput.value) / 100;\n"
      << "\n"
      << "    document.querySelectorAll('[data-model-col]').forEach((cell) => {\n"
      << "      cell.style.display = models.includes(cell.dataset.modelCol) ? '' : 'none';\n"
      << "    });\n"
      << "\n"
      << "    for (const row of rows) {\n"
      << "      const hasError = row.dataset.hasError === '1';\n"
      << "      let visible;\n"
      << "\n"
      << "      if (resultQuery === 'error') {\n"
      << "        visible = hasError;\n"
      << "      } else {\n"
      << "        const agree = rowAgreement(row, models);\n"
      << "        if (resultQuery === 'agree') visible = agree === true;\n"
      << "        else if (resultQuery === 'disagree') visible = agree === false;\n"
      << "        else visible = true;\n"
      << "      }\n"
      << "\n"
      << "      if (visible && classQuery) {\n"
      << "        visible = rowClassText(row, models).toLowerCase().includes(classQuery);\n"
      << "      }\n"
      << "\n"
      << "      if (visible && minConfidence !== null) {\n"
      << "        const maxConf = rowMaxConfidence(row, models);\n"
      << "        visible = maxConf !== null && maxConf >= minConfidence;\n"
      << "      }\n"
      << "\n"
      << "      const agreeCell = row.querySelector('.agree-cell');\n"
      << "      if (agreeCell) {\n"
      << "        const agree = rowAgreement(row, models);\n"
      << "        agreeCell.textContent = agree === null ? '\\u2014' : (agree ? 'agree' : "
         "'disagree');\n"
      << "      }\n"
      << "\n"
      << "      row.style.display = visible ? '' : 'none';\n"
      << "    }\n"
      << "\n"
      << "    applySort(models);\n"
      << "  }\n"
      << "\n"
      << "  function applySort(models) {\n"
      << "    const sortKey = sortSelect.value;\n"
      << "    const sorted = rows.slice();\n"
      << "    if (sortKey === 'confidence') {\n"
      << "      sorted.sort((a, b) => {\n"
      << "        const av = rowMaxConfidence(a, models);\n"
      << "        const bv = rowMaxConfidence(b, models);\n"
      << "        return (bv === null ? -1 : bv) - (av === null ? -1 : av);\n"
      << "      });\n"
      << "    } else if (sortKey === 'class') {\n"
      << "      sorted.sort((a, b) => rowClassText(a, models).localeCompare(rowClassText(b, "
         "models)));\n"
      << "    } else if (sortKey === 'result') {\n"
      << "      sorted.sort((a, b) => {\n"
      << "        const ra = a.dataset.hasError === '1' ? 2 : (rowAgreement(a, models) === "
         "false ? 1 : 0);\n"
      << "        const rb = b.dataset.hasError === '1' ? 2 : (rowAgreement(b, models) === "
         "false ? 1 : 0);\n"
      << "        return ra - rb;\n"
      << "      });\n"
      << "    } else {\n"
      << "      sorted.sort((a, b) => Number(a.dataset.idx) - Number(b.dataset.idx));\n"
      << "    }\n"
      << "    for (const row of sorted) tbody.appendChild(row);\n"
      << "  }\n"
      << "\n"
      << "  updateModelBtnLabel();\n"
      << "  classInput.addEventListener('input', applyFilters);\n"
      << "  resultSelect.addEventListener('change', applyFilters);\n"
      << "  confidenceInput.addEventListener('input', applyFilters);\n"
      << "  sortSelect.addEventListener('change', applyFilters);\n"
      << "  applyFilters();\n"
      << "</script>\n</body>\n</html>\n";
  close_or_throw(out, path);
}

// Write the complete report into a sibling staging directory, then swap the whole
// `output_dir` for it. Readers never see a mixture of old and new files, and a
// failure at any point leaves the previous report in place.
//
// `output_dir` is owned by this application: an existing directory is only
// replaced if it is empty or carries the marker written by a previous run and
// holds nothing else, so a customer directory (even one that happens to contain
// a `report.html`) can never be swapped away.
const char* const kReportMarker = ".image-classification-explorer-report";

// True when a process with this id still exists (it may be another run of this
// application mid-swap, whose backup must not be touched).
bool process_is_running(pid_t pid) {
  if (::kill(pid, 0) == 0)
    return true;
  return errno != ESRCH;
}

// Clean up after a publish that was killed part-way through its swap.
//
// publish_report renames the old report aside to `.<name>.previous-<pid>`,
// moves the new one into place, then deletes the backup. A process killed
// between those steps leaves either `output_dir` absent (restore the backup) or
// the backup orphaned (delete it). Backups belonging to a process that is still
// running are left alone, as is anything that does not carry the report marker.
void recover_interrupted_publish(const fs::path& output_dir) {
  const fs::path parent = output_dir.parent_path();
  if (!fs::is_directory(parent))
    return;

  const std::string prefix = "." + output_dir.filename().string() + ".previous-";
  // A backup whose process is still alive belongs to a publish that is mid-swap:
  // it still needs that directory to roll back, and its own rename will fill
  // output_dir shortly. Never restore or delete those.
  std::vector<fs::path> abandoned;
  for (const auto& entry : fs::directory_iterator(parent)) {
    if (!entry.is_directory())
      continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind(prefix, 0) != 0)
      continue;
    if (!fs::exists(entry.path() / kReportMarker))
      continue;
    // A backup bearing our own pid cannot belong to a concurrent invocation: it
    // is a stale one from a killed run whose pid the OS has since recycled onto
    // us. Treating it as live would leave it in place and then fail our own
    // rename onto that path, blocking every later run.
    const std::string suffix = name.substr(prefix.size());
    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; })) {
      const auto owner = static_cast<pid_t>(std::stol(suffix));
      if (owner != ::getpid() && process_is_running(owner))
        continue;
    }
    abandoned.push_back(entry.path());
  }
  if (abandoned.empty())
    return;

  if (!fs::exists(output_dir)) {
    auto newest = abandoned.begin();
    for (auto it = abandoned.begin(); it != abandoned.end(); ++it) {
      std::error_code ec;
      const auto written = fs::last_write_time(*it, ec);
      if (ec)
        continue;
      std::error_code best_ec;
      if (written > fs::last_write_time(*newest, best_ec))
        newest = it;
    }
    const fs::path restored = *newest;
    abandoned.erase(newest);
    fs::rename(restored, output_dir);
    std::cerr << "Recovered an interrupted report publication: restored "
              << restored.filename().string() << " to " << output_dir << "\n";
  }

  // Anything left belongs to a finished publish that never got to delete its
  // backup.
  for (const auto& leftover : abandoned) {
    std::error_code ec;
    fs::remove_all(leftover, ec);
    if (!ec)
      std::cerr << "Removed a leftover report backup: " << leftover.filename().string() << "\n";
  }
}

void publish_report(const fs::path& output_dir_arg, const std::vector<ImageResult>& results,
                    const std::vector<ModelProfile>& profiles,
                    const std::vector<std::string>& skipped, double total_ms) {
  static const std::set<std::string> kReportEntries = {"report.json", "report.csv", "report.html",
                                                       "thumbnails", kReportMarker};
  // weakly_canonical resolves symlinks in the existing part of the path (like
  // Python's Path.resolve()), so a symlinked output_dir has its *target*
  // replaced and the link itself is left in place.
  fs::path output_dir = fs::weakly_canonical(fs::absolute(output_dir_arg));
  if (output_dir.filename().empty()) // trailing slash
    output_dir = output_dir.parent_path();
  const fs::path parent = output_dir.parent_path();
  recover_interrupted_publish(output_dir);

  if (fs::exists(output_dir)) {
    if (!fs::is_directory(output_dir)) {
      throw std::runtime_error("output_dir " + output_dir.string() +
                               " exists and is not a directory");
    }
    std::vector<std::string> foreign;
    bool has_marker = false;
    bool has_entries = false;
    for (const auto& entry : fs::directory_iterator(output_dir)) {
      const std::string name = entry.path().filename().string();
      has_entries = true;
      if (name == kReportMarker)
        has_marker = true;
      else if (kReportEntries.count(name) == 0)
        foreign.push_back(name);
    }
    if (has_entries && !has_marker) {
      throw std::runtime_error("output_dir " + output_dir.string() +
                               " was not created by this application (missing " + kReportMarker +
                               "); use an empty or dedicated directory");
    }
    if (!foreign.empty()) {
      std::sort(foreign.begin(), foreign.end());
      std::string listed;
      for (size_t i = 0; i < foreign.size() && i < 3; ++i)
        listed += (i ? ", " : "") + foreign[i];
      throw std::runtime_error("output_dir " + output_dir.string() +
                               " contains entries that are not part of a previous report (" +
                               listed + "); use a dedicated directory");
    }
  }
  fs::create_directories(parent);

  const std::string tag = std::to_string(::getpid());
  const fs::path staging = parent / ("." + output_dir.filename().string() + ".staging-" + tag);
  const fs::path previous = parent / ("." + output_dir.filename().string() + ".previous-" + tag);
  std::error_code ignored;
  fs::remove_all(staging, ignored);
  fs::create_directories(staging);
  try {
    write_json_report(staging / "report.json", results, profiles, skipped, total_ms);
    write_csv_report(staging / "report.csv", results, profiles);
    write_html_report(staging / "report.html", results, profiles, skipped, staging);
    {
      std::ofstream marker(staging / kReportMarker);
      marker << "generated by image-classification-explorer\n";
      close_or_throw(marker, staging / kReportMarker);
    }

    const bool had_previous = fs::exists(output_dir);
    if (had_previous)
      fs::rename(output_dir, previous);
    try {
      fs::rename(staging, output_dir);
    } catch (...) {
      // A hard kill between the two renames is recovered on the next run by
      // recover_interrupted_publish().
      if (had_previous && !fs::exists(output_dir))
        fs::rename(previous, output_dir); // roll back to the previous report
      throw;
    }
    if (had_previous)
      fs::remove_all(previous, ignored);
  } catch (...) {
    fs::remove_all(staging, ignored);
    throw;
  }
  fs::remove_all(staging, ignored);
}

struct Args {
  fs::path config_path;
};

Args parse_args(int argc, char** argv) {
  Args args;
  args.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc)
        throw std::runtime_error("--config requires a path");
      args.config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return args;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Args args = parse_args(argc, argv);
    const auto raw = sima_examples::ScalarConfig::load(args.config_path);

    const std::string input_path = raw.string_or("io.input", "");
    const std::string fallback_url =
        raw.string_or("io.fallback_image_url",
                      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
                      "n01443537_goldfish.JPEG");
    auto extensions = split_csv(raw.string_or("io.extensions", ".jpg,.jpeg,.png,.bmp"));
    if (extensions.empty())
      extensions.assign(kDefaultExtensions.begin(), kDefaultExtensions.end());
    const fs::path output_dir = raw.string_or("io.output_dir", "report");
    const int timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
    if (timeout_ms <= 0) {
      throw std::runtime_error("runtime.timeout_ms must be positive, got " +
                               std::to_string(timeout_ms));
    }

    auto profiles = load_profiles(raw, args.config_path);
    for (auto& profile : profiles) {
      profile.labels = load_label_map(profile.label_map, profile.num_classes);
    }

    std::vector<std::string> skipped;
    const auto images = discover_images(input_path, extensions, fallback_url,
                                        sima_examples::default_goldfish_path(), skipped);
    for (const auto& s : skipped)
      std::cout << "Skipping " << s << "\n";

    std::vector<ImageResult> results;
    double total_ms = 0.0;
    if (images.empty()) {
      std::cerr << "No images to classify.\n";
    } else {
      std::cout << "Classifying " << images.size() << " image(s) with " << profiles.size()
                << " model(s)\n";

      const auto start = std::chrono::steady_clock::now();
      results = run_all(profiles, images, timeout_ms);
      total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count();
    }

    publish_report(output_dir, results, profiles, skipped, total_ms);

    const auto images_with_errors = std::count_if(
        results.begin(), results.end(), [](const ImageResult& r) { return !r.errors.empty(); });
    std::cout << "Done in " << total_ms << " ms. Report written to " << output_dir << "\n";
    std::cout << "  report.html, report.json, report.csv\n";
    if (images_with_errors > 0) {
      std::cout << "  " << images_with_errors
                << " image(s) had at least one model failure (see report.json)\n";
    }

    const auto expected_class_id_str = raw.string_value("validation.expected_class_id");
    const bool used_fallback_sample = input_path.empty();
    if (expected_class_id_str.has_value() && used_fallback_sample && images.size() == 1 &&
        !profiles.empty()) {
      const int expected_class_id = std::stoi(*expected_class_id_str);
      const auto& first = results.front();
      const auto it = first.predictions.find(profiles.front().name);
      if (it != first.predictions.end() && !it->second.top_k.empty()) {
        const auto& top1 = it->second.top_k.front();
        const double min_probability = raw.double_or("validation.min_probability", 0.0);
        if (top1.index != expected_class_id || top1.prob < min_probability) {
          std::cerr << "Note: " << profiles.front().name << " top1=" << top1.index << " ("
                    << top1.prob << ") did not match expected_class_id=" << expected_class_id
                    << " (min_probability=" << min_probability << "); see report for details.\n";
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }
}
