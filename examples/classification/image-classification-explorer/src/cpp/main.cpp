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

#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const std::vector<std::string> kDefaultExtensions = {".jpg", ".jpeg", ".png", ".bmp"};

struct ModelProfile {
  std::string name;
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

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

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
      const auto colon = trimmed.find(':');
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
  for (const auto& name : ordered) {
    ModelProfile profile;
    profile.name = name;
    profile.path = raw.string_or("models." + name + ".path", "");
    profile.input_width = raw.int_or("models." + name + ".input_width", 224);
    profile.input_height = raw.int_or("models." + name + ".input_height", 224);
    profile.preprocess = raw.string_or("models." + name + ".preprocess", "imagenet");
    profile.output = raw.string_or("models." + name + ".output", "softmax");
    profile.num_classes = raw.int_or("models." + name + ".num_classes", 1000);
    profile.label_map = raw.string_or("models." + name + ".label_map", "");
    profile.top_k = raw.int_or("models." + name + ".top_k", 5);
    if (name.find('.') != std::string::npos) {
      throw std::runtime_error("models." + name + ": profile names must not contain '.'");
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
  if (!fs::exists(label_path)) {
    // Bundled label maps live next to this example's source tree regardless of the
    // caller's cwd (model.path stays cwd-relative since it points at a downloaded file).
    fs::path bundled =
        fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR) / ".." / "common" / label_path.filename();
    if (fs::exists(bundled))
      label_path = bundled;
  }

  std::ifstream in(label_path);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open label map: " + label_path.string());
  }
  std::string line;
  while (std::getline(in, line)) {
    line = sima_examples::trim_copy(line);
    if (!line.empty())
      labels.push_back(line);
  }
  if (static_cast<int>(labels.size()) < num_classes) {
    throw std::runtime_error("label map " + label_path.string() + " has " +
                             std::to_string(labels.size()) + " entries, expected at least " +
                             std::to_string(num_classes));
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

fs::path fallback_source_path(const fs::path& fallback_dest) {
  return fallback_dest.string() + ".source-url";
}

bool fallback_cache_matches(const fs::path& fallback_dest, const std::string& fallback_url) {
  if (!fs::exists(fallback_dest))
    return false;
  std::ifstream in(fallback_source_path(fallback_dest));
  std::string cached_url;
  return static_cast<bool>(std::getline(in, cached_url)) && cached_url == fallback_url;
}

void record_fallback_source(const fs::path& fallback_dest, const std::string& fallback_url) {
  const fs::path source_path = fallback_source_path(fallback_dest);
  std::ofstream out(source_path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to record fallback image source: " + source_path.string());
  }
  out << fallback_url;
  close_or_throw(out, source_path);
}

// Download the fallback image to a temporary file and decode it before it
// replaces the cached copy, so a non-image payload served with HTTP 200 (e.g. a
// proxy error page) is never cached and silently reused by later runs.
void refresh_fallback_image(const std::string& fallback_url, const fs::path& fallback_dest) {
  const fs::path temporary = fallback_dest.string() + ".tmp";
  std::error_code ec;
  // download_file intentionally keeps a nonempty destination, so clear any
  // leftover partial download first.
  fs::remove(temporary, ec);
  if (ec) {
    throw std::runtime_error("failed to refresh fallback image: " + temporary.string() + ": " +
                             ec.message());
  }
  if (!sima_examples::download_file(fallback_url, temporary)) {
    throw std::runtime_error("failed to download fallback image: " + fallback_url);
  }
  if (cv::imread(temporary.string(), cv::IMREAD_COLOR).empty()) {
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to download fallback image: " + fallback_url +
                             ": downloaded file is not a decodable image");
  }
  fs::rename(temporary, fallback_dest, ec);
  if (ec) {
    fs::remove(temporary, ec);
    throw std::runtime_error("failed to refresh fallback image: " + fallback_dest.string() + ": " +
                             ec.message());
  }
  record_fallback_source(fallback_dest, fallback_url);
}

std::vector<fs::path> discover_images(const std::string& input_path,
                                      const std::vector<std::string>& extensions,
                                      const std::string& fallback_url,
                                      const fs::path& fallback_dest,
                                      std::vector<std::string>& skipped) {
  if (input_path.empty()) {
    if (!fallback_cache_matches(fallback_dest, fallback_url)) {
      refresh_fallback_image(fallback_url, fallback_dest);
    }
    return {fallback_dest};
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
std::optional<bool> agreement(const ImageResult& result,
                              const std::vector<ModelProfile>& profiles) {
  if (profiles.size() < 2)
    return std::nullopt;
  std::vector<std::string> labels;
  for (const auto& profile : profiles) {
    const auto it = result.predictions.find(profile.name);
    if (it == result.predictions.end() || it->second.top_k.empty())
      return std::nullopt;
    labels.push_back(profile.labels.at(static_cast<size_t>(it->second.top_k.front().index)));
  }
  return std::all_of(labels.begin(), labels.end(),
                     [&](const std::string& l) { return l == labels.front(); });
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
          const std::string label = label_for(profile, it->second.top_k.front().index);
          auto& count = class_summary[profile.name][label];
          count = count.is_number_integer() ? count.get<int>() + 1 : 1;
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
        top1_obj[profile.name] = {{"label", label_for(profile, it->second.top_k.front().index)},
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

  std::map<std::string, std::map<std::string, int>> class_summary;
  for (const auto& result : results) {
    for (const auto& profile : profiles) {
      const auto it = result.predictions.find(profile.name);
      if (it == result.predictions.end() || it->second.top_k.empty())
        continue;
      class_summary[profile.name][label_for(profile, it->second.top_k.front().index)]++;
    }
  }
  std::ostringstream summary_rows;
  for (const auto& [model, per_class] : class_summary) {
    for (const auto& [cls, count] : per_class) {
      summary_rows << "<tr><td>" << html_escape(model) << "</td><td>" << html_escape(cls)
                   << "</td><td>" << count << "</td></tr>";
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
      << "<h2>Per-class summary</h2>\n<table><thead><tr><th>Model</th><th>Predicted class</th>"
         "<th>Count</th></tr></thead><tbody>"
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
      << "    if (models.length < 2) return null;\n"
      << "    const top1 = rowTop1(row);\n"
      << "    const labels = models.map((m) => top1[m] && top1[m].label);\n"
      << "    if (labels.some((l) => l === undefined)) return null;\n"
      << "    return labels.every((l) => l === labels[0]);\n"
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
// `output_dir` is owned by this application: it is refused if it holds anything
// other than a previous report, so a shared directory (e.g. `.`) can never be
// swapped away.
void publish_report(const fs::path& output_dir_arg, const std::vector<ImageResult>& results,
                    const std::vector<ModelProfile>& profiles,
                    const std::vector<std::string>& skipped, double total_ms) {
  static const std::set<std::string> kReportEntries = {"report.json", "report.csv", "report.html",
                                                       "thumbnails"};
  fs::path output_dir = fs::absolute(output_dir_arg).lexically_normal();
  if (output_dir.filename().empty()) // trailing slash
    output_dir = output_dir.parent_path();
  const fs::path parent = output_dir.parent_path();

  if (fs::exists(output_dir)) {
    if (!fs::is_directory(output_dir)) {
      throw std::runtime_error("output_dir " + output_dir.string() +
                               " exists and is not a directory");
    }
    std::vector<std::string> foreign;
    for (const auto& entry : fs::directory_iterator(output_dir)) {
      const std::string name = entry.path().filename().string();
      if (kReportEntries.count(name) == 0)
        foreign.push_back(name);
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

    const bool had_previous = fs::exists(output_dir);
    if (had_previous)
      fs::rename(output_dir, previous);
    try {
      fs::rename(staging, output_dir);
    } catch (...) {
      if (had_previous)
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
