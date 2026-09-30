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
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// POSIX: the publication lock and the process-liveness check.
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const std::vector<std::string> kDefaultExtensions = {".jpg", ".jpeg", ".png", ".bmp"};

// The two entrypoints report the same failures with the same exit codes:
// 2 for anything wrong with the configuration or the command line, 3 for input
// that cannot be read, and 6 for a runtime or reporting failure.
class ConfigError : public std::runtime_error {
public:
  explicit ConfigError(const std::string& what) : std::runtime_error(what) {}
};

class InputError : public std::runtime_error {
public:
  explicit InputError(const std::string& what) : std::runtime_error(what) {}
};

// std::stoi reports an out-of-range value as an exception with no context; give
// the customer the key and the value instead, as Python does.
std::string lower_copy(std::string value); // defined below, used by parse_yaml_int

// PyYAML reads integers with the YAML 1.1 rules: a leading zero is octal, `0x`
// and `0b` are radix prefixes and underscores are separators. ScalarConfig uses
// std::stoi in base 10, so `num_classes: 010` is 8 to Python and 10 to C++ -
// no error on either side, just two different runs. Parse it the way PyYAML
// does so both read the same number out of the same file.
std::optional<long long> parse_yaml_int(const std::string& text) {
  std::string body = sima_examples::trim_copy(text);
  if (body.empty())
    return std::nullopt;
  bool negative = false;
  if (body.front() == '-' || body.front() == '+') {
    negative = body.front() == '-';
    body.erase(0, 1);
  }
  body.erase(std::remove(body.begin(), body.end(), '_'), body.end());
  if (body.empty())
    return std::nullopt;

  int base = 10;
  const std::string lowered = lower_copy(body);
  if (lowered.rfind("0x", 0) == 0) {
    base = 16;
    body = body.substr(2);
  } else if (lowered.rfind("0b", 0) == 0) {
    base = 2;
    body = body.substr(2);
  } else if (lowered.rfind("0o", 0) == 0) {
    base = 8;
    body = body.substr(2);
  } else if (body.size() > 1 && body.front() == '0') {
    base = 8; // YAML 1.1 bare octal
  }
  if (body.empty())
    return std::nullopt;

  try {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(body, &consumed, base);
    if (consumed != body.size())
      return std::nullopt;
    return negative ? -parsed : parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// PyYAML resolves `~` to null, so `input: ~` means "not set". ScalarConfig only
// recognises the spelling `null`, so without this C++ would look for a file
// literally named "~" while Python downloaded the fallback sample.
std::optional<std::string> config_scalar(const sima_examples::ScalarConfig& raw,
                                         const std::string& key) {
  auto value = raw.string_value(key);
  if (value.has_value() && sima_examples::trim_copy(*value) == "~")
    return std::nullopt;
  return value;
}

std::string config_scalar_or(const sima_examples::ScalarConfig& raw, const std::string& key,
                             const std::string& fallback) {
  const auto value = config_scalar(raw, key);
  if (!value.has_value())
    return fallback;
  // PyYAML resolves an unquoted numeric scalar to a number, and Python renders
  // that number as text, so `output_dir: 010` names the directory "8" there.
  // ScalarConfig keeps the text "010" and cannot see whether it was quoted, so
  // canonicalise the same way here: both entrypoints then agree for either
  // spelling. A value that is not a YAML integer is untouched.
  if (const auto number = parse_yaml_int(*value))
    return std::to_string(*number);
  return *value;
}

// Python prints a float with str(): the shortest text that reads back as the
// same value, and never bare digits - 0.0 prints as "0.0", not "0".
std::string python_float_text(double value) {
  char buffer[64];
  for (int precision = 1; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    if (std::strtod(buffer, nullptr) == value)
      break;
  }
  std::string text(buffer);
  if (text.find_first_of(".einf") == std::string::npos)
    text += ".0";
  return text;
}

int config_int(const sima_examples::ScalarConfig& raw, const std::string& key, int fallback) {
  const auto text = config_scalar(raw, key);
  if (!text.has_value())
    return fallback;
  const auto parsed = parse_yaml_int(*text);
  if (!parsed.has_value())
    throw ConfigError(key + " must be an integer, got " + *text);
  if (*parsed < std::numeric_limits<int>::min() || *parsed > std::numeric_limits<int>::max())
    throw ConfigError(key + " is out of range for a 32-bit integer: " + *text);
  return static_cast<int>(*parsed);
}

// pathlib drops "." components and redundant separators when it builds a Path,
// but never resolves "..". std::filesystem keeps the text as written, so
// `io.input: ./images` would put "./images/x.jpg" in the C++ report and
// "images/x.jpg" in the Python one - a difference in every path field, and in
// the thumbnail names, which are digests of those paths. Mirror pathlib exactly;
// lexically_normal() is not the same function, because it also collapses "..".
fs::path normalize_like_pathlib(const fs::path& path) {
  fs::path result;
  for (const auto& part : path) {
    if (part.empty() || part == ".")
      continue;
    result /= part;
  }
  return result.empty() ? fs::path(".") : result;
}

// The shipped config references the bundled label map by its in-package path,
// which is relative to the example directory rather than the caller's cwd. Only
// this exact reference falls back to the bundled copy; any other missing
// label_map path is a configuration error.
const char* const kBundledLabelMapRef = "src/common/imagenet_labels.txt";

// Python computes softmax in float32 and C++ accumulates in double, so the two
// agree only to about seven digits - well beyond the precision a float32 model
// output carries. Report a rounded value so both emit identical numbers.
constexpr int kProbabilityDecimals = 6;

double round_probability(double value) {
  const double factor = 1e6; // 10^kProbabilityDecimals
  static_assert(kProbabilityDecimals == 6, "factor must match kProbabilityDecimals");
  return std::round(value * factor) / factor;
}

// FNV-1a (64-bit). std::hash is not specified to be stable across builds, and
// Python's hash() is salted per process; this keeps thumbnail names identical
// across runs, machines and both implementations.
std::string stable_digest(const std::string& value) {
  std::uint64_t digest = 0xCBF29CE484222325ULL;
  for (unsigned char byte : value) {
    digest ^= static_cast<std::uint64_t>(byte);
    digest *= 0x100000001B3ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << digest;
  return out.str();
}

using FileFingerprint = std::pair<std::uintmax_t, fs::file_time_type>;

// --- Inputs: fingerprinting, decoding and the fallback image cache -------------
std::optional<FileFingerprint> file_fingerprint(const fs::path& path) {
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec)
    return std::nullopt;
  const auto written = fs::last_write_time(path, ec);
  if (ec)
    return std::nullopt;
  return FileFingerprint{size, written};
}

// Refuse to mix results from different versions of the same input file.
void check_unchanged(const fs::path& path, const std::optional<FileFingerprint>& expected) {
  if (!expected.has_value()) {
    // Fail closed. Returning here made every later check a no-op for this file,
    // so separate models and the thumbnail could each read a different
    // replacement under one report entry.
    throw std::runtime_error("input could not be fingerprinted: " + path.string());
  }
  const auto current = file_fingerprint(path);
  if (!current.has_value() || *current != *expected) {
    throw std::runtime_error("input changed while the run was in progress: " + path.string());
  }
}

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
  // Integer spellings, decided by the same parser the settings use. A radix
  // prefix only makes a key an integer when valid digits follow it: PyYAML
  // leaves `0xmodel` a string, so treating every `0x` as a number would reject
  // a profile name Python accepts.
  return parse_yaml_int(key).has_value();
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
  // (size, mtime) captured once, so every model and the thumbnail are known to
  // describe the same bytes even if the file is replaced mid-run.
  std::optional<std::pair<std::uintmax_t, fs::file_time_type>> fingerprint;
  std::map<std::string, Prediction> predictions;
  std::map<std::string, std::string> errors; // model name -> error message
};

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = lower_copy(sima_examples::trim_copy(item));
    if (item.empty())
      continue;
    // Accept `jpg` as well as `.jpg`: the extension compared against always
    // carries the dot, so an entry without one would silently match nothing.
    if (item.front() != '.')
      item.insert(item.begin(), '.');
    out.push_back(item);
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

// Directory holding this executable, used to find files shipped beside it
// regardless of the working directory.
fs::path executable_directory() {
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec)
    return {};
  return exe.parent_path();
}

// The directory holding the configuration file in use. config.yaml ships in
// src/common beside the label map and the report assets, so wherever the
// customer points --config, the rest of src/common is next to it. Set once in
// main() before anything reads a bundled file.
fs::path& config_directory() {
  static fs::path directory;
  return directory;
}

// Every file shipped under src/common is found the same way, by one function.
// The label map and the report assets each used to do their own lookup, and the
// two drifted: a packaged binary run from outside prebuilt-apps found its CSS
// but not its labels.
//
// The executable comes first, which covers the packaged layout
// (src/cpp/pre-built/<binary>) from any working directory.
// SIMANEAT_APPS_EXAMPLE_SOURCE_DIR is repository-relative, so it only resolves
// when the working directory is the repository or the installed prebuilt-apps
// root; it covers the development build tree, where the executable sits under
// build/. The last candidate covers running from the example directory itself.
std::vector<fs::path> bundled_candidates(const std::string& name) {
  std::vector<fs::path> candidates;
  // The configuration the customer actually named is the most reliable anchor:
  // config.yaml lives in src/common, so the assets are its neighbours. This is
  // what makes a build-tree binary work from any directory, where nothing else
  // below resolves.
  if (!config_directory().empty())
    candidates.push_back(config_directory() / name);
  const fs::path exe_dir = executable_directory();
  if (!exe_dir.empty()) {
    candidates.push_back(exe_dir / ".." / ".." / "common" / name);
    candidates.push_back(exe_dir / "common" / name);
  }
  candidates.push_back(fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR) / ".." / "common" / name);
  candidates.push_back(fs::path("src") / "common" / name);
  return candidates;
}

// The first candidate that exists, or an empty path.
fs::path find_bundled_file(const std::string& name) {
  for (const auto& candidate : bundled_candidates(name)) {
    std::error_code ec;
    if (fs::is_regular_file(candidate, ec) && !ec)
      return candidate;
  }
  return {};
}

std::string bundled_asset(const std::string& name) {
  const fs::path path = find_bundled_file(name);
  if (!path.empty()) {
    std::ifstream in(path);
    std::ostringstream contents;
    contents << in.rdbuf();
    if (in.good() || in.eof())
      return contents.str();
  }

  std::string tried;
  for (const auto& candidate : bundled_candidates(name)) {
    tried += "\n  " + candidate.lexically_normal().string();
  }
  throw std::runtime_error("failed to read bundled report asset '" + name +
                           "'; looked in:" + tried);
}

// --- Configuration: reading config.yaml and validating every setting -----------
std::vector<ModelProfile> load_profiles(const sima_examples::ScalarConfig& raw,
                                        const fs::path& config_path) {
  const auto known_names = profile_names(raw);
  auto ordered = ordered_model_keys(config_path);
  // Validate the declared spelling before reconciliation, so a key ScalarConfig
  // cannot represent (a colon inside the name) is reported by its real name.
  for (const auto& key : ordered) {
    if (!is_quoted_yaml_key(key) && looks_like_yaml_non_string(key)) {
      throw ConfigError("models: profile name " + key +
                        " is not a string; quote it in config.yaml");
    }
    const std::string declared = unquote_yaml_key(key);
    if (!is_valid_profile_name(declared)) {
      throw ConfigError("models." + declared +
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

  // `foo:` and `"foo":` are the same YAML key. PyYAML keeps the position of the
  // first and the value of the last; mirror that, or C++ would build two
  // profiles that then collide in the prediction and error maps.
  std::vector<std::string> deduped;
  for (const auto& key : ordered) {
    const std::string name = unquote_yaml_key(key);
    auto existing = std::find_if(deduped.begin(), deduped.end(), [&](const std::string& other) {
      return unquote_yaml_key(other) == name;
    });
    if (existing == deduped.end()) {
      deduped.push_back(key);
    } else {
      *existing = key; // later definition wins, position of the first is kept
    }
  }
  ordered = deduped;

  std::vector<ModelProfile> profiles;
  for (const auto& key : ordered) {
    ModelProfile profile;
    profile.config_key = key;
    profile.name = unquote_yaml_key(key);
    const std::string& name = profile.name;
    profile.path = config_scalar_or(raw, "models." + key + ".path", "");
    profile.input_width = config_int(raw, "models." + key + ".input_width", 224);
    profile.input_height = config_int(raw, "models." + key + ".input_height", 224);
    profile.preprocess = config_scalar_or(raw, "models." + key + ".preprocess", "imagenet");
    profile.output = config_scalar_or(raw, "models." + key + ".output", "softmax");
    profile.num_classes = config_int(raw, "models." + key + ".num_classes", 1000);
    profile.label_map = config_scalar_or(raw, "models." + key + ".label_map", "");
    profile.top_k = config_int(raw, "models." + key + ".top_k", 5);
    if (!is_valid_profile_name(name)) {
      throw ConfigError("models." + name +
                        ": profile names may only contain letters, digits, '_' and '-'");
    }
    if (profile.path.empty()) {
      throw ConfigError("models." + name + ".path is required");
    }
    if (profile.preprocess != "imagenet") {
      throw ConfigError("models." + name + ".preprocess=" + profile.preprocess +
                        " is not supported; only 'imagenet' is implemented");
    }
    if (profile.output != "softmax") {
      throw ConfigError("models." + name + ".output=" + profile.output +
                        " is not supported; only 'softmax' is implemented (raw "
                        "per-class scores, softmax applied, index i maps to "
                        "label_map[i])");
    }
    if (profile.top_k <= 0) {
      throw ConfigError("models." + name + ".top_k must be positive, got " +
                        std::to_string(profile.top_k));
    }
    if (profile.num_classes <= 0) {
      throw ConfigError("models." + name + ".num_classes must be positive, got " +
                        std::to_string(profile.num_classes));
    }
    if (profile.input_width <= 0 || profile.input_height <= 0) {
      throw ConfigError("models." + name +
                        ".input_width/input_height must be positive, "
                        "got " +
                        std::to_string(profile.input_width) + "x" +
                        std::to_string(profile.input_height));
    }
    profiles.push_back(std::move(profile));
  }
  if (profiles.empty()) {
    throw ConfigError("config.yaml must define at least one entry under `models`");
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

  fs::path label_path = normalize_like_pathlib(path);
  if (!fs::exists(label_path) && fs::path(path).generic_string() == kBundledLabelMapRef) {
    // Resolve the shipped reference through the same lookup the report assets
    // use, so it is found wherever the binary is run from (model.path stays
    // cwd-relative: it points at a file the customer downloaded). A missing
    // custom path is NOT redirected here.
    const fs::path bundled = find_bundled_file(fs::path(kBundledLabelMapRef).filename().string());
    if (!bundled.empty())
      label_path = bundled;
  }

  errno = 0;
  std::ifstream in(label_path);
  const int open_errno = errno;
  // A directory opens successfully on glibc, so is_open() alone would let it
  // through and report "0 entries" instead of naming the real problem.
  if (!in.is_open() || !fs::is_regular_file(label_path)) {
    const bool is_directory = fs::is_directory(label_path);
    const std::string reason = is_directory ? "Is a directory"
                               : open_errno ? std::strerror(open_errno)
                                            : "No such file or directory";
    throw ConfigError("failed to open label map " + label_path.string() + ": " + reason);
  }
  // Positional: physical line index == class id, so blank lines are never dropped
  // (that would silently shift every later label).
  std::string line;
  while (std::getline(in, line)) {
    labels.push_back(sima_examples::trim_copy(line));
  }
  if (static_cast<int>(labels.size()) < num_classes) {
    throw ConfigError("label map " + label_path.string() + " has " + std::to_string(labels.size()) +
                      " entries, expected at least " + std::to_string(num_classes));
  }
  for (int class_id = 0; class_id < num_classes; ++class_id) {
    if (labels[static_cast<size_t>(class_id)].empty()) {
      throw ConfigError("label map " + label_path.string() + " line " +
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
  // Same FNV-1a digest the Python implementation uses, so both share one cache
  // entry rather than downloading the same image twice.
  const std::string stem = base.stem().string();
  const std::string extension = base.extension().string();
  return base.parent_path() / (stem + "-" + stable_digest(url) + extension);
}

// Download a fallback image into a URL-keyed cache entry. The download lands on
// a process-private temporary file, is decoded before it is published, and is
// then moved into place with a single rename, so concurrent runs cannot observe
// or leave a half-updated cache entry and a non-image payload served with HTTP
// 200 (e.g. a proxy error page) is never cached.
fs::path download_fallback_image(const std::string& url, const fs::path& base) {
  const fs::path dest = fallback_cache_path(url, base);
  if (fs::exists(dest)) {
    if (!cv::imread(dest.string(), cv::IMREAD_COLOR).empty())
      return dest;
    // Truncated or corrupted since it was cached: refetch rather than classify it.
    std::error_code stale;
    fs::remove(dest, stale);
  }

  const fs::path temporary = dest.string() + ".tmp-" + std::to_string(::getpid());
  std::error_code ec;
  // download_file intentionally keeps a nonempty destination, so clear any
  // leftover partial download from a previous crash first.
  fs::remove(temporary, ec);
  if (ec) {
    throw InputError("failed to refresh fallback image: " + temporary.string() + ": " +
                     ec.message());
  }
  if (!sima_examples::download_file(url, temporary)) {
    throw InputError("failed to download fallback image: " + url);
  }
  if (cv::imread(temporary.string(), cv::IMREAD_COLOR).empty()) {
    fs::remove(temporary, ec);
    throw InputError("failed to download fallback image: " + url +
                     ": downloaded file is not a decodable image");
  }
  fs::rename(temporary, dest, ec);
  if (ec) {
    fs::remove(temporary, ec);
    throw InputError("failed to refresh fallback image: " + dest.string() + ": " + ec.message());
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

  const fs::path path = normalize_like_pathlib(input_path);
  if (fs::is_regular_file(path)) {
    const std::string raw_ext = path.extension().string();
    const std::string ext = lower_copy(raw_ext);
    if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
      skipped.push_back(path.string() + ": unsupported extension " +
                        (raw_ext.empty() ? "(none)" : raw_ext));
      return {};
    }
    return {path};
  }

  if (!fs::is_directory(path)) {
    throw InputError("input path does not exist: " + path.string());
  }

  std::vector<fs::path> entries;
  {
    // Constructing the iterator throws when the directory cannot be read; that
    // is an input failure, and Python reports it as one.
    std::error_code ec;
    fs::directory_iterator it(path, ec);
    if (ec) {
      throw InputError("failed to read input directory " + path.string() + ": " + ec.message());
    }
    for (const auto& entry : it) {
      // The throwing overload aborts the entire scan when one entry cannot be
      // stat'ed - a dangling or self-referential symlink - and the run exits
      // with no report at all. Python's Path.is_file() reports false for the
      // same entry and carries on, so skip it here too. (find_bundled_file
      // already uses this overload; this was the one place that did not.)
      std::error_code entry_ec;
      if (entry.is_regular_file(entry_ec) && !entry_ec)
        entries.push_back(normalize_like_pathlib(entry.path()));
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename() < b.filename(); });

  std::vector<fs::path> images;
  for (const auto& entry : entries) {
    const std::string raw_ext = entry.extension().string();
    const std::string ext = lower_copy(raw_ext);
    if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) {
      skipped.push_back(entry.string() + ": unsupported extension " +
                        (raw_ext.empty() ? "(none)" : raw_ext));
      continue;
    }
    images.push_back(entry);
  }

  if (images.empty() && skipped.empty()) {
    throw InputError("no image files found under " + path.string());
  }
  return images;
}

// --- Neat inference: preprocessing, model construction and execution -----------
simaai::neat::Model build_model(const ModelProfile& profile) {
  // preprocess is validated in load_profiles, before any input is read.
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
  if (!std::all_of(scores.begin(), scores.end(), [](float v) { return std::isfinite(v); })) {
    // NaN/inf would propagate through softmax into the report.
    throw std::runtime_error("model produced non-finite scores");
  }

  Prediction pred;
  pred.top_k = sima_examples::topk_with_softmax(scores, profile.top_k);
  pred.inference_ms = inference_ms;
  return pred;
}

std::vector<ImageResult> run_all(std::vector<ModelProfile>& profiles,
                                 const std::vector<fs::path>& images, int timeout_ms) {
  std::vector<ImageResult> results;
  results.reserve(images.size());
  // Fingerprint every input up front: models are loaded one at a time, so each
  // image is re-read per model, and agreement is only meaningful if those reads
  // saw the same bytes.
  for (const auto& image : images) {
    ImageResult result;
    result.image_path = image;
    result.fingerprint = file_fingerprint(image);
    results.push_back(std::move(result));
  }

  for (auto& profile : profiles) {
    std::cout << "Loading model '" << profile.name << "': " << profile.path << "\n";
    simaai::neat::Model model = build_model(profile);
    for (auto& result : results) {
      try {
        check_unchanged(result.image_path, result.fingerprint);
        Prediction prediction = classify(model, profile, result.image_path, timeout_ms);
        // Only keep the prediction once the bytes behind it are proven unchanged,
        // so a swapped input leaves an error and no result.
        check_unchanged(result.image_path, result.fingerprint);
        result.predictions[profile.name] = std::move(prediction);
      } catch (const std::exception& e) {
        std::cerr << "  " << result.image_path.string() << ": " << profile.name
                  << " failed: " << e.what() << "\n";
        result.errors[profile.name] = e.what();
      }
    }
  }
  return results;
}

// --- Reporting: agreement, per-class counts and the JSON/CSV/HTML writers ------
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
                           {"probability", round_probability(scored.prob)}});
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
                                          const std::optional<FileFingerprint>& fingerprint,
                                          int max_side = 160) {
  // The thumbnail must show the bytes the predictions were made from.
  try {
    check_unchanged(image_path, fingerprint);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (img.empty())
    return std::nullopt;
  try {
    // Re-check after the read, as inference does: a file replaced between the
    // check above and this read would otherwise put a thumbnail of the new
    // bytes next to predictions made from the old ones.
    check_unchanged(image_path, fingerprint);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  // Shrink only; never upscale a small image.
  const double scale = std::min(1.0, static_cast<double>(max_side) / std::max(img.cols, img.rows));
  cv::Mat resized;
  cv::resize(img, resized,
             cv::Size(std::max(1, static_cast<int>(img.cols * scale)),
                      std::max(1, static_cast<int>(img.rows * scale))));
  fs::create_directories(thumb_dir);
  const std::string name = stable_digest(image_path.string()) + ".jpg";
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
    const auto thumb =
        make_thumbnail(result.image_path, output_dir / "thumbnails", result.fingerprint);
    // alt="" marks the thumbnail decorative: the path is in the next cell, so a
    // screen reader should not announce it twice.
    const std::string img_cell = thumb.has_value() ? "<img src=\"" + *thumb + "\" alt=\"\">" : "";

    json top1_obj = json::object();
    std::ostringstream cells;
    for (const auto& profile : profiles) {
      const auto it = result.predictions.find(profile.name);
      const auto err_it = result.errors.find(profile.name);
      if (it != result.predictions.end() && !it->second.top_k.empty()) {
        top1_obj[profile.name] = {{"class_id", it->second.top_k.front().index},
                                  {"label", label_for(profile, it->second.top_k.front().index)},
                                  {"prob", round_probability(it->second.top_k.front().prob)}};
        std::ostringstream top_str;
        bool first = true;
        for (const auto& s : it->second.top_k) {
          if (!first)
            top_str << "<br>";
          first = false;
          top_str << html_escape(label_for(profile, s.index)) << " (" << std::fixed
                  << std::setprecision(2) << (s.prob * 100.0) << "%)";
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
    // Most frequent first, then by class id, matching the Python writer.
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second)
        return a.second > b.second;
      return a.first < b.first;
    });
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

  const std::string report_css = bundled_asset("report.css");
  const std::string report_js = bundled_asset("report.js");

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open for writing: " + path.string());
  }
  out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
      << "<title>Image Classification Explorer Report</title>\n<style>\n"
      << report_css << "</style>\n</head>\n<body>\n"
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
      << report_js << "</script>\n</body>\n</html>\n";
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

// --- Publication: swapping the report directory into place safely --------------
// True when a process with this id still exists (it may be another run of this
// application mid-swap, whose backup must not be touched).
// One run at a time per output directory, which the README states. There is no
// cross-process locking or recovery protocol: the two scratch names are fixed,
// are the only paths outside output_dir this touches, and are cleared on entry,
// so a run interrupted mid-swap costs the previous report and nothing else.
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
  fs::create_directories(parent);

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
  // Named exactly, never matched by prefix: a directory that merely starts with
  // one of these names belongs to someone else and is left alone. Leftovers here
  // are from a run of ours that was interrupted mid-swap.
  const fs::path staging = parent / ("." + output_dir.filename().string() + ".staging");
  const fs::path previous = parent / ("." + output_dir.filename().string() + ".previous");
  std::error_code ignored;
  fs::remove_all(staging, ignored);
  fs::remove_all(previous, ignored);
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
        throw ConfigError("--config requires a path");
      args.config_path = normalize_like_pathlib(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw ConfigError("unknown argument: " + arg);
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
    config_directory() = fs::absolute(args.config_path).parent_path();

    // An unreadable or malformed config file is a configuration failure.
    const sima_examples::ScalarConfig raw = [&] {
      try {
        return sima_examples::ScalarConfig::load(args.config_path);
      } catch (const std::exception& e) {
        throw ConfigError(e.what());
      }
    }();

    // `io: /images` or `runtime: 5000` leaves ScalarConfig holding a scalar at
    // the section name, and every nested lookup below would quietly fall back to
    // its default. Python rejects these, so reject them here too.
    for (const char* section : {"io", "runtime", "validation", "models"}) {
      if (const auto value = raw.string_value(section)) {
        throw ConfigError(std::string("`") + section + "` must be a mapping, got " + *value);
      }
    }

    // Not guarded: a collection written where a scalar belongs, such as
    // `output_dir: [a, b]`. Python rejects it; ScalarConfig keeps the text
    // verbatim, so C++ would create a directory literally named "[a, b]".
    //
    // Guarding it here was tried twice and withdrawn both times. ScalarConfig
    // flattens the file to scalars, which loses what is needed to do this
    // correctly: a block sequence leaves no entry at all, a quoted "[a, b]" is
    // indistinguishable from an unquoted one, and a check over every scalar
    // also rejects keys neither entrypoint reads - `notes: [see docs]` became
    // a hard error in C++ while Python ignored it. Each attempt broke a
    // configuration that works today in order to reject one nobody writes.
    // Doing it properly means a real YAML parser in the shared reader, which
    // is its owners' call, not this example's.

    const std::string input_path = config_scalar_or(raw, "io.input", "");
    const std::string fallback_url = config_scalar_or(
        raw, "io.fallback_image_url",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
        "n01443537_goldfish.JPEG");
    auto extensions = split_csv(config_scalar_or(raw, "io.extensions", ".jpg,.jpeg,.png,.bmp"));
    if (extensions.empty())
      extensions.assign(kDefaultExtensions.begin(), kDefaultExtensions.end());
    const fs::path output_dir =
        normalize_like_pathlib(config_scalar_or(raw, "io.output_dir", "report"));
    // Parse the validation block up front: std::stoi on a malformed value would
    // otherwise throw after the report had already been written.
    std::optional<int> expected_class_id;
    if (config_scalar(raw, "validation.expected_class_id").has_value()) {
      expected_class_id = config_int(raw, "validation.expected_class_id", 0);
    }

    const double min_probability = [&] {
      if (!config_scalar(raw, "validation.min_probability").has_value())
        return 0.0;
      try {
        return raw.double_or("validation.min_probability", 0.0);
      } catch (const std::exception&) {
        throw ConfigError("validation.min_probability must be a number, got " +
                          raw.string_or("validation.min_probability", "0.0"));
      }
    }();

    const int timeout_ms = config_int(raw, "runtime.timeout_ms", 20000);
    if (timeout_ms <= 0) {
      throw ConfigError("runtime.timeout_ms must be positive, got " + std::to_string(timeout_ms));
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
      std::string joined;
      for (const auto& profile : profiles) {
        if (!joined.empty())
          joined += ", ";
        joined += profile.name;
      }
      std::cout << "Classifying " << images.size() << " image(s) with " << profiles.size()
                << " model(s): " << joined << "\n";

      const auto start = std::chrono::steady_clock::now();
      results = run_all(profiles, images, timeout_ms);
      total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count();
    }

    try {
      publish_report(output_dir, results, profiles, skipped, total_ms);
    } catch (const std::exception& e) {
      throw std::runtime_error("failed to write report to " + output_dir.string() + ": " +
                               e.what());
    }

    const auto images_with_errors = std::count_if(
        results.begin(), results.end(), [](const ImageResult& r) { return !r.errors.empty(); });
    std::cout << "Done in " << std::fixed << std::setprecision(1) << total_ms
              << " ms. Report written to " << output_dir.string() << "\n";
    std::cout << "  report.html, report.json, report.csv\n";
    if (images_with_errors > 0) {
      std::cout << "  " << images_with_errors
                << " image(s) had at least one model failure (see report.json)\n";
    }

    // Parsed with the rest of the configuration further up, so a malformed value
    // cannot turn a completed run into a failure after the report is written.
    const bool used_fallback_sample = input_path.empty();
    if (expected_class_id.has_value() && used_fallback_sample && images.size() == 1 &&
        !profiles.empty()) {
      const auto& first = results.front();
      const auto it = first.predictions.find(profiles.front().name);
      if (it != first.predictions.end() && !it->second.top_k.empty()) {
        const auto& top1 = it->second.top_k.front();
        if (top1.index != *expected_class_id || top1.prob < min_probability) {
          // Formatted separately: std::fixed and setprecision are sticky, and
          // applied inline they would also reformat min_probability below.
          std::ostringstream probability;
          probability << std::fixed << std::setprecision(4) << top1.prob;
          std::cerr << "Note: " << profiles.front().name << " top1=" << top1.index << " ("
                    << probability.str()
                    << ") did not match expected_class_id=" << *expected_class_id
                    << " (min_probability=" << python_float_text(min_probability)
                    << "); see report for details.\n";
        }
      }
    }

    return 0;
  } catch (const ConfigError& e) {
    std::cerr << "Invalid configuration: " << e.what() << "\n";
    return 2;
  } catch (const InputError& e) {
    std::cerr << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }
}
