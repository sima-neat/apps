#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace sima_test {

namespace fs = std::filesystem;

inline std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

inline bool move_to_tmp(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  ec.clear();
  fs::rename(src, dst, ec);
  if (!ec)
    return true;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec)
    return false;
  fs::remove(src, ec);
  return true;
}

inline bool download_file(const std::string& url, const fs::path& out_path) {
  if (fs::exists(out_path)) {
    std::error_code ec;
    if (fs::file_size(out_path, ec) > 0 && !ec)
      return true;
  }

  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());

  std::string cmd = "curl -L --fail --silent --show-error -o " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0)
    return true;

  cmd = "wget -O " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0)
    return true;

  std::error_code rm_ec;
  fs::remove(out_path, rm_ec);
  return false;
}

inline fs::path default_goldfish_path() {
  try {
    return fs::temp_directory_path() / "sima_imagenet_goldfish.jpg";
  } catch (...) {
    return fs::path("tmp") / "imagenet_goldfish.jpg";
  }
}

inline std::string resolve_resnet50_tar_local_only(const fs::path& root_in = {}) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const char* env = std::getenv("SIMA_RESNET50_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const fs::path local = root / "tmp" / "resnet_50_mpk.tar.gz";
  if (fs::exists(local))
    return local.string();

  const std::vector<fs::path> candidates = {
      root / "resnet_50_mpk.tar.gz",
      root / "resnet-50_mpk.tar.gz",
  };
  for (const auto& candidate : candidates) {
    if (fs::exists(candidate) && move_to_tmp(candidate, local)) {
      return local.string();
    }
  }

  return "";
}

inline std::string resolve_resnet50_tar(const fs::path& root_in = {}) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const char* env = std::getenv("SIMA_RESNET50_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const fs::path local = root / "tmp" / "resnet_50_mpk.tar.gz";
  if (fs::exists(local))
    return local.string();

  const int rc = std::system("sima-cli modelzoo get resnet_50");
  if (rc != 0)
    return "";

  if (fs::exists(local))
    return local.string();

  const std::vector<fs::path> candidates = {
      root / "resnet_50_mpk.tar.gz",
      root / "resnet-50_mpk.tar.gz",
  };
  for (const auto& candidate : candidates) {
    if (fs::exists(candidate) && move_to_tmp(candidate, local)) {
      return local.string();
    }
  }

  return "";
}

inline std::string resolve_yolov8s_tar_local_first(const fs::path& root_in, bool skip_download) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const fs::path tmp_tar = root / "tmp" / "yolo_v8s_mpk.tar.gz";

  const char* env = std::getenv("SIMA_YOLO_TAR");
  if (env && *env && fs::exists(env)) {
    return std::string(env);
  }

  const fs::path direct_tar = root / "yolo_v8s_mpk.tar.gz";
  if (fs::exists(direct_tar))
    return direct_tar.string();

  if (fs::exists(tmp_tar))
    return tmp_tar.string();

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      root,
      fs::current_path(),
      root / "tmp",
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  const std::vector<std::string> names = {
      "yolo_v8s_mpk.tar.gz",
      "yolo-v8s_mpk.tar.gz",
      "yolov8s_mpk.tar.gz",
      "yolov8_s_mpk.tar.gz",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (fs::exists(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  if (!skip_download) {
    const int rc = std::system("sima-cli modelzoo get yolo_v8s");
    if (rc == 0 && fs::exists(tmp_tar))
      return tmp_tar.string();
  }

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (fs::exists(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  return "";
}

inline std::string resolve_yolov8s_tar(const fs::path& root = {}) {
  return resolve_yolov8s_tar_local_first(root, false);
}

inline std::string resolve_modelzoo_tar(const std::string& model_name,
                                        const fs::path& root_in = {}) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const fs::path tmp_dir = root / "tmp";
  std::error_code ec;
  fs::create_directories(tmp_dir, ec);

  auto append_unique = [](std::vector<std::string>& out, const std::string& v) {
    if (v.empty())
      return;
    for (const auto& existing : out) {
      if (existing == v)
        return;
    }
    out.push_back(v);
  };

  const std::string base = [&]() {
    const std::size_t pos = model_name.find_last_of('/');
    return (pos == std::string::npos) ? model_name : model_name.substr(pos + 1);
  }();

  std::string flat = model_name;
  for (char& c : flat) {
    if (c == '/')
      c = '_';
  }
  std::string flat_dash = model_name;
  for (char& c : flat_dash) {
    if (c == '/')
      c = '-';
  }

  std::vector<std::string> stems;
  append_unique(stems, base);
  append_unique(stems, flat);
  append_unique(stems, flat_dash);

  const std::size_t initial = stems.size();
  for (std::size_t i = 0; i < initial; ++i) {
    std::string alt = stems[i];
    bool changed = false;
    for (char& c : alt) {
      if (c == '_') {
        c = '-';
        changed = true;
      }
    }
    if (changed)
      append_unique(stems, alt);
  }

  std::vector<std::string> names;
  names.reserve(stems.size() * 3);
  for (const auto& stem : stems) {
    append_unique(names, stem + "_mpk.tar.gz");
    append_unique(names, stem + "-mpk.tar.gz");
    append_unique(names, stem + ".tar.gz");
  }

  const fs::path local = tmp_dir / (base + "_mpk.tar.gz");
  if (fs::exists(local))
    return local.string();

  auto try_candidate = [&](const fs::path& candidate) -> std::string {
    if (!fs::exists(candidate))
      return "";
    if (candidate == local)
      return local.string();
    if (move_to_tmp(candidate, local))
      return local.string();
    return "";
  };

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      tmp_dir,
      root,
      fs::current_path(),
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      const std::string found = try_candidate(dir / name);
      if (!found.empty())
        return found;
    }
  }

  const std::string cmd = "sima-cli modelzoo get " + shell_quote(model_name);
  if (std::system(cmd.c_str()) != 0)
    return "";

  if (fs::exists(local))
    return local.string();

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      const std::string found = try_candidate(dir / name);
      if (!found.empty())
        return found;
    }
  }

  return "";
}

inline fs::path ensure_coco_sample(const fs::path& root_in = {}) {
  const fs::path root = root_in.empty() ? fs::current_path() : root_in;
  const char* url_env = std::getenv("SIMA_COCO_URL");
  const std::string url =
      (url_env && *url_env)
          ? std::string(url_env)
          : "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";
  const fs::path out_path = root / "tmp" / "coco_sample.jpg";
  if (!download_file(url, out_path))
    return {};
  return out_path;
}

} // namespace sima_test
