#pragma once

#include <filesystem>
#include <string>

namespace sima_examples {

std::string shell_quote(const std::string& s);
bool move_to_tmp(const std::filesystem::path& src, const std::filesystem::path& dst);
bool download_file(const std::string& url, const std::filesystem::path& out_path);
std::filesystem::path default_goldfish_path();
std::string resolve_resnet50_tar_local_only(const std::filesystem::path& root_in);
std::string resolve_resnet50_tar(const std::filesystem::path& root_in);
std::string resolve_yolov8s_tar_local_first(const std::filesystem::path& root_in,
                                            bool skip_download);
std::string resolve_yolov8s_tar(const std::filesystem::path& root);
std::string resolve_modelzoo_tar(const std::string& model_name,
                                 const std::filesystem::path& root_in);
std::filesystem::path ensure_coco_sample(const std::filesystem::path& root_in);

} // namespace sima_examples
