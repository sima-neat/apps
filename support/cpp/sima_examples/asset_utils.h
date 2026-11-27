#pragma once

#include <filesystem>
#include <string>

namespace sima_examples {

std::string shell_quote(const std::string& s);
bool download_file(const std::string& url, const std::filesystem::path& out_path);
std::filesystem::path default_goldfish_path();

} // namespace sima_examples
