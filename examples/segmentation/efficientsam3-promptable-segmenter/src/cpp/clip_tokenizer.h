// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// CLIP tokenizer for the EfficientSAM3 text encoder, after OpenAI CLIP (MIT, see
// src/common/LICENSE-CLIP.txt).
#pragma once

#include <zlib.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class ClipTokenizer {
public:
  explicit ClipTokenizer(const std::filesystem::path& vocab) {
    // The vocabulary lists the printable bytes first, then the others, as bytes_to_unicode().
    std::vector<std::string> pieces;
    std::vector<std::string> others;
    for (int b = 0, n = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
      byte_chars_[b] = utf8(printable ? b : 256 + n++);
      (printable ? pieces : others).push_back(byte_chars_[b]);
    }
    pieces.insert(pieces.end(), others.begin(), others.end());
    for (int i = 0; i < 256; ++i) {
      pieces.push_back(pieces[i] + "</w>");
    }
    std::istringstream lines(gunzip(vocab));
    std::string line;
    std::getline(lines, line);
    for (int rank = 0; rank < 49152 - 256 - 2 && std::getline(lines, line); ++rank) {
      const std::size_t space = line.find(' ');
      ranks_[line] = rank;
      pieces.push_back(line.substr(0, space) + line.substr(space + 1));
    }
    pieces.push_back("<start_of_text>");
    pieces.push_back("<end_of_text>");
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      ids_[pieces[i]] = static_cast<int64_t>(i);
    }
  }

  std::vector<int64_t> encode(const std::string& text, int length) const {
    std::vector<int64_t> ids{ids_.at("<start_of_text>")};
    for (const std::u32string& word : words(text)) {
      std::string symbols;
      for (const char byte : utf8_of(word)) {
        symbols += byte_chars_[static_cast<unsigned char>(byte)];
      }
      for (const std::string& piece : bpe(symbols)) {
        ids.push_back(ids_.at(piece));
      }
    }
    ids.resize(std::min<std::size_t>(ids.size(), length - 1));
    ids.push_back(ids_.at("<end_of_text>"));
    ids.resize(length, 0);
    return ids;
  }

private:
  static std::string utf8(char32_t c) {
    std::string out;
    if (c < 0x80) {
      out += static_cast<char>(c);
    } else if (c < 0x800) {
      out += static_cast<char>(0xC0 | (c >> 6));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      out += static_cast<char>(0xE0 | (c >> 12));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (c >> 18));
      out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (c & 0x3F));
    }
    return out;
  }

  static std::string utf8_of(const std::u32string& text) {
    std::string out;
    for (const char32_t c : text) {
      out += utf8(c);
    }
    return out;
  }

  static std::u32string code_points(const std::string& text) {
    std::u32string out;
    for (std::size_t i = 0; i < text.size();) {
      const auto lead = static_cast<unsigned char>(text[i]);
      const int extra = lead < 0x80 ? 0 : lead < 0xE0 ? 1 : lead < 0xF0 ? 2 : 3;
      char32_t c = extra == 0 ? lead : lead & (0x3F >> extra);
      for (int k = 1; k <= extra && i + k < text.size(); ++k) {
        c = (c << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
      }
      out += c;
      i += extra + 1;
    }
    return out;
  }

  static std::string gunzip(const std::filesystem::path& path) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == nullptr) {
      throw std::runtime_error("cannot open " + path.string());
    }
    std::string out;
    std::array<char, 1 << 16> buffer{};
    int n = 0;
    while ((n = gzread(file, buffer.data(), buffer.size())) > 0) {
      out.append(buffer.data(), n);
    }
    gzclose(file);
    return out;
  }

  // The word pattern of clip_tokenizer.py: 's|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+
  static std::vector<std::u32string> words(const std::string& text) {
    static const std::locale utf8_locale("C.UTF-8");
    const auto letter = [](char32_t c) {
      return std::isalpha(static_cast<wchar_t>(c), utf8_locale);
    };
    const auto digit = [](char32_t c) { return c >= U'0' && c <= U'9'; };
    const auto space = [](char32_t c) {
      return std::isspace(static_cast<wchar_t>(c), utf8_locale);
    };
    const auto symbol = [&](char32_t c) {
      return !space(c) && (c == U'_' || !(letter(c) || digit(c)));
    };
    std::u32string lower = code_points(text);
    for (char32_t& c : lower) {
      c = static_cast<char32_t>(std::tolower(static_cast<wchar_t>(c), utf8_locale));
    }
    static const std::array<std::u32string, 7> contractions{U"'s", U"'t",  U"'re", U"'ve",
                                                            U"'m", U"'ll", U"'d"};
    std::vector<std::u32string> out;
    for (std::size_t i = 0; i < lower.size();) {
      const auto contraction =
          std::find_if(contractions.begin(), contractions.end(),
                       [&](const auto& c) { return lower.compare(i, c.size(), c) == 0; });
      std::size_t end = i + 1;
      if (contraction != contractions.end()) {
        end = i + contraction->size();
      } else if (letter(lower[i])) {
        while (end < lower.size() && letter(lower[end]))
          ++end;
      } else if (symbol(lower[i])) {
        while (end < lower.size() && symbol(lower[end]))
          ++end;
      } else if (!digit(lower[i])) {
        i = end;
        continue;
      }
      out.push_back(lower.substr(i, end - i));
      i = end;
    }
    return out;
  }

  std::vector<std::string> bpe(const std::string& word) const {
    std::vector<std::string> parts;
    for (const char32_t c : code_points(word)) {
      parts.push_back(utf8(c));
    }
    parts.back() += "</w>";
    while (parts.size() > 1) {
      int best = INT_MAX;
      std::size_t at = 0;
      for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        const auto rank = ranks_.find(parts[i] + " " + parts[i + 1]);
        if (rank != ranks_.end() && rank->second < best) {
          best = rank->second;
          at = i;
        }
      }
      if (best == INT_MAX) {
        break;
      }
      const std::string first = parts[at];
      const std::string second = parts[at + 1];
      std::vector<std::string> merged;
      for (std::size_t i = 0; i < parts.size();) {
        if (i + 1 < parts.size() && parts[i] == first && parts[i + 1] == second) {
          merged.push_back(first + second);
          i += 2;
        } else {
          merged.push_back(parts[i]);
          i += 1;
        }
      }
      parts = std::move(merged);
    }
    return parts;
  }

  std::array<std::string, 256> byte_chars_;
  std::unordered_map<std::string, int> ranks_;
  std::unordered_map<std::string, int64_t> ids_;
};
