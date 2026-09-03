#pragma once

/// Minimal, self-contained SHA-256 (FIPS 180-4). No external crypto dependency --
/// used only to pin a memory bank's `bank_meta.json` to the model package it was
/// built against (see patchcore_memory_bank.h), not for any security-sensitive
/// purpose.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace patchcore {

/// Lowercase hex SHA-256 digest of a byte buffer.
std::string sha256_hex(const void* data, std::size_t size);

/// Lowercase hex SHA-256 digest of a file's contents, read in chunks.
/// Throws `std::runtime_error` if the file cannot be opened.
std::string sha256_file(const std::filesystem::path& path);

} // namespace patchcore
