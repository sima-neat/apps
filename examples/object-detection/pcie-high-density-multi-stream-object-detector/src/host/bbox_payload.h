#pragma once

#include <gst/gst.h>
#include <simaai/scheduler/pcie/neat_pcie_return_layout.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie_high_density {

inline std::vector<std::uint8_t> copy_bbox_payload(GstSample* sample) {
  GstCaps* caps = sample ? gst_sample_get_caps(sample) : nullptr;
  const GstStructure* structure =
      caps && gst_caps_get_size(caps) == 1 ? gst_caps_get_structure(caps, 0) : nullptr;
  if (!structure || !gst_structure_has_name(structure, "application/vnd.simaai.tensor") ||
      g_strcmp0(gst_structure_get_string(structure, "format"), "BBOX") != 0) {
    throw std::runtime_error("PCIe result must have BBOX tensor caps");
  }

  NeatPcieReturnLayout layout{};
  GError* error = nullptr;
  if (!neat_pcie_return_layout_read_caps(caps, &layout, &error)) {
    const std::string message = error ? error->message : "invalid return caps";
    g_clear_error(&error);
    throw std::runtime_error("invalid PCIe BBOX return layout: " + message);
  }
  if (layout.count > 1) {
    throw std::runtime_error("PCIe BBOX return layout must contain one logical region");
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("PCIe result sample has no buffer");
  }
  const gsize raw_size = gst_buffer_get_size(buffer);
  if (!neat_pcie_return_layout_validate_payload(&layout, raw_size, &error)) {
    const std::string message = error ? error->message : "invalid allocation size";
    g_clear_error(&error);
    throw std::runtime_error("invalid PCIe BBOX payload: " + message);
  }

  // A layout with no regions is the Scheduler's legacy variable-length byte stream.
  const gsize offset = layout.count ? layout.regions[0].offset : 0;
  const gsize size = layout.count ? layout.regions[0].size : raw_size;
  if (offset > raw_size || size > raw_size - offset) {
    throw std::runtime_error("PCIe BBOX region exceeds the received allocation");
  }
  std::vector<std::uint8_t> bytes(size);
  if (gst_buffer_extract(buffer, offset, bytes.data(), size) != size) {
    throw std::runtime_error("failed to copy PCIe BBOX region");
  }
  return bytes;
}

} // namespace pcie_high_density
