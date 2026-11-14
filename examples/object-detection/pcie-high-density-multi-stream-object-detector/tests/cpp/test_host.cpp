#include "../../src/host/bbox_payload.h"

#include <cstring>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "[FAIL] " << message << '\n';
  }
}

GstCaps* bbox_caps(guint regions = 1) {
  GstCaps* caps = gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                      "BBOX", nullptr);
  NeatPcieReturnLayout layout{};
  layout.raw_bytes = 4096;
  layout.count = regions;
  for (guint i = 0; i < regions; ++i) {
    layout.regions[i] = {1408 + i * 1204, 1204};
    layout.packed_bytes += 1204;
  }
  neat_pcie_return_layout_set_caps(caps, &layout);
  return caps;
}

void check_payload(GstCaps* caps, gsize raw_size, bool accepted, const std::string& description) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, raw_size, nullptr);
  gst_buffer_memset(buffer, 0, 0, raw_size);
  const guint32 wrong_count = 1051;
  const guint32 correct_count = 5;
  if (raw_size >= 2612) {
    gst_buffer_fill(buffer, 0, &wrong_count, sizeof(wrong_count));
    gst_buffer_fill(buffer, 1408, &correct_count, sizeof(correct_count));
  } else if (raw_size == 1204) {
    gst_buffer_fill(buffer, 0, &correct_count, sizeof(correct_count));
  }
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  try {
    const auto bytes = pcie_high_density::copy_bbox_payload(sample);
    expect(accepted, description + " must fail");
    if (accepted) {
      expect(bytes.size() == 1204, "only the logical BBOX span is copied");
      guint32 count = 0;
      if (bytes.size() >= sizeof(count)) {
        std::memcpy(&count, bytes.data(), sizeof(count));
      }
      expect(count == correct_count, "count comes from offset 1408, not the allocation prefix");
    }
  } catch (const std::exception& error) {
    expect(!accepted, description + ": " + error.what());
  }
  gst_sample_unref(sample);
  gst_buffer_unref(buffer);
  if (caps) {
    gst_caps_unref(caps);
  }
}

} // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  check_payload(bbox_caps(), 4096, true, "padded BBOX return allocation");
  check_payload(gst_caps_new_simple("application/vnd.simaai.tensor", "format", G_TYPE_STRING,
                                    "BBOX", nullptr),
                1204, true, "legacy BBOX byte stream");
  check_payload(bbox_caps(), 4095, false, "truncated return allocation");
  check_payload(bbox_caps(), 4097, false, "oversized return allocation");
  check_payload(bbox_caps(2), 4096, false, "multiple logical BBOX regions");

  GstCaps* caps = bbox_caps();
  gst_structure_set(gst_caps_get_structure(caps, 0), "pcie-return-layout-version", G_TYPE_UINT, 99U,
                    nullptr);
  check_payload(caps, 4096, false, "unsupported layout version");

  caps = bbox_caps();
  gst_structure_remove_field(gst_caps_get_structure(caps, 0), "pcie-return-regions");
  check_payload(caps, 4096, false, "partial layout fields");

  caps = bbox_caps();
  gst_structure_set(gst_caps_get_structure(caps, 0), "pcie-return-raw-bytes", G_TYPE_UINT64,
                    guint64{2048}, nullptr);
  check_payload(caps, 4096, false, "region extends past allocation");

  caps = bbox_caps();
  gst_structure_set(gst_caps_get_structure(caps, 0), "format", G_TYPE_STRING, "MLA", nullptr);
  check_payload(caps, 4096, false, "unexpected tensor format");
  check_payload(nullptr, 4096, false, "missing caps");

  if (failures) {
    return 1;
  }
  std::cout << "all PCIe BBOX return-layout checks passed\n";
  return 0;
}
