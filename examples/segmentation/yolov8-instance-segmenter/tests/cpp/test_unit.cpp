#define main yolov8_segmenter_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  // Exercise the production decoder with CPU tensors, without a model or device.
  namespace neat = simaai::neat;
  std::vector<neat::Tensor> tensors;
  for (const int channels : {64, 64, 64, 80, 80, 80, 32, 32, 32}) {
    tensors.push_back(neat::Tensor::from_vector(std::vector<float>(channels, 0.0f),
                                                {1, 1, 1, channels}, neat::TensorMemory::CPU));
  }
  std::vector<float> probabilities(80, 0.0f);
  probabilities[3] = 0.74f;
  tensors[3] = neat::Tensor::from_vector(probabilities, {1, 1, 1, 80}, neat::TensorMemory::CPU);
  tensors.push_back(neat::Tensor::from_vector(std::vector<float>(32, 0.0f), {1, 1, 1, 32},
                                              neat::TensorMemory::CPU));
  TensorHWC proto;
  const auto boxes =
      decode_yolov8_instances_from_detess(tensors, 640, 0.70f, 0.45f, 10, false, proto);
  if (boxes.size() != 1 || boxes[0].class_id != 3 || std::abs(boxes[0].score - 0.74f) > 1e-6f) {
    std::cerr << "[FAIL] decoder must preserve probability 0.74 above threshold 0.70\n";
    ++failures;
  }
  if (!decode_yolov8_instances_from_detess(tensors, 640, 0.75f, 0.45f, 10, false, proto).empty()) {
    std::cerr << "[FAIL] decoder must reject probability 0.74 below threshold 0.75\n";
    ++failures;
  }
  // Test 1: --help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] --help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] --help printed usage\n";
    }
  }

  // Test 2: unknown flag is rejected.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --bogus: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag rejected\n";
    }
  }

  // Test 3: bad config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad config: expected nonzero exit\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
