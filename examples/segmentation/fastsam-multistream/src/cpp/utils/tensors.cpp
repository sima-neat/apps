#include "utils/tensors.h"

#include <cstring>
#include <stdexcept>

namespace neat = simaai::neat;

namespace app {

std::vector<float> tensor_to_floats(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("tensor_to_floats: expected Float32 tensor");
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<std::uint8_t> tensor_to_u8(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::UInt8) {
    throw std::runtime_error("tensor_to_u8: expected UInt8 tensor");
  }
  return tensor.copy_dense_bytes_tight();
}

}  // namespace app
