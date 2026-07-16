#pragma once

#include "neat.h"

#include <cstdint>
#include <vector>

namespace app {

// Copy a dense tensor payload to a typed host vector (throws on dtype mismatch).
std::vector<float> tensor_to_floats(const simaai::neat::Tensor& tensor);
std::vector<std::uint8_t> tensor_to_u8(const simaai::neat::Tensor& tensor);

}  // namespace app
