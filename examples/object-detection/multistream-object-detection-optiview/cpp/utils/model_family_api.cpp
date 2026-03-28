#pragma once

#include <string>

namespace multistream_object_detection_optiview {

enum class ModelFamily {
  Auto,
  YoloV8,
};

std::string to_string(ModelFamily family);
ModelFamily parse_model_family(const std::string& value);
ModelFamily resolve_model_family(const std::string& model_path, ModelFamily hint);

} // namespace multistream_object_detection_optiview
