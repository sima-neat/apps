#pragma once

#include <string>

namespace multistream_yolox_yolov8_optiview {

enum class ModelFamily {
  Auto,
  YoloX,
  YoloV8,
};

std::string to_string(ModelFamily family);
ModelFamily parse_model_family(const std::string& value);
ModelFamily resolve_model_family(const std::string& model_path, ModelFamily hint);

} // namespace multistream_yolox_yolov8_optiview
