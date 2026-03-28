#include "model_family_api.cpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace multistream_yolox_yolov8_optiview {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

} // namespace

std::string yolox_not_supported_message() {
  return "YOLOX model packs are not supported yet by this example; future support is planned.";
}

std::string to_string(ModelFamily family) {
  switch (family) {
  case ModelFamily::Auto:
    return "auto";
  case ModelFamily::YoloX:
    return "yolox";
  case ModelFamily::YoloV8:
    return "yolov8";
  }
  return "auto";
}

ModelFamily parse_model_family(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "auto") {
    return ModelFamily::Auto;
  }
  if (lowered == "yolox") {
    throw std::runtime_error(yolox_not_supported_message());
  }
  if (lowered == "yolov8") {
    return ModelFamily::YoloV8;
  }
  throw std::runtime_error("model.family must be one of [auto, yolov8]");
}

ModelFamily resolve_model_family(const std::string& model_path, ModelFamily hint) {
  if (hint == ModelFamily::YoloX) {
    throw std::runtime_error(yolox_not_supported_message());
  }
  if (hint != ModelFamily::Auto) {
    return hint;
  }

  const std::string lowered = lower_copy(model_path);
  if (lowered.find("yolox") != std::string::npos) {
    throw std::runtime_error(yolox_not_supported_message());
  }
  if (lowered.find("yolo_v8") != std::string::npos || lowered.find("yolov8") != std::string::npos) {
    return ModelFamily::YoloV8;
  }
  throw std::runtime_error("unable to infer model family from model path: " + model_path);
}

} // namespace multistream_yolox_yolov8_optiview
