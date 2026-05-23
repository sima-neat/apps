#pragma once

#include <string>

namespace multistream_object_detection_insight {

enum class ModelFamily {
  Auto,
  YoloV8,
};

std::string to_string(ModelFamily family);
ModelFamily resolve_model_family(const std::string& model_path,
                                 ModelFamily hint = ModelFamily::Auto);

} // namespace multistream_object_detection_insight
