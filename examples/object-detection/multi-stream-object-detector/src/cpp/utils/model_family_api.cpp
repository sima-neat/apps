#pragma once

#include <string>

namespace multi_stream_object_detector {

enum class ModelFamily {
  Auto,
  YoloV26,
};

std::string to_string(ModelFamily family);
ModelFamily resolve_model_family(const std::string& model_path,
                                 ModelFamily hint = ModelFamily::Auto);

} // namespace multi_stream_object_detector
