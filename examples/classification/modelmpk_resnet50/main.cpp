#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<simaai::neat::Tensor> tensors_from_sample(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    out.reserve(sample.fields.size());
    for (const auto& field : sample.fields) {
      if (field.kind != simaai::neat::SampleKind::Tensor || !field.tensor.has_value()) {
        throw std::runtime_error("bundle field missing tensor");
      }
      out.push_back(*field.tensor);
    }
    return out;
  }
  throw std::runtime_error("expected tensor output");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <image>\n";
    return 2;
  }

  const std::string tar_gz = argv[1];
  const std::string image_path = argv[2];

  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    std::cerr << "Failed to read image: " << image_path << "\n";
    return 3;
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "RGB";
  model_opt.input_max_width = rgb.cols;
  model_opt.input_max_height = rgb.rows;
  model_opt.input_max_depth = 3;
  simaai::neat::Model model(tar_gz, model_opt);

  try {
    {
      simaai::neat::Session p;
      p.add(model.session());
      std::cout << "Pipeline:\n" << p.describe_backend() << "\n" << std::flush;
    }
    auto run = model.build();
    simaai::neat::Tensor input = simaai::neat::from_cv_mat(
        rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);
    if (!run.push(input)) {
      throw std::runtime_error("push failed");
    }
    auto out_opt = run.pull();
    if (!out_opt.has_value()) {
      throw std::runtime_error("pull returned no outputs");
    }
    const std::vector<simaai::neat::Tensor> outputs = tensors_from_sample(*out_opt);
    if (outputs.empty()) {
      throw std::runtime_error("pull returned no outputs");
    }
    const simaai::neat::Tensor& t = outputs.front();
    run.close();
    if (t.semantic.image.has_value() &&
        t.semantic.image->format == simaai::neat::ImageSpec::PixelFormat::NV12) {
      std::cout << "OK: NV12 output " << t.width() << "x" << t.height() << "\n";
    } else {
      std::cout << "OK: tensor output, dims=" << t.shape.size() << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 5;
  }
}
