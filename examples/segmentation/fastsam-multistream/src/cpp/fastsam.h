#pragma once

#include "config.h"
#include "neat.h"
#include "neat/models.h"

#include <opencv2/core.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace app {

class Fastsam {
 public:
  struct Geometry {
    double scale = 1.0;
    int pad_x = 0;
    int pad_y = 0;
  };

  struct Segmentation {
    std::vector<std::array<float, 6>> boxes;  // x1,y1,x2,y2,score,cls
    std::vector<cv::Mat> masks;               // 160x160 CV_8UC1
  };

  struct Crop {
    cv::Mat window;   // ROI view into the source frame (shares data to keep the frame alive)
    cv::Mat submask;  // proto submask, CV_8UC1 (owned copy)
  };

  Fastsam(const AppConfig& cfg, const simaai::neat::RunOptions& run_opt, int frame_w, int frame_h);

  simaai::neat::TensorList run(const cv::Mat& rgb, int timeout_ms);
  void close();

  static Geometry get_letterbox_geometry(int orig_w, int orig_h, int infer_size);

  static Segmentation decode(const simaai::neat::TensorList& model_out, int top_k);

  static std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const Geometry& geom,
                                             int max_points = 80, double eps_frac = 0.004);

  static std::optional<Crop> object_crop(const cv::Mat& proto_mask, const cv::Mat& frame_rgb,
                                         const Geometry& geom, double min_area, double max_frac,
                                         double max_box_frac = 1.0, double margin = 0.1);

 private:
  std::unique_ptr<simaai::neat::Model> model_;
  simaai::neat::Model::Runner runner_;
  std::mutex mu_;
};

}  // namespace app
