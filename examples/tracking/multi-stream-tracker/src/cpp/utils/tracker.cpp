#include "examples/tracking/multi-stream-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

// The Kalman filter follows ByteTrack's constant-velocity model over
// (center_x, center_y, aspect, height). Its matrices keep each coordinate
// independent of the others, so the filter is stored as four 2x2 blocks
// (position, velocity) instead of one 8x8 matrix. The math is identical.

namespace multi_stream_tracker {
namespace {

constexpr double kGatedCost = 1.0e6;

using BBox = std::array<double, 4>;

double iou_xyxy(const BBox& a, const BBox& b) {
  const double xx1 = std::max(a[0], b[0]);
  const double yy1 = std::max(a[1], b[1]);
  const double xx2 = std::min(a[2], b[2]);
  const double yy2 = std::min(a[3], b[3]);
  const double inter = std::max(0.0, xx2 - xx1) * std::max(0.0, yy2 - yy1);
  const double area_a = std::max(0.0, a[2] - a[0]) * std::max(0.0, a[3] - a[1]);
  const double area_b = std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
  const double denom = area_a + area_b - inter;
  return denom > 0.0 ? inter / denom : 0.0;
}

std::array<double, 4> to_xyah(const BBox& box) {
  const double w = std::max(1e-6, box[2] - box[0]);
  const double h = std::max(1e-6, box[3] - box[1]);
  return {box[0] + w / 2, box[1] + h / 2, w / h, h};
}

class KalmanBox {
public:
  KalmanBox(const BBox& box, double position_noise, double velocity_noise)
      : wp_(position_noise), wv_(velocity_noise), pos_(to_xyah(box)) {
    const double h = pos_[3];
    const std::array<double, 4> pos_std{2 * wp_ * h, 2 * wp_ * h, 1e-2, 2 * wp_ * h};
    const std::array<double, 4> vel_std{10 * wv_ * h, 10 * wv_ * h, 1e-5, 10 * wv_ * h};
    for (int i = 0; i < 4; ++i) {
      cov_[i] = {pos_std[i] * pos_std[i], 0.0, vel_std[i] * vel_std[i]};
    }
  }

  void predict() {
    const double h = pos_[3];
    const std::array<double, 4> pos_std{wp_ * h, wp_ * h, 1e-2, wp_ * h};
    const std::array<double, 4> vel_std{wv_ * h, wv_ * h, 1e-5, wv_ * h};
    for (int i = 0; i < 4; ++i) {
      pos_[i] += vel_[i];
      const auto [a, b, c] = cov_[i];
      cov_[i] = {a + 2 * b + c + pos_std[i] * pos_std[i], b + c, c + vel_std[i] * vel_std[i]};
    }
  }

  void update(const BBox& box) {
    const auto measurement = to_xyah(box);
    const double h = pos_[3];
    const std::array<double, 4> meas_std{wp_ * h, wp_ * h, 1e-1, wp_ * h};
    for (int i = 0; i < 4; ++i) {
      const auto [a, b, c] = cov_[i];
      const double s = a + meas_std[i] * meas_std[i];
      const double k_pos = a / s;
      const double k_vel = b / s;
      const double innovation = measurement[i] - pos_[i];
      pos_[i] += k_pos * innovation;
      vel_[i] += k_vel * innovation;
      cov_[i] = {a - k_pos * a, b - k_pos * b, c - k_vel * b};
    }
  }

  void freeze_height_velocity() { vel_[3] = 0.0; }

  BBox box() const {
    const double w = pos_[2] * pos_[3];
    const double h = pos_[3];
    return {pos_[0] - w / 2, pos_[1] - h / 2, pos_[0] + w / 2, pos_[1] + h / 2};
  }

private:
  double wp_;
  double wv_;
  std::array<double, 4> pos_;
  std::array<double, 4> vel_{0.0, 0.0, 0.0, 0.0};
  // Per coordinate covariance [[a, b], [b, c]] of (position, velocity).
  std::array<std::array<double, 3>, 4> cov_{};
};

struct Track {
  int track_id = 0;
  KalmanBox kalman;
  double score = 0.0;
  int last_frame_index = 0;
  int hits = 1;
  bool confirmed = false;
  bool lost = false;
};

struct Candidate {
  BBox box;
  double score;
};

struct MatchResult {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> unmatched_tracks;
  std::vector<int> unmatched_boxes;
};

MatchResult match(const std::vector<Track*>& tracks, const std::vector<Candidate>& candidates,
                  double iou_threshold) {
  std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(candidates.size()));
  for (std::size_t t = 0; t < tracks.size(); ++t) {
    const BBox predicted = tracks[t]->kalman.box();
    for (std::size_t d = 0; d < candidates.size(); ++d) {
      const double iou = iou_xyxy(predicted, candidates[d].box);
      cost[t][d] = iou >= iou_threshold ? 1.0 - iou : kGatedCost;
    }
  }
  MatchResult result;
  result.pairs = linear_assignment(cost, 1.0 - iou_threshold);
  std::vector<bool> track_used(tracks.size(), false);
  std::vector<bool> box_used(candidates.size(), false);
  for (const auto& [t, d] : result.pairs) {
    track_used[static_cast<std::size_t>(t)] = true;
    box_used[static_cast<std::size_t>(d)] = true;
  }
  for (std::size_t t = 0; t < tracks.size(); ++t) {
    if (!track_used[t])
      result.unmatched_tracks.push_back(static_cast<int>(t));
  }
  for (std::size_t d = 0; d < candidates.size(); ++d) {
    if (!box_used[d])
      result.unmatched_boxes.push_back(static_cast<int>(d));
  }
  return result;
}

std::string lower_trim(const std::string& value) {
  const auto start = value.find_first_not_of(" \t");
  if (start == std::string::npos)
    return {};
  const auto end = value.find_last_not_of(" \t");
  std::string out = value.substr(start, end - start + 1);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

bool parse_number(const std::string& value, double& out) {
  try {
    std::size_t index = 0;
    out = std::stod(value, &index);
    return index == value.size();
  } catch (const std::exception&) {
    return false;
  }
}

} // namespace

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> labels = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush",
  };
  return labels;
}

std::pair<int, std::string> resolve_class(const std::string& value) {
  const auto& labels = coco_labels();
  const std::string name = lower_trim(value);
  if (!name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      })) {
    const long id = name.size() > 3 ? -1 : std::stol(name);
    if (id < 0 || id >= static_cast<long>(labels.size())) {
      throw std::runtime_error("unsupported class id " + name + "; expected 0.." +
                               std::to_string(labels.size() - 1));
    }
    return {static_cast<int>(id), labels[static_cast<std::size_t>(id)]};
  }
  const auto it = std::find(labels.begin(), labels.end(), name);
  if (name.empty() || it == labels.end()) {
    throw std::runtime_error("unsupported class: '" + value + "'; use a COCO class name or id");
  }
  return {static_cast<int>(it - labels.begin()), name};
}

void ClassTrackerConfig::validate() const {
  const std::string name = "tracking.classes[" + label + "]";
  const std::pair<const char*, double> unit_values[] = {
      {"high_score_threshold", high_score_threshold},
      {"low_score_threshold", low_score_threshold},
      {"new_track_threshold", new_track_threshold},
      {"match_iou_threshold", match_iou_threshold},
      {"low_match_iou_threshold", low_match_iou_threshold},
  };
  for (const auto& [key, value] : unit_values) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
      throw std::runtime_error(name + "." + key + " must be between 0 and 1");
    }
  }
  if (low_score_threshold > high_score_threshold) {
    throw std::runtime_error(name + ".low_score_threshold must be <= high_score_threshold");
  }
  if (new_track_threshold < high_score_threshold) {
    throw std::runtime_error(name + ".new_track_threshold must be >= high_score_threshold");
  }
  if (max_missing_frames < 0) {
    throw std::runtime_error(name + ".max_missing_frames must be >= 0");
  }
  if (min_confirmed_hits < 1) {
    throw std::runtime_error(name + ".min_confirmed_hits must be >= 1");
  }
  const std::pair<const char*, double> noise_values[] = {
      {"position_noise", position_noise},
      {"velocity_noise", velocity_noise},
  };
  for (const auto& [key, value] : noise_values) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error(name + "." + key + " must be > 0");
    }
  }
}

std::vector<ClassTrackerConfig> parse_class_configs(const std::vector<ClassEntry>& entries) {
  if (entries.empty() || entries.size() > kMaxClasses) {
    throw std::runtime_error("tracking.classes must list 1 to " + std::to_string(kMaxClasses) +
                             " classes");
  }
  std::vector<ClassTrackerConfig> configs;
  std::set<int> seen;
  for (std::size_t index = 0; index < entries.size(); ++index) {
    const std::string prefix = "tracking.classes[" + std::to_string(index) + "]";
    const auto class_it =
        std::find_if(entries[index].begin(), entries[index].end(),
                     [](const auto& item) { return item.first == "class"; });
    if (class_it == entries[index].end()) {
      throw std::runtime_error(prefix + " must be a mapping with a 'class' key");
    }
    ClassTrackerConfig config;
    std::tie(config.class_id, config.label) = resolve_class(class_it->second);
    if (!seen.insert(config.class_id).second) {
      throw std::runtime_error(prefix + ": duplicate class '" + config.label + "'");
    }
    for (const auto& [key, value] : entries[index]) {
      if (key == "class")
        continue;
      double number = 0.0;
      const bool is_integer_key = key == "max_missing_frames" || key == "min_confirmed_hits";
      if (key != "high_score_threshold" && key != "low_score_threshold" &&
          key != "new_track_threshold" && key != "match_iou_threshold" &&
          key != "low_match_iou_threshold" && key != "position_noise" &&
          key != "velocity_noise" && !is_integer_key) {
        throw std::runtime_error(prefix + ": unknown key '" + key + "'");
      }
      if (!parse_number(value, number)) {
        throw std::runtime_error(prefix + "." + key + " must be numeric");
      }
      if (is_integer_key && (value.find_first_of(".eE") != std::string::npos ||
                             std::floor(number) != number)) {
        throw std::runtime_error(prefix + "." + key + " must be an integer");
      }
      if (key == "high_score_threshold")
        config.high_score_threshold = number;
      else if (key == "low_score_threshold")
        config.low_score_threshold = number;
      else if (key == "new_track_threshold")
        config.new_track_threshold = number;
      else if (key == "match_iou_threshold")
        config.match_iou_threshold = number;
      else if (key == "low_match_iou_threshold")
        config.low_match_iou_threshold = number;
      else if (key == "position_noise")
        config.position_noise = number;
      else if (key == "velocity_noise")
        config.velocity_noise = number;
      else if (key == "max_missing_frames")
        config.max_missing_frames = static_cast<int>(number);
      else
        config.min_confirmed_hits = static_cast<int>(number);
    }
    config.validate();
    configs.push_back(config);
  }
  return configs;
}

std::vector<std::pair<int, int>> linear_assignment(const std::vector<std::vector<double>>& cost,
                                                   double max_cost) {
  const std::size_t rows = cost.size();
  const std::size_t cols = rows > 0 ? cost[0].size() : 0;
  if (rows == 0 || cols == 0) {
    return {};
  }
  // Shortest augmenting path with potentials; requires n <= m.
  const bool transposed = rows > cols;
  const std::size_t n = transposed ? cols : rows;
  const std::size_t m = transposed ? rows : cols;
  const auto at = [&](std::size_t r, std::size_t c) {
    return transposed ? cost[c][r] : cost[r][c];
  };
  constexpr double kInf = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<std::size_t> owner(m + 1, 0), way(m + 1, 0);
  for (std::size_t row = 1; row <= n; ++row) {
    owner[0] = row;
    std::size_t col0 = 0;
    std::vector<double> min_value(m + 1, kInf);
    std::vector<bool> used(m + 1, false);
    do {
      used[col0] = true;
      const std::size_t row0 = owner[col0];
      double delta = kInf;
      std::size_t col1 = 0;
      for (std::size_t col = 1; col <= m; ++col) {
        if (used[col])
          continue;
        const double reduced = at(row0 - 1, col - 1) - u[row0] - v[col];
        if (reduced < min_value[col]) {
          min_value[col] = reduced;
          way[col] = col0;
        }
        if (min_value[col] < delta) {
          delta = min_value[col];
          col1 = col;
        }
      }
      for (std::size_t col = 0; col <= m; ++col) {
        if (used[col]) {
          u[owner[col]] += delta;
          v[col] -= delta;
        } else {
          min_value[col] -= delta;
        }
      }
      col0 = col1;
    } while (owner[col0] != 0);
    do {
      const std::size_t col1 = way[col0];
      owner[col0] = owner[col1];
      col0 = col1;
    } while (col0 != 0);
  }

  std::vector<std::pair<int, int>> pairs;
  for (std::size_t col = 1; col <= m; ++col) {
    const std::size_t row = owner[col];
    if (row == 0)
      continue;
    const int r = static_cast<int>(transposed ? col - 1 : row - 1);
    const int c = static_cast<int>(transposed ? row - 1 : col - 1);
    if (cost[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] <= max_cost) {
      pairs.emplace_back(r, c);
    }
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

struct ByteTracker::Impl {
  std::vector<std::unique_ptr<Track>> tracks;
};

ByteTracker::ByteTracker(ClassTrackerConfig config, std::shared_ptr<int> next_track_id)
    : config_(std::move(config)),
      next_track_id_(next_track_id ? std::move(next_track_id) : std::make_shared<int>(1)),
      impl_(std::make_unique<Impl>()) {
  config_.validate();
}

ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

int ByteTracker::active_track_count() const {
  return static_cast<int>(impl_->tracks.size());
}

std::vector<TrackedDetection> ByteTracker::update(const std::vector<Detection>& detections,
                                                  int frame_index) {
  const ClassTrackerConfig& cfg = config_;
  std::vector<Candidate> high;
  std::vector<Candidate> low;
  for (const auto& det : detections) {
    if (det.class_id != cfg.class_id)
      continue;
    const BBox box{det.x1, det.y1, det.x2, det.y2};
    if (box[2] <= box[0] || box[3] <= box[1])
      continue;
    const double score = det.score;
    if (score >= cfg.high_score_threshold) {
      high.push_back({box, score});
    } else if (score >= cfg.low_score_threshold) {
      low.push_back({box, score});
    }
  }

  auto& tracks = impl_->tracks;
  for (auto& track : tracks) {
    if (track->lost)
      track->kalman.freeze_height_velocity();
    track->kalman.predict();
  }

  std::vector<Track*> updated;
  const auto apply = [&](Track* track, const Candidate& candidate) {
    track->kalman.update(candidate.box);
    track->score = candidate.score;
    track->last_frame_index = frame_index;
    track->hits += 1;
    track->lost = false;
    track->confirmed = track->confirmed || track->hits >= cfg.min_confirmed_hits;
    updated.push_back(track);
  };

  // Stage 1: confirmed tracks (active or lost) against high-score detections.
  std::vector<Track*> pool;
  for (auto& track : tracks) {
    if (track->confirmed)
      pool.push_back(track.get());
  }
  const MatchResult first = match(pool, high, cfg.match_iou_threshold);
  for (const auto& [t, d] : first.pairs) {
    apply(pool[static_cast<std::size_t>(t)], high[static_cast<std::size_t>(d)]);
  }

  // Stage 2: still-active tracks recover through low-score detections.
  std::vector<Track*> active;
  for (const int t : first.unmatched_tracks) {
    if (!pool[static_cast<std::size_t>(t)]->lost)
      active.push_back(pool[static_cast<std::size_t>(t)]);
  }
  const MatchResult second = match(active, low, cfg.low_match_iou_threshold);
  for (const auto& [t, d] : second.pairs) {
    apply(active[static_cast<std::size_t>(t)], low[static_cast<std::size_t>(d)]);
  }
  for (const int t : second.unmatched_tracks) {
    active[static_cast<std::size_t>(t)]->lost = true;
  }

  // Stage 3: tentative tracks against the remaining high-score detections.
  std::vector<Track*> tentative;
  for (auto& track : tracks) {
    if (!track->confirmed)
      tentative.push_back(track.get());
  }
  std::vector<Candidate> remaining;
  for (const int d : first.unmatched_boxes) {
    remaining.push_back(high[static_cast<std::size_t>(d)]);
  }
  const MatchResult third = match(tentative, remaining, cfg.match_iou_threshold);
  for (const auto& [t, d] : third.pairs) {
    apply(tentative[static_cast<std::size_t>(t)], remaining[static_cast<std::size_t>(d)]);
  }
  std::unordered_set<const Track*> dropped;
  for (const int t : third.unmatched_tracks) {
    dropped.insert(tentative[static_cast<std::size_t>(t)]);
  }

  std::vector<std::unique_ptr<Track>> survivors;
  for (auto& track : tracks) {
    if (dropped.count(track.get()) == 0 &&
        frame_index - track->last_frame_index <= cfg.max_missing_frames) {
      survivors.push_back(std::move(track));
    }
  }
  for (const int d : third.unmatched_boxes) {
    const Candidate& candidate = remaining[static_cast<std::size_t>(d)];
    if (candidate.score < cfg.new_track_threshold)
      continue;
    auto track = std::make_unique<Track>(Track{
        (*next_track_id_)++,
        KalmanBox(candidate.box, cfg.position_noise, cfg.velocity_noise),
        candidate.score,
        frame_index,
        1,
        cfg.min_confirmed_hits <= 1,
        false,
    });
    updated.push_back(track.get());
    survivors.push_back(std::move(track));
  }
  tracks = std::move(survivors);

  std::vector<TrackedDetection> output;
  for (const Track* track : updated) {
    if (!track->confirmed)
      continue;
    const BBox box = track->kalman.box();
    output.push_back(TrackedDetection{
        track->track_id,
        static_cast<float>(box[0]),
        static_cast<float>(box[1]),
        static_cast<float>(box[2]),
        static_cast<float>(box[3]),
        static_cast<float>(track->score),
        cfg.class_id,
        cfg.label,
    });
  }
  std::sort(output.begin(), output.end(),
            [](const auto& a, const auto& b) { return a.track_id < b.track_id; });
  return output;
}

MultiClassTracker::MultiClassTracker(const std::vector<ClassTrackerConfig>& configs) {
  auto next_track_id = std::make_shared<int>(1);
  trackers_.reserve(configs.size());
  for (const auto& config : configs) {
    trackers_.emplace_back(config, next_track_id);
  }
}

int MultiClassTracker::active_track_count() const {
  int count = 0;
  for (const auto& tracker : trackers_) {
    count += tracker.active_track_count();
  }
  return count;
}

std::vector<TrackedDetection> MultiClassTracker::update(const std::vector<Detection>& detections,
                                                        int frame_index) {
  std::vector<TrackedDetection> tracked;
  for (auto& tracker : trackers_) {
    auto class_tracks = tracker.update(detections, frame_index);
    tracked.insert(tracked.end(), class_tracks.begin(), class_tracks.end());
  }
  return tracked;
}

} // namespace multi_stream_tracker
