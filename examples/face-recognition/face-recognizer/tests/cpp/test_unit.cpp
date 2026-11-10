// Unit test for face-recognizer: CLI handling, the configuration rules --validate-config-only
// can reach without a model or a source, SCRFD decoding, face alignment, matching and the
// gallery file. The Python unit test decodes the same tensors and reads the same
// tests/fixtures/gallery.bin, so both implementations agree on the faces and the file.
#define main face_recognizer_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::validate_config_body;

namespace {

constexpr const char* kExampleName = "face-recognizer";
const fs::path kGalleryFixture =
    "examples/face-recognition/face-recognizer/tests/fixtures/gallery.bin";

const std::string kRtsp = "  type: rtsp\n"
                          "  rtsp:\n"
                          "    url: rtsp://127.0.0.1:8554/src1\n";
const std::string kInsight = "output:\n"
                             "  insight:\n"
                             "    host: 127.0.0.1\n";

std::string config_body(const std::string& source, const std::string& extra = kInsight) {
  return "model:\n"
         "  scrfd: models/scrfd.tar.gz\n"
         "  arcface: models/arcface.tar.gz\n"
         "gallery:\n"
         "  path: gallery.bin\n"
         "source:\n" +
         source + extra;
}

bool close_to(float actual, float expected, float tolerance = 1e-3f) {
  return std::fabs(actual - expected) <= tolerance;
}

bool test_help_runs(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(r.exit_code == 0, "help exits with code 0") &&
         expect_contains(r.stdout_text, "--enroll", "help lists --enroll");
}

bool test_unknown_flag_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(r.exit_code != 0, "unknown flag exits non-zero");
}

bool test_missing_config_file_exits_with_code_2(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
  return expect_true(r.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(r.stderr_text, "config file not found", "missing config is named");
}

bool test_validate_config_only_reports_the_mode(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "rtsp", config_body(kRtsp));
  const auto file = validate_config_body(kExampleName, binary, "video_file",
                                         config_body("  type: video_file\n"
                                                     "  video_file:\n"
                                                     "    path: clip.mp4\n",
                                                     ""));
  return expect_true(r.exit_code == 0, "an RTSP source validates") &&
         expect_contains(r.stdout_text, "mode=recognize", "the mode is reported") &&
         expect_true(file.exit_code == 0, "a video file validates without an Insight host");
}

bool test_configuration_rules_are_enforced(const std::string& binary) {
  struct Case {
    const char* name;
    std::string body;
    const char* message;
  };
  const std::vector<Case> cases = {
      {"bad_source_type", config_body("  type: http\n"),
       "source.type must be rtsp, usb or video_file"},
      {"no_rtsp_url", config_body("  type: rtsp\n"),
       "source.rtsp.url must be set when source.type=rtsp"},
      {"no_usb_camera", config_body("  type: usb\n"),
       "source.usb.camera must be set when source.type=usb"},
      {"odd_usb_width", config_body("  type: usb\n  usb:\n    camera: camera\n    width: 1281\n"),
       "source.usb.width and height must be multiples of 8 and at most 2040"},
      {"no_video_path", config_body("  type: video_file\n"),
       "source.video_file.path must be set when source.type=video_file"},
      {"no_insight_host", config_body(kRtsp, ""),
       "output.insight.host must be set for source.type=rtsp/usb"},
      {"min_score_of_one", config_body(kRtsp, kInsight + "inference:\n  min_score: 1.0\n"),
       "inference.min_score must be greater than 0 and less than 1"},
      {"zero_max_faces", config_body(kRtsp, kInsight + "inference:\n  max_faces: 0\n"),
       "inference.max_faces must be > 0"},
      {"negative_frames", config_body(kRtsp, kInsight + "inference:\n  frames: -1\n"),
       "inference.frames must be >= 0"},
      {"margin_above_one", config_body(kRtsp, kInsight + "match:\n  margin: 1.5\n"),
       "match.margin must be between 0 and 1"},
  };
  bool ok = true;
  for (const auto& c : cases) {
    const auto r =
        validate_config_body(kExampleName, binary, std::string("rule_") + c.name, c.body);
    ok &= expect_true(r.exit_code != 0, std::string(c.name) + " is rejected") &&
          expect_contains(r.stderr_text, c.message, std::string(c.name) + " names its key");
  }
  return ok;
}

// --enroll reads the enrollment section instead of the source.
bool test_enrollment_rules_are_enforced() {
  const auto problem = [](const AppConfig& cfg) -> std::string {
    try {
      validate_config(cfg, true);
    } catch (const std::exception& e) {
      return e.what();
    }
    return "";
  };
  AppConfig cfg;
  cfg.scrfd_model = "models/scrfd.tar.gz";
  cfg.arcface_model = "models/arcface.tar.gz";
  cfg.gallery_path = "gallery.bin";
  bool ok =
      expect_contains(problem(cfg), "enrollment.images_dir or enrollment.video_path must be set",
                      "enrollment needs images or a video");
  cfg.enroll_video_path = "person.mp4";
  ok &= expect_contains(problem(cfg), "enrollment.name must be set with enrollment.video_path",
                        "a video needs the person's name");
  cfg.enroll_name = "PersonA";
  ok &= expect_true(problem(cfg).empty(), "a named video validates without a source");
  cfg.enroll_sample_every = 0;
  ok &= expect_contains(problem(cfg), "enrollment.sample_every must be > 0",
                        "sample_every is checked");
  return ok;
}

// SCRFD output for a 640x640 input with every anchor silent: scores, boxes and landmarks
// at strides 8, 16 and 32, two anchors per cell.
std::vector<std::vector<float>> silent_scrfd_output() {
  std::vector<std::vector<float>> output(9);
  const int strides[3] = {8, 16, 32};
  for (int s = 0; s < 3; ++s) {
    const std::size_t anchors = static_cast<std::size_t>(640 / strides[s]) * (640 / strides[s]) * 2;
    output[s].assign(anchors, -20.0f);
    output[3 + s].assign(anchors * 4, 0.0f);
    output[6 + s].assign(anchors * 10, 0.0f);
  }
  return output;
}

std::vector<simaai::neat::Tensor> as_tensors(const std::vector<std::vector<float>>& output) {
  std::vector<simaai::neat::Tensor> tensors;
  for (const auto& values : output) {
    tensors.push_back(simaai::neat::Tensor::from_vector(
        values, {static_cast<int64_t>(values.size())}, simaai::neat::TensorMemory::CPU));
  }
  return tensors;
}

bool test_scrfd_output_decodes_to_source_pixels() {
  auto output = silent_scrfd_output();
  // Stride 16, cell (column 10, row 12): both anchors fire on the same box.
  const std::size_t strong = (12 * 40 + 10) * 2;
  output[1][strong] = 3.0f;
  output[1][strong + 1] = 2.0f;
  for (const std::size_t anchor : {strong, strong + 1}) {
    const float box[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::copy(box, box + 4, output[4].begin() + anchor * 4);
    for (int k = 0; k < 5; ++k) {
      output[7][anchor * 10 + k * 2] = 0.5f * k;
      output[7][anchor * 10 + k * 2 + 1] = 1.0f;
    }
  }
  // Stride 32, cell (column 2, row 15): a weaker face elsewhere.
  const std::size_t weak = (15 * 20 + 2) * 2;
  output[2][weak] = 1.0f;
  std::fill_n(output[5].begin() + weak * 4, 4, 1.0f);

  // A 1280x720 frame is halved and padded by 140 rows above and below.
  const auto letterbox = face_recog::letterbox_for(1280, 720);
  face_recog::ScrfdConfig cfg;
  const auto faces = face_recog::decode_scrfd(as_tensors(output), cfg, letterbox);
  bool ok = expect_true(faces.size() == 2, "overlapping anchors yield one face each");
  if (!ok) {
    return false;
  }
  const auto& face = faces.front();
  ok &= expect_true(close_to(face.score, 0.9526f), "the stronger face comes first");
  ok &= expect_true(close_to(face.x1, 288.0f) && close_to(face.y1, 40.0f) &&
                        close_to(face.x2, 416.0f) && close_to(face.y2, 232.0f),
                    "the box is in source-frame pixels");
  ok &= expect_true(close_to(face.landmarks[0], 320.0f) && close_to(face.landmarks[1], 136.0f) &&
                        close_to(face.landmarks[8], 384.0f) && close_to(face.landmarks[9], 136.0f),
                    "the landmarks are in source-frame pixels");
  ok &= expect_true(close_to(faces[1].x1, 64.0f) && close_to(faces[1].y1, 616.0f),
                    "the weaker face keeps its own box");

  cfg.max_faces = 1;
  ok &= expect_true(face_recog::decode_scrfd(as_tensors(output), cfg, letterbox).size() == 1,
                    "max_faces limits the result");
  cfg.max_faces = 100;
  cfg.min_score = 0.8f;
  ok &= expect_true(face_recog::decode_scrfd(as_tensors(output), cfg, letterbox).size() == 1,
                    "min_score drops the weaker face");
  return ok;
}

bool test_alignment_moves_the_landmarks_onto_the_template() {
  // The ArcFace template doubled in size and shifted, marked with white dots.
  const float tmpl[5][2] = {{38.2946f, 51.6963f},
                            {73.5318f, 51.5014f},
                            {56.0252f, 71.7366f},
                            {41.5493f, 92.3655f},
                            {70.7299f, 92.2041f}};
  cv::Mat frame(360, 400, CV_8UC3, cv::Scalar(0, 0, 0));
  face_recog::Landmarks landmarks{};
  for (int i = 0; i < 5; ++i) {
    landmarks[i * 2] = tmpl[i][0] * 2.0f + 50.0f;
    landmarks[i * 2 + 1] = tmpl[i][1] * 2.0f + 30.0f;
    cv::circle(frame, cv::Point2f(landmarks[i * 2], landmarks[i * 2 + 1]), 4,
               cv::Scalar(255, 255, 255), cv::FILLED);
  }
  const cv::Mat crop = face_recog::align_face(frame, landmarks);
  bool ok = expect_true(crop.cols == 112 && crop.rows == 112, "the crop is 112x112");
  for (int i = 0; i < 5; ++i) {
    const auto pixel = crop.at<cv::Vec3b>(static_cast<int>(tmpl[i][1] + 0.5f),
                                          static_cast<int>(tmpl[i][0] + 0.5f));
    ok &= expect_true(pixel[0] > 200, "landmark " + std::to_string(i) + " lands on the template");
  }
  return ok;
}

face_recog::Embedding unit_embedding(int axis, float along_next = 0.0f) {
  face_recog::Embedding embedding(face_recog::kEmbeddingDim, 0.0f);
  embedding[axis] = 1.0f;
  embedding[axis + 1] = along_next;
  face_recog::l2_normalize(embedding);
  return embedding;
}

bool test_matching_names_only_a_clear_winner() {
  face_recog::Gallery gallery;
  gallery.add("PersonA", unit_embedding(0));
  gallery.add("PersonB", unit_embedding(1));
  const face_recog::MatchConfig cfg;

  const auto known = face_recog::match_embedding(unit_embedding(0, 0.2f), gallery, cfg);
  const auto distant = face_recog::match_embedding(unit_embedding(2), gallery, cfg);
  const auto between = face_recog::match_embedding(unit_embedding(0, 0.9f), gallery, cfg);
  const auto nobody = face_recog::match_embedding(unit_embedding(0), face_recog::Gallery{}, cfg);
  return expect_true(known.name == "PersonA" && close_to(known.similarity, 0.9806f),
                     "a close embedding is named") &&
         expect_true(distant.name == "Unknown", "a similarity below the threshold is unknown") &&
         expect_true(between.name == "Unknown" && between.similarity > cfg.threshold,
                     "a lead smaller than the margin is unknown") &&
         expect_true(nobody.name == "Unknown", "an empty gallery names nobody");
}

bool test_gallery_file_round_trips() {
  const auto fixture = face_recog::load_gallery(kGalleryFixture);
  bool ok = expect_true(fixture.people.size() == 2 && fixture.people[0].name == "PersonA" &&
                            fixture.people[0].samples == 3 && fixture.people[1].name == "PersonB" &&
                            fixture.people[1].samples == 1,
                        "the shared fixture lists PersonA and PersonB") &&
            expect_true(close_to(fixture.people[0].mean[256], 0.5f) &&
                            close_to(fixture.people[1].mean[256], 0.5f) &&
                            close_to(fixture.people[1].mean[0], 1.0f),
                        "the shared fixture holds their embeddings");

  face_recog::Gallery gallery;
  gallery.add("PersonA", unit_embedding(0));
  gallery.add("PersonA", unit_embedding(1));
  ok &= expect_true(gallery.people.size() == 1 && gallery.people[0].samples == 2 &&
                        close_to(gallery.people[0].mean[0], 0.5f) &&
                        close_to(gallery.people[0].mean[1], 0.5f),
                    "a second sample is averaged into the same person");

  const fs::path path = fs::temp_directory_path() / "face_recognizer_unit_gallery.bin";
  face_recog::save_gallery(gallery, path);
  const auto saved = face_recog::load_gallery(path);
  ok &= expect_true(saved.people.size() == 1 && saved.people[0].name == "PersonA" &&
                        saved.people[0].samples == 2 && close_to(saved.people[0].mean[1], 0.5f),
                    "a saved gallery loads back");

  // A save that fails must leave the previous gallery in place.
  face_recog::Gallery unsavable = gallery;
  unsavable.add(std::string(70000, 'x'), unit_embedding(2));
  bool failed = false;
  try {
    face_recog::save_gallery(unsavable, path);
  } catch (const std::exception&) {
    failed = true;
  }
  ok &= expect_true(failed && face_recog::load_gallery(path).people.size() == 1,
                    "a failed save keeps the previous gallery");

  std::ofstream(path, std::ios::binary | std::ios::trunc) << "not a gallery";
  failed = false;
  try {
    face_recog::load_gallery(path);
  } catch (const std::exception&) {
    failed = true;
  }
  fs::remove(path);
  return ok && expect_true(failed, "a file that is not a gallery is rejected");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  bool ok = test_scrfd_output_decodes_to_source_pixels();
  ok &= test_alignment_moves_the_landmarks_onto_the_template();
  ok &= test_matching_names_only_a_clear_winner();
  ok &= test_gallery_file_round_trips();
  ok &= test_enrollment_rules_are_enforced();
  ok &= test_help_runs(binary);
  ok &= test_unknown_flag_fails(binary);
  ok &= test_missing_config_file_exits_with_code_2(binary);
  ok &= test_validate_config_only_reports_the_mode(binary);
  ok &= test_configuration_rules_are_enforced(binary);
  return ok ? 0 : 1;
}
