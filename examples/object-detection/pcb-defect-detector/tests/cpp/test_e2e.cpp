// E2E test for pcb-defect-detector.
// Runs the binary with a real model pack and the bundled PCB test images,
// then verifies that one annotated image is written per input image.
#include <opencv2/opencv.hpp>

#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr char kExample[] = "pcb-defect-detector";
// Mirrors kOutputTag in the application.
constexpr const char* kOutputTag = "_pcb";

// Annotated output keeps the source extension, so a JPEG board is re-encoded on
// write. That perturbs most pixels slightly, which a plain "any difference" test
// would mistake for drawing. Box outlines and labels instead replace pixels
// outright, so only a large per-channel delta counts as drawn.
constexpr int kAnnotationDelta = 60;
constexpr int kMinAnnotatedPixels = 200;

int count_input_images(const std::string& input_dir) {
  int count = 0;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      ++count;
    }
  }
  return count;
}

} // namespace

// Return the single line of `text` that begins with `prefix`, or "" if absent.
std::string extract_line(const std::string& text, const std::string& prefix) {
  const std::size_t at = text.find(prefix);
  if (at == std::string::npos) {
    return {};
  }
  const std::size_t end = text.find('\n', at);
  return text.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

// Read "<key><int>" out of the application's summary line.
int summary_field(const std::string& text, const std::string& key) {
  const std::size_t at = text.find(key);
  if (at == std::string::npos) {
    return -1;
  }
  return std::atoi(text.c_str() + at + key.size());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path(kExample, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured PCB defect model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }

  const std::string example_dir = fs::path(binary).parent_path().string();
  const std::vector<std::string> label_search = {
      "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt",
      example_dir + "/src/common/pcb_label.txt",
  };
  for (const auto& candidate : label_search) {
    if (!labels_file.empty()) {
      break;
    }
    if (fs::exists(candidate)) {
      labels_file = candidate;
    }
  }
  if (labels_file.empty()) {
    return skip_or_fail("src/common/pcb_label.txt not found; set SIMANEAT_APPS_TEST_LABELS_FILE "
                        "or ensure the example label file is available");
  }

  // PCB defects are not present in the shared COCO fixtures, so this example
  // uses its own test images instead of SIMANEAT_APPS_TEST_INPUT_DIR.
  const std::string input_dir = "assets/datasets-test/pcb";
  if (!fs::is_directory(input_dir) || fs::is_empty(input_dir)) {
    return skip_or_fail("PCB test images are missing or empty: " + input_dir);
  }

  const auto out_dir = create_test_output_dir(kExample, "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config(kExample, config_path,
                   {{"model.path", model_path},
                    {"model.labels", labels_file},
                    {"io.input_dir", input_dir},
                    {"io.output_dir", out_dir}});

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  const int expected_files = count_input_images(input_dir);
  const int output_files = count_output_files(out_dir);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (output_files != expected_files) {
    std::cerr << "[FAIL] expected " << expected_files << " annotated images, found " << output_files
              << "\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else if (result.stdout_text.find("Per-class totals:") == std::string::npos) {
    std::cerr << "[FAIL] run did not report per-class defect totals\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else {
    // The checks above all pass when every detection list is empty, because the
    // application writes an annotated copy either way. Assert the actual
    // inspection result so a broken decoder, an incompatible model package or a
    // missing BBOX payload cannot pass as a clean board.
    const int images_with_defects = summary_field(result.stdout_text, "images_with_defects=");
    const int total_defects = summary_field(result.stdout_text, "total_defects=");
    const int failed = summary_field(result.stdout_text, "failed=");

    if (failed != 0) {
      std::cerr << "[FAIL] " << failed << " image(s) failed to process\n";
      std::cerr << "stdout:\n" << result.stdout_text << "\n";
      rc = 1;
    } else if (images_with_defects != expected_files) {
      std::cerr << "[FAIL] every bundled board carries defects, but only " << images_with_defects
                << " of " << expected_files << " produced any\n";
      std::cerr << "stdout:\n" << result.stdout_text << "\n";
      rc = 1;
    } else if (total_defects < expected_files) {
      std::cerr << "[FAIL] expected at least one defect per board, got " << total_defects << "\n";
      std::cerr << "stdout:\n" << result.stdout_text << "\n";
      rc = 1;
    } else {
      // Each fixture is named after the defect it contains, so the reported
      // classes must include it. This is what proves PCB detection works rather
      // than that the process merely ran.
      //
      // Scope the search to the "Per-class totals:" line. Searching the whole of
      // stdout would be inert: the application echoes each input filename as it
      // works, so "missing_hole" is present whatever the model reported, and the
      // check could never fail.
      const std::string totals_line = extract_line(result.stdout_text, "Per-class totals:");

      std::vector<std::string> missing;
      std::vector<std::string> expected_classes;
      std::vector<std::string> unparsable;
      for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
          continue;
        }
        const std::string stem = entry.path().stem().string();       // pcb_01_missing_hole
        const std::size_t first = stem.find('_');
        const std::size_t second = stem.find('_', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
          unparsable.push_back(stem);
          continue;
        }
        const std::string defect = stem.substr(second + 1);          // missing_hole
        expected_classes.push_back(defect);
        // Match a whole key, not a substring: "spur" is a prefix of
        // "spurious_copper", so a raw find() would let one satisfy the other.
        if (totals_line.find(defect + ":") == std::string::npos) {
          missing.push_back(defect);
        }
      }

      if (expected_classes.empty()) {
        // Without this the loop above asserts nothing and the test passes
        // vacuously.
        std::cerr << "[FAIL] no defect class could be derived from the fixture names; "
                     "the class assertion would pass vacuously\n";
        for (const auto& stem : unparsable) {
          std::cerr << "       unparsable fixture: " << stem << "\n";
        }
        rc = 1;
      } else if (!missing.empty()) {
        std::cerr << "[FAIL] fixtures are named for the defects they contain, but these were "
                     "never detected:";
        for (const auto& defect : missing) {
          std::cerr << ' ' << defect;
        }
        std::cerr << "\nstdout:\n" << result.stdout_text << "\n";
        rc = 1;
      } else {
        // Everything above reads stdout or file metadata. An application that
        // reported detections but wrote an untouched copy of its input would
        // still pass all of it -- exactly the "saves unchanged images" failure
        // this test exists to catch. Compare decoded pixels so the overlay has
        // to be present in the file that ships.
        int unannotated = 0;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
          if (!entry.is_regular_file()) {
            continue;
          }
          const fs::path written =
              fs::path(out_dir) /
              (entry.path().stem().string() + kOutputTag + entry.path().extension().string());
          const cv::Mat before = cv::imread(entry.path().string());
          const cv::Mat after = cv::imread(written.string());
          if (before.empty() || after.empty()) {
            std::cerr << "[FAIL] could not read " << (before.empty() ? entry.path() : written)
                      << "\n";
            rc = 1;
            break;
          }
          if (before.size() != after.size()) {
            std::cerr << "[FAIL] " << written.filename().string() << " is " << after.cols << "x"
                      << after.rows << " but its source is " << before.cols << "x" << before.rows
                      << "\n";
            rc = 1;
            break;
          }
          cv::Mat diff;
          cv::absdiff(before, after, diff);
          std::vector<cv::Mat> channels;
          cv::split(diff, channels);
          cv::Mat strongest = cv::max(channels[0], cv::max(channels[1], channels[2]));
          const int changed = cv::countNonZero(strongest > kAnnotationDelta);
          if (changed < kMinAnnotatedPixels) {
            std::cerr << "[FAIL] " << written.filename().string()
                      << " has only " << changed
                      << " pixel(s); the board was reported as defective but the saved image "
                         "carries no visible overlay\n";
            ++unannotated;
            rc = 1;
          }
        }
        if (rc == 0 && unannotated == 0) {
          std::cout << "[OK] PCB defect detection produced " << output_files
                    << " annotated images, " << total_defects << " defects across "
                    << images_with_defects
                    << " board(s), every named defect class detected, every output drawn on\n";
        }
      }
    }
  }

  // Core letterboxes and maps coordinates for input that is not 640x640.
  //
  // Every bundled fixture is exactly the model's own 640x640 input, so the
  // letterbox is an identity operation and the happy-path check would pass even
  // if the application did the geometry itself. Rescaling the fixtures to three
  // different sizes exercises Core's on-device letterbox and its mapping of
  // boxes back to source coordinates, through the single graph seeded at the
  // configured ingress capacity.
  //
  // Each fixture is scaled *up*, so the defects are still present at full detail
  // and the same classes must still be reported. A wrong mapping would place
  // boxes in letterbox space instead of source space, and the annotated image
  // would no longer match the source dimensions.
  if (rc == 0) {
    const double scales[] = {2.0, 1.5, 1.25};
    const auto mixed_input = fs::path(out_dir).parent_path() / "mixed-input";
    fs::create_directories(mixed_input);

    std::vector<std::pair<std::string, cv::Size>> expected_sizes;
    std::set<std::string> expected_classes;
    int index = 0;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const cv::Mat source = cv::imread(entry.path().string());
      if (source.empty()) {
        std::cerr << "[FAIL] could not read fixture " << entry.path() << "\n";
        rc = 1;
        break;
      }
      const double scale = scales[index % (sizeof(scales) / sizeof(scales[0]))];
      cv::Mat resized;
      cv::resize(source, resized,
                 cv::Size(static_cast<int>(source.cols * scale),
                          static_cast<int>(source.rows * scale)),
                 0, 0, cv::INTER_CUBIC);
      cv::imwrite((mixed_input / entry.path().filename()).string(), resized);

      const std::string stem = entry.path().stem().string();
      expected_sizes.emplace_back(stem + kOutputTag + entry.path().extension().string(),
                                  resized.size());

      // Fixtures are named "<n>_<index>_<defect>", so the defect follows the
      // second underscore.
      const std::size_t first = stem.find('_');
      const std::size_t second = first == std::string::npos
                                     ? std::string::npos
                                     : stem.find('_', first + 1);
      if (second != std::string::npos) {
        expected_classes.insert(stem.substr(second + 1));
      }
      ++index;
    }

    std::set<cv::Size, bool (*)(const cv::Size&, const cv::Size&)> distinct(
        [](const cv::Size& a, const cv::Size& b) {
          return a.width != b.width ? a.width < b.width : a.height < b.height;
        });
    for (const auto& [name, size] : expected_sizes) {
      (void)name;
      distinct.insert(size);
    }
    if (rc == 0 && distinct.size() < 2) {
      std::cerr << "[FAIL] the rescaled fixtures must span more than one size or this test "
                   "does not exercise mixed resolutions at all\n";
      rc = 1;
    }
    if (rc == 0 && expected_classes.empty()) {
      std::cerr << "[FAIL] no defect class could be derived from the fixture names in "
                << input_dir << "; the class assertion would pass vacuously\n";
      rc = 1;
    }

    if (rc == 0) {
      const auto mixed_out = fs::path(out_dir).parent_path() / "mixed-output";
      fs::create_directories(mixed_out);
      const fs::path mixed_config = fs::path(out_dir).parent_path() / "config_mixed.yaml";
      write_e2e_config(kExample, mixed_config,
                       {{"model.path", model_path},
                        {"model.labels", labels_file},
                        {"io.input_dir", mixed_input.string()},
                        {"io.output_dir", mixed_out.string()}});

      const auto mixed = spawn_and_wait(binary, {"--config", mixed_config.string()}, timeout);
      const int mixed_failed = summary_field(mixed.stdout_text, "failed=");
      const int mixed_with_defects = summary_field(mixed.stdout_text, "images_with_defects=");
      const int mixed_defects = summary_field(mixed.stdout_text, "total_defects=");

      if (mixed.exit_code != 0) {
        std::cerr << "[FAIL] mixed-resolution input must run cleanly through Core's "
                     "preprocessing, exit was "
                  << mixed.exit_code << "\n";
        std::cerr << "stdout:\n" << mixed.stdout_text << "\n";
        rc = 1;
      } else if (mixed_failed != 0) {
        std::cerr << "[FAIL] " << mixed_failed << " rescaled image(s) failed to process\n";
        std::cerr << "stdout:\n" << mixed.stdout_text << "\n";
        rc = 1;
      } else if (mixed_with_defects != static_cast<int>(expected_sizes.size())) {
        std::cerr << "[FAIL] every board still carries its defects after rescaling, but only "
                  << mixed_with_defects << " of " << expected_sizes.size() << " produced any\n";
        std::cerr << "stdout:\n" << mixed.stdout_text << "\n";
        rc = 1;
      } else if (mixed_defects < static_cast<int>(expected_sizes.size())) {
        std::cerr << "[FAIL] expected at least one defect per board, got " << mixed_defects
                  << "\n";
        std::cerr << "stdout:\n" << mixed.stdout_text << "\n";
        rc = 1;
      }

      // Boxes are drawn on the source frame, so a correct mapping keeps the
      // annotated image at the source size. A 640x640 output here would mean the
      // letterboxed canvas leaked into the result.
      if (rc == 0) {
        for (const auto& [name, size] : expected_sizes) {
          const fs::path written = mixed_out / name;
          if (!fs::is_regular_file(written)) {
            std::cerr << "[FAIL] no annotated image for " << name << "\n";
            rc = 1;
            break;
          }
          const cv::Mat annotated = cv::imread(written.string());
          if (annotated.empty()) {
            std::cerr << "[FAIL] could not read annotated output " << written << "\n";
            rc = 1;
            break;
          }
          if (annotated.cols != size.width || annotated.rows != size.height) {
            std::cerr << "[FAIL] " << name << " was written at " << annotated.cols << "x"
                      << annotated.rows << " but its source is " << size.width << "x"
                      << size.height
                      << "; detections were not mapped back to source coordinates\n";
            rc = 1;
            break;
          }
        }
      }

      // Same content, only larger, so the model must still name the same defects.
      if (rc == 0) {
        const std::string totals_line = extract_line(mixed.stdout_text, "Per-class totals:");
        std::vector<std::string> missing;
        for (const auto& defect : expected_classes) {
          if (totals_line.find(defect + ":") == std::string::npos) {
            missing.push_back(defect);
          }
        }
        if (!missing.empty()) {
          std::cerr << "[FAIL] after rescaling, these defects were no longer detected:";
          for (const auto& defect : missing) {
            std::cerr << " " << defect;
          }
          std::cerr << "\n  Core's letterbox or its coordinate mapping is not handling "
                       "non-native input sizes\n";
          std::cerr << "stdout:\n" << mixed.stdout_text << "\n";
          rc = 1;
        } else {
          std::cout << "[OK] mixed resolutions (" << distinct.size()
                    << " distinct sizes) went through Core's preprocessing: " << mixed_defects
                    << " defects, every named class still detected, outputs at source size\n";
        }
      }
      remove_dir(mixed_out.string());
    }
    remove_dir(mixed_input.string());
  }

  // An input that cannot be processed must produce a non-zero exit, not a
  // success with a quietly shorter result set. Asserting failed==0 on the happy
  // path does not prove the failure path reports anything, so exercise it
  // directly.
  if (rc == 0) {
    const auto broken_input = fs::path(out_dir).parent_path() / "broken-input";
    fs::create_directories(broken_input);
    for (const auto& entry : fs::directory_iterator(input_dir)) {
      if (entry.is_regular_file()) {
        fs::copy_file(entry.path(), broken_input / entry.path().filename(),
                      fs::copy_options::overwrite_existing);
      }
    }
    // Passes the extension filter, but decodes to nothing.
    std::ofstream(broken_input / "corrupt.jpg") << "not a jpeg";

    const auto broken_out = fs::path(out_dir).parent_path() / "broken-output";
    fs::create_directories(broken_out);
    const fs::path broken_config = fs::path(out_dir).parent_path() / "config_broken.yaml";
    write_e2e_config(kExample, broken_config,
                     {{"model.path", model_path},
                      {"model.labels", labels_file},
                      {"io.input_dir", broken_input.string()},
                      {"io.output_dir", broken_out.string()}});

    const auto broken = spawn_and_wait(binary, {"--config", broken_config.string()}, timeout);
    const int broken_failed = summary_field(broken.stdout_text, "failed=");
    if (broken.exit_code == 0) {
      std::cerr << "[FAIL] an unreadable input must not exit 0\n";
      std::cerr << "stdout:\n" << broken.stdout_text << "\n";
      rc = 1;
    } else if (broken_failed < 1) {
      std::cerr << "[FAIL] an unreadable input must be counted in failed=, got " << broken_failed
                << "\n";
      std::cerr << "stdout:\n" << broken.stdout_text << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] an unreadable input is reported (failed=" << broken_failed
                << ") and exits " << broken.exit_code << "\n";
    }
    remove_dir(broken_input.string());
    remove_dir(broken_out.string());
  }

  // A timeout stops the run instead of being attributed to another image.
  //
  // A frame that times out is still in the pipeline, so continuing would hand its
  // result to the next image's pull. A 1 ms timeout is far below the ~200 ms
  // inference, so nothing can complete and the run must abort with no image
  // reported as processed.
  if (rc == 0) {
    const auto timeout_out = fs::path(out_dir).parent_path() / "timeout-output";
    fs::create_directories(timeout_out);
    const fs::path timeout_config = fs::path(out_dir).parent_path() / "config_timeout.yaml";
    write_e2e_config(kExample, timeout_config,
                     {{"model.path", model_path},
                      {"model.labels", labels_file},
                      {"io.input_dir", input_dir},
                      {"io.output_dir", timeout_out.string()},
                      {"runtime.timeout_ms", "1"}});

    const auto timed = spawn_and_wait(binary, {"--config", timeout_config.string()}, timeout);
    const std::string aborted = extract_line(timed.stderr_text, "Aborted after ");
    const int processed = summary_field(aborted, "Aborted after ");

    if (timed.exit_code == 0) {
      std::cerr << "[FAIL] a run where inference timed out must not exit 0\n";
      std::cerr << "stdout:\n" << timed.stdout_text << "\n";
      rc = 1;
    } else if (timed.stderr_text.find("timeout") == std::string::npos &&
               timed.stderr_text.find("Timeout") == std::string::npos) {
      std::cerr << "[FAIL] the timeout must be named on stderr\n";
      std::cerr << "stderr:\n" << timed.stderr_text << "\n";
      rc = 1;
    } else if (processed != 0) {
      std::cerr << "[FAIL] no image can complete under a 1 ms timeout, so none may be "
                   "reported as processed; got "
                << processed << "\n";
      std::cerr << "stderr:\n" << timed.stderr_text << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] a timeout aborts the run with no image reported as processed\n";
    }
    remove_dir(timeout_out.string());
  }

  // A save that fails must be reported, not counted as a processed image. The
  // application checks the image writer's return value; nothing else in this
  // suite drives that branch.
  if (rc == 0) {
    const int image_total = count_input_images(input_dir);
    const auto readonly_out = fs::path(out_dir).parent_path() / "readonly-output";
    fs::create_directories(readonly_out);
    // Written before the directory is sealed; the config does not live in it.
    const fs::path readonly_config = fs::path(out_dir).parent_path() / "config_readonly.yaml";
    write_e2e_config(kExample, readonly_config,
                     {{"model.path", model_path},
                      {"model.labels", labels_file},
                      {"io.input_dir", input_dir},
                      {"io.output_dir", readonly_out.string()}});

    std::error_code perm_ec;
    fs::permissions(readonly_out,
                    fs::perms::owner_read | fs::perms::owner_exec | fs::perms::group_read |
                        fs::perms::group_exec | fs::perms::others_read | fs::perms::others_exec,
                    fs::perm_options::replace, perm_ec);

    // root ignores the mode bits, which would let the checks below pass without a
    // single write ever failing.
    bool sealed = !perm_ec;
    if (sealed) {
      const fs::path probe = readonly_out / ".probe";
      std::ofstream probe_stream(probe);
      if (probe_stream.good()) {
        sealed = false;
        probe_stream.close();
        fs::remove(probe, perm_ec);
      }
    }

    if (!sealed) {
      // Strict mode must not pass on an unverified scenario; the Python twin
      // routes the same condition through skip_unless_e2e_ready.
      const int outcome = skip_or_fail(readonly_out.string() +
                                       " is still writable, so the write-failure path cannot be "
                                       "exercised (running as root?)");
      if (outcome != kSkipCode) {
        rc = outcome;
      }
    } else {
      const auto denied = spawn_and_wait(binary, {"--config", readonly_config.string()}, timeout);
      const std::string done = extract_line(denied.stdout_text, "Done: ");
      const int processed = summary_field(done, "Done: ");
      const int failed = summary_field(denied.stdout_text, "failed=");

      if (denied.exit_code == 0) {
        std::cerr << "[FAIL] images that could not be saved must not exit 0\n";
        std::cerr << "stdout:\n" << denied.stdout_text << "\n";
        rc = 1;
      } else if (processed != 0 || failed != image_total) {
        std::cerr << "[FAIL] no image could be saved, so all " << image_total
                  << " must be counted as failed and none as processed; got processed="
                  << processed << " failed=" << failed << "\n";
        std::cerr << "stdout:\n" << denied.stdout_text << "\n";
        rc = 1;
      } else {
        std::cout << "[OK] unsaveable output is reported (0/" << image_total
                  << ", failed=" << failed << ") and exits " << denied.exit_code << "\n";
      }
    }

    fs::permissions(readonly_out, fs::perms::owner_all | fs::perms::group_read |
                                      fs::perms::group_exec | fs::perms::others_read |
                                      fs::perms::others_exec,
                    fs::perm_options::replace, perm_ec);
    remove_dir(readonly_out.string());
  }

  remove_dir(out_dir);
  return rc;
}
