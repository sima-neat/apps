// Unit test for clip-zero-shot-classifier: the same config, prompts, token ids, token table,
// text-row, embedding, ranking and result panel checks as tests/python/test_unit.py, plus CLI
// handling.
#define main clip_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_e2e_config;

namespace {

constexpr const char* kExampleName = "clip-zero-shot-classifier";

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << (ok ? "[OK] " : "[FAIL] ") << what << "\n";
  failures += ok ? 0 : 1;
}

// The model package's tokenizer.json is not in the repository, so the test builds a small one with
// the same structure as tests/python/test_unit.py: printable ASCII characters at their code points,
// the same characters ending a word at code point + 100, each merge's result at 300 + its rank, and
// the start and end tokens at 400 and 401. "é" is the bytes C3 A9, which the byte-level vocabulary
// spells "Ã" and "©", so those two get the same scheme.
// "e l" ranks after "l l", so "elle" shows that the lowest-ranked pair merges first, not the
// leftmost.
const std::vector<std::string> kMerges = {"h e", "l l",      "he ll",   "hell o</w>", "' s</w>",
                                          "c a", "ca t</w>", "l o</w>", "e l"};
constexpr int64_t kStart = 400;
constexpr int64_t kEnd = 401;
const std::vector<std::pair<std::string, std::vector<int64_t>>> kTokens = {
    {"elle", {101, 301, 201}},
    {"hello", {303}},
    {"Hello,  CAT's!", {303, 144, 306, 304, 133}},
    {"hell 12", {300, 108, 208, 149, 150}},
    {"\tlo  hello\n", {307, 303}},
    {"hellohello", {302, 111, 303}},
    {"cats", {305, 116, 215}},
    {"It's 7.", {105, 216, 304, 155, 146}},
    // "é" is a letter, so "café" is one word and "f" does not end it.
    {"caf\xC3\xA9", {305, 102, 195, 269}},
};
// Prompt vectors whose cosine with the image is exactly 0.28, 0.30, 0.20 and 0.29. The image is
// not unit length.
constexpr std::array<float, 4> kCosines = {0.28F, 0.30F, 0.20F, 0.29F};
const nlohmann::json kTop3 = nlohmann::json::parse(
    R"({"top_classes":[{"label":"b","confidence":0.6652,"similarity":0.3},)"
    R"({"label":"d","confidence":0.2447,"similarity":0.29},{"label":"a","confidence":0.09,"similarity":0.28}]})");

std::string repeat(const std::string& text, int times) {
  std::string out;
  for (int i = 0; i < times; ++i) {
    out += text;
  }
  return out;
}

fs::path write_tokenizer_json(const fs::path& dir) {
  nlohmann::json vocab = nlohmann::json::object();
  for (int c = 33; c < 127; ++c) {
    vocab[std::string(1, static_cast<char>(c))] = c;
    vocab[std::string(1, static_cast<char>(c)) + "</w>"] = c + 100;
  }
  for (std::size_t rank = 0; rank < kMerges.size(); ++rank) {
    std::string merged = kMerges[rank];
    merged.erase(merged.find(' '), 1);
    vocab[merged] = 300 + rank;
  }
  vocab["\xC3\x83"] = 195;
  vocab["\xC2\xA9</w>"] = 269;
  vocab["<|startoftext|>"] = kStart;
  vocab["<|endoftext|>"] = kEnd;
  const fs::path path = dir / "tokenizer.json";
  std::ofstream(path) << nlohmann::json{{"model", {{"vocab", vocab}, {"merges", kMerges}}}}.dump();
  return path;
}

// A version 1 .npy header as numpy.save writes it. `rows` of 512 float32 values follow it, the
// first at row `first_row` (the rows before it are left as a hole in the file).
void write_npy(const fs::path& path, const std::string& descr, const std::string& shape,
               int64_t first_row, const std::vector<std::vector<float>>& rows) {
  std::string header =
      "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  header.resize(((10 + header.size() + 1 + 63) / 64) * 64 - 10 - 1, ' ');
  header += '\n';
  std::ofstream file(path, std::ios::binary);
  file.write("\x93NUMPY\x01\x00", 8);
  file.put(static_cast<char>(header.size() & 0xFF));
  file.put(static_cast<char>(header.size() >> 8));
  file << header;
  file.seekp(static_cast<std::streamoff>(10 + header.size()) +
             first_row * kEmbedDim * static_cast<std::streamoff>(sizeof(float)));
  for (const std::vector<float>& row : rows) {
    file.write(reinterpret_cast<const char*>(row.data()),
               static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
}

template <typename Call> std::string error_of(Call call) {
  try {
    call();
  } catch (const std::exception& error) {
    return error.what();
  }
  return "";
}

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

std::vector<std::vector<float>> prompt_vectors() {
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < kCosines.size(); ++i) {
    std::vector<float> vector(kEmbedDim, 0.0F);
    vector[0] = kCosines[i];
    vector[i + 1] = std::sqrt(1.0F - kCosines[i] * kCosines[i]);
    vectors.push_back(vector);
  }
  return vectors;
}

neat::Tensor cpu_tensor(const std::vector<float>& values, std::vector<int64_t> shape) {
  return neat::Tensor::from_vector(values, std::move(shape), neat::TensorMemory::CPU);
}

bool same_top(const std::vector<TopPrompt>& top,
              const std::vector<std::tuple<std::string, double, double>>& expected) {
  if (top.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < top.size(); ++i) {
    const auto& [label, confidence, similarity] = expected[i];
    if (top[i].label != label || top[i].confidence != confidence ||
        top[i].similarity != similarity) {
      return false;
    }
  }
  return true;
}

// The result panel is drawn on a flat frame: luma 200, chroma U 90 and V 160, away from neutral
// gray.
const std::vector<TopPrompt> kPanelTop = {{"a photo of a dog", 0.5, 0.0},
                                          {"a photo of a cat", 0.3, 0.0},
                                          {"a photo of a bird", 0.2, 0.0}};
// The same prompts in another order, so "a photo of a cat" moves to the best row.
const std::vector<TopPrompt> kPanelTopNext = {{"a photo of a cat", 0.7, 0.0},
                                              {"a photo of a bird", 0.2, 0.0},
                                              {"a photo of a dog", 0.1, 0.0}};

cv::Mat flat_nv12(int width, int height) {
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, cv::Scalar(200));
  cv::Mat chroma(height / 2, width / 2, CV_8UC2, nv12.ptr(height));
  chroma.setTo(cv::Scalar(90, 160));
  return nv12;
}

struct DrawnPanel {
  cv::Mat before;
  cv::Mat after;
  PanelLayout layout;
};

DrawnPanel drawn_panel(int width, int height, const std::vector<TopPrompt>& top,
                       TextImages* texts = nullptr) {
  DrawnPanel panel{flat_nv12(width, height), cv::Mat(),
                   panel_layout(width, height, static_cast<int>(top.size()))};
  panel.after = panel.before.clone();
  TextImages new_texts;
  draw_result_panel(panel.after, top, panel.layout, texts != nullptr ? *texts : new_texts);
  return panel;
}

bool same_pixels(const cv::Mat& a, const cv::Mat& b) {
  return cv::countNonZero(a != b) == 0;
}

// True when every changed pixel lies inside the panel's box, and the panel changed luma and chroma.
bool changes_only_its_box(const DrawnPanel& panel, int height) {
  const cv::Rect& box = panel.layout.box;
  const cv::Rect chroma_box(box.x, height + box.y / 2, box.width, box.height / 2);
  bool luma_changed = false;
  bool chroma_changed = false;
  for (int y = 0; y < panel.after.rows; ++y) {
    for (int x = 0; x < panel.after.cols; ++x) {
      if (panel.before.at<uchar>(y, x) == panel.after.at<uchar>(y, x)) {
        continue;
      }
      if (!(y < height ? box : chroma_box).contains(cv::Point(x, y))) {
        return false;
      }
      (y < height ? luma_changed : chroma_changed) = true;
    }
  }
  return luma_changed && chroma_changed;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  const fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  const fs::path scratch = create_test_scratch_dir(kExampleName, "unit");

  const Config packaged = load_config(config_path);
  check(packaged.codec == "h264" && packaged.tcp && packaged.latency_ms == 100 &&
            packaged.frames == 0 && !packaged.profile && packaged.profile_interval == 100 &&
            packaged.video_port == 9000 && packaged.metadata_port == 9100 &&
            packaged.prompts_path ==
                "examples/classification/clip-zero-shot-classifier/src/common/prompts.txt" &&
            packaged.tokenizer == "<tokenizer-path>" &&
            packaged.token_embeddings == "<token-embeddings-path>",
        "packaged config loads");
  const fs::path config = scratch / "config.yaml";
  for (const auto& [key, value, message] : std::vector<std::array<std::string, 3>>{
           {"runtime.profile_interval", "0", "runtime.profile_interval must be > 0"},
           {"source.codec", "vp9", "source.codec must be h264, h265 or mjpeg"}}) {
    write_e2e_config(kExampleName, config, {{key, value}});
    check(contains(error_of([&] { (void)load_config(config); }), message),
          key + " " + value + " is rejected");
  }

  const std::vector<std::string> shipped = load_prompts(packaged.prompts_path);
  check(shipped.size() >= 2 && std::all_of(shipped.begin(), shipped.end(),
                                           [](const std::string& prompt) {
                                             return !prompt.empty() &&
                                                    prompt == sima_examples::trim_copy(prompt);
                                           }),
        "packaged prompts load");
  const fs::path prompts_path = scratch / "prompts.txt";
  std::ofstream(prompts_path) << "\n  a dog \n\n\ta cat\n   \n";
  check(load_prompts(prompts_path.string()) == std::vector<std::string>{"a dog", "a cat"},
        "prompts skip blank lines and are trimmed");
  check(contains(error_of([&] { (void)load_prompts((scratch / "missing.txt").string()); }),
                 "missing.txt does not exist"),
        "a missing prompts file is rejected");
  for (const auto& [content, count] :
       std::vector<std::pair<std::string, int>>{{"", 0}, {"\n \n", 0}, {"a dog\n", 1}}) {
    std::ofstream(prompts_path) << content;
    check(contains(error_of([&] { (void)load_prompts(prompts_path.string()); }),
                   "has " + std::to_string(count) + " prompts; write at least two"),
          "a prompts file with " + std::to_string(count) + " prompts is rejected");
  }

  const fs::path tokenizer_json = write_tokenizer_json(scratch);
  const ClipTokenizer tokenizer(tokenizer_json);
  for (const auto& [prompt, ids] : kTokens) {
    std::vector<int64_t> expected{kStart};
    expected.insert(expected.end(), ids.begin(), ids.end());
    expected.push_back(kEnd);
    expected.resize(12, kEnd);
    check(tokenizer.encode(prompt, 12) == expected, "tokens of '" + prompt + "'");
  }
  std::vector<int64_t> truncated(7, 303);
  truncated.front() = kStart;
  truncated.push_back(kEnd);
  check(tokenizer.encode(repeat("hello ", 20), 8) == truncated,
        "the tokenizer truncates and keeps the end token");
  const std::vector<std::vector<int64_t>> tokens =
      prompt_tokens(tokenizer_json.string(), {"hello", "cats"});
  std::vector<int64_t> cats{kStart, 305, 116, 215, kEnd};
  cats.resize(kTextTokens, kEnd);
  check(tokens.size() == 2 && tokens[1] == cats, "prompt tokens fill the text encoder input");
  const std::string fits = repeat("hello ", 75);
  check(prompt_tokens(tokenizer_json.string(), {fits, "cats"})[0][76] == kEnd &&
            contains(error_of([&] {
                       (void)prompt_tokens(tokenizer_json.string(), {"cats", fits + "hello"});
                     }),
                     "prompt '" + fits + "hello' is 76 tokens; the text encoder reads at most 75"),
        "a prompt longer than the text encoder is rejected");
  check(contains(error_of([&] {
                   (void)prompt_tokens((scratch / "missing").string(), {"a dog", "a cat"});
                 }),
                 "does not exist; point it at text_encoder/tokenizer.json in the model package"),
        "a missing tokenizer is rejected");

  const fs::path table_path = scratch / "token_embeddings.npy";
  std::vector<float> end_row(kEmbedDim, 0.0F);
  end_row[0] = 1.0F;
  end_row[1] = 2.0F;
  end_row[2] = 3.0F;
  write_npy(table_path, "<f4", "(49408, 512)", kVocabSize - 1, {end_row});
  const TokenEmbeddings table = load_token_embeddings(table_path.string());
  const std::vector<float> rows = token_embeds(table, {49406, 49407});
  check(rows.size() == 2U * kEmbedDim && rows[kEmbedDim] == 1.0F && rows[kEmbedDim + 1] == 2.0F &&
            rows[kEmbedDim + 2] == 3.0F && rows[0] == 0.0F,
        "the token table returns the prompts' rows");
  check(contains(error_of([&] { (void)load_token_embeddings((scratch / "missing").string()); }),
                 "does not exist; point it at text_encoder/token_embeddings.npy in the model "
                 "package"),
        "a missing token table is rejected");
  for (const std::string shape : {"(10, 512)", "(49408, 256)", "(49408,)"}) {
    write_npy(table_path, "<f4", shape, 0, {});
    check(contains(error_of([&] { (void)load_token_embeddings(table_path.string()); }),
                   "has shape " + shape + ", not (49408, 512)"),
          "a token table of shape " + shape + " is rejected");
  }
  write_npy(table_path, "<f8", "(49408, 512)", 0, {});
  check(contains(error_of([&] { (void)load_token_embeddings(table_path.string()); }),
                 "is not little-endian float32 in C order"),
        "a token table that is not float32 is rejected");
  std::ofstream(table_path) << "not a table";
  check(contains(error_of([&] { (void)load_token_embeddings(table_path.string()); }),
                 "is not a NumPy .npy file"),
        "a file that is not .npy is rejected");

  std::vector<float> encoder_rows(kTextTokens * kEmbedDim);
  std::iota(encoder_rows.begin(), encoder_rows.end(), 0.0F);
  std::vector<int64_t> ids{49406, 320, 1125, 49407};
  ids.resize(kTextTokens, 49407);
  const std::vector<float> vector = end_of_text_row(encoder_rows.data(), ids);
  double norm = 0.0;
  bool row_three = true;
  const float* row = encoder_rows.data() + 3 * kEmbedDim;
  const float row_norm = std::sqrt(std::inner_product(row, row + kEmbedDim, row, 0.0F));
  for (int i = 0; i < kEmbedDim; ++i) {
    norm += static_cast<double>(vector[i]) * vector[i];
    row_three = row_three && std::abs(vector[i] - row[i] / row_norm) <= 1e-6F * row[i] / row_norm;
  }
  check(row_three && std::abs(std::sqrt(norm) - 1.0) < 1e-6,
        "the text vector is the unit row at the first end token");

  std::vector<float> embedding_values(kEmbedDim);
  std::iota(embedding_values.begin(), embedding_values.end(), 0.0F);
  const neat::Tensor embedding = cpu_tensor(embedding_values, {1, kEmbedDim});
  const neat::Tensor logits = cpu_tensor(std::vector<float>(1000, 0.0F), {1, 1000});
  bool found_by_size = true;
  for (const neat::TensorList& outputs :
       {neat::TensorList{embedding}, neat::TensorList{logits, embedding},
        neat::TensorList{embedding, logits}}) {
    neat::Sample sample;
    sample.kind = neat::SampleKind::TensorSet;
    sample.tensors = outputs;
    const neat::Mapping view = image_embedding(sample).view_read();
    const auto* values = static_cast<const float*>(view.data);
    found_by_size =
        found_by_size && std::equal(values, values + kEmbedDim, embedding_values.begin());
  }
  neat::Sample single;
  single.kind = neat::SampleKind::Tensor;
  single.tensor = embedding;
  check(found_by_size && image_embedding(single).shape == embedding.shape,
        "the image embedding is found by its size");
  neat::Sample logits_only;
  logits_only.kind = neat::SampleKind::TensorSet;
  logits_only.tensors = {logits};
  check(contains(error_of([&] { (void)image_embedding(logits_only); }),
                 "outputs of sizes [1000], none of 512 values"),
        "an image encoder without an embedding is rejected");

  const std::vector<std::vector<float>> vectors = prompt_vectors();
  std::vector<float> image(kEmbedDim, 0.0F);
  image[0] = 2.0F;
  const std::vector<TopPrompt> top = top_prompts(image.data(), vectors, {"a", "b", "c", "d"});
  check(same_top(top, {{"b", 0.6652, 0.3}, {"d", 0.2447, 0.29}, {"a", 0.09, 0.28}}),
        "top prompts rank by the softmax of 100 times the cosine");
  check(same_top(top_prompts(image.data(), {vectors[0], vectors[1]}, {"a", "b"}),
                 {{"b", 0.8808, 0.3}, {"a", 0.1192, 0.28}}),
        "two prompts give two top prompts");
  check(same_top(top_prompts(image.data(), {vectors[1], vectors[1], vectors[0]},
                             {"first", "second", "third"}),
                 {{"first", 0.4683, 0.3}, {"second", 0.4683, 0.3}, {"third", 0.0634, 0.28}}),
        "equal scores keep prompt order");
  check(nlohmann::json::parse(top_classes_json(top)) == kTop3,
        "top prompts serialize as classification metadata");

  for (const auto& [width, height, prompt_count] :
       std::vector<std::array<int, 3>>{{1280, 720, 3}, {640, 360, 2}}) {
    const DrawnPanel panel = drawn_panel(
        width, height, std::vector<TopPrompt>(kPanelTop.begin(), kPanelTop.begin() + prompt_count));
    const cv::Rect& box = panel.layout.box;
    check(static_cast<int>(panel.layout.rows.size()) == prompt_count && box.x > 0 && box.y > 0 &&
              box.x + box.width < width && box.y + box.height < height &&
              changes_only_its_box(panel, height),
          "the panel changes only its box at " + std::to_string(width) + "x" +
              std::to_string(height) + " with " + std::to_string(prompt_count) + " prompts");
  }
  check(panel_layout(640, 360, 3).box.width * 2 == 640 &&
            panel_layout(1280, 720, 3).box.width == 640,
        "the panel scales with the frame height");
  const DrawnPanel panel = drawn_panel(1280, 720, kPanelTop);
  const cv::Rect& box = panel.layout.box;
  // The box's top left corner holds no text or bar.
  check(panel.after.at<uchar>(box.y + 1, box.x + 1) == 60 &&
            panel.after.at<uchar>(720 + box.y / 2, box.x) == 117 &&
            panel.after.at<uchar>(720 + box.y / 2, box.x + 1) == 138,
        "the panel darkens its box and pulls chroma toward gray");
  bool bars_follow = true;
  for (std::size_t i = 0; i < kPanelTop.size(); ++i) {
    const cv::Rect& bar = panel.layout.rows[i].bar;
    const int fill = even(bar.width * kPanelTop[i].confidence);
    const uchar* line = panel.after.ptr(bar.y + bar.height / 2);
    for (int x = 0; x < bar.width; ++x) {
      const int expected = x < fill ? (i == 0 ? kAccentYuv[0] : kBarLuma) : kTrackLuma;
      bars_follow = bars_follow && line[bar.x + x] == expected;
    }
  }
  check(bars_follow, "the panel bars follow the probabilities");
  bool best_stands_out = panel.layout.rows[0].font_scale > panel.layout.rows[1].font_scale &&
                         panel.layout.rows[0].bar.height > panel.layout.rows[1].bar.height;
  for (std::size_t i = 0; i < 2; ++i) {
    const PanelRow& panel_row = panel.layout.rows[i];
    const uchar* uv = panel.after.ptr(720 + panel_row.bar.y / 2) + panel_row.bar.x;
    const bool accent = uv[0] == kAccentYuv[1] && uv[1] == kAccentYuv[2];
    double text_max = 0.0;
    cv::minMaxLoc(panel.after(cv::Rect(panel_row.bar.x, panel_row.baseline_y - 10,
                                       panel_row.label_width, 10)),
                  nullptr, &text_max);
    best_stands_out = best_stands_out && accent == (i == 0) &&
                      static_cast<int>(text_max) == (i == 0 ? kBestTextLuma : kTextLuma);
  }
  check(best_stands_out, "the panel's best row stands out");
  bool percentage_at_bar_end = true;
  for (const PanelRow& panel_row : panel.layout.rows) {
    const cv::Mat text =
        panel.after(cv::Rect(panel_row.bar.x, panel_row.baseline_y - 10, panel_row.bar.width, 10));
    int last_lit = -1;
    bool label_gap = true;
    for (int x = 0; x < text.cols; ++x) {
      double column_max = 0.0;
      cv::minMaxLoc(text.col(x), nullptr, &column_max);
      last_lit = column_max > 100 ? x : last_lit;
      label_gap = label_gap && (x != panel_row.label_width - 1 || column_max <= 100);
    }
    // The percentage's last lit column lies at the bar's right end, and a gap separates it from the
    // label.
    percentage_at_bar_end = percentage_at_bar_end &&
                            last_lit >= panel_row.bar.width - 2 * panel_row.thickness &&
                            last_lit < panel_row.bar.width + panel_row.thickness &&
                            panel_row.label_width < last_lit && label_gap;
  }
  check(percentage_at_bar_end, "the panel's percentage ends at the bar end");
  TextImages shared_texts;
  const cv::Mat first = drawn_panel(1280, 720, kPanelTop, &shared_texts).after;
  const cv::Mat second = drawn_panel(1280, 720, kPanelTopNext, &shared_texts).after;
  check(same_pixels(first, drawn_panel(1280, 720, kPanelTop).after) &&
            same_pixels(second, drawn_panel(1280, 720, kPanelTopNext).after),
        "the panel's text cache draws like a new cache");
  const PanelRow& best = panel.layout.rows[0];
  const std::string long_prompt = "a photo of " + repeat("a very ", 20) + "long prompt";
  const std::string cut = fit_text(long_prompt, best.font_scale, best.thickness, best.label_width);
  check(cut.rfind("a photo of a very", 0) == 0 && cut.size() > 3 &&
            cut.compare(cut.size() - 3, 3, "...") == 0 && cut.size() < long_prompt.size() &&
            text_width(cut, best.font_scale, best.thickness) <= best.label_width &&
            fit_text("a photo of a dog", best.font_scale, best.thickness, best.label_width) ==
                "a photo of a dog" &&
            // The panel font draws ASCII only.
            fit_text("caf\xC3\xA9", best.font_scale, best.thickness, best.label_width) == "caf?",
        "a long prompt is cut to its column");
  TextImages long_texts;
  const PanelLayout long_layout =
      drawn_panel(1280, 720, {{long_prompt, 1.0, 0.0}}, &long_texts).layout;
  std::vector<int> long_widths;
  int long_image_width = 0;
  for (const auto& [key, value] : long_texts) {
    if (std::get<0>(key) == long_prompt) {
      long_widths.push_back(std::get<4>(key));
      long_image_width = value.first.cols;
    }
  }
  check(long_widths == std::vector<int>{long_layout.rows[0].label_width} &&
            long_image_width <= long_layout.rows[0].label_width + 2 * long_layout.rows[0].thickness,
        "a long prompt is cut once per text and width");

  const auto fourcc = [](const std::string& tag) {
    int value = 0;
    for (std::size_t i = 0; i < tag.size(); ++i) {
      value |= static_cast<unsigned char>(tag[i]) << (8 * i);
    }
    return static_cast<double>(value);
  };
  check(fourcc_codec(fourcc("h264")) == "h264" && fourcc_codec(fourcc("avc1")) == "h264" &&
            fourcc_codec(fourcc("hevc")) == "h265" && fourcc_codec(fourcc("hvc1")) == "h265" &&
            fourcc_codec(fourcc("MJPG")) == "mjpeg" && fourcc_codec(fourcc("VP90")).empty() &&
            fourcc_codec(0.0).empty(),
        "the stream codec comes from the FOURCC");

  const ProcessResult help = spawn_and_wait(binary, {"--help"}, 20000);
  check(help.exit_code == 0 && contains(help.stdout_text, "Usage"), "--help prints usage");
  const ProcessResult bogus = spawn_and_wait(binary, {"--bogus"}, 20000);
  check(bogus.exit_code != 0 && contains(bogus.stderr_text, "unknown argument"),
        "unknown flags are rejected");
  const ProcessResult no_path = spawn_and_wait(binary, {"--config"}, 20000);
  check(no_path.exit_code != 0 && contains(no_path.stderr_text, "--config requires a path"),
        "--config without a path is rejected");

  remove_dir(scratch.string());
  return failures > 0 ? 1 : 0;
}
