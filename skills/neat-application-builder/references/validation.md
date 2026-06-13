# Validation

Validate at the level the environment supports. Do not claim Neat Library
runtime behavior is verified from source inspection alone.

## C++ Build Check

For standalone C++ apps, use the installed package:

```cmake
cmake_minimum_required(VERSION 3.16)
project(neat_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(neat_app main.cpp)
target_link_libraries(neat_app PRIVATE SimaNeat::sima_neat)
```

Build in the Neat Development Environment or target-appropriate environment. A
laptop host compile is not proof of Modalix runtime behavior unless it uses the
correct Neat Development Environment toolchain and runtime target.

## Python Import Check

Use the installed Python binding:

```bash
python3 - <<'PY'
import pyneat
print(pyneat.__name__)
PY
```

On a DevKit or Neat Development Environment image, activate the packaged
environment if the installation requires it.

## Artifact Checks

Before running:

- Confirm classic `Model` inputs point at a compiled model archive.
- Confirm GenAI inputs point at a deployed LLiMa model directory.
- Confirm image, video, audio, config, and output paths exist or are created by the app.
- Confirm the app uses public endpoint names that exist in the built graph.

## Runtime Checks

When hardware Neat Library runtime behavior matters, run on Modalix or the
connected DevKit.

Exercise at least:

- one successful request
- missing model path
- bad input path
- timeout or empty-output behavior when the app exposes a timeout
- wrong endpoint name for multi-input or multi-output graphs
- invalid GenAI request shape when working with GenAI APIs

For graph failures, read structured diagnostics first:

1. `error_code`
2. `repro_note`
3. first terminal entry in `bus`
4. `repro_gst_launch`

## Apps Examples

Use public apps examples for reference commands and full application patterns
after the Neat Library API shape is selected. Do not treat reference-repository build,
publication, documentation, or automation rules as required for standalone
applications.
