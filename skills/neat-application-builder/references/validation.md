# Validation

Choose the checks that establish the changed behavior in the available environment. A source
read, a build, an import, and a run on the target each establish a different fact, so state
which ones you obtained.

## C++ build

A standalone C++ application links the installed package with CMake:

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

Build in the Neat Development Environment with its toolchain, against the Core the target runs.
The `libsima_neat.so.<abi>` that `readelf -d <app> | grep NEEDED` shows must match the
target's installed library.

## Python import

A Python application imports the installed binding. On a DevKit, activate its virtual
environment first with `source ~/pyneat/bin/activate`. `pyneat.build_info()` reports the Core,
platform, and ABI versions.

## Prerequisites

- Classic `Model` paths point at a compiled model archive, and GenAI paths at a deployed LLiMa
  model directory.
- The versions `neat` reported in step 1 of the skill come from one installation, and the
  platform version matches the DevKit software. Reinstall Neat as a set rather than mixing
  packages.
- Image, video, audio, configuration, and output paths exist, or the application creates them.

## Run on the target

Run the application on the Modalix DevKit and inspect useful output against its contract:
saved detections, metadata received in Insight, or a generated answer. A process that starts or
a request that returns is not evidence of correct output.

Choose failure cases from the changed behavior and the inputs the application exposes: missing
model or input paths, timeouts and empty output, and user-supplied GenAI requests. When
lifecycle behavior changes, exercise stop, cleanup, and restart. When a local window changes,
check viewing and closing it on the target.

For a graph failure, read the structured diagnostics in this order: `error_code`, `repro_note`,
the first terminal entry in `bus`, then `repro_gst_launch` to replay the pipeline.
