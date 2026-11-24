# Validation

Choose the checks that establish the changed behavior in the available environment. A source
read, a build, an import, and a run on the target each establish a different fact, so state
which ones you obtained.

## C++ build

A standalone C++ application links the installed package with CMake. The `SYSROOT` block lets
the same file build in the Palette SDK and natively on the DevKit:

```cmake
cmake_minimum_required(VERSION 3.16)
project(neat_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

find_package(SimaNeat REQUIRED CONFIG)

add_executable(neat_app main.cpp)
target_link_libraries(neat_app PRIVATE SimaNeat::sima_neat)
```

Configure and build with no toolchain file:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

In the Palette SDK, the shell's `SYSROOT`, `CC`, `CXX`, `CXXFLAGS`, and `LDFLAGS` select the
ARM64 cross compiler and the Modalix sysroot. Do not pass the Apps repository's
`cmake/toolchains/aarch64-modalix.cmake` from `/neat-resources/apps-src`: it exists only in
Core-bundled images. The build is correct when `file build/neat_app` reports ARM aarch64,
`SimaNeat_DIR` in `build/CMakeCache.txt` points under `$SYSROOT`, and the
`libsima_neat.so.<abi>` that `readelf -d build/neat_app | grep NEEDED` shows matches the
DevKit's installed library.

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
