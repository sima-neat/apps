# Validation

Choose checks that establish the requested behavior in the available
environment. Source inspection, a successful build, and a Python import establish
different facts from target runtime execution. State which evidence was obtained.

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
correct Neat Development Environment toolchain and runtime target. Build against
the Core the target runs: the `libsima_neat.so.<abi>` that
`readelf -d <app> | grep NEEDED` shows must match the target's installed
library.

## Python Import Check

Use the installed Python binding:

```bash
python3 - <<'PY'
import pyneat
print(pyneat.build_info())
PY
```

On a DevKit, PyNeat runs from its virtual environment:
`source ~/pyneat/bin/activate`. `build_info()` reports the Core version,
platform version, and ABI version.

## Artifact Checks

Before running, check the prerequisites used by the application:

- Confirm classic `Model` inputs point at a compiled model archive.
- Confirm GenAI inputs point at a deployed LLiMa model directory.
- Confirm the installed Neat release set with `neat`: Core, PyNeat, Scheduler, Runner, and LLiMa come from one installation, and the platform version matches the DevKit software. Reinstall Neat as a set rather than mixing packages.
- Confirm image, video, audio, config, and output paths exist or are created by the app.
- Confirm the app uses public endpoint names that exist in the built graph.
- For encoded media, confirm the selected source, decoder, and passthrough sender use the same codec.
- For camera capture, confirm the zero-copy versus `allow_cpu_fallback` choice and validate any changed capture buffer count on the target.

## Runtime Checks

When hardware Neat Library runtime behavior matters, run on Modalix or the
connected DevKit.

Run the supported workflow and inspect useful output against the application's
contract, such as saved detections, streamed metadata, or a generated answer.
A process starting or a request returning successfully is insufficient when the
output itself has not been checked.

Select failure cases from the behavior changed and the inputs the application
exposes. Relevant cases include missing model or input paths, timeout or empty
output handling, configurable endpoint names, and user-supplied GenAI requests.
When lifecycle behavior changes, exercise stop, cleanup, and restart where
supported. When local display changes, verify viewing and close/exit behavior on
the target using the selected backend.

For graph failures, read structured diagnostics first:

1. `error_code`
2. `repro_note`
3. first terminal entry in `bus`
4. `repro_gst_launch`

## Evidence to return

Record the build/run commands, expected output, observed output, and results of
selected failure checks. If hardware, models, or services are unavailable,
identify the missing prerequisite and the checks left unverified. Complete the
independent checks the environment supports.
