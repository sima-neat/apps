#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="${ROOT_DIR}/examples"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

prompt_nonempty() {
  local label="$1"
  local value=""
  while [[ -z "${value}" ]]; do
    read -r -p "${label}: " value
  done
  printf '%s' "${value}"
}

prompt_choice() {
  local label="$1"
  shift
  local options=("$@")
  local count="${#options[@]}"
  local choice=""

  printf '%s\n' "${label}" >&2
  for ((i = 0; i < count; ++i)); do
    printf '  %d) %s\n' "$((i + 1))" "${options[i]}" >&2
  done

  while true; do
    read -r -p "Enter choice [1-${count}]: " choice
    if [[ "${choice}" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= count )); then
      printf '%s' "${options[choice - 1]}"
      return 0
    fi
    printf 'Invalid choice. Please enter a number between 1 and %d.\n' "${count}" >&2
  done
}

validate_name() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9._-]+$ ]]
}

ensure_dir() {
  mkdir -p "$1"
}

render_cpp_main() {
  local example_name="$1"
  cat <<EOF_CPP
// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::cout << "${example_name}\\n";
  return 0;
}
EOF_CPP
}

render_cpp_cmakelists() {
  local example_name="$1"
  cat <<EOF_CMAKE
cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(${example_name} LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("\${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("${example_name}" UNIT_TEST E2E_TEST)
EOF_CMAKE
}

render_cpp_test_cmakelists() {
  cat <<'EOF_CMAKE'
# Test sources are registered by ../../../../cmake/ExampleModule.cmake.
# Keep this file to make the expected test layout explicit per example.
EOF_CMAKE
}

render_cpp_unit_test() {
  cat <<'EOF_CPP'
#include <iostream>

int main() {
  std::cerr << "[SKIP] TODO: implement example unit test\\n";
  return 77;
}
EOF_CPP
}

render_cpp_e2e_test() {
  cat <<'EOF_CPP'
#include <iostream>

int main() {
  std::cerr << "[SKIP] TODO: implement example e2e test\\n";
  return 77;
}
EOF_CPP
}

render_python_main() {
  local example_name="$1"
  cat <<EOF_PY
#!/usr/bin/env python3

def main() -> int:
    print("${example_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF_PY
}

render_python_requirements() {
  cat <<'EOF_REQ'
numpy
opencv-python
EOF_REQ
}

render_python_unit_test() {
  cat <<'EOF_PY'
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(EXAMPLE_DIR),
    )
    assert result.returncode == 0
EOF_PY
}

render_python_e2e_test() {
  cat <<'EOF_PY'
#!/usr/bin/env python3
import pytest


@pytest.mark.e2e
def test_e2e_placeholder() -> None:
    pytest.skip("TODO: implement example e2e test")
EOF_PY
}

render_readme() {
  local example_name="$1"
  local category="$2"
  cat <<EOF_MD
# ${example_name}

## Metadata
| Field | Value |
| --- | --- |
| Category | ${category} |
| Difficulty | Intermediate |
| Tags | ${category} |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | ${example_name} |
| Model | TODO or TODO [https://host/path/model_mpk.tar.gz] |

## Concept
TODO: describe what this example demonstrates.

## Build
### Build From The Apps Repo
\`\`\`bash
cd <apps-repo-root>
./build.sh
\`\`\`

### Build This Example Directly With CMake
\`\`\`bash
cd <apps-repo-root>
cmake -S examples/${category}/${example_name}/cpp -B build/${example_name}
cmake --build build/${example_name} -j
\`\`\`

## Run
### C++
\`\`\`bash
./build/examples/${category}/${example_name}/${example_name}
\`\`\`

### Python
\`\`\`bash
source ~/pyneat/bin/activate
pip install -r examples/${category}/${example_name}/python/requirements.txt
python3 examples/${category}/${example_name}/python/main.py
\`\`\`

## Source Files
- C++: \`cpp/main.cpp\`
- C++ tests: \`cpp/tests/unit_test.cpp\`, \`cpp/tests/e2e_test.cpp\`
- Python: \`python/main.py\`
- Python tests: \`python/tests/test_unit.py\`, \`python/tests/test_e2e.py\`
- Shared assets: \`common/\`
EOF_MD
}

register_cpp_example() {
  local category_file="$1"
  local example_name="$2"
  local line="add_subdirectory(${example_name}/cpp ${example_name})"

  grep -Fqx "${line}" "${category_file}" && return 0
  printf '%s\n' "${line}" >> "${category_file}"
}

create_example() {
  local category="$1"
  local example_name="$2"
  local module_dir="${EXAMPLES_DIR}/${category}/${example_name}"
  local category_cmake="${EXAMPLES_DIR}/${category}/CMakeLists.txt"

  [[ -d "${EXAMPLES_DIR}/${category}" ]] || die "Unknown category: ${category}"
  [[ -f "${category_cmake}" ]] || die "Missing category CMakeLists: ${category_cmake}"
  [[ ! -e "${module_dir}" ]] || die "Example already exists: ${module_dir}"

  ensure_dir "${module_dir}/cpp/tests"
  ensure_dir "${module_dir}/python/tests"
  ensure_dir "${module_dir}/common"
  : > "${module_dir}/common/.gitkeep"

  render_cpp_main "${example_name}" > "${module_dir}/cpp/main.cpp"
  render_cpp_cmakelists "${example_name}" > "${module_dir}/cpp/CMakeLists.txt"
  render_cpp_test_cmakelists > "${module_dir}/cpp/tests/CMakeLists.txt"
  render_cpp_unit_test > "${module_dir}/cpp/tests/unit_test.cpp"
  render_cpp_e2e_test > "${module_dir}/cpp/tests/e2e_test.cpp"

  render_python_main "${example_name}" > "${module_dir}/python/main.py"
  render_python_requirements > "${module_dir}/python/requirements.txt"
  render_python_unit_test > "${module_dir}/python/tests/test_unit.py"
  render_python_e2e_test > "${module_dir}/python/tests/test_e2e.py"

  render_readme "${example_name}" "${category}" > "${module_dir}/README.md"

  chmod +x \
    "${module_dir}/python/main.py" \
    "${module_dir}/python/tests/test_unit.py" \
    "${module_dir}/python/tests/test_e2e.py"

  register_cpp_example "${category_cmake}" "${example_name}"
}

main() {
  local category
  local example_name

  mapfile -t categories < <(find "${EXAMPLES_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  category="$(prompt_choice "Select example category" "${categories[@]}")"

  example_name="$(prompt_nonempty "Enter example name")"
  validate_name "${example_name}" || die "Example name must match [A-Za-z0-9._-]+"

  create_example "${category}" "${example_name}"
  echo "Created example: ${EXAMPLES_DIR}/${category}/${example_name}"
}

main "$@"
