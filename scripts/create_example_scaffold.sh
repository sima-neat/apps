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
  cat <<EOF
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
  std::cout << "${example_name}\n";
  return 0;
}
EOF
}

render_cpp_cmakelists() {
  local example_name="$1"
  cat <<EOF
cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(${example_name} LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

include("\${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ExampleModule.cmake")
sima_neat_apps_module("${example_name}")

if (BUILD_TESTING)
  # Register example-specific smoke or e2e tests here.
endif()
EOF
}

render_cpp_readme() {
  local example_name="$1"
  local category="$2"
  cat <<EOF
# ${example_name}

## Metadata
| Field | Value |
| --- | --- |
| Category | ${category} |
| Difficulty | Intermediate |
| Tags | ${category} |
| Status | experimental |
| Binary Name | ${example_name} |
| Model | TODO |

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
cd <apps-repo-root>/examples/${category}/${example_name}
cmake -S . -B build
cmake --build build -j
\`\`\`

## Run
\`\`\`bash
./build/${example_name}
\`\`\`

## Source Files
- C++: \`main.cpp\`
- Tests: \`tests/e2e_test.cpp\`
EOF
}

render_cpp_e2e_test() {
  cat <<'EOF'
#include <iostream>

int main() {
  std::cerr << "[SKIP] TODO: implement example e2e test\n";
  return 77;
}
EOF
}

render_python_main() {
  local example_name="$1"
  cat <<EOF
#!/usr/bin/env python3

def main() -> int:
    print("${example_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
}

render_python_readme() {
  local example_name="$1"
  local category="$2"
  cat <<EOF
# ${example_name}

## Metadata
| Field | Value |
| --- | --- |
| Category | ${category} |
| Difficulty | Intermediate |
| Tags | ${category}, python |
| Status | experimental |
| Binary Name | ${example_name} |
| Model | TODO |

## Concept
TODO: describe what this Python example demonstrates.

## Run
\`\`\`bash
python3 main.py
\`\`\`

## Source Files
- Python: \`main.py\`
- Tests: \`tests/e2e_test.py\`
EOF
}

render_python_e2e_test() {
  cat <<'EOF'
#!/usr/bin/env python3
import sys


def main() -> int:
    print("[SKIP] TODO: implement example e2e test", file=sys.stderr)
    return 77


if __name__ == "__main__":
    raise SystemExit(main())
EOF
}

register_cpp_example() {
  local category_file="$1"
  local example_name="$2"

  grep -Fqx "add_subdirectory(${example_name})" "${category_file}" && return 0
  printf 'add_subdirectory(%s)\n' "${example_name}" >> "${category_file}"
}

create_cpp_example() {
  local category="$1"
  local example_name="$2"
  local module_dir="${EXAMPLES_DIR}/${category}/${example_name}"
  local category_cmake="${EXAMPLES_DIR}/${category}/CMakeLists.txt"

  [[ -d "${EXAMPLES_DIR}/${category}" ]] || die "Unknown C++ category: ${category}"
  [[ -f "${category_cmake}" ]] || die "Missing category CMakeLists: ${category_cmake}"
  [[ ! -e "${module_dir}" ]] || die "Example already exists: ${module_dir}"

  ensure_dir "${module_dir}/tests"
  render_cpp_main "${example_name}" > "${module_dir}/main.cpp"
  render_cpp_cmakelists "${example_name}" > "${module_dir}/CMakeLists.txt"
  render_cpp_readme "${example_name}" "${category}" > "${module_dir}/README.md"
  render_cpp_e2e_test > "${module_dir}/tests/e2e_test.cpp"
  register_cpp_example "${category_cmake}" "${example_name}"
}

create_python_example() {
  local category="$1"
  local example_name="$2"
  local module_dir="${EXAMPLES_DIR}/${category}/${example_name}"

  [[ ! -e "${module_dir}" ]] || die "Example already exists: ${module_dir}"

  ensure_dir "${module_dir}/tests"
  render_python_main "${example_name}" > "${module_dir}/main.py"
  render_python_readme "${example_name}" "${category}" > "${module_dir}/README.md"
  render_python_e2e_test > "${module_dir}/tests/e2e_test.py"
  chmod +x "${module_dir}/main.py" "${module_dir}/tests/e2e_test.py"
}

main() {
  local language
  language="$(prompt_choice "Select example language" "C++" "Python")"

  local category
  local example_name

  mapfile -t categories < <(find "${EXAMPLES_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  category="$(prompt_choice "Select example category" "${categories[@]}")"

  example_name="$(prompt_nonempty "Enter example name")"
  validate_name "${example_name}" || die "Example name must match [A-Za-z0-9._-]+"

  if [[ "${language}" == "C++" ]]; then
    create_cpp_example "${category}" "${example_name}"
    echo "Created C++ example: ${EXAMPLES_DIR}/${category}/${example_name}"
  else
    create_python_example "${category}" "${example_name}"
    echo "Created Python example: ${EXAMPLES_DIR}/${category}/${example_name}"
  fi
}

main "$@"
