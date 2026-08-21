cmake_minimum_required(VERSION 3.16)

project(pcie-high-density-multi-stream-object-detector-host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(SimaPCIeHost CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-rtp-1.0
  gstreamer-rtsp-1.0
  glib-2.0
)

get_filename_component(APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
get_filename_component(APP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(APP_COMMON_DIR "${APP_ROOT}/src/common")

add_executable(pcie-high-density-multi-stream-object-detector-host
  main.cpp
  "${APP_COMMON_DIR}/app_config.cpp"
  "${APPS_ROOT}/support/runtime/config_utils.cpp"
)

target_link_libraries(pcie-high-density-multi-stream-object-detector-host
  PRIVATE
    SimaPCIeHost::sima_neat_pcie_host
    PkgConfig::GST
    nlohmann_json::nlohmann_json
)

target_include_directories(pcie-high-density-multi-stream-object-detector-host
  PRIVATE
    "${APP_COMMON_DIR}"
    "${APPS_ROOT}"
)

target_compile_definitions(pcie-high-density-multi-stream-object-detector-host
  PRIVATE
    PCIE_HIGH_DENSITY_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
)
