cmake_minimum_required(VERSION 3.16)
project(SimaNeatApps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SIMANEAT_APPS_BUILD_CPP "Build C++ examples" ON)
option(SIMANEAT_APPS_BUILD_PYTHON "Reserved toggle for future Python example tooling" OFF)
option(SIMANEAT_APPS_BUILD_OPTIVIEW_SUPPORT "Build shared OptiView support library" ON)

message(STATUS "Configuring SimaNeatApps")
message(STATUS "  Build C++ examples      : ${SIMANEAT_APPS_BUILD_CPP}")
message(STATUS "  Build Python tooling    : ${SIMANEAT_APPS_BUILD_PYTHON}")
message(STATUS "  Build OptiView support  : ${SIMANEAT_APPS_BUILD_OPTIVIEW_SUPPORT}")

if (SIMANEAT_APPS_BUILD_CPP)
  find_package(SimaNeat REQUIRED)
  find_package(nlohmann_json REQUIRED)
  add_subdirectory(support)
  add_subdirectory(examples)
endif()

if (SIMANEAT_APPS_BUILD_PYTHON)
  message(STATUS "Python examples are source-only placeholders in this initial implementation.")
endif()
