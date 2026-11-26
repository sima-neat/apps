# Headers for the sysroot's libyaml-cpp.so.0.8, which Core links. The SDK ships no -dev package.
# Offline builds: README "Build From Source".
include_guard(GLOBAL)

# The function below records these policy settings where it is defined.
# Extracted files get the extraction time, without DOWNLOAD_EXTRACT_TIMESTAMP (CMake 3.24).
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
# Keep FetchContent_Populate(<name>), deprecated in CMake 3.30, until the minimum CMake is 3.18.
if (POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

function(sima_neat_apps_ensure_yaml_cpp)
  if (TARGET SimaNeatApps::yaml_cpp)
    return()
  endif()

  # Keep this library and the tarball URL and hash below at the same yaml-cpp version (0.8).
  find_library(SIMA_NEAT_APPS_YAML_CPP_LIBRARY NAMES libyaml-cpp.so.0.8)
  if (NOT SIMA_NEAT_APPS_YAML_CPP_LIBRARY)
    message(FATAL_ERROR
      "libyaml-cpp.so.0.8 not found: the SDK sysroot ships it as a dependency of Neat Core")
  endif()

  if (NOT "$ENV{FETCHCONTENT_SOURCE_DIR_YAML_CPP}" STREQUAL "")
    set(FETCHCONTENT_SOURCE_DIR_YAML_CPP "$ENV{FETCHCONTENT_SOURCE_DIR_YAML_CPP}"
      CACHE PATH "Unpacked yaml-cpp 0.8.0 sources" FORCE)
  endif()
  if (NOT "${FETCHCONTENT_SOURCE_DIR_YAML_CPP}" STREQUAL ""
      AND NOT IS_ABSOLUTE "${FETCHCONTENT_SOURCE_DIR_YAML_CPP}")
    message(FATAL_ERROR "FETCHCONTENT_SOURCE_DIR_YAML_CPP must be an absolute path: "
      "${FETCHCONTENT_SOURCE_DIR_YAML_CPP}")
  endif()
  set(YAML_CPP_SOURCE_URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz"
    CACHE STRING "yaml-cpp 0.8.0 source tarball")
  if (NOT "$ENV{YAML_CPP_SOURCE_URL}" STREQUAL "")
    set(YAML_CPP_SOURCE_URL "$ENV{YAML_CPP_SOURCE_URL}" CACHE STRING
      "yaml-cpp 0.8.0 source tarball" FORCE)
  endif()
  if (NOT YAML_CPP_SOURCE_URL MATCHES "^[A-Za-z][A-Za-z0-9+.-]*://"
      AND NOT IS_ABSOLUTE "${YAML_CPP_SOURCE_URL}")
    message(FATAL_ERROR "YAML_CPP_SOURCE_URL must be a URL or an absolute path: "
      "${YAML_CPP_SOURCE_URL}")
  endif()

  include(FetchContent)
  FetchContent_Declare(yaml_cpp
    URL "${YAML_CPP_SOURCE_URL}"
    URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
  )
  FetchContent_GetProperties(yaml_cpp)
  if (NOT yaml_cpp_POPULATED)
    FetchContent_Populate(yaml_cpp)
  endif()
  if (NOT EXISTS "${yaml_cpp_SOURCE_DIR}/include/yaml-cpp/yaml.h")
    message(FATAL_ERROR "${yaml_cpp_SOURCE_DIR} has no include/yaml-cpp/yaml.h: "
      "FETCHCONTENT_SOURCE_DIR_YAML_CPP must name an unpacked yaml-cpp-0.8.0")
  endif()

  add_library(SimaNeatApps::yaml_cpp SHARED IMPORTED GLOBAL)
  # Consumers get this directory as -isystem, which GCC searches before <sysroot>/usr/include.
  set_target_properties(SimaNeatApps::yaml_cpp PROPERTIES
    IMPORTED_LOCATION "${SIMA_NEAT_APPS_YAML_CPP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${yaml_cpp_SOURCE_DIR}/include"
  )
endfunction()
