# yaml-cpp for the support library without the -dev package in the SDK sysroot.
#
# Neat Core already links libyaml-cpp.so.0.8, so the board and the SDK sysroot carry that
# runtime library; only the headers are missing. They come from the matching upstream
# release, pinned by hash, and the examples link the same runtime library Core uses.
# One yaml-cpp in the process, and no change to the sysroot.
#
# Offline builds: point FETCHCONTENT_SOURCE_DIR_YAML_CPP at an unpacked yaml-cpp-0.8.0,
# or YAML_CPP_SOURCE_URL at a local copy of the tarball, as a -D option or an environment
# variable, so `YAML_CPP_SOURCE_URL=<tarball> ./build.sh` works. A non-empty environment
# variable replaces the cached value on every configure.
include_guard(GLOBAL)

# The function below records these policy settings where it is defined.
# Extracted files get the extraction time, without DOWNLOAD_EXTRACT_TIMESTAMP (CMake 3.24).
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
# FetchContent_Populate(<name>) downloads without add_subdirectory() on every CMake from
# 3.16; CMake 3.30 deprecates it in favour of FetchContent_MakeAvailable, which needs 3.18
# to skip yaml-cpp's own CMakeLists.txt.
if (POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

function(sima_neat_apps_ensure_yaml_cpp)
  if (TARGET SimaNeatApps::yaml_cpp)
    return()
  endif()

  # The ABI of these headers must match the runtime library below: change both together.
  find_library(SIMA_NEAT_APPS_YAML_CPP_LIBRARY NAMES libyaml-cpp.so.0.8)
  if (NOT SIMA_NEAT_APPS_YAML_CPP_LIBRARY)
    message(FATAL_ERROR
      "libyaml-cpp.so.0.8 not found: the SDK sysroot ships it as a dependency of Neat Core")
  endif()

  if (NOT "$ENV{FETCHCONTENT_SOURCE_DIR_YAML_CPP}" STREQUAL "")
    set(FETCHCONTENT_SOURCE_DIR_YAML_CPP "$ENV{FETCHCONTENT_SOURCE_DIR_YAML_CPP}"
      CACHE PATH "Unpacked yaml-cpp 0.8.0 sources" FORCE)
  endif()
  set(YAML_CPP_SOURCE_URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz"
    CACHE STRING "yaml-cpp 0.8.0 source tarball")
  if (NOT "$ENV{YAML_CPP_SOURCE_URL}" STREQUAL "")
    set(YAML_CPP_SOURCE_URL "$ENV{YAML_CPP_SOURCE_URL}" CACHE STRING
      "yaml-cpp 0.8.0 source tarball" FORCE)
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

  add_library(SimaNeatApps::yaml_cpp SHARED IMPORTED GLOBAL)
  # Consumers get this directory with -isystem, which GCC searches before the sysroot's
  # /usr/include. The SDK's -I<sysroot>/usr/include is dropped as a duplicate of that
  # system directory, so yaml-cpp headers in the sysroot do not take precedence.
  set_target_properties(SimaNeatApps::yaml_cpp PROPERTIES
    IMPORTED_LOCATION "${SIMA_NEAT_APPS_YAML_CPP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${yaml_cpp_SOURCE_DIR}/include"
  )
endfunction()
