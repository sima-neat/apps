cmake_minimum_required(VERSION 3.16)
project(SimaNeatApps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(SIMANEAT_APPS_BUILD_CPP "Build C++ examples" ON)

function(_sima_neat_apps_ensure_neat_target)
  if (TARGET SimaNeatApps::sima_neat)
    return()
  endif()

  find_package(SimaNeat CONFIG QUIET)
  if (TARGET SimaNeat::sima_neat_shared)
    set(_core_target SimaNeat::sima_neat_shared)
  elseif (TARGET SimaNeat::sima_neat)
    set(_core_target SimaNeat::sima_neat)
  else()
    set(_core_root "${CMAKE_CURRENT_LIST_DIR}/../core")
    set(_core_lib "${_core_root}/build/libsima_neat.so")
    set(_core_include "${_core_root}/include")
    if (NOT EXISTS "${_core_lib}" OR NOT EXISTS "${_core_include}")
      message(FATAL_ERROR
        "Could not find a usable SimaNeat package or local core build. "
        "Expected ${_core_lib} and ${_core_include}.")
    endif()

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
      gstreamer-1.0
      gstreamer-app-1.0
      gstreamer-video-1.0
      gstreamer-sdp-1.0
      gstreamer-rtsp-server-1.0
      glib-2.0
    )
    pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

    add_library(SimaNeat::sima_neat_shared SHARED IMPORTED GLOBAL)
    set_target_properties(SimaNeat::sima_neat_shared PROPERTIES
      IMPORTED_LOCATION "${_core_lib}"
      INTERFACE_COMPILE_DEFINITIONS "SIMA_WITH_OPENCV;SIMA_HAS_SIMAAI_POOL=1"
      INTERFACE_INCLUDE_DIRECTORIES "${_core_include};${_core_include}/pipeline"
      INTERFACE_LINK_LIBRARIES "PkgConfig::GST;PkgConfig::OPENCV;gstsimaallocator;gstsimaaibufferpool"
    )
    set(_core_target SimaNeat::sima_neat_shared)
  endif()

  add_library(sima_neat_apps_neat_core INTERFACE)
  add_library(SimaNeatApps::sima_neat ALIAS sima_neat_apps_neat_core)
  target_link_libraries(sima_neat_apps_neat_core INTERFACE "${_core_target}")
endfunction()

message(STATUS "Configuring SimaNeatApps")
message(STATUS "  Build C++ examples      : ${SIMANEAT_APPS_BUILD_CPP}")

if (SIMANEAT_APPS_BUILD_CPP)
  _sima_neat_apps_ensure_neat_target()
  find_package(nlohmann_json REQUIRED)
  add_subdirectory(support)
  add_subdirectory(examples)
endif()
