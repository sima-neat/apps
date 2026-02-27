set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Avoid link checks against host during toolchain probing.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  set(_SYSROOT "$ENV{SYSROOT}")
elseif(DEFINED SYSROOT AND NOT "${SYSROOT}" STREQUAL "")
  set(_SYSROOT "${SYSROOT}")
endif()

if(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
  set(CMAKE_C_COMPILER "$ENV{CC}" CACHE FILEPATH "C compiler")
elseif(DEFINED ENV{CROSS_COMPILE} AND NOT "$ENV{CROSS_COMPILE}" STREQUAL "")
  set(CMAKE_C_COMPILER "$ENV{CROSS_COMPILE}gcc" CACHE FILEPATH "C compiler")
endif()

if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE FILEPATH "CXX compiler")
elseif(DEFINED ENV{CROSS_COMPILE} AND NOT "$ENV{CROSS_COMPILE}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "$ENV{CROSS_COMPILE}g++" CACHE FILEPATH "CXX compiler")
endif()

if(DEFINED _SYSROOT)
  set(CMAKE_SYSROOT "${_SYSROOT}" CACHE PATH "Cross sysroot")
  set(CMAKE_FIND_ROOT_PATH "${_SYSROOT}" CACHE PATH "CMake find root path")
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY CACHE STRING "Find packages only in sysroot")
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY CACHE STRING "Find libraries only in sysroot")
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY CACHE STRING "Find headers only in sysroot")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER CACHE STRING "Find programs on host")

  if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
    set(
      CMAKE_PREFIX_PATH
      "${_SYSROOT}/usr;${_SYSROOT}/usr/lib/cmake;${_SYSROOT}/usr/lib/aarch64-linux-gnu/cmake"
      CACHE STRING "Cross package prefixes"
    )
  endif()

  # Make host pkg-config resolve target .pc metadata and sysroot all paths.
  if(NOT DEFINED ENV{PKG_CONFIG_SYSROOT_DIR} OR "$ENV{PKG_CONFIG_SYSROOT_DIR}" STREQUAL "")
    set(ENV{PKG_CONFIG_SYSROOT_DIR} "${_SYSROOT}")
  endif()
  if(NOT DEFINED ENV{PKG_CONFIG_LIBDIR} OR "$ENV{PKG_CONFIG_LIBDIR}" STREQUAL "")
    set(
      ENV{PKG_CONFIG_LIBDIR}
      "${_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${_SYSROOT}/usr/lib/pkgconfig:${_SYSROOT}/usr/share/pkgconfig"
    )
  endif()
  set(ENV{PKG_CONFIG_DIR} "")
endif()

if(NOT DEFINED PKG_CONFIG_EXECUTABLE OR PKG_CONFIG_EXECUTABLE STREQUAL "")
  if(DEFINED ENV{PKG_CONFIG_EXECUTABLE} AND NOT "$ENV{PKG_CONFIG_EXECUTABLE}" STREQUAL "")
    set(PKG_CONFIG_EXECUTABLE "$ENV{PKG_CONFIG_EXECUTABLE}" CACHE FILEPATH "pkg-config executable")
  elseif(EXISTS "/usr/bin/pkg-config")
    set(PKG_CONFIG_EXECUTABLE "/usr/bin/pkg-config" CACHE FILEPATH "pkg-config executable")
  endif()
endif()
