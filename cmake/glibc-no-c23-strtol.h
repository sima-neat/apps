#pragma once

// Build-time compatibility shim for cross-building Modalix applications with
// newer host glibc headers. GCC's C++ driver predefines _GNU_SOURCE, which can
// make recent glibc headers redirect strtol-family APIs to __isoc23_* symbols.
// Older Modalix runtimes do not export those symbols. Keep _GNU_SOURCE enabled
// for libstdc++ and pthread feature visibility, but disable only that redirect.
#include <features.h>
#ifdef __GLIBC_USE_C2X_STRTOL
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
#endif
