// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

// finite.hpp -- a NaN/Inf test that survives -ffast-math.
//
// The extension is built with -ffast-math (GCC/Clang) and /fp:fast (MSVC). On GCC/Clang that
// implies -ffinite-math-only, which lets the compiler assume no value is ever NaN or Inf and
// fold std::isfinite(x) to `true`. The guards that depend on it -- the lean refactorization's
// pivot check and the batch input validation -- then silently stop detecting NaN, on Linux
// builds only (checked in the generated assembly with GCC 12). Turning finite-math off for the
// whole file is not an option: it measured 3-5% slower single-threaded.
//
// This test looks at the exponent bits instead, which no floating-point mode can reason away:
// a double is NaN or +-Inf exactly when all 11 exponent bits are set.
#pragma once

#include <cstdint>
#include <cstring>

inline bool p3s_isfinite(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
