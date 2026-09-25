# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18)
project(p3s_cpp LANGUAGES CXX)

# Portable build for the nr_klu pybind11 extension (fast single-grid Newton-Raphson
# power flow on KLU). Driven by scikit-build-core via p3s/cpp/pyproject.toml, so
# `pip install p3s[cpp]` builds it the same way on Linux, macOS and Windows.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimisation flags, per-compiler (the old build.sh used -O3 -march=native -ffast-math).
#
# -march=native / /arch:AVX2 are great for a local build but break redistributable
# wheels (the binary refuses to start on a host without the ISA), so both compilers
# gate host tuning behind the same option.
option(P3S_CPP_NATIVE "Tune for the build host CPU (-march=native / /arch:AVX2)" ON)
if(MSVC)
  add_compile_options(/O2 /fp:fast /EHsc)
  # MSVC's default x64 target is SSE2: 2-wide doubles and no FMA. eval_F_and_J is a
  # streaming sincos + multiply-add loop over the Ybus nonzeros, so the width matters.
  if(P3S_CPP_NATIVE)
    add_compile_options(/arch:AVX2)
  endif()
else()
  add_compile_options(-O3 -funroll-loops -ffast-math)
  if(P3S_CPP_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# --- Python + pybind11 -------------------------------------------------------
# Only request Interpreter + Development.Module. We deliberately do NOT request the
# NumPy component of FindPython: its header-location heuristic is unreliable (newer
# numpy moved its headers to numpy/_core/include, which trips CMake's FindPython and
# yields "Could NOT find Python (missing: NumPy)"). Instead we ask the interpreter for
# the include dir directly below -- that always works, across conda/venv/system.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Resolve the numpy headers via an interpreter (robust to numpy's layout changes).
# numpy is a build requirement, so it lives in the (isolated) build env -- which may be
# a DIFFERENT interpreter than the one FindPython selected for linking. Try the build
# interpreter (sys.executable of the process running CMake / scikit-build-core) first,
# then the FindPython interpreter, then PATH. The headers are ABI-compatible across
# these for the numpy C-API we use.
set(_numpy_candidates "")
if(DEFINED ENV{PYTHON})            # scikit-build-core / pip often set this to the build python
  list(APPEND _numpy_candidates "$ENV{PYTHON}")
endif()
list(APPEND _numpy_candidates "${Python_EXECUTABLE}" "python" "python3")

set(NUMPY_INCLUDE_DIR "")
foreach(_pyexe ${_numpy_candidates})
  execute_process(
    COMMAND "${_pyexe}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE _numpy_inc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _numpy_rc
    ERROR_QUIET
  )
  if(_numpy_rc EQUAL 0 AND EXISTS "${_numpy_inc}/numpy/arrayobject.h")
    set(NUMPY_INCLUDE_DIR "${_numpy_inc}")
    break()
  endif()
endforeach()

if(NOT NUMPY_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not locate numpy headers. numpy must be importable in the build "
    "environment (it is listed in build-system.requires; if you build with "
    "--no-build-isolation, install numpy into the active env first).")
endif()
message(STATUS "nr_klu: numpy headers at ${NUMPY_INCLUDE_DIR}")

# --- KLU / SuiteSparse -------------------------------------------------------
# conda-forge and SuiteSparse >= 6 ship CMake config packages (lib/cmake/KLU, etc.),
# which is the clean cross-platform path. Fall back to a manual FindKLU for older /
# system installs that only provide headers + libraries.

# Make sure CMake searches the active conda environment. When pip builds in an isolated
# subprocess the activated env's CMAKE_PREFIX_PATH may not propagate, so add CONDA_PREFIX
# explicitly (Library/ on Windows, the prefix root on Linux/macOS).
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(KLU CONFIG QUIET)
if(NOT KLU_FOUND AND NOT TARGET SuiteSparse::KLU)
  find_package(KLU MODULE REQUIRED)
endif()

# The config package exposes the SuiteSparse::KLU imported target (which pulls in its
# own AMD/BTF/COLAMD/SuiteSparse_config dependencies). The MODULE fallback defines the
# same target name so the rest of this file is identical either way.

# --- OpenMP (optional) -------------------------------------------------------
# Enables the multi-threaded batch solve (Solver.solve_batch with n_threads>1).
# Optional: without it the batch path still works, just serially.
find_package(OpenMP QUIET)

pybind11_add_module(nr_klu nr_klu.cpp)
set_target_properties(nr_klu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(nr_klu PRIVATE "${NUMPY_INCLUDE_DIR}")
target_link_libraries(nr_klu PRIVATE SuiteSparse::KLU)
if(OpenMP_CXX_FOUND)
  target_link_libraries(nr_klu PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "nr_klu: OpenMP enabled (multi-threaded batch available)")
else()
  message(STATUS "nr_klu: OpenMP NOT found (batch solve will run single-threaded)")
endif()

# Install the compiled module *into the p3s package* so `from p3s import
# nr_klu` resolves with no path juggling. scikit-build-core places it under the wheel
# root; we target the p3s/ subdirectory.
install(TARGETS nr_klu LIBRARY DESTINATION p3s RUNTIME DESTINATION p3s)
