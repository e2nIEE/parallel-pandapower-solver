# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

# FindKLU.cmake -- locate SuiteSparse KLU and its dependencies on installs that do
# NOT ship CMake config packages (e.g. system libsuitesparse-dev). conda-forge and
# SuiteSparse >= 6 provide KLUConfig.cmake and are used in preference (see CMakeLists).
#
# Defines the imported target SuiteSparse::KLU (matching the config-package name) so
# the consuming CMakeLists is identical in both code paths.
#
# Honours KLU_ROOT / SuiteSparse_ROOT / CONDA_PREFIX hints.

set(_klu_hints
    ${KLU_ROOT} $ENV{KLU_ROOT}
    ${SuiteSparse_ROOT} $ENV{SuiteSparse_ROOT}
    $ENV{CONDA_PREFIX}
    $ENV{CONDA_PREFIX}/Library   # conda on Windows
)

find_path(KLU_INCLUDE_DIR
    NAMES klu.h
    HINTS ${_klu_hints}
    PATH_SUFFIXES include include/suitesparse suitesparse
)

# KLU links against BTF, AMD, COLAMD and SuiteSparse_config.
set(_klu_components klu btf amd colamd suitesparseconfig)
set(_klu_libs "")
set(_klu_missing "")
foreach(_c ${_klu_components})
  find_library(KLU_${_c}_LIBRARY
      NAMES ${_c}
      HINTS ${_klu_hints}
      PATH_SUFFIXES lib lib64 Library/lib
  )
  if(KLU_${_c}_LIBRARY)
    list(APPEND _klu_libs ${KLU_${_c}_LIBRARY})
  else()
    list(APPEND _klu_missing ${_c})
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KLU
    REQUIRED_VARS KLU_INCLUDE_DIR KLU_klu_LIBRARY
    FAIL_MESSAGE "Could not find KLU. Install SuiteSparse (e.g. `conda install -c conda-forge suitesparse`) or set KLU_ROOT. Missing libs: ${_klu_missing}"
)

if(KLU_FOUND AND NOT TARGET SuiteSparse::KLU)
  add_library(SuiteSparse::KLU UNKNOWN IMPORTED)
  set_target_properties(SuiteSparse::KLU PROPERTIES
      IMPORTED_LOCATION "${KLU_klu_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${KLU_INCLUDE_DIR}"
  )
  # link the remaining dependency libraries (btf/amd/colamd/suitesparseconfig)
  list(REMOVE_ITEM _klu_libs "${KLU_klu_LIBRARY}")
  if(_klu_libs)
    set_target_properties(SuiteSparse::KLU PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_klu_libs}")
  endif()
endif()

mark_as_advanced(KLU_INCLUDE_DIR)
