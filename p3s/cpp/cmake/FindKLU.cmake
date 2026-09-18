# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

# FindKLU.cmake -- locate SuiteSparse KLU and its dependencies on installs that do
# NOT ship CMake config packages (e.g. system libsuitesparse-dev on Debian <= 12).
# conda-forge and SuiteSparse >= 6 provide KLUConfig.cmake and are used in preference
# (see CMakeLists).
#
# Defines the imported target SuiteSparse::KLU (matching the config-package name) so
# the consuming CMakeLists is identical in both code paths.
#
# Honours KLU_ROOT / SuiteSparse_ROOT / CONDA_PREFIX hints.
#
# Result variables:
#   KLU_FOUND         -- TRUE if KLU and all its dependencies were found
#   KLU_INCLUDE_DIRS  -- include directory containing klu.h
#   KLU_LIBRARIES     -- klu + btf + amd + colamd + suitesparseconfig
#   KLU_VERSION       -- e.g. 1.3.9 (parsed from klu.h)

set(_klu_hints
    ${KLU_ROOT} $ENV{KLU_ROOT}
    ${SuiteSparse_ROOT} $ENV{SuiteSparse_ROOT}
    $ENV{CONDA_PREFIX}
    $ENV{CONDA_PREFIX}/Library   # conda on Windows
)

find_path(KLU_INCLUDE_DIR
    NAMES klu.h
    HINTS ${_klu_hints}
    PATH_SUFFIXES include include/suitesparse suitesparse SuiteSparse
)

# KLU links against BTF, AMD, COLAMD and SuiteSparse_config.
set(_klu_components klu btf amd colamd suitesparseconfig)
set(_klu_libs "")
set(_klu_missing "")
set(_klu_required_vars KLU_INCLUDE_DIR)

foreach(_c IN LISTS _klu_components)
  find_library(KLU_${_c}_LIBRARY
      NAMES ${_c}
      HINTS ${_klu_hints}
      PATH_SUFFIXES lib lib64 Library/lib
  )
  list(APPEND _klu_required_vars KLU_${_c}_LIBRARY)
  if(KLU_${_c}_LIBRARY)
    list(APPEND _klu_libs "${KLU_${_c}_LIBRARY}")
  else()
    list(APPEND _klu_missing ${_c})
  endif()
endforeach()

# Version, from klu.h (present in both SuiteSparse 5.x and >= 6).
set(KLU_VERSION "")
if(KLU_INCLUDE_DIR AND EXISTS "${KLU_INCLUDE_DIR}/klu.h")
  file(STRINGS "${KLU_INCLUDE_DIR}/klu.h" _klu_version_lines
       REGEX "^#define[ \t]+KLU_(MAIN|SUB|SUBSUB)_VERSION[ \t]+[0-9]+")
  foreach(_part MAIN SUB SUBSUB)
    string(REGEX MATCH "KLU_${_part}_VERSION[ \t]+([0-9]+)" _m "${_klu_version_lines}")
    if(_m)
      list(APPEND _klu_version_parts "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(_klu_version_parts)
    string(REPLACE ";" "." KLU_VERSION "${_klu_version_parts}")
  endif()
  unset(_klu_version_parts)
  unset(_klu_version_lines)
endif()

# IMPORTANT: never interpolate a CMake *list* into an argument of
# find_package_handle_standard_args(). FPHSA forwards its arguments internally as an
# unquoted ${ARGN}, which re-splits the string on its semicolons -- the trailing
# elements then surface as "Unknown keywords given to
# FIND_PACKAGE_HANDLE_STANDARD_ARGS()". Flatten to a plain string first.
if(_klu_missing)
  string(REPLACE ";" ", " _klu_missing_str "${_klu_missing}")
  set(_klu_missing_str "Missing libraries: ${_klu_missing_str}.")
elseif(NOT KLU_INCLUDE_DIR)
  set(_klu_missing_str "klu.h was not found.")
else()
  set(_klu_missing_str "")
endif()

set(_klu_version_arg "")
if(KLU_VERSION)
  set(_klu_version_arg VERSION_VAR KLU_VERSION)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KLU
    REQUIRED_VARS ${_klu_required_vars}
    ${_klu_version_arg}
    FAIL_MESSAGE "Could not find SuiteSparse KLU. ${_klu_missing_str} Install SuiteSparse (Debian/Ubuntu: `apt-get install libsuitesparse-dev`, conda: `conda install -c conda-forge suitesparse`) or point KLU_ROOT / SuiteSparse_ROOT at an existing install."
)

if(KLU_FOUND)
  set(KLU_INCLUDE_DIRS "${KLU_INCLUDE_DIR}")
  set(KLU_LIBRARIES "${_klu_libs}")

  if(NOT TARGET SuiteSparse::KLU)
    add_library(SuiteSparse::KLU UNKNOWN IMPORTED)
    set_target_properties(SuiteSparse::KLU PROPERTIES
        IMPORTED_LOCATION "${KLU_klu_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KLU_INCLUDE_DIR}"
    )
    # Link the remaining dependency libraries (btf/amd/colamd/suitesparseconfig).
    set(_klu_deps "${_klu_libs}")
    list(REMOVE_ITEM _klu_deps "${KLU_klu_LIBRARY}")
    if(_klu_deps)
      set_target_properties(SuiteSparse::KLU PROPERTIES
          INTERFACE_LINK_LIBRARIES "${_klu_deps}")
    endif()
    unset(_klu_deps)
  endif()
endif()

mark_as_advanced(KLU_INCLUDE_DIR ${_klu_required_vars})

unset(_klu_hints)
unset(_klu_components)
unset(_klu_libs)
unset(_klu_missing)
unset(_klu_missing_str)
unset(_klu_required_vars)
unset(_klu_version_arg)
