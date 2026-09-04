# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from numpy import abs, argwhere, dtype, isclose, ndarray, nonzero, signedinteger


def differing_entries(a: ndarray, b: ndarray, rtol=1e-05, atol=1e-08):
    """
    Return a list of (row, col, a_val, b_val) where a and b differ
    beyond the given tolerance.
    """
    if a.shape != b.shape:
        raise ValueError("Shapes of a and b must match")

    mask = ~isclose(a, b, rtol=rtol, atol=atol)  # False = close, True = differ
    idxs = argwhere(mask)  # 2-D indices where they differ
    # return [(int(i), int(j), a[i, j], b[i, j]) for i, j in idxs]
    return idxs


def _compare_arrays(arr1, arr2, tolerance=0.1) -> tuple[bool, ndarray[Any, dtype[signedinteger[Any]]]]:
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same shape.")

    # Calculate the absolute difference
    diff = abs(arr1 - arr2)

    # Find positions where the difference exceeds the tolerance
    indices = nonzero(diff > tolerance)[0]

    # Return the positions of differences that are too large
    # indices contains the row indices for 1D arrays
    return len(indices) == 0, indices


def compare_arrays(table, arr1, arr2, atol=0.1) -> bool:
    is_equal, where = _compare_arrays(arr1, arr2, atol)
    if not is_equal:
        name = ""
        if "name" in dir(arr1):
            name = arr1.name
        print(f"In {table}.{name}, the following elements are not equal: {where}")
        return False
    return True
