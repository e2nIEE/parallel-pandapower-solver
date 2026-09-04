# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone cuDSS availability + functionality probe.

Does NOT need p3s. Steps, each printed so a crash localizes the failure:
  1. locate libcudss.so (pip-wheel path, LD_LIBRARY_PATH, or bare name)
  2. dlopen it
  3. cudssGetProperty -> print version (proves the lib is real, no GPU needed yet)
  4. cudssCreate/cudssDestroy a handle (proves it links against CUDA + the driver works)

    python -m p3s.cuda.check_cudss
"""

import ctypes
import glob
import os
import sys


def find_libcudss():
    """Return a loadable path to libcudss.so, searching the pip wheel first."""
    cands = []
    # pip wheel: site-packages/nvidia/**/libcudss.so*
    try:
        import site

        bases = list(site.getsitepackages())
        try:
            bases.append(site.getusersitepackages())
        except Exception:
            pass
        for b in bases:
            cands += glob.glob(os.path.join(b, "nvidia", "**", "libcudss.so*"), recursive=True)
    except Exception:
        pass
    # LD_LIBRARY_PATH
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if d:
            cands += glob.glob(os.path.join(d, "libcudss.so*"))
    # bare names (loader's own path)
    cands += ["libcudss.so", "libcudss.so.0"]
    # prefer the most specific (versioned) real files first
    seen, ordered = set(), []
    for c in sorted(cands, reverse=True):
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def main():
    print(f"python: {sys.version.split()[0]}", flush=True)
    print("[1] searching for libcudss.so ...", flush=True)
    cands = find_libcudss()
    for c in cands:
        print(f"    candidate: {c}", flush=True)

    lib = None
    for c in cands:
        try:
            lib = ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            print(f"[2] PASS loaded: {c}", flush=True)
            break
        except OSError as e:
            last = e
    if lib is None:
        print(f"[2] FAIL: could not load libcudss.so. Last error: {last}", flush=True)
        print("    -> pip install nvidia-cudss-cu12  (and/or add its lib dir to LD_LIBRARY_PATH)", flush=True)
        return

    # 3. version (cudssGetProperty(libraryPropertyType, int*)); enum: MAJOR=0,MINOR=1,PATCH=2
    print("[3] querying cuDSS version via cudssGetProperty ...", flush=True)
    try:
        lib.cudssGetProperty.restype = ctypes.c_int
        lib.cudssGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        vals = []
        for t in (0, 1, 2):
            v = ctypes.c_int(-1)
            st = lib.cudssGetProperty(t, ctypes.byref(v))
            if st != 0:
                print(f"    cudssGetProperty(type={t}) status {st}", flush=True)
            vals.append(v.value)
        print(f"[3] PASS cuDSS version: {vals[0]}.{vals[1]}.{vals[2]}", flush=True)
        if (vals[0], vals[1]) < (0, 6):
            print(
                "    NOTE: uniform batched solve needs cuDSS >= 0.6.0; this is older -- "
                "the general batch API still works but without the uniform fast path.",
                flush=True,
            )
    except AttributeError:
        print("[3] WARN: cudssGetProperty symbol missing (very old/renamed build)", flush=True)

    # 4. create + destroy a handle (needs a working CUDA context/driver)
    print("[4] cudssCreate/cudssDestroy a handle ...", flush=True)
    try:
        lib.cudssCreate.restype = ctypes.c_int
        lib.cudssCreate.argtypes = [ctypes.c_void_p]
        lib.cudssDestroy.restype = ctypes.c_int
        lib.cudssDestroy.argtypes = [ctypes.c_void_p]
        h = ctypes.c_void_p()
        st = lib.cudssCreate(ctypes.byref(h))
        if st != 0:
            print(f"[4] FAIL cudssCreate status {st} (nonzero = error; check GPU/driver)", flush=True)
            return
        lib.cudssDestroy(h)
        print("[4] PASS cudssCreate/Destroy OK -- cuDSS is functional in this env.", flush=True)
        print("\nALL PASS: cuDSS is installed and working. Safe to build the CudssBatch backend.\n", flush=True)
    except Exception as e:
        print(f"[4] FAIL: {e!r}", flush=True)


if __name__ == "__main__":
    main()
