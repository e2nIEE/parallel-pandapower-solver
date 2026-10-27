# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the KLU subset of SuiteSparse as shared libraries for the p3s-cpp wheels.

cibuildwheel runs this once per build machine (``before-all``, see the
``[tool.cibuildwheel]`` table in p3s/cpp/pyproject.toml) so that ``nr_klu`` can link
against KLU. The wheel repair step (auditwheel / delocate / delvewheel) then copies the
shared libraries into the wheel, so users need neither a compiler nor SuiteSparse.

Only KLU and its dependencies (BTF, AMD, COLAMD, SuiteSparse_config) are built: no BLAS,
CHOLMOD, OpenMP, CUDA or Fortran, so CMake and a C compiler are enough on every platform.

KLU and BTF are LGPL-2.1-or-later, AMD, COLAMD and SuiteSparse_config BSD-3-Clause. They
are shipped as separate, unmodified shared libraries (so they stay replaceable), and their
license texts are copied to PREFIX/share/licenses/SuiteSparse, from where the p3s-cpp
CMake build installs them into the wheel's .dist-info.

Usage:  python build_klu.py PREFIX [--force]

Stdlib only, so it runs on whatever Python the build host provides (including the
manylinux container's python3).
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

SUITESPARSE_VERSION = "7.14.1"
SUITESPARSE_SHA256 = "81e560e1f74546df139edb765b3f5bc865866da23062312ffe8fd821063c8397"
SUITESPARSE_URL = f"https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v{SUITESPARSE_VERSION}.tar.gz"

# Top-level entries of the SuiteSparse tree the CMake build needs. The release tarball is
# the whole repository (~100 MB, GraphBLAS etc.), so only these are extracted.
_NEEDED = ("CMakeLists.txt", "LICENSE.txt", "SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU")

# (source inside the tree, file name under PREFIX/share/licenses/SuiteSparse)
_LICENSES = (
    ("KLU/Doc/License.txt", "KLU-License.txt"),
    ("KLU/Doc/lesser.txt", "LGPL-2.1.txt"),
    ("BTF/Doc/License.txt", "BTF-License.txt"),
    ("AMD/Doc/License.txt", "AMD-License.txt"),
    ("COLAMD/Doc/License.txt", "COLAMD-License.txt"),
    ("SuiteSparse_config/README.txt", "SuiteSparse_config-README.txt"),
)

_NOTICE = f"""\
The p3s-cpp binary wheels bundle the following components of SuiteSparse
{SUITESPARSE_VERSION} as separate, unmodified shared libraries:

  KLU, BTF                      LGPL-2.1-or-later   (KLU-License.txt, BTF-License.txt, LGPL-2.1.txt)
  AMD, COLAMD                   BSD-3-Clause        (AMD-License.txt, COLAMD-License.txt)
  SuiteSparse_config            BSD-3-Clause        (SuiteSparse_config-README.txt)

Source code: https://github.com/DrTimothyAldenDavis/SuiteSparse/releases/tag/v{SUITESPARSE_VERSION}
Build script: p3s/cpp/ci/build_klu.py in https://github.com/e2nIEE/parallel-pandapower-solver

You may replace the bundled libraries with your own build of the same SuiteSparse
components (same major library versions).
"""


def _log(msg: str) -> None:
    print(f"[build_klu] {msg}", flush=True)


def _download(url: str, dest: str, attempts: int = 4) -> None:
    for i in range(1, attempts + 1):
        try:
            _log(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return
        except OSError as exc:
            if i == attempts:
                raise
            _log(f"download failed ({exc}), retrying")
            time.sleep(5 * i)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive: str, dest: str) -> str:
    """Extract the needed part of the source tree; return the source root."""
    root = f"SuiteSparse-{SUITESPARSE_VERSION}"
    prefixes = tuple(f"{root}/{name}" for name in _NEEDED)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name == root or m.name.startswith(prefixes)]
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 (and security backports)
            tar.extractall(dest, members=members, filter="data")
        else:
            tar.extractall(dest, members=members)
    return os.path.join(dest, root)


def _run(cmd: list[str]) -> None:
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build(prefix: str, force: bool = False) -> None:
    prefix = os.path.abspath(prefix)
    marker = os.path.join(prefix, "lib", "cmake", "KLU", "KLUConfig.cmake")
    if os.path.exists(marker) and not force:
        _log(f"KLU already installed in {prefix} (pass --force to rebuild)")
        return

    work = tempfile.mkdtemp(prefix="suitesparse-klu-")
    try:
        archive = os.path.join(work, "suitesparse.tar.gz")
        _download(SUITESPARSE_URL, archive)
        digest = _sha256(archive)
        if digest != SUITESPARSE_SHA256:
            raise SystemExit(f"SHA-256 mismatch for {SUITESPARSE_URL}: got {digest}, expected {SUITESPARSE_SHA256}")
        src = _extract(archive, work)

        build_dir = os.path.join(work, "build")
        configure = [
            "cmake",
            "-S",
            src,
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            # Keep lib/ (GNUInstallDirs would pick lib64 on the RHEL-based manylinux images).
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DSUITESPARSE_ENABLE_PROJECTS=suitesparse_config;amd;colamd;btf;klu",
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_STATIC_LIBS=OFF",
            "-DBUILD_TESTING=OFF",
            "-DSUITESPARSE_DEMOS=OFF",
            "-DKLU_USE_CHOLMOD=OFF",
            "-DSUITESPARSE_REQUIRE_BLAS=OFF",
            # With the BLAS off, SuiteSparse_config still evaluates ${BLA_VENDOR} unquoted
            # and fails when it is empty. No BLAS is searched for or linked either way.
            "-DBLA_VENDOR=Generic",
            "-DSUITESPARSE_USE_OPENMP=OFF",
            "-DSUITESPARSE_USE_CUDA=OFF",
            "-DSUITESPARSE_USE_FORTRAN=OFF",
        ]
        if sys.platform == "darwin":
            # Absolute install names, so delocate can resolve the libraries without
            # DYLD_LIBRARY_PATH. MACOSX_DEPLOYMENT_TARGET is picked up from the environment.
            configure.append(f"-DCMAKE_INSTALL_NAME_DIR={prefix}/lib")
        _run(configure)
        _run(["cmake", "--build", build_dir, "--config", "Release", "--parallel"])
        _run(["cmake", "--install", build_dir, "--config", "Release"])

        lic_dir = os.path.join(prefix, "share", "licenses", "SuiteSparse")
        os.makedirs(lic_dir, exist_ok=True)
        for rel, name in _LICENSES:
            shutil.copyfile(os.path.join(src, *rel.split("/")), os.path.join(lic_dir, name))
        with open(os.path.join(lic_dir, "NOTICE.txt"), "w", encoding="utf-8") as fh:
            fh.write(_NOTICE)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not os.path.exists(marker):
        raise SystemExit(f"KLU install incomplete: {marker} is missing")
    _log(f"SuiteSparse {SUITESPARSE_VERSION} (KLU subset) installed to {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix", help="install prefix for the KLU libraries, headers and CMake configs")
    parser.add_argument("--force", action="store_true", help="rebuild even if KLU is already installed")
    args = parser.parse_args()
    build(args.prefix, force=args.force)


if __name__ == "__main__":
    main()
