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
are shipped as separate shared libraries (so they stay replaceable). Everything the wheel
needs for them and for the runtime libraries the repair step bundles alongside (OpenMP, and
the C++ standard library on Windows) is staged in PREFIX/share/licenses/p3s-cpp: a
NOTICE.txt for this platform, the SuiteSparse license texts and the runtime license texts
from p3s/cpp/licenses. The p3s-cpp CMake build installs that directory into the wheel's
.dist-info/licenses.

``--source-archive DIR`` instead writes SuiteSparse-<version>-KLU-subset.tar.gz to DIR:
exactly the part of the SuiteSparse release that is built here, plus the build script. The
release_assets workflow attaches it to each published GitHub release; NOTICE.txt points
there.

Usage:  python build_klu.py PREFIX [--force]
        python build_klu.py --source-archive DIR

Stdlib only, so it runs on whatever Python the build host provides (including the
manylinux container's python3).
"""

import argparse
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

SUITESPARSE_VERSION = "7.14.1"
SUITESPARSE_SHA256 = "81e560e1f74546df139edb765b3f5bc865866da23062312ffe8fd821063c8397"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_URL = "https://github.com/e2nIEE/parallel-pandapower-solver"

# Top-level entries of the SuiteSparse tree the CMake build needs. The release tarball is
# the whole repository (~100 MB, GraphBLAS etc.), so only these are extracted, and only
# these go into the source archive.
_NEEDED = ("CMakeLists.txt", "LICENSE.txt", "SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU")

# (source inside the tree, file name under PREFIX/share/licenses/p3s-cpp)
_LICENSES = (
    ("KLU/Doc/License.txt", "SuiteSparse/KLU-License.txt"),
    ("KLU/Doc/lesser.txt", "SuiteSparse/LGPL-2.1.txt"),
    ("BTF/Doc/License.txt", "SuiteSparse/BTF-License.txt"),
    ("AMD/Doc/License.txt", "SuiteSparse/AMD-License.txt"),
    ("COLAMD/Doc/License.txt", "SuiteSparse/COLAMD-License.txt"),
    ("SuiteSparse_config/README.txt", "SuiteSparse/SuiteSparse_config-README.txt"),
)

# Runtime libraries the repair step bundles next to SuiteSparse, per build platform:
# (file in p3s/cpp/licenses, file name under PREFIX/share/licenses/p3s-cpp). The MSVC
# runtime DLLs on Windows come with no license text to ship (see _RUNTIME_NOTICE).
_RUNTIME_LICENSES = {
    "linux": (
        ("LICENSE-GPL-3.0.txt", "OpenMP/GPL-3.0.txt"),
        ("LICENSE-GCC-exception-3.1.txt", "OpenMP/GCC-exception-3.1.txt"),
    ),
    "darwin": (("LICENSE-LLVM-OpenMP.txt", "OpenMP/LLVM-OpenMP-LICENSE.txt"),),
    "win32": (),
}

_NOTICE = """\
p3s-cpp binary wheel: bundled third-party libraries
===================================================

p3s-cpp itself is licensed under BSD-3-Clause (see LICENSE). So that it works without
a separate SuiteSparse or OpenMP installation, this wheel bundles the shared libraries
below, dynamically linked to the nr_klu extension. Each stays under its own license,
whose text is in this directory.

SuiteSparse {version}, only these components:

  KLU, BTF             LGPL-2.1-or-later   SuiteSparse/KLU-License.txt, SuiteSparse/BTF-License.txt,
                                           SuiteSparse/LGPL-2.1.txt
  AMD, COLAMD          BSD-3-Clause        SuiteSparse/AMD-License.txt, SuiteSparse/COLAMD-License.txt
  SuiteSparse_config   BSD-3-Clause        SuiteSparse/SuiteSparse_config-README.txt

  Source code: SuiteSparse-{version}-KLU-subset.tar.gz, attached to the release
    {release_url}
  It holds exactly the SuiteSparse files these libraries were built from, plus the
  build script build_klu.py with the CMake configuration used. The same files are
  part of the upstream SuiteSparse release:
    {suitesparse_url}
    (SHA-256 {sha256})

{runtime}
Location in the wheel and replacing a library
---------------------------------------------

{layout}
"""

_RUNTIME_NOTICE = {
    "linux": """\
Runtime library:

  libgomp              GPL-3.0-or-later    OpenMP/GPL-3.0.txt, OpenMP/GCC-exception-3.1.txt
                       WITH GCC-exception-3.1
  The GNU OpenMP runtime of the GCC in the manylinux build image.
  Source code: https://gcc.gnu.org
""",
    "darwin": """\
Runtime library:

  libomp               Apache-2.0          OpenMP/LLVM-OpenMP-LICENSE.txt
                       WITH LLVM-exception
  The LLVM OpenMP runtime, from Homebrew's libomp package.
  Source code: https://github.com/llvm/llvm-project (openmp/)
""",
    "win32": """\
Runtime libraries:

  msvcp140.dll (C++ standard library) and vcomp140.dll (OpenMP runtime) from the
  Microsoft Visual C++ Redistributable. They are proprietary, (c) Microsoft
  Corporation, and redistributed as "Distributable Code" under the Microsoft Visual
  Studio license terms.
""",
}

_LAYOUT_NOTICE = {
    "linux": """\
The libraries are in p3s_cpp.libs/. auditwheel copied them there under new file names
with a hash suffix (e.g. libklu-<hash>.so.2.3.6) and rewrote their SONAMEs, their
references to each other and the RPATH of nr_klu ($ORIGIN/../p3s_cpp.libs) to match.
Their code is unchanged. To use your own build of a library, replace the file and keep
its hashed file name, because that is the name nr_klu and the other bundled libraries
load. If your build depends on other bundled libraries, point those references to the
hashed names as well (e.g. with patchelf --replace-needed).
""",
    "darwin": """\
The libraries are in p3s_cpp.dylibs/ under their original file names. delocate rewrote
their install names and the references of nr_klu to them
(@loader_path/../p3s_cpp.dylibs/...). Their code is unchanged.
To use your own build of a library, replace the file and keep its file name. On Apple
silicon the replacement must be signed; an ad hoc signature (codesign --force --sign -)
is enough.
""",
    "win32": """\
delvewheel copied the DLLs under new file names with a hash suffix (e.g.
klu-<hash>.dll) and patched the import tables of nr_klu and of the bundled DLLs to
match. Their code is unchanged. The wheel contains two identical copies of each DLL:
the ones in p3s/, next to nr_klu.pyd, are the ones that are loaded. The copies in
p3s_cpp.libs/ are not used (delvewheel writes both because p3s is a namespace
package). To use your own build of a library, replace the file in p3s/ and keep its
hashed file name, because that is the name nr_klu and the other bundled DLLs load.
""",
}


def _log(msg: str) -> None:
    print(f"[build_klu] {msg}", flush=True)


def _platform() -> str:
    return "linux" if sys.platform.startswith("linux") else sys.platform


def _suitesparse_url(version: str) -> str:
    return f"https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v{version}.tar.gz"


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


def _fetch(version: str, sha256: str, dest: str) -> None:
    """Download the SuiteSparse release tarball to dest and verify its checksum."""
    url = _suitesparse_url(version)
    _download(url, dest)
    digest = _sha256(dest)
    if digest != sha256:
        raise SystemExit(f"SHA-256 mismatch for {url}: got {digest}, expected {sha256}")


def _needed(tar: tarfile.TarFile, root: str) -> list[tarfile.TarInfo]:
    """Members of the SuiteSparse tarball (rooted at root) that the build needs."""
    prefixes = tuple(f"{root}/{name}" for name in _NEEDED)
    return [m for m in tar.getmembers() if m.name == root or m.name.startswith(prefixes)]


def _extract(archive: str, dest: str) -> str:
    """Extract the needed part of the source tree; return the source root."""
    root = f"SuiteSparse-{SUITESPARSE_VERSION}"
    with tarfile.open(archive, "r:gz") as tar:
        members = _needed(tar, root)
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 (and security backports)
            tar.extractall(dest, members=members, filter="data")
        else:
            tar.extractall(dest, members=members)
    return os.path.join(dest, root)


def _release_url() -> str:
    """GitHub release page of this p3s-cpp version (its tag is v<version>)."""
    try:
        with open(os.path.join(_HERE, os.pardir, "pyproject.toml"), encoding="utf-8") as fh:
            match = re.search(r'^version = "([^"]+)"', fh.read(), re.MULTILINE)
    except OSError:
        match = None
    return f"{_REPO_URL}/releases/tag/v{match.group(1)}" if match else f"{_REPO_URL}/releases"


def _stage_licenses(src: str, lic_dir: str) -> None:
    """Copy all license material the wheel needs into lic_dir and write its NOTICE.txt."""
    platform = _platform()
    files = [(os.path.join(src, *rel.split("/")), name) for rel, name in _LICENSES]
    files += [
        (os.path.join(_HERE, os.pardir, "licenses", rel), name) for rel, name in _RUNTIME_LICENSES.get(platform, ())
    ]
    for path, name in files:
        dest = os.path.join(lic_dir, *name.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(path, dest)
    notice = _NOTICE.format(
        version=SUITESPARSE_VERSION,
        release_url=_release_url(),
        suitesparse_url=_suitesparse_url(SUITESPARSE_VERSION),
        sha256=SUITESPARSE_SHA256,
        runtime=_RUNTIME_NOTICE.get(platform, ""),
        layout=_LAYOUT_NOTICE.get(platform, "See the documentation of the wheel repair tool used.\n"),
    )
    with open(os.path.join(lic_dir, "NOTICE.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(notice)


def _run(cmd: list[str]) -> None:
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build(prefix: str, force: bool = False) -> None:
    prefix = os.path.abspath(prefix)
    marker = os.path.join(prefix, "lib", "cmake", "KLU", "KLUConfig.cmake")
    lic_dir = os.path.join(prefix, "share", "licenses", "p3s-cpp")
    notice = os.path.join(lic_dir, "NOTICE.txt")
    if os.path.exists(marker) and os.path.exists(notice) and not force:
        _log(f"KLU already installed in {prefix} (pass --force to rebuild)")
        return

    work = tempfile.mkdtemp(prefix="suitesparse-klu-")
    try:
        archive = os.path.join(work, "suitesparse.tar.gz")
        _fetch(SUITESPARSE_VERSION, SUITESPARSE_SHA256, archive)
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

        _stage_licenses(src, lic_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not os.path.exists(marker):
        raise SystemExit(f"KLU install incomplete: {marker} is missing")
    _log(f"SuiteSparse {SUITESPARSE_VERSION} (KLU subset) installed to {prefix}")


_ARCHIVE_README = """\
Source code of the SuiteSparse {version} libraries bundled in the p3s-cpp binary wheels
({repo}).

SuiteSparse-{version}/ is the part of the upstream release
  {url}
  (SHA-256 {sha256})
that the wheels are built from, unchanged: the top-level CMakeLists.txt and LICENSE.txt
and the SuiteSparse_config, AMD, COLAMD, BTF and KLU directories. KLU and BTF are
LGPL-2.1-or-later, AMD, COLAMD and SuiteSparse_config BSD-3-Clause; see the License.txt
files in their Doc/ directories.

build_klu.py is the script that builds them for the wheels. The CMake configure,
build and install commands it runs (see build()) work on this tree as is, with
-S SuiteSparse-{version}.
"""


def write_source_archive(out_dir: str) -> str:
    """Write SuiteSparse-<version>-KLU-subset.tar.gz to out_dir; return its path.

    The archive is reproducible: member order, times and owners are taken from the
    upstream tarball, and the gzip header carries no timestamp.
    """
    upstream_root = f"SuiteSparse-{SUITESPARSE_VERSION}"
    name = f"{upstream_root}-KLU-subset"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.tar.gz")

    work = tempfile.mkdtemp(prefix="suitesparse-src-")
    try:
        archive = os.path.join(work, "suitesparse.tar.gz")
        _fetch(SUITESPARSE_VERSION, SUITESPARSE_SHA256, archive)
        readme = _ARCHIVE_README.format(
            version=SUITESPARSE_VERSION,
            repo=_REPO_URL,
            url=_suitesparse_url(SUITESPARSE_VERSION),
            sha256=SUITESPARSE_SHA256,
        )
        with open(__file__, "rb") as fh:
            script = fh.read()

        with tarfile.open(archive, "r:gz") as src, open(out, "wb") as raw:
            gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
            dst = tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT)
            members = _needed(src, upstream_root)
            mtime = members[0].mtime  # the upstream root directory

            top = tarfile.TarInfo(name)
            top.type, top.mode, top.mtime = tarfile.DIRTYPE, 0o755, mtime
            dst.addfile(top)
            for path, data in (("README.txt", readme.encode()), ("build_klu.py", script)):
                info = tarfile.TarInfo(f"{name}/{path}")
                info.size, info.mode, info.mtime = len(data), 0o644, mtime
                dst.addfile(info, io.BytesIO(data))
            for member in members:
                fileobj = src.extractfile(member) if member.isfile() else None
                member.name = f"{name}/{member.name}"
                dst.addfile(member, fileobj)
            dst.close()
            gz.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    _log(f"wrote {out} (SHA-256 {_sha256(out)})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix", nargs="?", help="install prefix for the KLU libraries, headers and CMake configs")
    parser.add_argument("--force", action="store_true", help="rebuild even if KLU is already installed")
    parser.add_argument(
        "--source-archive", metavar="DIR", help="write the KLU-subset source archive to DIR instead of building"
    )
    args = parser.parse_args()
    if args.source_archive:
        write_source_archive(args.source_archive)
    elif args.prefix:
        build(args.prefix, force=args.force)
    else:
        parser.error("PREFIX or --source-archive is required")


if __name__ == "__main__":
    main()
