# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guards around the batched C++ solver that differ between platforms / compilers.

* Non-finite inputs must be detected. The extension is built with -ffast-math, under which
  GCC folds ``std::isfinite`` to ``true``; the checks therefore use a bit-level test
  (``p3s/cpp/finite.hpp``). These tests fail on a Linux build that regresses to
  ``std::isfinite``.
* The default worker count must come from OpenMP (OMP_NUM_THREADS / the process CPU mask),
  not from ``std::thread::hardware_concurrency()``, which ignores the CPU mask on Linux and
  oversubscribed SLURM allocations.

The thread-count test is Linux-only and runs in a subprocess, because OpenMP reads its
environment once per process and its thread pool outlives each call.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pandapower.networks import case14

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp
from p3s.timeseries import dc_initial_voltage

try:
    from p3s import nr_klu  # type: ignore[attr-defined]
except ImportError:
    from p3s.cpp import nr_klu  # type: ignore[attr-defined]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IS_LINUX = sys.platform.startswith("linux")


def _net():
    net = case14()
    net.trafo.shift_degree = 0.0
    calculate_trafo_characteristic(net, inplace=True)
    return net


def _solver_and_inputs(net, T=8):
    npf = NewtonPowerflowCpp(net)
    Y = npf._YBus.tocsr()
    s = nr_klu.Solver(
        np.ascontiguousarray(Y.indptr, dtype=np.int32),
        np.ascontiguousarray(Y.indices, dtype=np.int32),
        np.ascontiguousarray(Y.data, dtype=np.complex128),
        np.ascontiguousarray(npf.busses["pv"], dtype=np.int32),
        np.ascontiguousarray(npf.busses["pq"], dtype=np.int32),
    )
    Sbus = np.repeat(npf._sBus[:, None], T, axis=1)
    return s, Y, Sbus, dc_initial_voltage(npf), np.asarray(npf.busses["pq"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_non_finite_start_voltage_is_rejected(bad, n_threads):
    s, _, Sbus, V0, _ = _solver_and_inputs(_net())
    V0 = V0.copy()
    V0[3] = complex(bad, 0.0)
    with pytest.raises(RuntimeError, match="invalid initial voltage"):
        s.solve_batch(Sbus, V0, n_threads=n_threads)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_nan_admittance_is_never_reported_converged(n_threads):
    """A NaN in Ybus makes every Jacobian pivot NaN. The lean refactorization must reject it
    and the column must end non-converged -- not crash, hang, or claim convergence."""
    s, Y, Sbus, V0, pq = _solver_and_inputs(_net())
    Yx = np.ascontiguousarray(Y.data, dtype=np.complex128).copy()
    # Poison the diagonal of a PQ bus: it enters both its P and Q rows of the Jacobian.
    # (An entry of the slack row would be harmless -- that row is not part of the system.)
    row = int(pq[0])
    k = Y.indptr[row] + int(np.flatnonzero(Y.indices[Y.indptr[row] : Y.indptr[row + 1]] == row)[0])
    Yx[k] = complex(np.nan, np.nan)
    s.update_Y(Yx)
    r = s.solve_batch(Sbus, V0, max_iter=10, n_threads=n_threads)
    assert not r["converged"].any()


def _run_child(code: str, env_extra: dict[str, str], cpus: list[int] | None = None) -> str:
    """Run `code` after a common prelude in a fresh interpreter. `cpus` restricts the child's
    CPU mask before it starts -- as SLURM / taskset do -- so OpenMP initialises inside it."""
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"):  # OpenMP defaults only
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    prelude = textwrap.dedent(
        """
        import os, numpy as np
        from pandapower.networks import case14
        from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
        from p3s.NewtonPowerflowCpp import NewtonPowerflow
        from p3s.timeseries import dc_initial_voltage
        try:
            from p3s import nr_klu
        except ImportError:
            from p3s.cpp import nr_klu
        net = case14()
        net.trafo.shift_degree = 0.0
        calculate_trafo_characteristic(net, inplace=True)
        npf = NewtonPowerflow(net)
        Y = npf._YBus.tocsr()
        s = nr_klu.Solver(Y.indptr.astype(np.int32), Y.indices.astype(np.int32),
                          Y.data.astype(np.complex128), np.asarray(npf.busses["pv"], np.int32),
                          np.asarray(npf.busses["pq"], np.int32))
        Sbus = np.repeat(npf._sBus[:, None], 256, axis=1)
        V0 = dc_initial_voltage(npf)
        def n_os_threads():
            return len(os.listdir("/proc/self/task"))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        env=env,
        cwd=REPO_ROOT,
        preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.skipif(not IS_LINUX, reason="CPU masks / hardware_concurrency mismatch are Linux-specific")
@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 4 if IS_LINUX else True, reason="needs >= 4 CPUs")
def test_default_thread_count_respects_cpu_mask():
    """n_threads=0 inside a 2-CPU mask must start at most 2 workers. The old default
    (hardware_concurrency) started one per CPU of the whole machine."""
    out = _run_child(
        """
        before = n_os_threads()
        r = s.solve_batch(Sbus, V0, n_threads=0)
        assert r["converged"].all()
        print(n_os_threads() - before)  # libgomp keeps its pool threads alive after the region
        """,
        {},
        cpus=sorted(os.sched_getaffinity(0))[:2],
    )
    assert int(out) <= 1, f"default batch started {int(out) + 1} workers inside a 2-CPU mask"
