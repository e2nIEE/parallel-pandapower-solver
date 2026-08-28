# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""N-1 contingency analysis for p3s's batched power flow.

See ``docs/n_minus_1_plan.md``. A contingency is one named ``outage_group`` (all
lines/trafos in the group taken out together); branches with no group stay in
service. Each contingency is a values-only change to Ybus on a shared sparsity
pattern, so the batch engines (CPU ``nr_klu``, GPU ``cuSolverRf``) keep their
amortized symbolic factorization and the time-series speedup.

This package is built in phases (see the plan). Phase 0 ships fixtures and a
pandapower ground-truth oracle; later phases add the backend-agnostic case
generator and the CPU/GPU solve paths.
"""
