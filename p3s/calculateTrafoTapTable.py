# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandapower.auxiliary import pandapowerNet

INDEX = ["id_characteristic", "step"]
COLUMNS = [
    "voltage_ratio",
    "angle_deg",
    "vk_percent",
    "vkr_percent",
    "vk_hv_percent",
    "vkr_hv_percent",
    "vk_mv_percent",
    "vkr_mv_percent",
    "vk_lv_percent",
    "vkr_lv_percent",
]
# short-circuit voltages copied from the element into each of its characteristic rows
VK_COLUMNS = {
    "trafo": ["vk_percent", "vkr_percent"],
    "trafo3w": [
        "vk_hv_percent",
        "vkr_hv_percent",
        "vk_mv_percent",
        "vkr_mv_percent",
        "vk_lv_percent",
        "vkr_lv_percent",
    ],
}


def _column(table: pd.DataFrame, name: str, default) -> NDArray:
    if name not in table:
        return np.full(len(table), default)
    if default is None:
        return table[name].to_numpy(dtype=object)
    return table[name].fillna(default).to_numpy(dtype=float)


def _uses_table(table: pd.DataFrame) -> NDArray:
    """Elements that already take their tap data from the characteristic table."""
    if "tap_dependency_table" not in table or "id_characteristic_table" not in table:
        return np.zeros(len(table), dtype=bool)
    tap_dependency_table = table["tap_dependency_table"].fillna(False).to_numpy(dtype=bool)
    return tap_dependency_table & table["id_characteristic_table"].notna().to_numpy()


def _steps(table: pd.DataFrame) -> tuple[NDArray, NDArray]:
    """(element position, step) of every row to generate: each step from tap_min to tap_max, so a
    tap change needs no new table, plus the current tap position (NaN -> 0, as the models read it)."""
    tap_pos = _column(table, "tap_pos", 0.0)
    tap_min = _column(table, "tap_min", np.nan)
    tap_max = _column(table, "tap_max", np.nan)
    positions, steps = [], []
    for i, (pos, lo, hi) in enumerate(zip(tap_pos, tap_min, tap_max, strict=True)):
        element_steps = {int(pos)}
        if np.isfinite(lo) and np.isfinite(hi):
            element_steps.update(range(int(lo), int(hi) + 1))
        positions.extend([i] * len(element_steps))
        steps.extend(sorted(element_steps))
    return np.asarray(positions, dtype=int), np.asarray(steps, dtype=int)


def _characteristic_rows(element_table: pd.DataFrame, ids: NDArray, vk_columns: list[str]) -> pd.DataFrame:
    """The characteristic rows of the given elements (one per step, see _steps) under the given ids."""
    position, step = _steps(element_table)
    tap_diff = step - _column(element_table, "tap_neutral", 0.0)[position]
    s = tap_diff * _column(element_table, "tap_step_percent", 0.0)[position] / 100.0
    step_degree = _column(element_table, "tap_step_degree", 0.0)[position]
    tap_changer_type = _column(element_table, "tap_changer_type", None)[position]
    has_side = np.isin(_column(element_table, "tap_side", None)[position], ("hv", "mv", "lv"))

    voltage_ratio = np.ones(len(step))
    angle_deg = np.zeros(len(step))

    def _complex(idx):
        # "Ratio" / "Symmetrical": pandapower's complex tap changer (_get_trafo_shift, ideal=False)
        phi = np.deg2rad(step_degree[idx])
        re = 1.0 + s[idx] * np.cos(phi)
        im = s[idx] * np.sin(phi)
        voltage_ratio[idx] = np.hypot(re, im)
        angle_deg[idx] = np.rad2deg(np.arctan(im / re))

    def _ideal(idx):
        # "Ideal": pure phase shifter (_get_trafo_shift, ideal=True)
        by_degree = tap_diff[idx] * step_degree[idx]
        by_percent = 2.0 * np.rad2deg(np.arcsin(np.clip(s[idx] / 2.0, -1.0, 1.0)))
        angle_deg[idx] = np.where(step_degree[idx] != 0.0, by_degree, by_percent)

    dispatch = {
        "Ratio": _complex,
        "Symmetrical": _complex,
        "Ideal": _ideal,
    }
    for changer_type, func in dispatch.items():
        mask = has_side & (tap_changer_type == changer_type)
        if np.any(mask):
            func(mask)

    rows = pd.DataFrame(
        {
            "id_characteristic": ids[position],
            "step": step,
            "voltage_ratio": voltage_ratio,
            "angle_deg": angle_deg,
        }
    )
    for col in vk_columns:
        rows[col] = element_table[col].to_numpy(dtype=float)[position]
    return rows.set_index(INDEX)


def calculate_trafo_characteristic(net: pandapowerNet, inplace: bool = False):
    """Converts the tap changers of all transformers (trafo and trafo3w) into the transformer
    characteristic table (net.trafo_characteristic_table, MultiIndex id_characteristic / step).

    Transformers that already use the table (tap_dependency_table True and an
    id_characteristic_table set, e.g. a "Tabular" tap changer from CGMES) keep their id and their
    rows unchanged. Every other transformer gets a new id and one row per step from tap_min to
    tap_max (plus its current tap_pos), with the voltage ratio and angle pandapower computes for
    its tap changer type (build_branch._calc_tap_from_dataframe / _get_trafo_shift):

    * "Ratio", "Symmetrical": complex tap 1 + s*exp(j*phi) with s = (step - tap_neutral) *
      tap_step_percent / 100 and phi = tap_step_degree;
      voltage_ratio = |1 + s*exp(j*phi)|, angle_deg = arctan(s*sin(phi) / (1 + s*cos(phi)))
    * "Ideal": voltage_ratio = 1, angle_deg = (step - tap_neutral) * tap_step_degree, or
      2*arcsin(s/2) when only tap_step_percent is set
    * no tap_side, "Tabular" without table, no tap changer: voltage_ratio = 1, angle_deg = 0

    voltage_ratio and angle_deg refer to the winding on tap_side, as in pandapower's table. Rows
    generated by an earlier call are replaced, so the function can be called again after changing
    the tap data.

    Args:
        net: The pandapowerNet object containing the transformer and associated network data.
        inplace: Boolean flag indicating whether results should be updated directly in the
            pandapowerNet object (True) or returned as a DataFrame (False).
            net.<trafo|trafo3w>.id_characteristic_table is written in both cases.

    Returns:
        DataFrame with transformer characteristic parameters for all transformers in the
        network. Returns None if `inplace` is True.
    """
    existing = net.get("trafo_characteristic_table")
    if not isinstance(existing, pd.DataFrame) or existing.empty:
        existing = pd.DataFrame(columns=INDEX + COLUMNS).set_index(INDEX)
    elif existing.index.names != INDEX:
        existing = existing.set_index(INDEX)

    elements = [et for et in ("trafo", "trafo3w") if et in net and len(net[et])]

    # Ids still in use through the table are kept; ids this function generated earlier for the
    # other elements are dropped and regenerated.
    table_ids, generated_ids = set(), set()
    for et in elements:
        if "id_characteristic_table" not in net[et]:
            continue
        uses_table = _uses_table(net[et])
        ids = net[et]["id_characteristic_table"]
        table_ids |= set(ids[uses_table].astype(int))
        generated_ids |= set(ids[~uses_table].dropna().astype(int))
    stale = generated_ids - table_ids
    existing_ids = existing.index.get_level_values("id_characteristic")
    kept = existing[~existing_ids.isin(list(stale))]

    # every element that uses the table needs a row for its current tap position
    for et in elements:
        table = net[et][_uses_table(net[et])]
        if not len(table):
            continue
        wanted = pd.MultiIndex.from_arrays(
            [table["id_characteristic_table"].astype(int), _column(table, "tap_pos", 0.0).astype(int)]
        )
        missing = wanted[kept.index.get_indexer(wanted) < 0]
        if len(missing):
            raise KeyError(f"trafo_characteristic_table has no row for {et} (id_characteristic, step) {list(missing)}")

    next_id = int(kept.index.get_level_values("id_characteristic").max()) + 1 if len(kept) else 0
    generated = [kept]
    for et in elements:
        element_table = net[et]
        convert = ~_uses_table(element_table)
        new_ids = np.arange(next_id, next_id + int(convert.sum()))
        next_id += len(new_ids)

        ids = pd.Series(pd.NA, index=element_table.index, dtype="Int64")
        if "id_characteristic_table" in element_table:
            ids[:] = element_table["id_characteristic_table"].astype("Int64")
        ids[convert] = new_ids
        element_table["id_characteristic_table"] = ids

        generated.append(_characteristic_rows(element_table[convert], new_ids, VK_COLUMNS[et]))

    df = pd.concat(generated).reindex(columns=COLUMNS + [c for c in kept.columns if c not in COLUMNS]).sort_index()

    if inplace:
        net["trafo_characteristic_table"] = df
        return None
    else:
        return df
