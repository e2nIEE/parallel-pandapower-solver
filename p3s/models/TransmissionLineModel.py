# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numba
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from p3s.models.TwoPort import TwoPort


@numba.njit
def pi_model(R, X, C, G, length, vn_kv, parallel, f_hz=50.0, sn_mva=1.0):
    """
    Compute the per‐phase π‐model of a line.

    Inputs
    ------
    R      : float or array, series resistance [Ω/m]
    X      : float or array, series reactance of the line [Ω/m]
    C      : float or array, capacitance per unit length [nF/m]
    tap    : float or array, line length [m]
    phi    : float, radian frequency = 2*pi*freq

    Returns
    -------
    Zs      : complex array, series impedance per phase [Ω]
    Ysh_half: complex array, half the shunt admittance per phase [S]
    """
    # Calculate scaling value, scale everything in p.u.
    Z_N = (vn_kv**2) / sn_mva

    # 1) Calculate B
    B = 2 * C * f_hz * np.pi * 1e-9 * Z_N

    # 2) Calculate Z Series impedance, everything needs to be scaled to p.u.
    x_s = 1j * X * length / Z_N
    Z_sr = 1 / (R * length / Z_N + x_s) / parallel

    # 3) Total shunt admittance, split equally at both ends
    Y_sr = 0.5 * (G * 1e-6 + 1j * B) * length * parallel

    # 4) Calculate admittances
    Y_ff = Z_sr + Y_sr
    Y_tt = Z_sr + Y_sr
    Y_ft = -Z_sr
    Y_tf = -Z_sr

    return Y_ff, Y_ft, Y_tf, Y_tt, 1 / x_s


@numba.njit
def t_model(R, X, C, G, length, vn_kv, parallel, f_hz=50.0, sn_mva=1.0):
    """
    Compute the per‐phase T‐model of a line.

    Inputs
    ------
    R      : float or array, series resistance [Ω/m]
    X      : float or array, series reactance of the line [Ω/m]
    B      : float or array, capacitance per unit length [F/m]
    tap    : float or array, line length [m]
    phi    : float, radian frequency = 2*pi*freq

    Returns
    -------
    Zs      : complex array, series impedance per phase [Ω]
    Ysh_half: complex array, half the shunt admittance per phase [S]
    """
    # Calculate scaling value, scale everything in p.u.
    Z_N = (vn_kv**2) / sn_mva

    # 1) Calculate B
    B = 2 * C * f_hz * np.pi * 1e-9 * Z_N

    # 2) Calculate Z Series impedance, everything needs to be scaled to p.u.
    x_s = 1j * X * length / Z_N
    Z_sr = 1 / (R * length / Z_N + x_s) / parallel

    # 3) Total shunt admittance, split equally at both ends
    Y_sr = 0.5 * (G * 1e-6 + 1j * B) * length * parallel

    # T-Modell-Zweige
    Z1 = Z2 = Z_sr / 2
    Y0 = 2 * Y_sr

    # Vorberechnen
    a1 = 1.0 / Z1
    a2 = 1.0 / Z2
    D = Y0 + a1 + a2

    # Calculate admittances
    Y_ff = a1 - a1 * a1 / D
    Y_tt = a2 - a2 * a2 / D
    Y_ft = -(a1 * a2) / D
    Y_tf = Y_ft

    return Y_ff, Y_ft, Y_tf, Y_tt, 1 / x_s


class TransmissionLineModel(TwoPort):
    """
    This is a class to represent a branch model based on the pi approach.
    Main idea is:
             +------------------+
    -----+---| y = 1 / (R + jX) |---+-----
         |   +------------------+   |
         |                          |
    +----+----+                +----+----+
    | G+j*B/2 |                | G+j*B/2 |
    +----+----+                +----+----+
         |                          |
        ---                        ---

    """

    # @numba.njit
    def __init__(self, line_table: DataFrame, voltages: NDArray, f_hz: float = 50.0, sn_mva: float = 1.0):
        super().__init__()
        self._from_bus = line_table["from_bus"].values
        self._to_bus = line_table["to_bus"].values

        # Input Values
        length = line_table["length_km"].values
        R = line_table["r_ohm_per_km"].values
        C = line_table["c_nf_per_km"].values
        X = line_table["x_ohm_per_km"].values
        G = line_table["g_us_per_km"].values
        parallel = line_table["parallel"].values
        # self.voltages = line_table["vn_kv"].values
        self.voltages = voltages

        # return four matrices
        self._Y_ff, self._Y_ft, self._Y_tf, self._Y_tt, one_over_x = pi_model(
            R=R,
            X=X,
            C=C,
            G=G,
            f_hz=f_hz,
            length=length,
            vn_kv=self.voltages,
            sn_mva=sn_mva,
            parallel=parallel,
        )

        # for DC powerflow
        self._DC_Yff = self._DC_Ytt = -one_over_x
        self._DC_Yft = self._DC_Ytf = one_over_x

        # Zero out de-energised lines. An out-of-service line carries no current and must
        # contribute nothing to Ybus -- otherwise open tie switches (e.g. the 5 open ties
        # in case33bw) stay electrically closed and every solver silently converges on the
        # wrong network. Masking the stamps here (rather than dropping the rows) keeps the
        # arrays row-aligned with net.line, which yf_matrix/yt_matrix rely on for res_line.
        self._apply_in_service(line_table)
