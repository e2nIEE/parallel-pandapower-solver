# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from pandas import DataFrame

from p3s.models.TwoPort import TwoPort


def ward_model(pz_mw, qz_mvar, ps_mw, qs_mvar, in_service, sn_mva=1.0):
    """
    Compute the two halves of a pandapower ``ward`` equivalent.

    A ward is a network equivalent made of TWO electrically distinct parts that happen
    to share a table row:

    * ``pz_mw`` / ``qz_mvar`` -- a constant IMPEDANCE, i.e. a shunt admittance whose
      power draw scales with vm**2. It belongs on the Ybus diagonal.
    * ``ps_mw`` / ``qs_mvar`` -- a constant POWER demand, independent of voltage. It
      belongs in the bus injection vector Sbus, exactly like a load.

    pandapower keeps this split too: ``_calc_shunts_and_add_on_ppc`` writes pz/qz onto
    the bus GS/BS columns while ``_get_pq_results`` adds ps/qs to the PQ injections.

    Inputs
    ------
    pz_mw, qz_mvar : float or array, constant-impedance demand at 1.0 pu [MW / MVAr]
    ps_mw, qs_mvar : float or array, constant-power demand [MW / MVAr]
    in_service     : bool array, service status
    sn_mva         : float, network reference power [MVA]

    Returns
    -------
    y_shunt : complex array, shunt admittance for the Ybus diagonal [p.u.]
    s_ward  : complex array, constant-power demand [MVA], zero where out of service
    """
    # 1) Constant-impedance part -> shunt admittance.
    #    pandapower stores GS = pz_mw and BS = -qz_mvar, and makeYbus then forms
    #    Ysh = (GS + 1j*BS)/baseMVA. The reactive sign therefore FLIPS: a positive
    #    qz_mvar (inductive demand) becomes a negative susceptance. This is the same
    #    convention ShuntModel already uses for net.shunt.
    y_shunt = (pz_mw - 1j * qz_mvar) / sn_mva

    # 2) Constant-power part -> bus injection, in MVA (scaled to p.u. by the caller,
    #    which divides the whole Sbus by sn_mva). A ward has NO scaling column, unlike
    #    load/sgen -- pandapower hardcodes scaling = 1.0 for ward and xward.
    s_ward = ps_mw + 1j * qs_mvar

    # 3) An out-of-service ward is electrically absent in BOTH halves. The shunt stamp is
    #    masked by TwoPort._apply_in_service, but the power half never reaches Ybus, so it
    #    has to be zeroed here.
    zeros = np.zeros_like(s_ward)
    s_ward = np.where(in_service, s_ward, zeros)

    return y_shunt, s_ward


class WardModel(TwoPort):
    """
    This is a class to represent a pandapower ``ward`` equivalent: a network equivalent
    that draws both a constant power and a constant-impedance power at a single bus.

    Main idea is:

         bus
          |
          +-------------------+
          |                   |
    +-----+------+     +------+------+
    | ps + j*qs  |     | (pz - j*qz) |   <- shunt admittance, draws vm**2 * (pz + j*qz)
    | const. S   |     |   / sn_mva  |
    +-----+------+     +------+------+
          |                   |
         ---                 ---

    Unlike every other model in this package a ward is an ACTIVE element: only its
    constant-impedance half can be stamped into Ybus. The constant-power half is exposed
    as :attr:`s_bus`, a per-bus injection vector the solver adds to Sbus -- mirroring how
    :class:`TwoWindingTransformerModel` exposes ``p_shift`` for the DC phase-shift
    injection.

    Both terminals of the TwoPort stamp are the SAME bus (as in ShuntModel), so the
    element contributes only to the Ybus diagonal.
    """

    def __init__(self, ward_table: DataFrame, n_bus: int, sn_mva: float = 1.0):
        super().__init__()
        # A ward hangs off a single bus: from == to, so the stamp lands on the diagonal.
        self._from_bus = ward_table["bus"].values
        self._to_bus = ward_table["bus"].values

        # Input Values
        pz_mw = ward_table["pz_mw"].values
        qz_mvar = ward_table["qz_mvar"].values
        ps_mw = ward_table["ps_mw"].values
        qs_mvar = ward_table["qs_mvar"].values
        in_service = np.asarray(ward_table["in_service"].values, dtype=bool)

        y_shunt, s_ward = ward_model(
            pz_mw=pz_mw,
            qz_mvar=qz_mvar,
            ps_mw=ps_mw,
            qs_mvar=qs_mvar,
            in_service=in_service,
            sn_mva=sn_mva,
        )

        # Only the from-diagonal carries the shunt; the other three stamps stay zero so
        # create_y_matrix() adds nothing off-diagonal (same layout as ShuntModel).
        self._Y_ff = y_shunt
        self._Y_ft = np.zeros_like(y_shunt)
        self._Y_tf = np.zeros_like(y_shunt)
        self._Y_tt = np.zeros_like(y_shunt)

        # A ward shunt is a pure admittance with no series reactance, so it contributes
        # nothing to the DC B-matrix (pandapower's makeBdc builds B from branch 1/x only;
        # bus shunts are not part of it). Kept explicit so create_y_dc_matrix() stays safe
        # to call on this element.
        self._DC_Yff = np.zeros_like(y_shunt)
        self._DC_Yft = np.zeros_like(y_shunt)
        self._DC_Ytf = np.zeros_like(y_shunt)
        self._DC_Ytt = np.zeros_like(y_shunt)

        # Zero the shunt stamps of de-energised wards (see TwoPort._apply_in_service).
        # The constant-power half was already masked inside ward_model().
        self._apply_in_service(ward_table)

        # Constant-power demand accumulated per bus, in MVA. Several wards may share a
        # bus, hence np.add.at rather than plain indexing. The solver adds this to its
        # Sbus accumulator (which it later divides by sn_mva), so it is deliberately NOT
        # scaled here -- matching the units of net.load.p_mw as read by _setup_pf.
        self.s_bus: np.typing.NDArray = np.zeros(n_bus, dtype=complex)
        np.add.at(self.s_bus, self._from_bus, s_ward)

        # Kept for res_ward, which reports ps + vm**2 * pz (the two halves recombined).
        self._pz_mw = np.where(in_service, pz_mw, 0.0)
        self._qz_mvar = np.where(in_service, qz_mvar, 0.0)
        self._ps_mw = np.where(in_service, ps_mw, 0.0)
        self._qs_mvar = np.where(in_service, qs_mvar, 0.0)
