# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numba
from pandas import DataFrame

from p3s.models.TwoPort import TwoPort


@numba.njit
def impedance_model(rft_pu, xft_pu, rtf_pu, xtf_pu, gf_pu, bf_pu, gt_pu, bt_pu, impedance_sn_mva, sn_mva=1.0):
    """
    Compute the asymmetric two-port stamp of a pandapower ``impedance`` element.

    The element is a pure series impedance given directly in per unit of its OWN
    reference power ``sn_mva``, with an optional shunt admittance at each terminal.
    Unlike a line it is *asymmetric*: the impedance seen from the from-bus
    (``z_ft``) may differ from the one seen from the to-bus (``z_tf``), which is how
    pandapower represents network equivalents.

    Inputs
    ------
    rft_pu, xft_pu   : float or array, series R / X from -> to [p.u. of impedance_sn_mva]
    rtf_pu, xtf_pu   : float or array, series R / X to -> from [p.u. of impedance_sn_mva]
    gf_pu, bf_pu     : float or array, shunt G / B at the from bus [p.u. of impedance_sn_mva]
    gt_pu, bt_pu     : float or array, shunt G / B at the to bus [p.u. of impedance_sn_mva]
    impedance_sn_mva : float or array, reference power of the per unit values [MVA]
    sn_mva           : float, network reference power [MVA]

    Returns
    -------
    Y_ff, Y_ft, Y_tf, Y_tt : complex arrays, the branch admittance stamp
    one_over_x             : complex array, 1/(j*x_ft) for the DC B-matrix
    """
    # 1) Rebase the per unit values from the element's own reference power onto the
    #    network reference power:
    #        z_ohm    = z_pu * v**2 / impedance_sn_mva
    #        z_pu_net = z_ohm / (v**2 / sn_mva) = z_pu * sn_mva / impedance_sn_mva
    #    The voltage base cancels, so no bus voltage is needed here. Shunt admittances
    #    are the reciprocal of an impedance and therefore scale the other way round.
    z_scale = sn_mva / impedance_sn_mva
    y_scale = impedance_sn_mva / sn_mva

    z_ft = (rft_pu + 1j * xft_pu) * z_scale
    z_tf = (rtf_pu + 1j * xtf_pu) * z_scale

    # 2) Series admittances. Both directions are inverted separately -- for the
    #    symmetric case (z_ft == z_tf) this reduces to the ordinary branch stamp.
    y_ft = 1.0 / z_ft
    y_tf = 1.0 / z_tf

    # 3) Terminal shunt admittances. pandapower carries these as a full (not halved)
    #    admittance per terminal: _calc_impedance_parameters_from_dataframe multiplies
    #    them by 2 precisely to cancel the /2 that makeYbus applies to Bcf/Bct, so the
    #    value stamped on the diagonal is g + j*b itself.
    y_shunt_f = (gf_pu + 1j * bf_pu) * y_scale
    y_shunt_t = (gt_pu + 1j * bt_pu) * y_scale

    # 4) Assemble the stamp. Mirrors pypower's makeYbus.branch_vectors with tap == 1:
    #        Yff = Ysf + Bcf/2 ;  Yft = -Ysf
    #        Ytt = Yst + Bct/2 ;  Ytf = -Yst
    #    i.e. the from-side series admittance drives the from row and the to-side one
    #    drives the to row. An impedance has no tap changer and no phase shift.
    Y_ff = y_ft + y_shunt_f
    Y_ft = -y_ft
    Y_tf = -y_tf
    Y_tt = y_tf + y_shunt_t

    # DC powerflow uses the from->to reactance only (makeBdc reads BR_X, which
    # _calc_impedance_parameter fills with x_ft; the asymmetric part lives in
    # BR_X_ASYM and is ignored by the DC model).
    one_over_x = 1.0 / (1j * xft_pu * z_scale)

    return Y_ff, Y_ft, Y_tf, Y_tt, one_over_x


class ImpedanceModel(TwoPort):
    """
    This is a class to represent a pandapower ``impedance`` element: an asymmetric
    series impedance with optional shunt admittances at either terminal.

    Main idea is:
               +-----------------------+
      ---+-----| y_ft = 1 / z_ft  (f->t)|-----+---
         |     | y_tf = 1 / z_tf  (t->f)|     |
         |     +-----------------------+      |
         |                                    |
    +----+----+                          +----+----+
    | gf+j*bf |                          | gt+j*bt |
    +----+----+                          +----+----+
         |                                    |
        ---                                  ---

    All impedance values are already per unit, referred to the element's own
    ``sn_mva``, so they only need rebasing onto the network reference power.
    """

    def __init__(self, impedance_table: DataFrame, sn_mva: float = 1.0):
        super().__init__()
        self._from_bus = impedance_table["from_bus"].values
        self._to_bus = impedance_table["to_bus"].values

        # Input Values
        rft_pu = impedance_table["rft_pu"].values
        xft_pu = impedance_table["xft_pu"].values
        rtf_pu = impedance_table["rtf_pu"].values
        xtf_pu = impedance_table["xtf_pu"].values
        impedance_sn_mva = impedance_table["sn_mva"].values

        # The terminal shunts were added to net.impedance later than the series values,
        # so nets written by older pandapower versions (and hand-built test nets) can be
        # missing the columns entirely. Absent or NaN means "no shunt" -- defaulting to
        # zero here keeps those nets loadable instead of poisoning Ybus with NaN.
        gf_pu = impedance_table["gf_pu"].values
        bf_pu = impedance_table["bf_pu"].values
        gt_pu = impedance_table["gt_pu"].values
        bt_pu = impedance_table["bt_pu"].values

        # return four matrices
        self._Y_ff, self._Y_ft, self._Y_tf, self._Y_tt, one_over_x = impedance_model(
            rft_pu=rft_pu,
            xft_pu=xft_pu,
            rtf_pu=rtf_pu,
            xtf_pu=xtf_pu,
            gf_pu=gf_pu,
            bf_pu=bf_pu,
            gt_pu=gt_pu,
            bt_pu=bt_pu,
            impedance_sn_mva=impedance_sn_mva,
            sn_mva=sn_mva,
        )

        # for DC powerflow. Sign convention matches TransmissionLineModel
        # (diagonal -one_over_x, off-diagonal +one_over_x) so that line, trafo and
        # impedance diagonals add rather than cancel.
        self._DC_Yff = self._DC_Ytt = -one_over_x
        self._DC_Yft = self._DC_Ytf = one_over_x

        # Zero out de-energised impedances (see TwoPort._apply_in_service). An
        # out-of-service impedance must contribute nothing to Ybus or to the DC
        # B-matrix; masking the stamps rather than dropping the rows keeps the arrays
        # row-aligned with net.impedance, which yf_matrix/yt_matrix rely on to fill
        # res_impedance.
        self._apply_in_service(impedance_table)
