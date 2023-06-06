import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class BuckConverter_DAE(ptype_dae):
    """
    Example implementing the buck converter model in a manipulated fashion, modelled as system of differential-algebraic
    equations (DAEs).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.duty = 0.5
        self.fsw = 1e3
        self.Vs = 10.0
        self.Rs = 1.0
        self.C1 = 1.0
        self.Rp = 0.2
        self.Lp = 1.0
        self.C2 = 1.0
        self.Rl = 5.0
        self.V_refmin = 2
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):

        vRs, vC1, vRp, vLp, vC2, vRl = u[0], u[1], u[2], u[3], u[4], u[5]
        iRs, iC1, iRp, iLp, iC2, iRl = u[6], u[7], u[8], u[9], u[10], u[11]

        dvC1, dvC2, diLp = du[1], du[4], du[9]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if u[1] <= self.V_refmin or t >= t_switch:
            self.Vs = self.V_refmin

        first_state_f = (
            vRs - self.Rs * iC1,
            dvC1 - iC1 / self.C1,
            vRp - self.Rp * iRp,
            vRp + vLp + vC2,
            dvC2 - iC2 / self.C2,
            vRl - self.Rl * iRl,
            self.Vs - self.Rs * iRs - vC1,
            iC1 - iRp,
            iRp - iLp,
            diLp - vLp / self.Lp,
            iLp - iC2 - iRl,
            vC2 - self.Rl * iRl,
        )
        second_state_f = (
            vRs - self.Rs * iRs,
            dvC1 - iC1 / self.C1,
            vRp - self.Rp * iRp,
            vC1 - vRp - vLp - vC2,
            dvC2 - iC2 / self.C2,
            vRl - self.Rl * iRl,
            self.Vs - self.Rs * iRs - vC1,
            iRs - iC1 - iRp,
            iRp - iLp,
            diLp - vLp / self.Lp,
            iLp - iC2 - iRl,
            vC2 - self.Rl * iRl,
        )

        f = self.dtype_f(self.init)
        f[:] = first_state_f
        print('Decision:')
        #if t >= t_switch:
        #    if u[4] >= self.V_refmin:
        #        print(t, t_switch, u[4], self.V_refmin, 'if-if')
        #        f[:] = first_state_f
        #    else:
        #        print(t, t_switch, u[4], self.V_refmin, 'if-else')
        #        f[:] = second_state_f
        #else:
        #    if u[4] >= self.V_refmin:
        #        print(t, t_switch, u[4], self.V_refmin, 'else-if')
        #        f[:] = first_state_f #first_state_f
        #    else:
        #        print(t, t_switch, u[4], self.V_refmin, 'else-else')
        #        f[:] = second_state_f #second_state_f

        #if u[1] >= self.V_refmin or t >= t_switch:
        #    f[:] = first_state_f
        #    print('If')
        #else:
        #    f[:] = second_state_f
        #    print('Else')

        # S1 = 1, S2 = 0
        #if u[4] >= self.V_refmin or t >= t_switch:
            #print(t, 'If')
            #f[:] = (
            #    vRs - self.Rs * iC1,
            #    dvC1 - iC1 / self.C1,
            #    vRp - self.Rp * iRp,
            #    vRp + vLp + vC2,
            #    dvC2 - iC2 / self.C2,
            #    vRl - self.Rl * iRl,
            #    self.Vs - self.Rs * iRs - vC1,
            #    iC1 - iRp,
            #    iRp - iLp,
            #    diLp - vLp / self.Lp,
            #    iLp - iC2 - iRl,
            #    vC2 - self.Rl * iRl,
            #)
            #if t >= t_switch:
            #    f[:] = first_state_f
            #    print(t, 'First if')
            #else:
            #    f[:] = second_state_f
            #    print(t, 'First else')
            #f[:] = first_state_f
        #else:  # S1 = 0, S2 = 1
            #print(t, 'else')
            #f[:] = (
            #    vRs - self.Rs * iRs,
            #    dvC1 - iC1 / self.C1,
            #    vRp - self.Rp * iRp,
            #    vC1 - vRp - vLp - vC2,
            #    dvC2 - iC2 / self.C2,
            #    vRl - self.Rl * iRl,
            #    self.Vs - self.Rs * iRs - vC1,
            #    iRs - iC1 - iRp,
            #    iRp - iLp,
            #    diLp - vLp / self.Lp,
            #    iLp - iC2 - iRl,
            #    vC2 - self.Rl * iRl,
            #)
            #if t >= t_switch:
            #    f[:] = second_state_f
            #    print(t, 'Second if')
            #else:
            #    print(t, 'Second else')
            #    f[:] = first_state_f
            #f[:] = second_state_f
        return f

    def u_exact(self, t):
        me = self.dtype_u(self.init)
        me[0] = 0.0  # vRs
        me[1] = 0.0  # vC1
        me[2] = 0.0  # vRp
        me[3] = 0.0  # vLp
        me[4] = 0.0  # vC2
        me[5] = 0.0  # vRl
        me[6] = 0.0  # iRs
        me[7] = 0.0  # iC1
        me[8] = 0.0  # iRp
        me[9] = 0.0  # iLp
        me[10] = 0.0  # iC2
        me[11] = 0.0  # iRl
        return me

    def get_switching_info(self, u, t):
        switch_detected = False
        m_guess = -100

        for m in range(1, len(u)):
            #state_node_before = self.V_refmin < u[m - 1][4] < self.V_refmax
            #state_current_node_drop = self.V_refmin - u[m][4] >= 0
            #state_current_node_rise = u[m][4] - self.V_refmax >= 0

            #if state_node_before and (state_current_node_drop or state_current_node_drop):
            #    switch_detected = True
            #    m_guess = m - 1
            #    if state_current_node_drop:
            #        root_problem = [self.V_refmin - u[k][4] for k in range(1, len(u))]
            #    else:
            #        root_problem = [u[k][4] - self.Vrefmax for k in range(1, len(u))]
            #    break

            #state_node_before_drop = self.V_refmin - u[m - 1][4] >= 0
            #state_node_before_rise = u[m - 1][4] - self.V_refmax >= 0
            #state_current_node = self.V_refmin < u[m][4] < self.V_refmax

            #if (state_node_before_drop or state_node_before_rise) and state_current_node:
            #    switch_detected = True
            #    m_guess = m - 1
            #    if state_node_before_drop:
            #        root_problem = [self.V_refmin - u[k][4] for k in range(1, len(u))]
            #    else:
            #        root_problem = [u[k][4] - self.Vrefmax for k in range(1, len(u))]
            #    break

            if u[m][4] - self.V_refmin <= 0:
                switch_detected = True
                m_guess = m - 1

        vC_switch = [u[m][4] - self.V_refmin for m in range(1, len(u))] if switch_detected else []
        print(vC_switch)
        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        self.nswitches += 1
