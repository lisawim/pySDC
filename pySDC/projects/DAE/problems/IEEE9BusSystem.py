import numpy as np
from scipy.interpolate import interp1d
import scipy as sp
import pandapower
import pandapower.networks as nw
import pandas as pd
from pypower.api import case9, ppoption, runpf, printpf, makeYbus, ext2int

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import ParameterError


def IEEE9Bus():
    ppc = case9()
    ppc_res, success = runpf(ppc)
    ppci = ext2int(ppc )
    Ybus, yf, yt = makeYbus(ppci['baseMVA'],ppci['bus'],ppci['branch'])

    # mpc = {}

    # mpc['baseMVA'] = ppc['baseMVA']

    # mpc['bus'] = ppc['bus']
    # mpc['bus'] = [
    #     [1.0000, 3.0000, 0, 	   0,       0, 0, 1.0000, 1.0400, 0,       345.0000, 1.0000, 1.1000, 0.9000],
    #     [2.0000, 2.0000, 0, 	   0,       0, 0, 1.0000, 1.0250, 9.280005481642794,  345.0000, 1.0000, 1.1000, 0.9000],
    #     [3.0000, 2.0000, 0, 	   0,       0, 0, 1.0000, 1.0250, 4.664751333136771,  345.0000, 1.0000, 1.1000, 0.9000],
    #     [4.0000, 1.0000, 0, 	   0,       0, 0, 1.0000, 1.02578839284401, -2.216787799949789, 345.0000, 1.0000, 1.1000, 0.9000],
    #     [5.0000, 1.0000, 125.0000, 50.0000, 0, 0, 1.0000, 0.9956308580482948, -3.988805272851466, 345.0000, 1.0000, 1.1000, 0.9000],
    #     [6.0000, 1.0000, 90.0000,  30.0000, 0, 0, 1.0000, 1.012654324017776, -3.687396170157061, 345.0000, 1.0000, 1.1000, 0.9000],
    #     [7.0000, 1.0000, 0,        0,       0, 0, 1.0000, 1.025769372386454, 3.719701154621758,  345.0000, 1.0000, 1.1000, 0.9000],
    #     [8.0000, 1.0000, 100.0000, 35.0000, 0, 0, 1.0000, 1.015882583627499, 0.727536076874295,  345.0000, 1.0000, 1.1000, 0.9000],
    #     [9.0000, 1.0000, 0,        0,       0, 0, 1.0000, 1.032352949002368, 1.966716074449083,  345.0000, 1.0000, 1.1000, 0.9000],
    # ]

    # mpc['gen'] = ppc['gen']
    # mpc['gen'] = [
    #     [1, 71.641, 27.046, 300, -300, 1.04,  100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [2, 163,    6.6537, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [3, 85,     -10.86, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]

    # mpc['branch'] = ppc['branch']
    # mpc['branch'] = [
    #     [1, 4, 0,       0.0576,     0,      250, 250, 250, 0, 0, 1, -360, 360,  71.641,   27.046,   -71.641, -23.923],
    #     [4, 6, 0.017,   0.092,      0.158,  250, 250, 250, 0, 0, 1, -360, 360,  30.704,   1.03,     -30.537, -16.543],
    #     [6, 9, 0.039,   0.17,       0.358,  150, 150, 150, 0, 0, 1, -360, 360, -59.463,  -13.457,    60.817, -18.075],
    #     [3, 9, 0,       0.0586,     0,      300, 300, 300, 0, 0, 1, -360, 360,  85,      -10.86,    -85,      14.955],
    #     [8, 9, 0.0119,  0.1008,     0.209,  150, 150, 150, 0, 0, 1, -360, 360, -24.095,  -24.296,    24.183,  3.1195],
    #     [7, 8, 0.0085,  0.072,      0.149,  250, 250, 250, 0, 0, 1, -360, 360,  76.38,   -0.79733,  -75.905, -10.704],
    #     [2, 7, 0,       0.0625,     0,      250, 250, 250, 0, 0, 1, -360, 360,  163,      6.6537,   -163,     9.1781],
    #     [5, 7, 0.032,   0.161,      0.306,  250, 250, 250, 0, 0, 1, -360, 360, -84.32,   -11.313,    86.62,  -8.3808],
    #     [4, 5, 0.01,    0.085,      0.176,  250, 250, 250, 0, 0, 1, -360, 360,  40.937,   22.893,   -40.68,  -38.687],
    # ]

    # mpc['Ybus'] = Ybus.todense()
    ppc_res['Ybus'] = Ybus.todense()
    # mpc['Ybus'] = np.array([
    #     [0-17.36111111111111j,0+0j,0+0j,0+17.36111111111111j,0+0j,0+0j,0+0j,0+0j,0+0j],
    #     [0+0j,0-16j,0+0j,0+0j,0+0j,0+0j,0+0j,0+16j,0+0j],
    #     [0+0j,0+0j,0-17.06484641638225j,0+0j,0+0j,0+17.06484641638225j,0+0j,0+0j,0+0j],
    #     [0+17.36111111111111j,0+0j,0+0j,3.307378962025306-39.30888872611897j,-1.942191248714727+10.51068205186793j,0+0j,0+0j,0+0j,-1.36518771331058+11.60409556313993j],
    #     [0+0j,0+0j,0+0j,-1.942191248714727+10.51068205186793j,3.224200387138842-15.84092701422946j,-1.282009138424115+5.588244962361526j,0+0j,0+0j,0+0j],
    #     [0+0j,0+0j,0+17.06484641638225j,0+0j,-1.282009138424115+5.588244962361526j,2.437096619314212-32.15386180510696j,-1.155087480890097+9.784270426363173j,0+0j,0+0j],
    #     [0+0j,0+0j,0+0j,0+0j,0+0j,-1.155087480890097+9.784270426363173j,2.772209954136233-23.30324902327162j,-1.617122473246136+13.69797859690844j,0+0j],
    #     [0+0j,0+16j,0+0j,0+0j,0+0j,0+0j,-1.617122473246136+13.69797859690844j,2.804726852537284-35.44561313021703j,-1.187604379291148+5.975134533308591j],
    #     [0+0j,0+0j,0+0j,-1.36518771331058+11.60409556313993j,0+0j,0+0j,0+0j,-1.187604379291148+5.975134533308591j,2.552792092601728-17.33823009644852j],
    # ], dtype=complex)
    # mpc['Ybus'] = np.array([
    #         [0 - 17.361j, 0 + 0j,   0 + 0j,      0 + 17.361j,       0 + 0j,            0 + 0j,           0 + 0j,            0 + 0j,            0 + 0j],
    #         [0 + 0j,      0 - 16j,  0 + 0j,      0 + 0j,            0 + 0j,            0 + 0j,           0 + 16j,           0 + 0j,            0 + 0j],
    #         [0 + 0j,      0 + 0j,   0 - 17.065j, 0 + 0j,            0 + 0j,            0 + 0j,           0 + 0j,            0 + 0j,            0 + 17.065j],
    #         [0 + 17.361j, 0 + 0j,   0 + 0j,      3.3074 - 39.309j, -1.3652 + 11.604j, -1.9422 + 10.511j, 0 + 0j,            0 + 0j,            0 + 0j],
    #         [0 + 0j,      0 + 0j,   0 + 0j,     -1.3652 + 11.604j,  2.5528 - 17.338j,  0 + 0j,          -1.1876 + 5.9751j,  0 + 0j,            0 + 0j],
    #         [0 + 0j,      0 + 0j,   0 + 0j,     -1.9422 + 10.511j,  0 + 0j,            3.2242 - 15.841j, 0 + 0j,            0 + 0j,           -1.282 + 5.5882j],
    #         [0 + 0j,      0 + 16j,  0 + 0j,      0 + 0j,           -1.1876 + 5.9751j,  0 + 0j,           2.8047 - 35.446j, -1.6171 + 13.698j,  0 + 0j],
    #         [0 + 0j,      0 + 0j,   0 + 0j,      0 + 0j,            0 + 0j,            0 + 0j,          -1.6171 + 13.698j,  2.7722 - 23.303j, -1.1551 + 9.7843j],
    #         [0 + 0j,      0 + 0j,   0 + 17.065j, 0 + 0j,            0 + 0j,           -1.282 + 5.5882j,  0 + 0j,           -1.1551 + 9.7843j,  2.4371 - 32.154j]
    #     ], dtype=complex)

    return ppc_res


def get_YBus(ppc):

    ppci = ext2int(ppc)
    Ybus, yf, yt = makeYbus(ppci['baseMVA'],ppci['bus'],ppci['branch'])

    return Ybus.todense()


class IEEE9BusSystem(ptype_dae):
    """
    Example implementing the IEEE9BusSystem.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """Initialization routine"""

        m, n = 3, 9  # m is number of machines (adjust this according to your specific case)
        nvars = 11*m + 2*m + 2*n
        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)
        self.mpc = IEEE9Bus()

        # self.net = nw.case9()
        self.m = m
        self.n = n
        self.baseMVA = self.mpc['baseMVA']
        self.ws = 2 * np.pi * 60
        self.ws_vector = self.ws * np.ones(self.m)
        self.w0 = self.ws_vector

        # Machine data (MD) as a 2D NumPy array
        self.MD = np.array([
            [23.640,                      6.4000,                    3.0100],                      # 1 - H
            [0.1460,                      0.8958,                    1.3125],                      # 2 - Xd
            [0.0608,                      0.1198,                    0.1813],                      # 3 - Xdp
            [0.0489,                      0.0881,                    0.1133],                      # 4 - Xdpp
            [0.0969,                      0.8645,                    1.2578],                      # 5 - Xq
            [0.0969,                      0.1969,                    0.2500],                      # 6 - Xqp
            [0.0396,                      0.0887,                    0.0833],                      # 7 - Xqpp
            [8.960000000000001,           6.0000,                    5.8900],                      # 8 - Tdop
            [0.1150,                      0.0337,                    0.0420],                      # 9 - Td0pp
            [0.3100,                      0.5350,                    0.6000],                      # 10 - Tqop
            [0.0330,                      0.0780,                    0.1875],                      # 11 - Tq0pp
            [0.0041,                      0.0026,                    0.0035],                      # 12 - RS
            [0.1200,                      0.1020,                    0.0750],                      # 13 - Xls
            [0.1 * (2 * 23.64) / self.ws, 0.2 * (2 * 6.4) / self.ws, 0.3 * (2 * 3.01) / self.ws],  # 14 - Dm (ws should be defined)
        ])

        # Excitation data (ED) as a 2D NumPy array
        self.ED = np.array([
            20.000 * np.ones(self.m),   # 1- KA
            0.2000 * np.ones(self.m),   # 2- TA
            1.0000 * np.ones(self.m),   # 3- KE
            0.3140 * np.ones(self.m),   # 4- TE
            0.0630 * np.ones(self.m),   # 5- KF
            0.3500 * np.ones(self.m),   # 6- TF
            0.0039 * np.ones(self.m),   # 7- Ax
            1.5550 * np.ones(self.m),   # 8- Bx
        ])

        # Turbine data (TD) as a 2D NumPy array
        self.TD = np.array([
            0.10 * np.ones(self.m),     # 1- TCH
            0.05 * np.ones(self.m),     # 2- TSV
            0.05 * np.ones(self.m),     # 3- RD
        ])


        self.bus = self.mpc['bus']
        self.branch = self.mpc['branch']
        self.gen = self.mpc['gen']
        self.YBus = self.mpc['Ybus']  # get_YBus(0.0)

        # ---- line outage disturbance ----
        temp_mpc = self.mpc
        temp_mpc['branch'] = np.delete(temp_mpc['branch'],5,0)
        self.YBus_line7_8_outage = get_YBus(temp_mpc)

        temp_mpc = self.mpc
        temp_mpc['branch'] = np.delete(temp_mpc['branch'],6,0)
        self.YBus_line6_8_outage = get_YBus(temp_mpc)


        # ---- bus fault disturbance ----
        temp_mpc = self.mpc
        temp_mpc['bus'][4][5]=-150
        self.YBus_bus5_Xfault = get_YBus(temp_mpc)


        # ---- load step disturbance ----
        temp_mpc = self.mpc
        temp_mpc['bus'][3][4]= 30
        self.YBus_bus4_LoadStep = get_YBus(temp_mpc)


        # ---- excitation limiter vmax ----
        self.vmax = 2.1

        # self.Yabs = np.array([
        #     [17.361111111111111, 0, 0, 17.361111111111111, 0, 0, 0, 0, 0],
        #     [0, 16, 0, 0, 0, 0, 16, 0, 0],
        #     [0, 0, 17.064846416382249, 0, 0, 0, 0, 0, 17.064846416382252],
        #     [17.361111111111111, 0, 0, 39.447781794175079, 11.684124756739717, 10.688617499098877, 0, 0, 0],
        #     [0, 0, 0, 11.684124756739717, 17.525152505625829, 0, 6.0920141868554758, 0, 0],
        #     [0, 0, 0, 10.688617499098877, 0, 16.165717948998456, 0, 0, 5.7334133978252355],
        #     [0, 16, 0, 0, 6.0920141868554758, 0, 35.55640565206722, 13.793103448275863, 0],
        #     [0, 0, 0, 0, 0, 0, 13.793103448275863, 23.467564063413604, 9.8522167487684733],
        #     [0, 0, 17.064846416382252, 0, 0, 5.7334133978252355, 0, 9.8522167487684733, 32.246089203402462],
        # ])# abs(self.Ybus)
        # self.Yang = np.array([
        #     [-1.570796326794897, 0, 0, 1.570796326794897, 0, 0, 0, 0, 0],
        #     [0, -1.570796326794897, 0, 0, 0, 0, 1.570796326794897, 0, 0],
        #     [0, 0, -1.570796326794897, 0, 0, 0, 0, 0, 1.570796326794897],
        #     [1.570796326794897, 0, 0, -1.486855836984724, 1.687905071361761, 1.753517887556994, 0, 0, 0],
        #     [0, 0, 0, 1.687905071361761, -1.424611751458039, 0, 1.766997144083013, 0, 0],
        #     [0, 0, 0, 1.753517887556994, 0, -1.370003001562272, 0, 0, 1.796305961983321],
        #     [0, 1.570796326794897, 0, 0, 1.766997144083013, 0, -1.491833237569754, 1.688307972226342, 0],
        #     [0, 0, 0, 0, 0, 0, 1.688307972226342, -1.452390416852785, 1.688307972226342],
        #     [0, 0, 1.570796326794897, 0, 0, 1.796305961983321, 0, 1.688307972226342, -1.495146137990937],
        # ])  # np.angle(self.Ybus)  # rad

        self.IC1 = [row[7] for row in self.bus]  # Column 8 in MATLAB is indexed as 7 in Python (0-based index)
        self.IC2 = [row[8] for row in self.bus]  # Column 9 in MATLAB is indexed as 8 in Python

        n_prev, m_prev = self.n, self.m
        self.n = len(self.bus)  # Number of rows in 'bus' list; self.n already defined above?!
        self.m = len(self.gen)  # Number of rows in 'gen' list; self.m already defined above?!
        if n_prev != self.n or m_prev != self.m:
            raise ParameterError("Number of rows in bus or gen not equal to initialised n or m!")

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][1]
        self.genP = gen0
        self.IC3 = [val / self.baseMVA for val in self.genP]

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][2]
        genQ = gen0
        for i in range(self.n):
            genQ[i] += self.bus[i][5]  # Column 6 in MATLAB is indexed as 5 in Python
        self.IC4 = [val / self.baseMVA for val in genQ]

        self.IC5 = [row[2] for row in self.bus]  # Column 3 in MATLAB is indexed as 2 in Python
        self.IC5 = [val / self.baseMVA for val in self.IC5]

        self.IC6 = [row[3] for row in self.bus]  # Column 4 in MATLAB is indexed as 3 in Python
        self.IC6 = [val / self.baseMVA for val in self.IC6]

        self.IC = list(zip(self.IC1, self.IC2, self.IC3, self.IC4, self.IC5, self.IC6))

        self.PL = [row[4] for row in self.IC]  # Column 5 in MATLAB is indexed as 4 in Python
        self.QL = [row[5] for row in self.IC]  # Column 6 in MATLAB is indexed as 5 in Python

        self.PG = np.array([row[2] for row in self.IC])  # Column 3 in MATLAB is indexed as 2 in Python
        self.QG = np.array([row[3] for row in self.IC])  # Column 4 in MATLAB is indexed as 3 in Python

        self.TH0 = np.array([row[1] * np.pi / 180 for row in self.IC])
        self.V0 = np.array([row[0] for row in self.IC])
        self.VG0 = self.V0[:self.m]
        self.THG0 = self.TH0[:self.m]

        # Extracting values from the MD array
        self.H = self.MD[0, :]
        self.Xd = self.MD[1, :]
        self.Xdp = self.MD[2, :]
        self.Xdpp = self.MD[3, :]
        self.Xq = self.MD[4, :]
        self.Xqp = self.MD[5, :]
        self.Xqpp = self.MD[6, :]
        self.Td0p = self.MD[7, :]
        self.Td0pp = self.MD[8, :]
        self.Tq0p = self.MD[9, :]
        self.Tq0pp = self.MD[10, :]
        self.Rs = self.MD[11, :]
        self.Xls = self.MD[12, :]
        self.Dm = self.MD[13, :]

        # Extracting values from the ED array
        self.KA = self.ED[0, :]
        self.TA = self.ED[1, :]
        self.KE = self.ED[2, :]
        self.TE = self.ED[3, :]
        self.KF = self.ED[4, :]
        self.TF = self.ED[5, :]
        self.Ax = self.ED[6, :]
        self.Bx = self.ED[7, :]

        # Extracting values from the TD array
        self.TCH = self.TD[0, :]
        self.TSV = self.TD[1, :]
        self.RD = self.TD[2, :]

        # Calculate MH
        self.MH = 2 * self.H / self.ws

        # Represent QG as complex numbers
        self.QG = self.QG.astype(complex)

        # Convert VG0 and THG0 to complex phasors
        self.Vphasor = self.VG0 * np.exp(1j * self.THG0)

        # Calculate Iphasor
        self.Iphasor = np.conj(np.divide(self.PG[:m] + 1j * self.QG[:m], self.Vphasor))

        # Calculate E0
        self.E0 = self.Vphasor + (self.Rs + 1j * self.Xq) * self.Iphasor

        # Calculate Em, D0, Id0, and Iq0
        self.Em = np.abs(self.E0)
        self.D0 = np.angle(self.E0)
        self.Id0 = np.real(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))
        self.Iq0 = np.imag(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))

        # Calculate Edp0, Si2q0, Eqp0, and Si1d0
        self.Edp0 = (self.Xq - self.Xqp) * self.Iq0
        self.Si2q0 = (self.Xls - self.Xq) * self.Iq0
        self.Eqp0 = self.Rs * self.Iq0 + self.Xdp * self.Id0 + self.V0[:self.m] * np.cos(self.D0 - self.TH0[:self.m])
        self.Si1d0 = self.Eqp0 - (self.Xdp - self.Xls) * self.Id0

        # Calculate Efd0 and TM0
        self.Efd0 = self.Eqp0 + (self.Xd - self.Xdp) * self.Id0
        self.TM0 = ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * self.Eqp0 * self.Iq0 + ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * self.Si1d0 * self.Iq0 + \
            ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * self.Edp0 * self.Id0 - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * self.Si2q0 * self.Id0 + \
            (self.Xqpp - self.Xdpp) * self.Id0 * self.Iq0
        
        # Calculate VR0 and RF0
        self.VR0 = (self.KE + self.Ax * np.exp(self.Bx * self.Efd0)) * self.Efd0
        self.RF0 = (self.KF / self.TF) * self.Efd0

        # Calculate Vref and PSV0
        self.Vref = self.V0[:self.m] + self.VR0 / self.KA
        self.PSV0 = self.TM0
        self.PC = self.PSV0

        # Create a list to hold the initial values of state variables
        x0 = [0] * (11 * m)

        # Assign initial values to state variables as in the MATLAB code
        x0[0:m] = self.Eqp0
        x0[m:2 * m] = self.Si1d0
        x0[2 * m:3 * m] = self.Edp0
        x0[3 * m:4 * m] = self.Si2q0
        x0[4 * m:5 * m] = self.D0
        x0[5 * m:6 * m] = self.ws_vector
        x0[6 * m:7 * m] = self.Efd0
        x0[7 * m:8 * m] = self.RF0
        x0[8 * m:9 * m] = self.VR0
        x0[9 * m:10 * m] = self.TM0
        x0[10 * m:11 * m] = self.PSV0

        # Initial values of algebraic variables
        self.a0 = np.concatenate((self.Id0, self.Iq0, self.V0, self.TH0))

        self.alpha = 2
        self.beta = 2

        self.bb1, self.aa1 = np.meshgrid(np.arange(0, self.m), np.arange(0, self.n))
        self.bb1, self.aa1 = self.bb1.astype(int), self.aa1.astype(int)

        # Create matrix grid to eliminate for-loops (load buses)
        self.bb2, self.aa2 = np.meshgrid(np.arange(self.m, self.n), np.arange(0, self.n))
        self.bb2, self.aa2 = self.bb2.astype(int), self.aa2.astype(int)

        # disturbance
        self.event_happened = False

        self.t_switch = None
        self.nswitches = 0


    def eval_f(self, u, du, t):
        dEqp, dSi1d, dEdp = du[0:self.m], du[self.m:2*self.m], du[2*self.m:3*self.m]
        dSi2q, dDelta = du[3*self.m:4*self.m], du[4*self.m:5*self.m]
        dw, dEfd, dRF = du[5*self.m:6*self.m], du[6*self.m:7*self.m], du[7*self.m:8*self.m]
        dVR, dTM, dPSV = du[8*self.m:9*self.m], du[9*self.m:10*self.m], du[10*self.m:11*self.m]

        Eqp, Si1d, Edp = u[0:self.m], u[self.m:2*self.m], u[2*self.m:3*self.m]
        Si2q, Delta = u[3*self.m:4*self.m], u[4*self.m:5*self.m]
        w, Efd, RF = u[5*self.m:6*self.m], u[6*self.m:7*self.m], u[7*self.m:8*self.m]
        VR, TM, PSV = u[8*self.m:9*self.m], u[9*self.m:10*self.m], u[10*self.m:11*self.m]

        Id, Iq = u[11*self.m:11*self.m + self.m], u[11*self.m + self.m:11*self.m + 2*self.m]

        # V = []
        # TH = []

        # fault disturbance ver.1, bus5, via changing YBus, not working: 
        # if t <= 0.29 and t >= 0.1:
        #     # self.YBus = self.YBus_bus8_Xfault
        #     self.YBus = self.YBus_bus5_Xfault
        # #     self.mpc['bus'][7][5]=-10e3
        # #     self.YBus=get_YBus(self.mpc)
        # else:
        #     # self.mpc['bus'][7][5]=0
        # #     self.YBus=get_YBus(self.mpc)
        #     self.YBus = self.mpc['Ybus']

        # Load step disturbance: 
        # if  t >= 0.1:
        #     self.YBus = self.YBus_bus4_LoadStep


        # line outage disturbance: 
        if t >= 0.1:
            # temp_mpc = self.mpc
            # temp_mpc['branch'] = np.delete(temp_mpc['branch'],5,0)
            # self.YBus=get_YBus(temp_mpc)
            self.YBus = self.YBus_line7_8_outage
            # self.YBus = self.YBus_line6_8_outage
        
        # --- manual limiter event for all generators ---
        # Efd = np.array([min(x, 2.0) for x in Efd])


        # self.YBus = get_YBus(self.mpc)

        V = u[11*self.m + 2*self.m : 11*self.m + 2*self.m + self.n]
        TH = u[11*self.m + 2*self.m + self.n:11*self.m + 2*self.m + 2 *self.n]

        # fault disturbance ver.2, bus5: 
        # if t < 0.2 and t >= 0.1:
        #     # self.YBus = self.YBus_bus8_Xfault
        #     V[4] = 1e-9
        #     TH[4] = 1e-9


        self.Yang = np.angle(self.YBus)
        self.Yabs = np.abs(self.YBus)

        COI = np.sum(w * self.MH) / np.sum(self.MH)

        # Voltage-dependent active loads PL2, and voltage-dependent reactive loads QL2 
        # PL2 = self.PL * ((V / self.V0) ** self.alpha)
        # QL2 = self.QL * ((V / self.V0) ** self.beta)
        PL2 = np.array(self.PL)
        QL2 = np.array(self.QL)

        # fault disturbance ver.2, bus5: 
        # if t < 0.2 and t >= 0.1:
        #     # self.YBus = self.YBus_bus8_Xfault
        #     # PL2[4] = 1e-9
        #     QL2[4] = 100


        # load step ver.2, bus 4
        # if t >= 0.1:
        #     PL2[4] = 1.5


        # mechanical torque step disturbance
        # if t < 0.3 and t >= 0.1:
        #     TM[0] = self.TM0[0] * 0.7
        # else:
        #     TM[0] = self.TM0[0]
            

        # if(w==0.95):
        #     t1 = #...
        # if w <= 0.95:
        #     # do something
        # else:
            # else    
        # if t < 0.2 and t >= 0.1:
        #     PL2[3] = 1000.0
            # QL2[3] = 500.0



            # V[4] = 0.0
            # TH[4] = 0.0
            # PL2[4] = 0.0
            # QL2[4] = 0.0
            # V[4] = 0.2
            # TH[4] = 0.0

        V = V.T

        # Vectorized calculations
        Vectorized_angle1 = (np.array([TH.take(indices) for indices in self.bb1.T]) - np.array([TH.take(indices) for indices in self.aa1.T]) - self.Yang[:self.m, :self.n])
        Vectorized_mag1 = (V[:self.m] * V[:self.n].reshape(-1, 1)).T * self.Yabs[:self.m, :self.n]

        sum1 = np.sum(Vectorized_mag1 * np.cos(Vectorized_angle1), axis=1)
        sum2 = np.sum(Vectorized_mag1 * np.sin(Vectorized_angle1), axis=1)

        VG = V[:self.m]
        THG = TH[:self.m]
        Angle_diff = Delta - THG

        Vectorized_angle2 = (np.array([TH.take(indices) for indices in self.bb2.T]) - np.array([TH.take(indices) for indices in self.aa2.T]) - self.Yang[self.m:self.n, :self.n])
        Vectorized_mag2 = (V[self.m:self.n] * V[:self.n].reshape(-1, 1)).T * self.Yabs[self.m:self.n, : self.n]

        sum3 = np.sum(Vectorized_mag2 * np.cos(Vectorized_angle2), axis=1)
        sum4 = np.sum(Vectorized_mag2 * np.sin(Vectorized_angle2), axis=1)

        # Initialise f
        f = self.dtype_f(self.init)

        # # limiter event for gen1
        # if(Efd[1]>= 2.0):
        #     Efd[1] = 2.0
        #     if(dEfd[1] > 0):
        #         dEfd[1] = 0 #(7) == 0
        #     else:
        #         pass
        #         #(7)
        # else:
        #     #(7)
        #     pass

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # Equations as list
        eqs = []
        eqs.append((1.0 / self.Td0p) * (-Eqp - (self.Xd - self.Xdp) * (Id - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls) ** 2) * (Si1d + (self.Xdp - self.Xls) * Id - Eqp)) + Efd) - dEqp)  # (1)
        eqs.append((1.0 / self.Td0pp) * (-Si1d + Eqp - (self.Xdp - self.Xls) * Id) - dSi1d)  # (2)
        eqs.append((1.0 / self.Tq0p) * (-Edp + (self.Xq - self.Xqp) * (Iq - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls) ** 2) * (Si2q + (self.Xqp - self.Xls) * Iq + Edp))) - dEdp)  # (3)
        eqs.append((1.0 / self.Tq0pp) * (-Si2q - Edp - (self.Xqp - self.Xls) * Iq) - dSi2q)  # (4)
        eqs.append(w - COI - dDelta)  # (5)
        eqs.append((self.ws / (2.0 * self.H)) * (TM - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp * Iq - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d * Iq - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp * Id + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q * Id - (self.Xqpp - self.Xdpp) * Id * Iq - self.Dm * (w - self.ws)) - dw)  # (6)
        # eqs.append((1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)  # (7)
        # --- limiter event for gen1 ---
        if(Efd[1]>= self.vmax or t >= t_switch):
            Efd[1] = self.vmax
            # if(dEfd[1] > 0 or t > t_switch):
            _temp_dEfd0 = ((1.0 / self.TE[0]) * ((-(self.KE[0] + self.Ax[0] * np.exp(self.Bx[0] * Efd[0]))) * Efd[0] + VR[0]) - dEfd[0])
            # #    _temp_dEfd1 = 0
            dEfd[1] = 0
            #    _temp_dEfd1 = ((1.0 / self.TE[1]) * ((-(self.KE[1] + self.Ax[1] * np.exp(self.Bx[1] * Efd[1]))) * Efd[1] + VR[1]) - dEfd[1])
            _temp_dEfd2 = ((1.0 / self.TE[2]) * ((-(self.KE[2] + self.Ax[2] * np.exp(self.Bx[2] * Efd[2]))) * Efd[2] + VR[2]) - dEfd[2])
            eqs.append(np.array([_temp_dEfd0, 0, _temp_dEfd2]))
            # else:
            #    eqs.append((1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)
        else:
           eqs.append((1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)
        # --- limiter event for gen1 ---

        # --- OXL limiter in Milano 2010 Book ---
        # beta_p = self.Xq * self.PG


        eqs.append((1.0 / self.TF) * (-RF + (self.KF / self.TF) * Efd) - dRF)  # (8)
        eqs.append((1.0 / self.TA) * (-VR + self.KA * RF - ((self.KA * self.KF) / self.TF) * Efd + self.KA * (self.Vref - V[:self.m])) - dVR)  # (9)
        eqs.append((1.0 / self.TCH) * (-TM + PSV) - dTM)  # (10)
        eqs.append((1.0 / self.TSV) * (-PSV + self.PSV0 - (1.0 / self.RD) * (w / self.ws - 1)) - dPSV)  # (11)
        eqs.append(self.Rs * Id - self.Xqpp * Iq - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q + VG * np.sin(Angle_diff))  # (12)
        eqs.append(self.Rs * Iq + self.Xdpp * Id - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d + VG * np.cos(Angle_diff))  # (13)
        eqs.append((Id * VG.T * np.sin(Angle_diff) + Iq * VG.T * np.cos(Angle_diff)) - PL2[0:self.m] - sum1)  # (14)
        eqs.append((Id * VG.T * np.cos(Angle_diff) - Iq * VG.T * np.sin(Angle_diff))- QL2[0:self.m] - sum2)  # (15)
        eqs.append(-PL2[self.m:self.n] - sum3)  # (16)
        eqs.append(-QL2[self.m:self.n] - sum4)  # (17)
        eqs_flatten = [item for sublist in eqs for item in sublist]

        f[:] = eqs_flatten

        # --- manual limiter event for all generators ---
        # f[6*self.m:7*self.m][indices_to_change] = dEfd[indices_to_change]

        return f
    
    def u_exact(self, t):
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me[0:self.m] = self.Eqp0
        me[self.m:2 * self.m] = self.Si1d0
        me[2 * self.m:3 * self.m] = self.Edp0
        me[3 * self.m:4 * self.m] = self.Si2q0
        me[4 * self.m:5 * self.m] = self.D0
        me[5 * self.m:6 * self.m] = self.ws_vector
        me[6 * self.m:7 * self.m] = self.Efd0
        me[7 * self.m:8 * self.m] = self.RF0
        me[8 * self.m:9 * self.m] = self.VR0
        me[9 * self.m:10 * self.m] = self.TM0
        me[10 * self.m:11 * self.m] = self.PSV0
        me[11*self.m:11*self.m + self.m] = self.Id0
        me[11*self.m + self.m:11*self.m + 2*self.m] = self.Iq0
        me[11*self.m + 2*self.m:11*self.m + 2*self.m + self.n] = self.V0
        me[11*self.m + 2*self.m + self.n:11*self.m + 2*self.m + 2 *self.n] = self.TH0
        return me
    
    def get_switching_info(self, u, t, du=None):
        switch_detected = False
        already_detected = False
        m_guess = -100

        for m in range(1, len(u)):
            h_prev_node = u[m - 1][6*self.m+1] - self.vmax
            h_curr_node = u[m][6*self.m+1] - self.vmax
            h2_prev_node = du[m - 1][6*self.m+1]
            h2_curr_node = du[m][6*self.m+1]
            if h_prev_node < 0 and h_curr_node >= 0 and not already_detected:
                print('If')
                switch_detected = True
                m_guess = m - 1
                state_function = [u[m][6*self.m+1] - self.vmax for m in range(len(u))]
                already_detected = True
                break
            # elif h2_prev_node > 0 and h2_curr_node <= 0 and not already_detected:
            #     print('elif')
            #     switch_detected = True
            #     m_guess = m - 1
            #     state_function = [du[m][6*self.m+1] for m in range(len(du))]
            #     already_detected = True
            #     break
            # elif h2_prev_node >= 0 and h2_curr_node < 0 and not already_detected:
            #     print('elif2')
            #     switch_detected = True
            #     m_guess = m - 1
            #     state_function = [du[m][6*self.m+1] for m in range(len(du))]
            #     already_detected = True
            #     break
            else:
                print('else')
                # set first state function in the beginning, have to think about how to deal with wether it is
                # already close to zero or not
                state_function = [u[m][6*self.m+1] - self.vmax for m in range(len(u))]
        print(state_function)
        return switch_detected, m_guess, state_function

            
        
            

    
    def count_switches(self):
        self.nswitches +=1