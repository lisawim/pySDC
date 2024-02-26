import sys
sys.path.append("/home/jzh/data_jzh/pints_related/gits_2/pySDC")
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import statistics
import pySDC.helpers.plot_helper as plt_helper
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.ThreeInverterSystem import ThreeInverterSystem
# from pySDC.projects.DAE.problems.WSCC9BusSystem_noSE import WSCC9BusSystem
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.Runge_Kutta_DAE import BackwardEulerDAE, RungeKuttaDAE, ImplicitMidpointMethodIMEXDAE, DIRK43_2DAE

# from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
# from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats

class TimeSeries:
    """Stores data from different simulation sources.
        A TimeSeries object always consists of timestamps and datapoints.
    """
    def __init__(self, name, time, values, label=""):
        self.time = np.array(time)
        self.values = np.array(values)
        self.name = name
        if not label:
            self.label = name
        else:
            self.label = label
    
    def __mul__(self, ts):
        if isinstance(ts, TimeSeries):
            new_val = self.values * ts.values
        else:
            new_val = self.values * ts
        return TimeSeries(self.name+"_mul", self.time, new_val)
    
    def abs(self):
        """ Calculate absolute value of complex time series.
        """
        abs_values = []
        for value in self.values:
            abs_values.append(np.abs(value))
        ts_abs = TimeSeries(self.name+'_abs', self.time, abs_values, self.label+'_abs')
        return ts_abs

    def phase(self):
        """ Calculate phase of complex time series.
        """
        phase_values = []
        for value in self.values:
            phase_values.append(np.angle(value, deg=True))
        ts_phase = TimeSeries(self.name+'_phase', self.time, phase_values, self.label+'_phase')
        return ts_phase

    @staticmethod
    def rel_diff(name, ts1, ts2, normalize = None, point_wise = False, threshold = 0):
        """
        Returns relative difference between two time series objects to the first.
        calculated against the max of ts1.
        """
        diff_ts = TimeSeries.diff('diff', ts1, ts2)
        diff_val=diff_ts.values
        if normalize is not None:
            rel_diff_to_ts1 = diff_val/normalize
            ts_rel_diff_to_ts1 = TimeSeries(name, diff_ts.time, rel_diff_to_ts1)
            return ts_rel_diff_to_ts1
        # relative error to the max value of ts1
        if not point_wise:
            rel_diff_to_ts1 = diff_val/np.abs(ts1.values).max()
            ts_rel_diff_to_ts1 = TimeSeries(name, diff_ts.time, rel_diff_to_ts1)
        else:
            # calculate the relative point-wise error above threshold 
            index_=np.where(np.abs(ts1.values)>threshold)
            ts1_filtered_val_=ts1.values[index_]
            ts1_filtered_time_=ts1.time[index_]
            #ts2_filtered_val_=ts2.values[index_]
            diff_val_filtered=diff_val[index_]
            rel_diff_to_ts1 = diff_val_filtered/ts1_filtered_val_
            ts_rel_diff_to_ts1 = TimeSeries(name, ts1_filtered_time_, np.abs(rel_diff_to_ts1))
        return ts_rel_diff_to_ts1
    
    @staticmethod
    def diff(name, ts1, ts2):
        """Returns difference between values of two Timeseries objects.
        """
        if len(ts1.time) == len(ts2.time):
            ts_diff = TimeSeries(name, ts1.time, (ts1.values - ts2.values))
        else:  # different timestamps, common time vector and interpolation required before substraction
            time = sorted(set(list(ts1.time) + list(ts2.time)))
            interp_vals_ts1 = np.interp(time, ts1.time, ts1.values)
            interp_vals_ts2 = np.interp(time, ts2.time, ts2.values)
            ts_diff = TimeSeries(name, time, (interp_vals_ts2 - interp_vals_ts1))
        return ts_diff

    @staticmethod
    def abs_rel_diff(name, ts1, ts2, normalize = None, point_wise = False, threshold = 0):
        ts_rel_diff = TimeSeries.rel_diff(name, ts1, ts2, normalize, point_wise, threshold)
        ts_rel_diff.values = np.abs(ts_rel_diff.values)
        return ts_rel_diff


def single_run(run_name="test", time_step=1e-1, end_time=0.1, sweeper_type = RungeKuttaDAE):
    """
    A testing ground for the synchronous machine model
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-7
    level_params['dt'] = time_step

    # initialize sweeper parameters
    M_fix = 3
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'RADAU-RIGHT',
        'QI': 'IE',
    }

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-3  # tollerance for implicit solver
    problem_params['nvars'] = 36

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogSolution]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = ThreeInverterSystem
    # description['problem_class'] = WSCC9BusSystem
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_type
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = end_time

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)


    # store results
    sol = get_sorted(stats, type='approx_solution', sortby='time')
    sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    sol_data = np.array([[sol[j][1][i] for j in range(len(sol))] for i in range(problem_params['nvars'])])
    niter = filter_stats(stats, type='niter')
    niter = np.fromiter(niter.values(), int)

    t = np.array([me[0] for me in get_sorted(stats, type='u', sortby='time')])
    # print([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time', recomputed=False)])
    i_c1 = np.array([ me[1][0:2] for me in get_sorted(stats, type='u', sortby='time')])
    i_c2 = np.array([ me[1][8:10] for me in get_sorted(stats, type='u', sortby='time')])
    i_c3 = np.array([ me[1][16:18] for me in get_sorted(stats, type='u', sortby='time')])

    v_cc1 = np.array([ me[1][6:8] for me in get_sorted(stats, type='u', sortby='time')])
    v_cc2 = np.array([ me[1][14:16] for me in get_sorted(stats, type='u', sortby='time')])
    v_cc3 = np.array([ me[1][22:24] for me in get_sorted(stats, type='u', sortby='time')])

    delta1 = np.array([ me[1][5] for me in get_sorted(stats, type='u', sortby='time')])
    delta2 = np.array([ me[1][13] for me in get_sorted(stats, type='u', sortby='time')])
    delta3 = np.array([ me[1][21] for me in get_sorted(stats, type='u', sortby='time')])

    i_pccDQ = np.array([ me[1][24:26] for me in get_sorted(stats, type='u', sortby='time')])
    il_12DQ = np.array([ me[1][26:28] for me in get_sorted(stats, type='u', sortby='time')])
    il_23DQ = np.array([ me[1][28:30] for me in get_sorted(stats, type='u', sortby='time')])

    i_g1DQ = i_pccDQ - il_12DQ
    i_g2DQ = il_12DQ - il_23DQ
    i_g3DQ = il_23DQ

    i_g1=np.zeros_like(i_g1DQ)
    i_g2=np.zeros_like(i_g2DQ)
    i_g3=np.zeros_like(i_g3DQ)

    for k in range(len(delta1)):
        rotation_matrix1 = np.array([[np.cos(delta1[k]), np.sin(delta1[k])], [-np.sin(delta1[k]), np.cos(delta1[k])]])
        i_g1[k, :] = np.dot(rotation_matrix1, i_g1DQ[k, :])

        rotation_matrix2 = np.array([[np.cos(delta2[k]), np.sin(delta2[k])], [-np.sin(delta2[k]), np.cos(delta2[k])]])
        i_g2[k, :] = np.dot(rotation_matrix2, i_g2DQ[k, :])

        rotation_matrix3 = np.array([[np.cos(delta3[k]), np.sin(delta3[k])], [-np.sin(delta3[k]), np.cos(delta3[k])]])
        i_g3[k, :] = np.dot(rotation_matrix3, i_g3DQ[k, :])

    fs = 2000
    Ts = 1/fs
    Lf = 0.1e-3;    # L Filter
    Kp  = Lf*1.8/(3*Ts)


    v_g1 = Kp*(i_c1-i_g1) + v_cc1
    v_g2 = Kp*(i_c2-i_g2) + v_cc2 
    v_g3 = Kp*(i_c3-i_g3) + v_cc3 

    # Initialize i_pcc as a zeros matrix with the same shape as i_pccDQ
    i_pcc = np.zeros_like(i_pccDQ)

    # Perform the matrix operations
    for k in range(len(delta1)):
        rotation_matrix = np.array([[np.cos(delta1[k]), np.sin(delta1[k])],
                                    [-np.sin(delta1[k]), np.cos(delta1[k])]])
        i_pcc[k, :] = np.dot(rotation_matrix, i_pccDQ[k, :])

    # Set v_pcc to v_g1
    v_pcc = v_g1

    # Transpose i_pcc
    i_pcc = i_pcc.T

    # file_name_suffix = "invSys"
    # fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    # ax.plot(t, v_pcc[:, 0], label='v_pcc_d')
    # ax.plot(t, v_pcc[:, 1], label='v_pcc_q')
    # ax.legend(loc='upper right', fontsize=10)
    # ax.set_xlim(0.9,1.7)
    # # plt_helper.plt.show()
    # fig.savefig(f'data/{run_name}_{file_name_suffix}_v_pcc.png', dpi=300, bbox_inches='tight')
    return TimeSeries(run_name, t, v_pcc) # return d-, q-axis 



## Reference 

# reference = RungeKuttaDAE
reference = BackwardEulerDAE
# reference = fully_implicit_DAE
reference_res = single_run("reference", 1e-6, 5e-3, reference)


method_list = [DIRK43_2DAE, ImplicitMidpointMethodIMEXDAE, fully_implicit_DAE]

dt_list = [200e-6, 250e-6, 500e-6, 1000e-6]


## Candidates

res_candidates= {
    0:[],
    1:[],
    2:[]
}

# for i_meth in range(len(method_list)):
#     for dt_ in dt_list: 
#         res_ = single_run(f"{i_meth}_{dt_}", dt_, 5e-3, method_list[i_meth])
#         res_candidates[i_meth].append(res_)



## Store results

pickle.dump(reference_res, open("data/pintsime2/reference_eb_5us_ts.p", 'wb'))
# pickle.dump(res_candidates[0], open("data/pintsime2/DIRK43_2DAE_ts.p", 'wb'))
# pickle.dump(res_candidates[1], open("data/pintsime2/ImplicitMidpointMethodIMEXDAE_ts.p", 'wb'))
# pickle.dump(res_candidates[2], open("data/pintsime2/fully_implicit_DAE_ts.p", 'wb'))

