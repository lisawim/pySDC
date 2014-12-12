from pySDC import Step as stepclass
from pySDC import CollocationClasses as collclass



import numpy as np


from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.Methods_Serial import mlsdc_step, sdc_step
import pySDC.Methods_Parallel as mp


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['Tend'] = 2*0.125
    sparams['maxiter'] = 10
    sparams['parblocks'] = 2

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['nvars'] = 255

    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d


    S = stepclass.step(sparams)

    S.generate_hierarchy(description)

    S.dt = 0.125

    S.time = 0

    S.stats.niter = 0

    P = S.levels[0].prob
    uinit = P.u_exact(S.time)
    S.init_step(uinit)



    step_stats = []


    while S.time < S.params.Tend:

        uend = mlsdc_step(S)

        step_stats.append(S.stats)

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)


    uex = P.u_exact(S.time)

    print(step_stats[1].residual,step_stats[1].level_stats[0].residual)

    print('error at time %s: %s' %(S.time,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)))