import numpy as np

from pySDC.projects.DAE.problems.telegrapherDAE import TelegrapherDAE
from pySDC.projects.DAE.misc.meshDAE import MeshDAE

nvars = 128
problem_params = {"nvars": nvars, "newton_tol": 1e-12}

prob = TelegrapherDAE(**problem_params)

u = MeshDAE((nvars, None, np.dtype("float64")))
prob.eval_f(u, 0, 0)