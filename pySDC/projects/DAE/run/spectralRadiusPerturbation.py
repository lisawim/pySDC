import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP, LinearTestSPPMinion
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.projects.DAE.run.sweepEqualMatrix import getSweeperMatrix


def compute_dt_eps_ratios(eps_range, ratio_range, num_points=100):
    # Generate epsilon values
    eps_values = np.linspace(eps_range[0], eps_range[1], num_points)

    # Initialize the ratios
    ratios = np.zeros(num_points)
    
    # Set the first and last ratios explicitly
    ratios[0] = ratio_range[0]
    ratios[-1] = ratio_range[1]
    
    # Interpolate the remaining ratios
    ratios[1:-1] = np.logspace(np.log10(ratio_range[0]), np.log10(ratio_range[1]), num_points-2)
    
    # Compute dt values
    dt_values = ratios * eps_values
    
    # Sort dt, eps, and ratios by ratios
    sorted_indices = np.argsort(ratios)
    sorted_dt_values = dt_values[sorted_indices]
    sorted_eps_values = eps_values[sorted_indices]
    
    return sorted_dt_values, sorted_eps_values


def generate_powers_of_ten(ratioRange):
    startExponent = int(np.floor(np.log10(ratioRange[0])))
    endExponent = int(np.ceil(np.log10(ratioRange[1])))
    powersOfTen = [10**i for i in range(startExponent, endExponent + 1)]
    return powersOfTen


def main():
    QI = 'LU'
    sweeper = generic_implicit
    problem = LinearTestSPP
    M = 3
    quad_type = 'RADAU-RIGHT'

    epsValues = np.logspace(-11, 1, num=40)#[10 ** (-m) for m in range(2, 12)]
    # dtValues = [5 * 10 ** (-m) for m in range(1, 5)]
    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    dtValues = (Tend - t0) / nSteps

    spectralRadius = np.zeros((len(dtValues), len(epsValues)))
    cond = np.zeros((len(dtValues), len(epsValues)))
    ratio = np.zeros((len(dtValues), len(epsValues)))
    for d, dt in enumerate(dtValues):
        for e, eps in enumerate(epsValues):
            print(f"{dt=}, {eps=}")
            # initialize level parameters
            level_params = {
                'dt': dt,
            }

            problem_params = {
                'lintol': 1e-14,
                'eps': eps,
            }

            # initialize sweeper parameters
            sweeper_params = {
                'quad_type': quad_type,
                'num_nodes': M,
                'QI': QI,
                'initial_guess': 'spread',
            }

            step_params = {
                'maxiter': 1,
            }

            # fill description dictionary for easy step instantiation
            description = {
                'problem_class': problem,
                'problem_params': problem_params,
                'sweeper_class': sweeper,
                'sweeper_params': sweeper_params,
                'level_params': level_params,
                'step_params': step_params,
            }

            S = Step(description=description)

            L = S.levels[0]
            P = L.prob

            L.status.time = 0.0
            u0 = P.u_exact(L.status.time)
            S.init_step(u0)

            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]

            LHS, RHS = getSweeperMatrix(M, dt, QImat, Q, P.A)

            sysMatrix = np.kron(np.identity(M), np.identity(P.A.shape[0])) - dt * np.kron(Q, P.A)

            cond[d, e] = np.linalg.norm(sysMatrix, np.inf) * np.linalg.norm(np.linalg.inv(sysMatrix), np.inf)

            K = np.linalg.inv(LHS).dot(RHS)
            lambdas = np.linalg.eigvals(K)
            sR = np.linalg.norm(lambdas, np.inf)
            spectralRadius[d, e] = sR

            ratio[d, e] = dt / eps

    plotQuantity(
        dtValues,
        epsValues,
        spectralRadius,
        epsValues,
        (0.0, 0.6),
        'Spectral radius',
        description['problem_class'].__name__,
        f'plotSpectralRadiusPerturbation_{QI=}_{M=}.png',
    )

    plotQuantity(
        dtValues,
        epsValues,
        cond,
        None,
        None,
        r'Condition number $\kappa(I-\Delta t QA)$',
        description['problem_class'].__name__,
        f'plotConditionPerturbation_{QI=}_{M=}.png',
        'loglog',
    )

    plotQuantity(
        dtValues,
        epsValues,
        ratio,
        epsValues,
        None,
        r'Ratio $\frac{\Delta t}{\varepsilon}$',
        description['problem_class'].__name__,
        f'plotRatioPerturbation_{QI=}_{M=}.png',
        'loglog',
    )


def plotQuantity(dtValues, epsValues, quantity, xLim, yLim, quantity_label, prob_cls_name, file_name, plot_type='semilogx'):
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for d, dt in enumerate(dtValues):
        if plot_type == 'semilogx':
            ax.semilogx(epsValues, quantity[d, :], label=rf"$\Delta t=${dt}")
        elif plot_type == 'loglog':
            ax.loglog(epsValues, quantity[d, :], label=rf"$\Delta t=${dt}")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=20)
    if plot_type == 'loglog':
        ax.set_yscale('log', base=10)
    # if xLim is not None:
    #     ax.set_xlim(xLim[0], xLim[-1])

    #     powersOfTen = generate_powers_of_ten(xLim)
    #     ax.set_xticks(powersOfTen)
    #     ax.set_xticklabels([i for i in powersOfTen])

    if yLim is not None:
        ax.set_ylim(yLim[0], yLim[-1])

    ax.minorticks_off()

    ax.set_ylabel(quantity_label, fontsize=20)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{prob_cls_name}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
