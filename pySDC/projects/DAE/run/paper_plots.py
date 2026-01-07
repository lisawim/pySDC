from pySDC.projects.DAE.plotting.plot_spectral_radius import plot_spectral_radius_and_evd
from pySDC.projects.DAE.run.plot_order_iteration import (
    choose_time_step_sizes,
    plot_order_linear,
    plot_order_andrews,
    plot_order_reaction_diffusion,
)
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.plots_work_prec import plots_work_vs_error
from pySDC.projects.DAE.run.plot_error import plot_algebraic_error_vs_iteration
from pySDC.projects.DAE.run.plot_error_manifold import plot_manifold_value_vs_iteration


def make_work_precision_plots(format="png"):
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="work_precision")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="work_precision")
    config_reac_diff = get_configs(problem_name="REACTION-DIFFUSION", config_type="work_precision")

    plots_work_vs_error(metric_key="all_max_global_error", format=format, **config_linear)

    plots_work_vs_error(metric_key="q_max_final_error", format=format, **config_andrews)

    qDelta_best = ["LU", "MIN-SR-S"]
    include_dopri = False
    plots_work_vs_error(
        metric_key="all_max_global_error",
        format=format,
        qDelta_best=qDelta_best,
        include_dopri=include_dopri,
        **config_reac_diff,
    )

if __name__ == "__main__":
    format = "png"
    plot_spectral_radius_and_evd(format=format)  # Figure 2
    plot_order_linear(format=format) # Figure 3
    make_work_precision_plots(format=format)  # Figures 4, 5, 8, 9, 11, 12
    plot_order_andrews(format=format) # Figure 6
    plot_algebraic_error_vs_iteration(1e-3, 6, problem_name="ANDREWS-SQUEEZER", format="png")  # Figure 7
    plot_order_reaction_diffusion(format=format) # Figure 10

    dt_list, _ = choose_time_step_sizes(problem_name=problem_name)
    plot_manifold_value_vs_iteration(dt=dt_list[0], num_nodes=6, problem_name="REACTION-DIFFUSION")  # Figure 13
