from mpi4py import MPI

from pySDC.projects.DAE.misc.configurations import get_configs


def make_scaling_plots(global_comm, journal="SIAM_Scientific_Computing"):
    from pySDC.projects.DAE.run.plots_scaling_new import plots_scaling

    # Plots for LINEAR-TEST
    print("\nGenerating scaling plots for LINEAR-TEST...\n")
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    filename = "results_scaling_dt=0.05_linear_#7.pkl"
    plots_scaling(
        global_comm=global_comm, format="png", filename=filename, journal=journal, ref_QI="IE", **config_linear
    )

    # Plots for ANDREWS-SQUEEZER
    print("\nGenerating scaling plots for ANDREWS-SQUEEZER...\n")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    filename = "results_scaling_dt=0.001_andrews_#9.pkl"
    nodes_to_plot = range(2, 17)
    plots_scaling(
        global_comm=global_comm,
        format="png",
        filename=filename,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        ref_QI="EE",
        **config_andrews,
    )

    # Plots for REACTION-DIFFUSION
    print("\nGenerating scaling plots for REACTION-DIFFUSION...\n")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")
    filename = "results_scaling_dt=0.05_reaction_diffusion_#14.pkl"
    plots_scaling(
        global_comm=global_comm, format="png", filename=filename, journal=journal, ref_QI="IE", **config_reacdiff
    )


def make_speedup_at_accuracy_plots(global_comm, journal="SIAM_Scientific_Computing"):
    from pySDC.projects.DAE.run.plots_speedup_at_accuracy import plots_speedup_at_accuracy
    
    # Plots for LINEAR-TEST
    print("\nGenerating speedup plots for LINEAR-TEST...\n")
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="speedup_at_accuracy")
    filename = "results_speedup_at_acc_dt=0.05_linear_#5.pkl"
    plots_speedup_at_accuracy(
        global_comm=global_comm, filename=filename, journal=journal, ref_QI="IE", **config_linear
    )

    print("\nGenerating speedup plots for REACTION-DIFFUSION...\n")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="speedup_at_accuracy")
    # filename = "results_speedup_at_acc_dt=0.05_reaction_diffusion_#2.pkl"
    filename = "results_speedup_at_acc_dt=0.05_reaction_diffusion_#12.pkl"
    plots_speedup_at_accuracy(
        global_comm=global_comm, filename=filename, journal=journal, ref_QI="IE", **config_reacdiff
    )


if __name__ == "__main__":
    """
    Generates plots for paper 'Spectral deferred corrections parallelized
    across the method for differential-algebraic equations'.
    """

    global_comm = MPI.COMM_WORLD

    make_scaling_plots(global_comm=global_comm)  # Figures 2, 3, 5, 6, 7, 8, 9

    make_speedup_at_accuracy_plots(global_comm=global_comm)  # Figures 4, 10
