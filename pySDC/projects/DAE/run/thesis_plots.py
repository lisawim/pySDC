from mpi4py import MPI

from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.misc.configurations import get_configs


def make_plots_for_chapter_application(journal="BUW_thesis"):
    from pySDC.projects.PinTSimE.pwm_signal import plot_pwm
    from pySDC.projects.DAE.run.solution import (
        plot_solution_piline,
        plot_solution_buck_converter,
        plot_solution_battery,
    )

    plot_pwm(filename="pwm_signal", journal=journal)
    plot_solution_piline(filename="piline_solution", journal=journal)
    plot_solution_buck_converter(filename="buck_converter_solution", journal=journal)
    plot_solution_battery(filename="battery_solution", journal=journal)

def make_plots_for_chapter_test_problems(journal="BUW_thesis"):
    from pySDC.projects.DAE.run.solution import (
        plot_solution_linear,
        plot_solution_linear_embedded,
        plot_solution_andrews,
        plot_solution_reaction_diffusion,
        plot_solution_discontinuous_test,
        plot_solution_wscc9,
    )

    plot_solution_linear(filename="linear_test_solution", journal=journal)
    plot_solution_linear_embedded(filename="linear_test_embedded_solution", journal=journal)
    plot_solution_andrews(filename="andrews_squeezer_solution", journal=journal)
    plot_solution_reaction_diffusion(filename="reaction_diffusion_solution", journal=journal)
    plot_solution_discontinuous_test(filename="discontinuous_test_solution", journal=journal)
    plot_solution_wscc9(filename="wscc9_solution", journal=journal)


def make_plots_for_chapter_num_results(journal="BUW_thesis"):
    from pySDC.projects.DAE.plotting.spectral_radius import plot_spectral_radius_sdc_spp_and_sdc_e_and_sdc_c
    from pySDC.projects.DAE.plotting.singular_values import plot_svd_of_iteration_matrix_powers
    from pySDC.projects.DAE.run.study_embedding_linear import (
        convergence_plot_thesis, increment_plot_different_sweepers_thesis
    )
    from pySDC.projects.DAE.run.algebraic_parts import (
        absolute_values_g_thesis,
        dae_errors_thesis,
    )
    from pySDC.projects.DAE.run.plot_order_iteration import (
        plot_order_linear,
        plot_order_andrews,
        plot_order_reaction_diffusion,
    )
    from pySDC.projects.DAE.run.plots_work_prec import plots_work_vs_error
    from pySDC.projects.DAE.run.plots_scaling import plots_scaling
    from pySDC.projects.DAE.run.plots_speedup_at_accuracy import plots_speedup_at_accuracy

    num_nodes = 4

    dt_list, _ = choose_time_step_sizes("LINEAR-TEST")
    dt = dt_list[3]

    # Section 6.1
    plot_spectral_radius_sdc_spp_and_sdc_e_and_sdc_c(journal=journal)
    plot_svd_of_iteration_matrix_powers(num_nodes=num_nodes, journal=journal)
    convergence_plot_thesis(dt=dt, num_nodes=num_nodes, along="iterations", journal=journal)
    increment_plot_different_sweepers_thesis(dt=dt, num_nodes=num_nodes, journal=journal)

    # Section 6.2
    problem_name2 = "ANDREWS-SQUEEZER"
    dt_list_andrews, _ = choose_time_step_sizes(problem_name2)
    dt_andrews = dt_list_andrews[0]

    absolute_values_g_thesis(dt=dt_andrews, num_nodes=num_nodes, problem_name=problem_name2, journal=journal)
    dae_errors_thesis(dt=dt_andrews, num_nodes=num_nodes, problem_name=problem_name2, journal=journal)

    # Section 6.3
    plot_order_linear(journal=journal)
    plot_order_linear(sweeper_type="semiImplicitDAE", journal=journal)

    config_linear_work_prec = get_configs(problem_name="LINEAR-TEST", config_type="work_precision")
    filename = "results_experiment_6_linear_thesis.pkl"
    plots_work_vs_error(filename=filename, journal=journal, **config_linear_work_prec)

    config_linear_scaling = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    filename = "results_scaling_dt=0.05_linear_thesis.pkl"
    plots_scaling(
        global_comm=MPI.COMM_WORLD, filename=filename, journal=journal, **config_linear_scaling
    )
    
    # Plots for LINEAR-TEST
    print("\nGenerating speedup plots for LINEAR-TEST...\n")
    config_linear_speedup_acc = get_configs(problem_name="LINEAR-TEST", config_type="speedup_at_accuracy")
    filename = "results_speedup_at_acc_dt=0.05_linear_thesis.pkl"
    plots_speedup_at_accuracy(
        global_comm=MPI.COMM_WORLD, filename=filename, journal=journal, **config_linear_speedup_acc
    )

    # Section 6.4
    plot_order_andrews(journal=journal)
    plot_order_andrews(sweeper_type="semiImplicitDAE", journal=journal)

    config_andrews_work_prec = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="work_precision")
    filename = "results_experiment_6_andrews_thesis.pkl"
    plots_work_vs_error(filename=filename, journal=journal, **config_andrews_work_prec)

    config_andrews_scaling = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    filename = "results_scaling_dt=0.001_andrews_thesis.pkl"
    nodes_to_plot = range(2, 17)
    plots_scaling(
        global_comm=MPI.COMM_WORLD,
        filename=filename,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        **config_andrews_scaling,
    )

    # Section 6.5
    plot_order_reaction_diffusion(journal=journal)
    plot_order_reaction_diffusion(sweeper_type="semiImplicitDAE", journal=journal)
    plot_order_reaction_diffusion(sweeper_type="imexConstrainedDAE", journal=journal)

    problem_name3 = "REACTION-DIFFUSION"
    dt_list_reacdiff, _ = choose_time_step_sizes(problem_name3)
    dt_reacdiff = dt_list_reacdiff[2]

    absolute_values_g_thesis(dt=dt_reacdiff, num_nodes=num_nodes, problem_name=problem_name3, journal=journal)
    dae_errors_thesis(dt=dt_reacdiff, num_nodes=num_nodes, problem_name=problem_name3, journal=journal)

    config_reacdiff_work_prec = get_configs(problem_name="REACTION-DIFFUSION", config_type="work_precision")
    filename = "results_experiment_6_reaction_diffusion_thesis.pkl"
    plots_work_vs_error(
        include_dopri=False,
        qDelta_best=["LU", "MIN-SR-S"],
        filename=filename,
        journal=journal,
        **config_reacdiff_work_prec,
    )

    config_reacdiff_scaling = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")
    filename = "results_scaling_dt=0.05_reaction_diffusion_thesis.pkl"
    plots_scaling(
        global_comm=MPI.COMM_WORLD, filename=filename, journal=journal, **config_reacdiff_scaling
    )

    config_reacdiff_speedup_acc = get_configs(problem_name="REACTION-DIFFUSION", config_type="speedup_at_accuracy")
    filename = "results_speedup_at_acc_dt=0.05_reaction_diffusion_thesis.pkl"
    plots_speedup_at_accuracy(
        global_comm=MPI.COMM_WORLD, filename=filename, journal=journal, **config_reacdiff_speedup_acc
    )


def make_plots_for_chapter_num_results_SE(journal="BUW_thesis"):
    from pySDC.projects.PinTSimE.paper_PSCC2024.paper_plots import make_plots_for_test_DAE, make_plots_for_WSCC9_test_case

    make_plots_for_test_DAE(journal=journal)
    make_plots_for_WSCC9_test_case(journal=journal)


if __name__ == "__main__":
    # make_plots_for_chapter_application()
    # make_plots_for_chapter_test_problems()
    # make_plots_for_chapter_num_results()
    make_plots_for_chapter_num_results_SE()
