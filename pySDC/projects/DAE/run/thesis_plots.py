from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes


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
    from pySDC.projects.DAE.run.plot_order_iteration import plot_order_linear, plot_order_andrews, plot_order_reaction_diffusion

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

    # Section 6.4
    plot_order_andrews(journal=journal)
    plot_order_andrews(sweeper_type="semiImplicitDAE", journal=journal)

    # Section 6.5
    plot_order_reaction_diffusion(journal=journal)
    plot_order_reaction_diffusion(sweeper_type="semiImplicitDAE", journal=journal)

    problem_name3 = "REACTION-DIFFUSION"
    dt_list_reacdiff, _ = choose_time_step_sizes(problem_name3)
    dt_reacdiff = dt_list_reacdiff[2]

    absolute_values_g_thesis(dt=dt_reacdiff, num_nodes=num_nodes, problem_name=problem_name3, journal=journal)
    dae_errors_thesis(dt=dt_reacdiff, num_nodes=num_nodes, problem_name=problem_name3, journal=journal)


def make_plots_for_chapter_num_results_SE(journal="BUW_thesis"):
    from pySDC.projects.PinTSimE.paper_PSCC2024.paper_plots import make_plots_for_test_DAE, make_plots_for_WSCC9_test_case

    # make_plots_for_test_DAE(journal=journal)
    make_plots_for_WSCC9_test_case(journal=journal)


if __name__ == "__main__":
    # make_plots_for_chapter_application()
    # make_plots_for_chapter_test_problems()
    make_plots_for_chapter_num_results()
    # make_plots_for_chapter_num_results_SE()
