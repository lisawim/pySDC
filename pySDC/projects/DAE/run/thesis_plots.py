def make_plots_for_chapter_application(journal="BUW_thesis"):
    from pySDC.projects.PinTSimE import plot_pwm
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


if __name__ == "__main__":
    make_plots_for_chapter_application()
    # make_plots_for_chapter_test_problems()
