from pySDC.core.hooks import Hooks


class LogGlobalError(Hooks):
    def log_global_error(self, step, level_number, attr, variable, suffix=""):
        """
        Function to add the global error to the stats

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level
            suffix (str): Suffix for naming the variable in stats

        Returns:
            None
        """
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        u_ex = L.prob.u_exact(t=L.time + L.dt)

        u_ex_part = getattr(u_ex, attr)
        u_num_part = getattr(L.uend, attr)

        e_global = abs(u_ex_part - u_num_part)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f"e_global_{variable}{suffix}",
            value=e_global,
        )


class LogGlobalErrorDiffVar(LogGlobalError):
    def pre_iteration(self, step, level_number):
        super().pre_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_pre_iteration")

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_iteration")

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_pre_sweep")

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_sweep")

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_step")


class LogGlobalErrorAlgVar(LogGlobalError):
    def pre_iteration(self, step, level_number):
        super().pre_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_pre_iteration")

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_iteration")

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_pre_sweep")

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_sweep")

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_step")
