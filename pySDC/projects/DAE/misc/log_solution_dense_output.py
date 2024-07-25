import numpy as np
from pySDC.core.hooks import Hooks

class LogSolutionDenseOutput(Hooks):
    def post_step(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.level
            Current level number.
        """

        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        nodes = np.append([L.time], L.sweep.coll.nodes) if not L.sweep.coll.left_is_node else L.sweep.coll.nodes
        u = L.u if not L.sweep.coll.left_is_node else L.u[1:]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='log_solution_dense_output',
            value=u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='log_nodes_dense_output',
            value=nodes,
        )