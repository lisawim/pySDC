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

        coll_nodes = L.sweep.coll.nodes
        nodes = [L.time + L.dt * coll_nodes[m] for m in range(len(coll_nodes))]
        nodes = np.append([L.time], nodes) if not L.sweep.coll.left_is_node else nodes
        u = L.u if not L.sweep.coll.left_is_node else L.u[1:]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u_dense',
            value=u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='nodes_dense',
            value=nodes,
        )