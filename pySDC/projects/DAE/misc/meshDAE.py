from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh, imex_mesh


class MeshDAE(MultiComponentMesh):
    r"""
    Datatype for DAE problems. The solution of the problem can be splitted in the differential part
    and in an algebraic part.

    This data type can be used for the solution of the problem itself as well as for its derivative.
    """

    components = ['diff', 'alg']


class imex_dae_mesh(MultiComponentMesh):
    r"""
    Datatype for IMEX DAE problems.

    The differential part is split into an implicit and an explicit
    contribution, while the algebraic part is stored separately.

    Internally, the storage is flat:

        u.diff_impl
        u.diff_expl
        u.alg

    For convenience, the differential part can also be accessed as

        u.diff.impl
        u.diff.expl
    """

    components = ["diff_impl", "diff_expl", "alg"]

    @property
    def diff(self):
        r"""
        Differential part of the IMEX DAE.

        Returns a view with components

            diff.impl
            diff.expl

        corresponding to

            self.diff_impl
            self.diff_expl
        """
        return self[:2].view(imex_mesh)
