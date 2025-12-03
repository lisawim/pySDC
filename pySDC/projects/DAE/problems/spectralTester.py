import numpy as np
import scipy.sparse as sp
from numpy.fft import rfft, irfft

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class SpectralTester(ProblemDAE):
    """
    This class is intended to test the reaction-diffusion PDAE (for the first time), but
    I want to extend it to general problems.
    """

    def __init__(self, nvars, Nr, newton_tol, comm=None):
        super().__init__(nvars=2*nvars, newton_tol=newton_tol)

        self._makeAttributeAndRegister(
            "newton_tol",
            "nvars",
            "Nr",
            localVars=locals(),
        )

        self.spectral_init = ((2 * self.Nr,), comm, np.dtype('complex128'))
        self.physical_init = ((2 * self.nvars,), comm, np.dtype('float64'))

    def block_slice(self, matrix, block):
        if block == "11":
            return matrix[: self.Nr, : self.Nr]
        elif block == "12":
            return matrix[: self.Nr, self.Nr : 2 * self.Nr]
        elif block == "13":
            return matrix[: self.Nr, 2 * self.Nr : 3 * self.Nr]
        elif block == "21":
            return matrix[self.Nr : 2 * self.Nr, : self.Nr]
        elif block == "22":
            return matrix[self.Nr : 2 * self.Nr, self.Nr : 2 * self.Nr]
        elif block == "23":
            return matrix[self.Nr : 2 * self.Nr, 2 * self.Nr : 3 * self.Nr]
        elif block == "31":
            return matrix[2 * self.Nr : 3 * self.Nr, : self.Nr]
        elif block == "32":
            return matrix[2 * self.Nr : 3 * self.Nr, self.Nr : 2 * self.Nr]
        elif block == "33":
            return matrix[2 * self.Nr : 3 * self.Nr, 2 * self.Nr : 3 * self.Nr]

    def _approximate_jacobian_by_finite_differences(self, g_hat, factor, rhs_hat, t, u_hat, h=None):
        """
        Computes Jacobian via finite difference approximation.
        
        Parameters
        ----------
        x :
            Vector with values where the Jacobian is computed at.
        g : callable function
            Function to compute the Jacobian for.
        """
        m = 3 * self.Nr

        # g0_hat = np.asarray(g_hat(factor, rhs_hat, t, u_hat))

        u0_hat = np.asarray(u_hat).flatten()

        if h is None:
            # base = np.sqrt(np.finfo(np.float64).eps)  # gut für forward FD
            base = np.sqrt(np.finfo(np.float64).eps)  # gut für central
            h_vec = base * np.maximum(1.0, np.abs(u0_hat))
        else:
            h_vec = np.full(m, float(h), dtype=float)

        e = self.dtype_u(self.spectral_init)
        e_flat = e.flatten()
        e_flat[:] = 0.0
        jac_cols = []
        for j in range(m):
            e_flat[j] = 1.0
            ej = e_flat.reshape(e.shape).view(type(e))

            gp_hat = g_hat(factor, rhs_hat, t, u_hat + h_vec[j] * ej)
            gm_hat = g_hat(factor, rhs_hat, t, u_hat - h_vec[j] * ej)
            col = (np.asarray(gp_hat) - np.asarray(gm_hat)) / (2.0 * h_vec[j])
            # col = (np.asarray(gp_hat) - g0_hat) / h_vec[j]

            jac_cols.append(col)
            e_flat[j] = 0.0

        jac = np.column_stack(jac_cols).astype(np.complex128, copy=False)
        return jac
    
    def _approximate_jacobian_by_finite_differences_phys(self, g_func, factor, rhs, t, u, h=None):
        """
        Computes Jacobian via finite difference approximation.
        
        Parameters
        ----------
        x :
            Vector with values where the Jacobian is computed at.
        g : callable function
            Function to compute the Jacobian for.
        """
        m = 3 * self.nvars

        g0 = np.asarray(g_func(factor, rhs, t, u))

        u0 = np.asarray(u).flatten()

        if h is None:
            # base = np.sqrt(np.finfo(np.float64).eps)  # gut für forward FD
            base = np.sqrt(np.finfo(np.float64).eps)  # gut für central
            h_vec = base * np.maximum(1.0, np.abs(u0))
        else:
            h_vec = np.full(m, float(h), dtype=float)

        e = self.dtype_u(self.physical_init)
        e_flat = e.flatten()
        e_flat[:] = 0.0
        jac_cols = []
        for j in range(m):
            e_flat[j] = 1.0
            ej = e_flat.reshape(e.shape).view(type(e))

            gp = g_func(factor, rhs, t, u + h_vec[j] * ej)
            gm = g_func(factor, rhs, t, u - h_vec[j] * ej)
            col = (np.asarray(gp) - np.asarray(gm)) / (2.0 * h_vec[j])
            # col = (np.asarray(gp_hat) - g0_hat) / h_vec[j]

            jac_cols.append(col)
            e_flat[j] = 0.0

        jac = np.column_stack(jac_cols).astype(np.complex128, copy=False)
        return jac

    def _check_jacobian(self, dg_hat, g_hat, factor, rhs_hat, t, u_hat):
        jac_ref = self._approximate_jacobian_by_finite_differences(g_hat, factor, rhs_hat, t, u_hat)

        # norm_diff_jac = np.linalg.norm(dg_hat - jac_ref, ord='fro')
        # print(f"Norm difference of Jacobians: {norm_diff_jac}")

        self._check_blocks(dg_hat, jac_ref)

    def _check_blocks(self, dg_hat, jac_ref):
        for block in ["11", "12", "13", "21", "22", "23", "31", "32", "33"]:
            diff_block = self.block_slice(dg_hat, block) - self.block_slice(jac_ref, block)
            norm_diff_block = np.linalg.norm(diff_block, np.inf)
            print(f"Difference of block {block}: {norm_diff_block}")

        for block in ["33"]:#, "32", "33"]:
            block_matrix_dg = self.block_slice(dg_hat, block)
            diag_dg = np.diag(block_matrix_dg)
            block_matrix_ref = self.block_slice(jac_ref, block)
            diag_ref = np.diag(block_matrix_ref)
            # print(f"{diag_dg=}")
            # print(f"{diag_ref=}")
            # print()
        print()

    def get_expected_block(self, factor, name):
        if name == "J33":
            if type(self).__name__ in [
                "SemiImplicitReactionDiffusionPDAE_FFT", "ReactionDiffusionPDAE_FFT_Constrained"
            ]:
                J = -np.diag(self.Lx)
            elif type(self).__name__ == "ReactionDiffusionPDAE_FFT":
                J = -factor * np.diag(self.Lx)

            return J[np.ix_(self.mask_g3, self.mask_w)]
        elif name in ["J31", "J32"]:
            if type(self).__name__ in [
                "ReactionDiffusionPDAE_FFT", "SemiImplicitReactionDiffusionPDAE_FFT"
            ]:
                J = -factor * np.eye(self.Nr, dtype=complex)
            elif type(self).__name__ == "ReactionDiffusionPDAE_FFT_Constrained":
                J = -np.eye(self.Nr, dtype=complex)

            return J[self.mask_g3, :]

    def compare_blocks_with_expected_output(self, factor, J_ana_red):
        sl = self.block_slices_reduced()

        print("\nBlockweiser Vergleich zu erwartbaren Block (abs. Maximum):")
        for name, (r, c) in sl.items():
            if name in ["J31", "J32", "J33"]:
                J_block_expected = self.get_expected_block(factor, name)
                A = J_ana_red[r, c]
                diff = A - J_block_expected
                num = self.frob_sparse(diff)
                den = max(self.frob_sparse(A), self.frob_sparse(J_block_expected), 1e-30)
                # print(f"  {name}: {num/den:.3e}")
                print(f"  {name}: {num:.3e}")

    # ------------------------------- CHATGPT suggested routines ------------------------------------- #
    # 
    #
    # ------------------------------------------------------------------------------------------------ #
    def block_slices_reduced(self):
        r1 = slice(0, self.Nr)
        r2 = slice(self.Nr, 2 * self.Nr)
        r3 = slice(2 * self.Nr, 3 * self.Nr - 1)  # reduziert (DC in g3 entfernt)
        return {
            "J11": (r1, r1), "J12": (r1, r2), "J13": (r1, r3),
            "J21": (r2, r1), "J22": (r2, r2), "J23": (r2, r3),
            "J31": (r3, r1), "J32": (r3, r2), "J33": (r3, r3),
        }

    def frob_sparse(self, A):
        if sp.isspmatrix(A):
            d = A.data
            return float(np.sqrt((d.conj()*d).sum().real))
        A = np.asarray(A)
        return float(np.linalg.norm(A, ord=np.inf) )

    def fd_jacobian_full_from_g_hat(self, factor, g_hat, rhs_hat, t, u_hat_, h=1e-8):
        """
        Zentrale FD-Jacobi der *vollen* Abbildung g_hat an u_hat_.
        Gibt eine dichte (3Nr x 3Nr) Matrix zurück.
        """

        m  = 3 * self.Nr

        # Basiswert z0 = [û, v̂, ŵ] (alle Länge Nr), komplex
        z0 = np.concatenate([
            u_hat_.diff[: self.Nr], 
            u_hat_.diff[self.Nr : 2 * self.Nr], 
            u_hat_.alg[: self.Nr]
        ]).astype(np.complex128, copy=False)

        J = np.empty((m, m), dtype=np.complex128)

        # helper zum Bauen eines "pertubierten" dtype_u im rFFT-Raum
        def pack_to_state(z):
            uhat = z[: self.Nr]
            vhat = z[self.Nr : 2 * self.Nr]
            what = z[2 * self.Nr : 3 * self.Nr]

            st = self.dtype_u(self.spectral_init, val=0.0)
            st.diff[: self.Nr] = uhat
            st.diff[self.Nr : 2 * self.Nr] = vhat
            st.alg[: self.Nr] = what
            return st

        for k in range(m):
            e = np.zeros(m, dtype=np.complex128)
            e[k] = 1.0

            z_plus  = z0 + h*e
            z_minus = z0 - h*e

            up = pack_to_state(z_plus)
            um = pack_to_state(z_minus)

            g_plus  = g_hat(factor, rhs_hat, t, up)   # Länge 3Nr (voll)
            g_minus = g_hat(factor, rhs_hat, t, um)
            J[:, k] = (g_plus - g_minus) / (2 * h)

        return J
    
    def fd_jacobian_from_g_phys(self, factor, g_phys, rhs, t, u, h=1e-8):
        """
        Zentrale FD-Jacobi der *vollen* Abbildung g_hat an u_hat_.
        Gibt eine dichte (3Nr x 3Nr) Matrix zurück.
        """

        m  = 3 * self.nvars

        # Basiswert z0 = [û, v̂, ŵ] (alle Länge Nr), komplex
        z0 = np.concatenate([
            u.diff[: self.nvars], 
            u.diff[self.nvars :], 
            u.alg[: self.nvars]
        ]).astype(dtype=np.float64, copy=False)

        J = np.empty((m, m))

        # helper zum Bauen eines "pertubierten" dtype_u im rFFT-Raum
        def pack_to_state(z):
            u_ = z[: self.nvars]
            v_ = z[self.nvars : 2 * self.nvars]
            w_ = z[2 * self.nvars : 3 * self.nvars]

            st = self.dtype_u(self.physical_init, val=0.0)
            st.diff[: self.nvars] = u_
            st.diff[self.nvars :] = v_
            st.alg[: self.nvars] = w_
            return st

        for k in range(m):
            e = np.zeros(m)
            e[k] = 1.0

            z_plus  = z0 + h*e
            z_minus = z0 - h*e

            up = pack_to_state(z_plus)
            um = pack_to_state(z_minus)

            g_plus  = g_phys(factor, rhs, t, up)   # Länge 3*nvars (voll)
            g_minus = g_phys(factor, rhs, t, um)
            J[:, k] = (g_plus - g_minus) / (2 * h)

        return J

    def reduce_full_J(self, J_full):
        """
        Reduziert die volle (3Nr x 3Nr) Jacobi auf (3Nr-1 x 3Nr-1),
        indem in Block-3 die DC-Zeile (g3 bei k=0) und in den Variablen
        die w-DC-Spalte entfernt wird.
        Erwartet: self.mask_g3 (True außer k=0), self.mask_w (True außer k=0).
        """

        # Zeilen-Auswahl: g1 (alle), g2 (alle), g3 (mask_g3)
        row_idx = np.r_[
            np.arange(self.Nr),
            np.arange(self.Nr, 2 * self.Nr),
            2 * self.Nr + np.flatnonzero(self.mask_g3),
        ]

        # Spalten-Auswahl: û (alle), v̂ (alle), ŵ (mask_w)
        col_idx = np.r_[
            np.arange(self.Nr),
            np.arange(self.Nr, 2 * self.Nr),
            2 * self.Nr + np.flatnonzero(self.mask_w),
        ]

        J_red = J_full[np.ix_(row_idx, col_idx)]
        return np.array(J_red)

    def compare_blocks(self, J_ana_red, J_fd_red):
        sl = self.block_slices_reduced()

        print("\nBlockweiser Vergleich (abs. Maximum):")

        for name, (r, c) in sl.items():
            A = J_ana_red[r, c]
            B = J_fd_red[r, c]
            diff = A - B
            num = self.frob_sparse(diff)
            den = max(self.frob_sparse(A), self.frob_sparse(B), 1e-30)
            # print(f"  {name}: {num/den:.3e}")
            print(f"  {name}: {num:.3e}")
        num_tot = self.frob_sparse(J_ana_red - J_fd_red)
        den_tot = max(self.frob_sparse(J_ana_red), self.frob_sparse(J_fd_red), 1e-30)
        # print(f"\nGesamt: rel. Frobenius = {num_tot/den_tot:.3e}")
        print(f"\nGesamt: abs. Maximum = {num_tot:.3e}")

