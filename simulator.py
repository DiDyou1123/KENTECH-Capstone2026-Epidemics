"""
Metapopulation SIR model simulator module
"""

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp

from cut_graph import EasyCutGraph

from collections import deque
from pdb import set_trace

arr64 = npt.NDArray[np.float64]


class BlockJacobian:
    """
    Fast jacobian builder, written by Claude
    """

    def __init__(
        self,
        N: int,  # Half the degree of freedom
        S_pattern: sps.csr_array,  # Adjacency matrix sparsity pattern
    ):
        """
        S1_pattern, S2_pattern: sparse (N,N) with the *structural* nonzeros
        of S1, S2 (values irrelevant; only pattern matters; diagonal must
        be included so updates to d1/d2 land in existing slots).
        """
        self.N = N
        I = sps.eye(N, format="coo")

        # Force diagonal into each block's pattern
        S1f = (S_pattern + I).tocoo()
        S2f = (S_pattern + I).tocoo()
        D12 = I.copy()
        D21 = I.copy()

        # Tag each block's entries with row/col offsets and a "source id"
        rows = np.concatenate([S1f.row, S2f.row + N, D12.row, D21.row + N])
        cols = np.concatenate([S1f.col, S2f.col + N, D12.col + N, D21.col])
        # placeholder data, unique so we can locate each entry after CSR sums duplicates?
        # Better: ensure no duplicates by construction (pattern of S1 has diag, so
        # there are no duplicates with the I we added — use sum_duplicates safely below).
        data = np.zeros(rows.size)

        J = sps.coo_matrix((data, (rows, cols)), shape=(2 * N, 2 * N)).tocsr()
        J.sort_indices()
        self.J = J

        # Build index maps: for each (r, c) we want, find its position in J.data
        def locate(r, c):
            # vectorized lookup in CSR
            r = np.asarray(r)
            c = np.asarray(c)
            out = np.empty(r.size, dtype=np.intp)
            for k in range(r.size):
                start, end = J.indptr[r[k]], J.indptr[r[k] + 1]
                idx = start + np.searchsorted(J.indices[start:end], c[k])
                out[k] = idx
            return out

        idx = np.arange(N)
        self.idx_d1 = locate(idx, idx)  # A11 diagonal
        self.idx_d2 = locate(idx + N, idx + N)  # A22 diagonal
        self.idx_d12 = locate(idx, idx + N)  # A12 diagonal
        self.idx_d21 = locate(idx + N, idx)  # A21 diagonal
        self.idx_S1 = locate(S_pattern.tocoo().row, S_pattern.tocoo().col)
        self.idx_S2 = locate(S_pattern.tocoo().row + N, S_pattern.tocoo().col + N)

    def update(
        self,
        d1: arr64,
        d2: arr64,
        d12: arr64,
        d21: arr64,
        S_data: arr64,
    ) -> sps.csr_array:
        """S1_data / S2_data: values aligned with the COO order of the
        pattern you passed in __init__ (same row/col arrays)."""
        D = self.J.data
        D[:] = 0.0
        # off-diagonal blocks
        D[self.idx_d12] = d12
        D[self.idx_d21] = d21
        # on-diagonal blocks: sparse part + diagonal part (added, not overwritten)

        D[self.idx_S1] += S_data
        D[self.idx_S2] += S_data
        D[self.idx_d1] += d1
        D[self.idx_d2] += d2

        return self.J


class MetapopulationSIRSolver:
    """
    Matapopulation SIR model simulator
    """

    def __init__(
        self,
        graph: (
            nx.DiGraph | nx.Graph | EasyCutGraph
        ),  # Mobility directional network with population node attributes and mobility edge attributes
        pop_attr: str = "population",  # Population attribute name
        mob_attr: str = "mobility",  # Population attribute name
        tol: float = 1e-11,  # Internal tolerance
    ):
        # Setup network
        self.num_nodes = graph.number_of_nodes()
        self.adj_mat = nx.adjacency_matrix(graph, weight=mob_attr, dtype=np.float64)
        self.total_pops = np.array(
            list(nx.get_node_attributes(graph, pop_attr).values())
        )

        # Exceptions
        self.tol = tol
        asymmetry = abs(self.adj_mat.sum(axis=0) - self.adj_mat.sum(axis=1)) > self.tol
        if np.any(asymmetry):
            raise ValueError(
                f"Mobility is not balanced for nodes {list(np.where(asymmetry)[0])}"
            )

        overflow = self.adj_mat.sum(axis=1) > self.total_pops
        if np.any(overflow):
            raise ValueError(
                f"Mobility outflow exceeds population for nodes {list(np.where(overflow)[0])}"
            )

        self.jac = BlockJacobian(self.num_nodes, self.adj_mat)

    def get_velocity(
        self,
        time: float,
        pop_fracs: arr64,  # Population compartments fractions array
        i_rate: float,
        r_rate: float,
    ) -> arr64:
        """
        SIR model equations
        """

        i_fracs = pop_fracs[: self.num_nodes]  # Infected population fraction
        s_fracs = pop_fracs[self.num_nodes :]  # Susceptible population fraction

        d_i_fracs = (
            i_rate * s_fracs * i_fracs
            - r_rate * i_fracs
            + (self.adj_mat.T.dot(i_fracs) - self.adj_mat.sum(axis=1) * i_fracs)
            / self.total_pops
        )
        d_s_fracs = (
            -i_rate * s_fracs * i_fracs
            + (self.adj_mat.T.dot(s_fracs) - self.adj_mat.sum(axis=1) * s_fracs)
            / self.total_pops
        )

        return np.concatenate((d_i_fracs, d_s_fracs), axis=0)

    def get_jacobian(
        self,
        time: float,
        pop_fracs: arr64,  # Population compartments fractions array
        i_rate: float,
        r_rate: float,
    ) -> sps.csr_array:

        i_fracs = pop_fracs[: self.num_nodes]  # Infected population fraction
        s_fracs = pop_fracs[self.num_nodes :]  # Susceptible population fraction

        d1 = i_rate * s_fracs - r_rate - self.adj_mat.sum(axis=1) / self.total_pops
        d2 = -i_rate * i_fracs - self.adj_mat.sum(axis=1) / self.total_pops
        d12 = i_rate * i_fracs
        d21 = -i_rate * s_fracs

        S = self.adj_mat.T / self.total_pops.reshape(-1, 1)

        return self.jac.update(d1, d2, d12, d21, S.data)

    def terminal_simulation(
        self,
        basic_rep: float,  # R0 = beta/gamma
        recovery_time: float,  # T = 1/gamma
        init_node: int,  # Node index of initial infection
        init_i_pop: float,  # Initial infected population
        time_min: float,  # Minimum solve time
        time_max: float,  # Maximum solve time
        ttol: float | None = None,
        # Termination tolerance, fraction of global total population
        t_window: float | None = None,  # Time window to maintain termination tolerance
        method: str = "DOP853",  # Solver method
        rtol: float | None = None,  # Solver relative tolerance
        atol: float | None = None,  # Solver absolute tolerance
        **kwargs,  # Other solve kwargs
    ) -> dict:
        """
        Run SIR model simulation by solving equations numerically
        Solve until infected population does not change
        """

        # Setup epidemics
        i_rate = basic_rep / recovery_time  # Infection rate beta
        r_rate = 1 / recovery_time  # Removal rate gamma

        # Initial point
        init_i_fracs = np.zeros(self.num_nodes, dtype=np.float64)
        init_i_fracs[init_node] = init_i_pop / self.total_pops[init_node]
        init_s_fracs = np.ones(self.num_nodes, dtype=np.float64) - init_i_fracs

        # Tolerances
        if ttol == None:
            ttol = self.tol
        if rtol == None:
            rtol = self.tol
        if atol == None:
            atol = self.tol

        # Termination condition
        if t_window == None:
            t_window = 2 / i_rate  # Default t_window is two times recovery time

        history = deque(
            [(0.0, init_i_fracs.copy() * self.total_pops / self.total_pops.sum())]
        )

        def termination(
            time: float,
            pop_fracs: arr64,
            i_rate: float,
            r_rate: float,
        ) -> float:

            if time < time_min:
                return 1.0  # keep solving

            i_norm_fracs = (
                pop_fracs[: self.num_nodes].copy()
                * self.total_pops
                / self.total_pops.sum()
            )
            history.append((time, i_norm_fracs))

            # drop old samples outside window
            while len(history) >= 2 and history[1][0] < time - t_window:
                history.popleft()

            # need at least 2 points to compare change
            if len(history) < 2:
                return 1.0

            i_norm_frac_arr = np.stack(
                [x[1] for x in history], axis=0
            )  # shape (m, num_nodes)
            d_i_norm_frac = np.abs(np.diff(i_norm_frac_arr, axis=0)) > ttol

            # event must return float; terminate when no changes exceed ttol
            return np.sum(d_i_norm_frac, dtype=float)

        termination.terminal = True

        # Solve
        result = solve_ivp(
            self.get_velocity,
            (0, time_max),
            np.concatenate((init_i_fracs, init_s_fracs), axis=0),
            args=(i_rate, r_rate),
            events=termination,
            method=method,
            jac=self.get_jacobian if method in ["Radau", "BDF"] else None,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

        # Add populations to result
        result["I"] = result["y"][: self.num_nodes] * self.total_pops[:, None]
        result["S"] = result["y"][self.num_nodes :] * self.total_pops[:, None]
        result["R"] = (
            1 - result["y"][: self.num_nodes] - result["y"][self.num_nodes :]
        ) * self.total_pops[:, None]

        return result

    def unit_time_simulation(
        self,
        basic_rep: float,  # R0 = beta/gamma
        recovery_time: float,  # T = 1/gamma
        init_node: int,  # Node index of initial infection
        init_i_pop: np.float64,  # Initial infected population
        time_max: np.float64,  # Maximum solve time
        eval_rate: np.float64,  # Unit time evaluation frequency
        method: str = "DOP853",  # Solve method
        **kwargs,  # Other solve kwargs
    ) -> dict:
        """
        Run SIR model simulation by solving equations numerically
        """

        # Setup epidemics
        i_rate = basic_rep / recovery_time  # Infection rate beta
        r_rate = 1.0 / recovery_time  # Removal rate gamma

        # Initial point
        init_i_fracs = np.zeros(self.num_nodes, dtype=np.float64)
        init_i_fracs[init_node] = init_i_pop / self.total_pops[init_node]
        init_s_fracs = np.ones(self.num_nodes, dtype=np.float64) - init_i_fracs

        # Evaluation times
        eval_time = np.arange(0, eval_rate * time_max) / eval_rate

        # Solve
        result = solve_ivp(
            self.get_velocity,
            (0, time_max),
            np.concatenate((init_i_fracs, init_s_fracs), axis=0),
            args=(i_rate, r_rate),
            t_eval=eval_time,
            method=method,
            **kwargs,
        )

        # Add populations to result
        result["I"] = result["y"][: self.num_nodes] * self.total_pops[:, None]
        result["S"] = result["y"][self.num_nodes :] * self.total_pops[:, None]
        result["R"] = (
            1 - result["y"][: self.num_nodes] - result["y"][self.num_nodes :]
        ) * self.total_pops[:, None]

        return result

    def get_eff_rep(
        self,
        basic_rep: float,  # R0 = beta/gamma
        recovery_time: float,  # T = 1/gamma
        sparse: bool = False,  # Acceleration using sparse matrix tricks
    ) -> float:
        """
        Calculate global effective reproduction number
        """

        i_rate = basic_rep / recovery_time  # Infection rate beta
        r_rate = 1.0 / recovery_time  # Removal rate gamma

        if not sparse:
            in_jac = i_rate * np.identity(self.num_nodes) + self.adj_mat.T / np.reshape(
                self.total_pops, (-1, 1)
            )
            out_jac_inv = np.diag(
                (r_rate + self.adj_mat.sum(axis=1) / self.total_pops) ** (-1)
            )

            eigvals = np.linalg.eigvals(in_jac.dot(out_jac_inv))

            return np.max(np.abs(eigvals))
        else:  # Accelerate version of code above by Claude
            pops = np.asarray(self.total_pops).ravel()

            # in_jac = c*I + diag(1/pops) @ A.T     (A.T / pops[:,None] == left-scale rows of A.T)
            inv_pops = sps.diags(1.0 / pops)
            in_jac = (
                i_rate * sps.eye(self.num_nodes, format="csr")
                + inv_pops @ self.adj_mat.T
            )

            # out_jac_inv = diag( 1 / (1/recovery_time + rowsum(A)/pops) )
            row_sums = np.asarray(self.adj_mat.sum(axis=1)).ravel()
            d = 1.0 / (r_rate + row_sums / pops)
            out_jac_inv = sps.diags(d)

            next_gen_mat = in_jac @ out_jac_inv

            vals = eigs(
                next_gen_mat,
                k=1,
                which="LM",
                return_eigenvectors=False,
                tol=self.tol,  # No idea why this is marked
            )
            return float(np.abs(vals[0]))
