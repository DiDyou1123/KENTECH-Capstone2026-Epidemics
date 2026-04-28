"""
Metapopulation SIR model simulator module
"""

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse._csr import csr_array
from scipy.integrate import solve_ivp

from collections import deque
from pdb import set_trace

arr64 = npt.NDArray[np.float64]


class MetapopulationSIRSolver:
    """
    Matapopulation SIR model simulator
    """

    def __init__(
        self,
        graph: nx.DiGraph,  # Mobility directional network with population node attributes and mobilityedge attributes
        pop_attr: str = "population",  # Population attribute name
        mob_attr: str = "weight",  # Population attribute name
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

    def terminal_simulation(
        self,
        basic_rep: float,  # R0 = beta/mu
        recovery_time: float,  # T = 1/mu
        init_node: int,  # Node index of initial infection
        init_i_pop: np.float64,  # Initial infected population
        time_min: np.float64,  # Minimum solve time
        time_max: np.float64,  # Maximum solve time
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
        r_rate = 1 / recovery_time  # Removal rate mu

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
        basic_rep: float,  # R0 = beta/mu
        recovery_time: float,  # T = 1/mu
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
        r_rate = 1 / recovery_time  # Removal rate mu

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
