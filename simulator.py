"""
Metapopulation SIR model simulator module
"""

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse._csr import csr_array
from scipy.integrate import solve_ivp

from collections import deque
from typing import Callable
from time import time
from pdb import set_trace

arr64 = npt.NDArray[np.float64]


class MetapopulationSIR:
    """
    Matapopulation SIR model simulator
    """

    def __init__(
        self,
        graph: nx.DiGraph,  # Mobility directional network with population node attributes and weight (diffusion rate) edge attributes
        basic_rep: float,  # R0 = beta/mu
        recovery_time: float,  # T = 1/mu
        tol: float = 1e-11,  # Internal tolerance
    ):
        # Setup network
        self.num_nodes = graph.number_of_nodes()
        self.adj_mat = nx.adjacency_matrix(graph, weight="weight", dtype=np.float64)
        self.total_pops = np.array(
            list(nx.get_node_attributes(graph, "population").values())
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

        # Setup epidemics
        self.i_rate = basic_rep / recovery_time  # Infection rate beta
        self.r_rate = 1 / recovery_time  # Removal rate mu

    def get_velocity(
        self,
        time: float,
        pop_fracs: arr64,  # Population compartments fractions array
    ) -> arr64:
        """
        SIR model equations
        """

        i_fracs = pop_fracs[: self.num_nodes]  # Infected population fraction
        s_fracs = pop_fracs[self.num_nodes :]  # Susceptible population fraction

        d_i_fracs = (
            self.i_rate * s_fracs * i_fracs
            - self.r_rate * i_fracs
            + (self.adj_mat.T.dot(i_fracs) - self.adj_mat.sum(axis=1) * i_fracs)
            / self.total_pops
        )
        d_s_fracs = (
            -self.i_rate * s_fracs * i_fracs
            + (self.adj_mat.T.dot(s_fracs) - self.adj_mat.sum(axis=1) * s_fracs)
            / self.total_pops
        )

        return np.concatenate((d_i_fracs, d_s_fracs), axis=0)

    def run_simulation(
        self,
        init_node: int,  # Node index of initial infection
        init_i_pop: np.float64,  # Initial infected population
        time_min: np.float64,  # Minimum solve time
        time_max: np.float64,  # Maximum solve time
        ttol: float | None = None,  # Termination tolerance
        t_window: float | None = None,  # Termination tolerance time window
        method: str = "RK45",  # Solve method
        rtol: float | None = None,  # Solve relative tolerance
        atol: float | None = None,  # Solve absolute tolerance
        **kwargs,  # Other solve kwargs
    ) -> dict:
        """
        Run SIR model simulation by solving equations numerically
        """

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
            t_window = 2 / self.i_rate  # Default t_window is two times recovery time

        def termination(
            time: float,
            pop_fracs: arr64,  # Population compartments fractions array
        ) -> int:
            """
            Terminate solve when susceptible fraction change less than ttol for t_window
            """
            if time < time_min:
                return 1

            termination.time_queue.append(time)
            s_fracs = pop_fracs[self.num_nodes :]

            termination.s_fracs_queue = np.append(
                termination.s_fracs_queue, s_fracs, axis=1
            )

            if termination.time_queue[1] < time - t_window:
                termination.time_queue.pop()
                termination.s_fracs_queue = np.delete(
                    termination.s_fracs_queue, (0), axis=1
                )

            return np.sum(
                np.abs(termination.s_fracs_queue[1:] - termination.s_fracs_queue[:-1])
                > ttol,
                dtype=int,
            )

        termination.terminal = True  # Teminate solve when condition is met
        termination.time_queue = [0]
        termination.s_fracs_queue = init_s_fracs
        termination.start_time = time()

        # Solve
        result = solve_ivp(
            self.get_velocity,
            (0, time_max),
            np.concatenate((init_i_fracs, init_s_fracs), axis=0),
            events=termination,
            method=method,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

        result["y"][: self.num_nodes] = result["y"][: self.num_nodes] * self.total_pops
        result["y"][self.num_nodes :] = result["y"][self.num_nodes :] * self.total_pops

        return result
