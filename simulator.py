"""
Metapopulation SIR model simulator module
"""

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.sparse._csr import csr_array
from scipy.integrate import solve_ivp

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
        graph: nx.DiGraph,  # Mobility directional network with population node attributes and weight (diffusion probability) edge attributes
        basic_rep: float,  # R0 = beta/mu
        recovery_time: float,  # T = 1/mu
    ):
        self.num_nodes = graph.number_of_nodes()
        self.adj_mat = nx.adjacency_matrix(graph, weight="weight", dtype=np.float64)
        self.pops = nx.get_node_attributes(graph, "population")

        self.i_rate = basic_rep / recovery_time  # Infection rate beta
        self.r_rate = 1 / recovery_time  # Removal rate mu

    def get_velocity(
        self,
        time: float,
        pop_fracs: arr64,  # Population compartments fractions array
        i_rate: float,  # Infection rate beta
        r_rate: float,  # Removal rate mu
        adj_mat: csr_array,  # Mobility w_nm as adjacency matrix
        num_nodes: int,
    ) -> arr64:
        """
        SIR model equations
        """

        i_fracs = pop_fracs[:num_nodes]  # Infected population fraction
        s_fracs = pop_fracs[num_nodes:]  # Susceptible population fraction

        d_i_fracs = (
            i_rate * s_fracs * i_fracs
            - r_rate * i_fracs
            + (adj_mat.T.dot(i_fracs) - adj_mat.sum(axis=1) * i_fracs)
        )
        d_s_fracs = -i_rate * s_fracs * i_fracs + (
            adj_mat.T.dot(s_fracs) - adj_mat.sum(axis=1) * s_fracs
        )

        return np.concatenate((d_i_fracs, d_s_fracs), axis=0)

    def run_simulation(
        self,
        init_node: int,  # Node index of initial infection
        init_i_pop: np.float64,  # Initial infected population
        time_min: np.float64,  # Minimum solve time
        time_max: np.float64,  # Maximum solve time
        ttol: float = 1e-11,  # Termination tolerance
        t_window: float | None = None,  # Termination tolerance time window
        method: str = "RK45",  # Solve method
        rtol: float = 1e-11,  # Solve relative tolerance
        atol: float = 1e-11,  # Solve absolute tolerance
        **kwargs  # Other solve kwargs
    ) -> dict:
        """
        Run SIR model simulation by solving equations numerically
        """

        # Initial point

        init_i_fracs = np.zeros(self.num_nodes, dtype=np.float64)
        init_i_fracs[init_node] = init_i_pop / self.pops[init_node]
        init_s_fracs = np.ones(self.num_nodes, dtype=np.float64) - init_i_fracs

        # Termination condition

        if t_window == None:
            t_window = 2 / self.i_rate  # Default t_window is two times recovery time

        def termination(
            time: float,
            pop_fracs: arr64,  # Population compartments fractions array
            i_rate: float,  # Infection rate beta
            r_rate: float,  # Removal rate mu
            adj_mat: csr_array,  # Mobility w_nm as adjacency matrix
            num_nodes: int,
        ) -> int:
            """
            Terminate solve when susceptible fraction change less than ttol for t_window
            """
            if time < time_min:
                return 1

            termination.time_queue.append(time)
            s_fracs = pop_fracs[num_nodes:]

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
        termination.time_queue = [0]  #
        termination.s_fracs_queue = init_s_fracs
        termination.start_time = time()

        # Solve
        result = solve_ivp(
            self.get_velocity,
            (0, time_max),
            np.concatenate((init_i_fracs, init_s_fracs), axis=0),
            args=(self.i_rate, 1 / self.r_rate, self.adj_mat, self.num_nodes),
            event=termination,
            method=method,
            rtol=rtol,
            atol=atol,
            **kwargs
        )

        return result
