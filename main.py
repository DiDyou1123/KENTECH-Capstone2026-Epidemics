import numpy as np
import pandas as pd
import networkx as nx

from simulator import MetapopulationSIRSolver
from cut_graph import EasyCutGraph

from itertools import product
from time import time
from pdb import set_trace
import os.path as osp

# ==============================
# Simulation parameters
# 여기만 바꾸면 됨 (그러면 좋겠음)
# ==============================

name = "test_greedy_bc"  # Save file name

# Network parameters
mob_column = "Max. Number of Routes"  # Mobility column in edgelist
pop_column = "Population"  # Populaiton column in nodelist
unit_mob = 200  # Number of passangeres for each routes (people/days/routes)

# Epidemics parameters
basic_rep_list = [
    2.0,
    4.0,
]  # Single population basic reproduction number R0
r_time_list = [14.0, 28.0]  # Recovery time T (days)
init_i_pop = 5.0  # Initial infected populaiton (people)

# Simulation parameters
max_time = 10 * 365.0  # Maximum simulation time (days)
tolerance = 1e-11  # Numerical solver & terminaiton tolerance

# Edge cutting parameters
num_cut_steps = 10  # Number of edge cut numbers
cut_method = "greedy_bc"  # Edge cutting method
num_cut_seeds = 1  # Number of different edge cuts for a single edge cut number
# method 랜덤 아닐 때는 num_cut_seeds = 1

prog_period = 100  # Progress print period

# ==============================
# Load mobility network
# ==============================

edgelist = pd.read_csv("data/edgelist_symmetric.csv")
nodelist = pd.read_csv("data/nodelist_connected.csv")

edgelist["mobility"] = edgelist[mob_column] * unit_mob
nodelist = nodelist.set_index("ID")

node_attr_dict = {}
for i in nodelist.index:
    node_attr_dict[i] = {"population": nodelist.loc[i, pop_column]}

orig_graph = nx.from_pandas_edgelist(
    edgelist,
    "Source",
    "Target",
    edge_attr="mobility",
    create_using=EasyCutGraph,
)
nx.set_node_attributes(orig_graph, node_attr_dict)

num_edges = orig_graph.number_of_edges()
num_nodes = orig_graph.number_of_nodes()

# ==============================
# Iteraiton setup
# ==============================

# Results CSV file column names
path = osp.join("results", f"{name}-results.csv")
with open(path, "w") as f:
    f.write(
        "Number of edge cuts,Edge cut seed,Total mobility cut,Largest connected component size,Basic reproduction number,Recovery time,Global effective reproduciton number,Infection origin,Solver message,Peak severity,Peak time,Global attack rate\n"
    )

# Progression counter
total_jobs = (
    (num_cut_steps + 1)
    * num_cut_seeds
    * len(r_time_list)
    * len(basic_rep_list)
    * num_nodes
)
done_jobs = 0

# ==============================
# Main iteration
# ==============================

print("Starting")
clock_start = time()

# Edge cut iteration
for num_cuts in range(0, num_edges, num_edges // num_cut_steps):
    for cut_seed in range(num_cut_seeds):

        # Cut edges of mobility graph
        cut_graph, cut_mobs = orig_graph.get_edge_cut(
            cut_method, num_cuts, seed=cut_seed, get_cut_weight=True, weight="mobility"
        )

        # Network measures
        connected_comps = nx.connected_components(cut_graph)
        lccs = max(connected_comps, key=len)  # Largest connected component size
        # ========== Effective distance 관련 구현 필요 ==========
        # ========== Effective reproduction number 구현??? ==========

        simulator = MetapopulationSIRSolver(cut_graph, tol=tolerance)

        # Epidemics iteration
        for r_time, basic_rep in product(r_time_list, basic_rep_list):
            eff_rep = simulator.get_eff_rep(
                basic_rep=basic_rep, recovery_time=r_time, sparse=True
            )  # Global effective reproduction number

            # Initial point iteration
            for init_node in orig_graph.nodes:

                # Main solve
                result = simulator.terminal_simulation(
                    basic_rep, r_time, init_node, init_i_pop, 2 * r_time, max_time
                )

                # Successful solve until termination
                if (
                    result["success"]
                    and result["message"] == "A termination event occurred."
                ):

                    # Epidemics measures
                    global_pop = simulator.total_pops.sum()  # Global total population
                    peak_i_frac = (
                        result["I"].sum(axis=0).max() / global_pop
                    )  # Peak severity I_max
                    peak_i_time = result["t"][
                        result["I"].sum(axis=0).argmax()
                    ]  # Peak time t_max
                    inf_r_frac = 1.0 - (
                        result["I"].sum(axis=0)[-1] / global_pop
                        + result["S"].sum(axis=0)[-1] / global_pop
                    )  # Global attack rate R(inf)
                    # ========== Effective reproduction number 구현??? ==========

                    # Save results to CSV file
                    with open(path, "a") as f:
                        f.write(
                            f"{num_cuts},{cut_seed},{cut_mobs},{lccs},{basic_rep},{r_time},{init_node},A termination event occurred.,{peak_i_frac},{peak_i_time},{inf_r_frac}\n"
                        )

                else:
                    with open(path, "a") as f:
                        f.write(
                            f"{num_cuts},{cut_seed},{cut_mobs},{lccs},{basic_rep},{r_time},{eff_rep},{init_node},{result['message']},{0},{0},{0}\n"
                        )

                # Prin progression
                done_jobs += 1

                if done_jobs % prog_period == 0:
                    print(
                        f"{done_jobs}/{total_jobs} done, {round(time() - clock_start)} s passed"
                    )

print(f"Finished, results saved at {path}")
