import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path

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

name = "test_random_2"  # Save file name

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
cut_method = "uniform_random"  # Edge cutting method
num_cut_seeds = 3  # Number of different edge cuts for a single edge cut number
# method 랜덤 아닐 때는 num_cut_seeds = 1

prog_period = 100  # Progress print period

# ==============================
# Load mobility network
# ==============================

nodelist = pd.read_csv(osp.join("data", "nodelist_connected.csv"))
edgelist = pd.read_csv(osp.join("data", "edgelist_symmetric.csv"))

nodelist = nodelist.set_index("ID")
edgelist["mobility"] = edgelist[mob_column] * unit_mob

nodes = [
    (node, {"population": nodelist.loc[node, pop_column]}) for node in nodelist.index
]
edges = [
    (
        edgelist.loc[edge, "Source"],
        edgelist.loc[edge, "Target"],
        edgelist.loc[edge, "mobility"],
    )
    for edge in edgelist.index
]

orig_graph = EasyCutGraph()
orig_graph.add_nodes_from(nodes)
orig_graph.add_weighted_edges_from(edges, weight="mobility")

num_edges = orig_graph.number_of_edges()
num_nodes = orig_graph.number_of_nodes()

# ==============================
# Iteraiton setup
# ==============================

# Results CSV file column names
path = osp.join("results", f"{name}-results.csv")
with open(path, "w") as f:
    f.write(
        "Number of edge cuts,Edge cut seed,Total mobility cut,Largest connected component size,Maximum connected component population,Diameter,Average distance,Basic reproduction number,Recovery time,Global effective reproduciton number,Infection origin,Solver message,Peak severity,Peak time,Global attack rate\n"
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

        # Setup simulator
        simulator = MetapopulationSIRSolver(cut_graph, tol=tolerance)

        # Network measures
        lcc = max(nx.connected_components(cut_graph), key=len)  # Largest connected component
        lcc_num = len(lcc)  # LCCS

        set_trace()
        max_cc_pop = 0  # Maximum conncected component population
        for cc in nx.connected_components(cut_graph):
            cc_pop = sum([simulator.total_pops[node] for node in cc])
            set_trace()
            if cc_pop > max_cc_pop:
                max_cc_pop = cc_pop
        set_trace()

        adj_mat_unweighted = nx.to_scipy_sparse_array(cut_graph, format="csr")
        dist_matrix = shortest_path(
            adj_mat_unweighted, method="D", unweighted=True, directed=False
        )  # Distance matrix
        dist_mat_fin = np.isfinite(dist_matrix)
        diameter = dist_matrix[dist_mat_fin].max()  # Network diameter
        avg_dist = dist_matrix[dist_mat_fin].sum() / (
            dist_mat_fin.sum() - len(dist_matrix)
        )  # Average distance

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

                    # Save results to CSV file
                    with open(path, "a") as f:
                        f.write(
                            f"{num_cuts},{cut_seed},{cut_mobs},{lcc_num},{max_cc_pop},{diameter},{avg_dist},{basic_rep},{r_time},{eff_rep},{init_node},A termination event occurred.,{peak_i_frac},{peak_i_time},{inf_r_frac}\n"
                        )

                else:
                    with open(path, "a") as f:
                        f.write(
                            f"{num_cuts},{cut_seed},{cut_mobs},{lcc_num},{max_cc_pop},{diameter},{avg_dist},{basic_rep},{r_time},{eff_rep},{init_node},{result['message']},{0},{0},{0}\n"
                        )

                # Prin progression
                done_jobs += 1

                if done_jobs % prog_period == 0:
                    print(
                        f"{done_jobs}/{total_jobs} done, {round(time() - clock_start)} s passed"
                    )

print(f"Finished, results saved at {path}")
