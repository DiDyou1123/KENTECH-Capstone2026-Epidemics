import numpy as np
import pandas as pd
import networkx as nx

from simulator import MetapopulationSIRSolver
from cut_graph import EasyCutGraph

from time import time

# ==============================
# Simulation parameters
# 여기만 바꾸면 됨 (그러면 좋겠음)
# ==============================

name = "test_random"  # Save file name

# Network

mob_column = "Max. Number of Routes"  # Mobility column in edgelist
pop_column = "Population"  # Populaiton column in nodelist
unit_mob = 200  # Number of passangeres for each routes (people/days/routes)

# Epidemics

bas_rep_list = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]  # Single population basic reproduction number R0
r_time_list = [14, 28]  # Recovery time T (days)
init_i_pop = 5  # Initial infected populaiton (people)

# Edge cutting

num_cut_steps = 50  # Number of edge cut numbers
cut_method = "uniform_random"  # Edge cutting method
num_cut_seeds = 50  # Number of different edge cuts for a single edge cut number
# method 랜덤 아닐 때는 num_cut_seeds = 1

# ==============================
# Load mobility network
# ==============================

edgelist = pd.read_csv("data\\edgelist_symmetric.csv")
nodelist = pd.read_csv("data\\nodelist_connected.csv")

edgelist["mobility"] = edgelist[mob_column] * unit_mob
nodelist = nodelist.set_index("ID")

node_attr_dict = {}
for i in nodelist.index:
    node_attr_dict[i] = {"population": nodelist.loc[i, pop_column]}

orig_graph = nx.from_pandas_edgelist(
    edgelist,
    "Source",
    "Target",
    edge_attr=["mobility", "eff_dist"],
    create_using=EasyCutGraph,
)
nx.set_node_attributes(orig_graph, node_attr_dict)

num_edges = orig_graph.number_of_edges()
num_nodes = orig_graph.number_of_nodes()


# ==============================
# Main iteration
# ==============================

clock_start = time()

for num_cuts in range(0, num_edges, num_edges // num_cut_steps):
    for cut_seed in range(num_cut_seeds):

        # Cut edges of mobility graph

        cut_graph, cut_weights = orig_graph.get_edge_cut(
            cut_method, num_cuts, seed=cut_seed, get_cut_weight=True, weight="mobility"
        )

        # Network measures

        connected_comps = nx.connected_components(cut_graph)
        lccs = max(connected_comps, key=len)  # Largest connected component size
