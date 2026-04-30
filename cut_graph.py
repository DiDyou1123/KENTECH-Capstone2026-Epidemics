"""
Graph cutting methods module
"""

import numpy as np
import networkx as nx

from pdb import set_trace


class EasyCutGraph(nx.Graph):
    """
    nx.Graph with internal graph cutting method implemented
    """

    def get_edge_cut(
        self,
        method: str,  # Edge cutting method
        num_cuts: int,  # Number of edges to cut
        seed: int = 0,  # RNG seed for random methods
        get_cut_weight: bool = False,  # Whether to return total weights of cut edges (return tuple if True)
        weight: str | None = None,  # Weight attribute to measure
        **kwargs,  # Other method-specific keyword arguments
    ) -> "EasyCutGraph | tuple[EasyCutGraph, float]":
        """
        Return edge-cut graph using various methods
        """

        # Raise error if edges cannot be cut
        if num_cuts > self.number_of_edges():
            raise ValueError(
                f"Cannot cut {num_cuts} edges from this graph with {self.number_of_edges()} edges"
            )

        # Select edges to cut based on the given method
        if method == "uniform_random":
            rng = np.random.default_rng(seed=seed)
            cuts = self.get_random_edge_cut(num_cuts, rng)

        elif method == "weighted_random":
            rng = np.random.default_rng(seed=seed)
            cuts = self.get_random_edge_cut(
                num_cuts, rng, prob_weights=[]
            )  # 1/mobility random은 여기서 확률 weight만 바꾸면 될듯?

        # ========== 여기에 다른 method if문 추가 ==========

        else:
            raise ValueError(f"Undefined edge cutting method: {method}")

        # Generate edge cut graph
        cut_graph = self.copy()
        cut_graph.remove_edges_from(cuts)

        # Return cut graph
        if not get_cut_weight:
            return cut_graph

        # Count total weights of cut edges
        else:
            if weight == None:
                raise ValueError("No weight attribute specified to count")

            cut_weights = 0.0
            for edge in cuts:
                cut_weights += self[edge[0]][edge[1]][weight]

            return (cut_graph, cut_weights)

    def get_random_edge_cut(
        self,
        num_cuts: int,  # Number of edges to cut
        rng: np.random.Generator,  # RNG
        prob_weights=None,  # Edge cuting probability weights
    ) -> np.ndarray:
        """
        Select random edges to cut
        """

        return rng.choice(np.array(self.edges), num_cuts, replace=False, p=prob_weights)

    # ========== 아래에 다른 자를 edge 선택 방법 방법 구현 ==========
