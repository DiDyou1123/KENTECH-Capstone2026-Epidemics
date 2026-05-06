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
            mobilities = np.array([self[u][v]["mobility"] for u, v in self.edges()])
            inv_mob = 1.0 / mobilities
            prob_weights = inv_mob / inv_mob.sum()
            cuts = self.get_random_edge_cut(num_cuts, rng, prob_weights=prob_weights) 
            # ========== 1/mobility random은 위에서 확률 weight만 바꾸면 될듯? ==========

        elif method == "static_bc":
            cuts = self.get_static_bc_edge_cut(num_cuts, **kwargs)

        elif method == "greedy_bc":
            cuts = self.get_greedy_bc_edge_cut(num_cuts, **kwargs)

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

    def get_static_bc_edge_cut(
        self,
        num_cuts: int,
        weight: str | None = "mobility",
    ) -> list[tuple]:
        
        # BC를 한 번 계산한 뒤, 높은 순서대로 num_cuts개 엣지 선택
        
        inv_weight = "inv_" + weight
        for u, v, data in self.edges(data=True):
            data[inv_weight] = 1.0 / data[weight]

        bc = nx.edge_betweenness_centrality(self, weight=inv_weight)
        sorted_edges = sorted(bc, key=bc.get, reverse=True)
        return sorted_edges[:num_cuts]


    def get_greedy_bc_edge_cut(
        self,
        num_cuts: int,
        weight: str | None = "mobility",
    ) -> list[tuple]:
        
        # 매 스텝마다 BC를 재계산하며 가장 높은 엣지를 하나씩 num_cuts번 자르기

        working_graph = self.copy()
        inv_weight = "inv_" + weight
        cuts = []

        for _ in range(num_cuts):
            if working_graph.number_of_edges() == 0:
                break

            for u, v, data in working_graph.edges(data=True):
                data[inv_weight] = 1.0 / data[weight]

            bc = nx.edge_betweenness_centrality(working_graph, weight=inv_weight)
            top_edge = max(bc, key=bc.get)
            cuts.append(top_edge)
            working_graph.remove_edge(*top_edge)

        return cuts

def get_population_product_edge_cut(
        self,
        num_cuts: int,
        pop_attr: str = "population",
    ) -> list[tuple]:
        """
        Select edges to cut based on the product of endpoint populations.

        Edge score:
            score(u, v) = population[u] * population[v]

        Edges with larger score are cut first.
        """

        edge_scores = {}

        for u, v in self.edges:
            pop_u = self.nodes[u].get(pop_attr)
            pop_v = self.nodes[v].get(pop_attr)

            if pop_u is None or pop_v is None:
                raise ValueError(
                    f"Node population attribute '{pop_attr}' is missing for edge ({u}, {v})"
                )

            edge_scores[(u, v)] = pop_u * pop_v

        sorted_edges = sorted(edge_scores, key=edge_scores.get, reverse=True)

        return sorted_edges[:num_cuts]
