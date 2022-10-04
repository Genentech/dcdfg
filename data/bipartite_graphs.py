import networkx as nx
import numpy as np


class DagModuleGenerator:
    """
    Create the structure of a module graph

    Args:
    features (int): Number of features in the graph to generate.
    modules (int): Number of modules
    expected_density (int): Expected number of edge per node.
    """

    def __init__(
        self, features=100, modules=10, expected_density=3, p_vertex=None, p_module=None
    ) -> None:
        self.features = features
        self.modules = modules
        self.total_vertices = self.features + self.modules
        self.adjacency_matrix = np.zeros((self.total_vertices, self.total_vertices))
        self.expected_density = expected_density
        if p_vertex:
            self.p_vertex = p_vertex
        else:
            self.p_vertex = (
                2 * self.modules * self.expected_density / (self.features - 1)
            )
        if p_module:
            self.p_module = p_module
        else:
            self.p_module = 2 * self.expected_density / (self.modules - 1)

    def __call__(self):
        # create partition of nodes into modules and features
        causal_order = np.random.permutation(np.arange(self.total_vertices))
        modules_ind = np.random.choice(
            causal_order[1:-1], size=(self.modules), replace=False
        )
        modules_ind.sort()
        features_ind = np.setdiff1d(causal_order, modules_ind, assume_unique=True)
        features_ind.sort()

        for i in range(self.total_vertices - 1):
            vertex = causal_order[i]
            if vertex in modules_ind:
                # parent must be a node
                possible_parents = np.intersect1d(causal_order[(i + 1) :], features_ind)
                prob_connection = self.p_module
            else:
                # parent must be a module
                possible_parents = np.intersect1d(causal_order[(i + 1) :], modules_ind)
                prob_connection = self.p_vertex
            num_parents = np.random.binomial(
                n=possible_parents.shape[0], p=prob_connection
            )
            parents = np.random.choice(
                possible_parents, size=num_parents, replace=False
            )
            self.adjacency_matrix[parents, vertex] = 1

        try:
            self.U = self.adjacency_matrix[features_ind][:, modules_ind]
            self.V = self.adjacency_matrix[modules_ind][:, features_ind]
            self.bipartite_half = np.where(np.dot(self.U, self.V) > 0, 1, 0)
            self.module_list = modules_ind
            self.g = nx.DiGraph(self.adjacency_matrix)
            for node in features_ind:
                self.g.nodes[node]["type"] = "node"
            for module in modules_ind:
                self.g.nodes[module]["type"] = "module"
            assert not list(nx.simple_cycles(self.g))

        except AssertionError:
            print("Regenerating, graph non valid...")
            return self()

        return self.g
