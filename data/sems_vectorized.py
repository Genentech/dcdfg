import networkx as nx
import numpy as np
from numba import njit


def init_params(graph, n_hidden=0):
    """
    This function will use the generated graph to initialize the parameters
    """
    causal_order = np.array(list(nx.topological_sort(graph)))
    nodes = len(causal_order)
    if n_hidden == 0:
        # linear model: mask with adjacency to depend only on parents
        weights = np.random.uniform(0.25, 1, size=(nodes, nodes))
        weights *= 2 * np.random.binomial(1, 0.5, size=(nodes, nodes)) - 1
        weights *= nx.adjacency_matrix(graph).A
    else:
        # non-linear model: jsut sample several with the right mask to make a MLP
        weights_1 = np.random.normal(size=(nodes, nodes, n_hidden))

        for hid in range(n_hidden):
            weights_1[:, :, hid] *= nx.adjacency_matrix(graph).A
        # add more params to put all together
        weights_2 = np.random.normal(size=(nodes, n_hidden))
        weights = (weights_1, weights_2)
    return causal_order, weights


@njit
def simulate_data_linear(
    n_samples, weights, causal_order, intervention_set, module_set
):
    n_nodes = causal_order.shape[0]
    data = np.zeros(shape=(n_samples, n_nodes))
    for node in causal_order:
        # each node is a function of its parents
        if node in intervention_set:
            data[:, node] = np.random.normal(0, 1, size=(n_samples,))
        else:
            # should be all zeros in the first node, and at most n-1 in the nth round
            data[:, node] = np.dot(data, weights[:, node])
            if node not in module_set:
                data[:, node] += np.random.normal(0, 0.4, size=(n_samples,))
    return data

@njit
def simulate_data_linear_unif(
    n_samples, weights, causal_order, intervention_set, module_set
):
    n_nodes = causal_order.shape[0]
    data = np.zeros(shape=(n_samples, n_nodes))
    for node in causal_order:
        # each node is a function of its parents
        if node in intervention_set:
            data[:, node] = np.random.normal(0, 1, size=(n_samples,))
        else:
            # should be all zeros in the first node, and at most n-1 in the nth round
            data[:, node] = np.dot(data, weights[:, node])
            if node not in module_set:
                data[:, node] += np.random.uniform(-0.4 * 1.73, 0.4 * 1.73, size=(n_samples,))
    return data

@njit
def simulate_data_nn(n_samples, weights_1, weights_2, causal_order, intervention_set):
    n_nodes = causal_order.shape[0]
    data = np.zeros(shape=(n_samples, n_nodes))
    for node in causal_order:
        # each node is a function of its parents
        if node in intervention_set:
            data[:, node] = np.random.normal(0, 1, size=(n_samples,))
        else:
            # should be all zeros in the first node, and at most n-1 in the nth round
            # TODO contiguous array warning
            temp = np.dot(data, weights_1[:, node])  # shape n_samples times n_hidden
            data[:, node] = np.dot(
                np.tanh(temp),
                weights_2[
                    node,
                ],
            )
            data[:, node] += np.random.normal(0, 1, size=(n_samples,))
    return data

@njit
def simulate_data_nn_uniform(n_samples, weights_1, weights_2, causal_order, intervention_set):
    n_nodes = causal_order.shape[0]
    data = np.zeros(shape=(n_samples, n_nodes))
    for node in causal_order:
        # each node is a function of its parents
        if node in intervention_set:
            data[:, node] = np.random.normal(0, 1, size=(n_samples,))
        else:
            # should be all zeros in the first node, and at most n-1 in the nth round
            # TODO contiguous array warning
            temp = np.dot(data, weights_1[:, node])  # shape n_samples times n_hidden
            data[:, node] = np.dot(
                np.tanh(temp),
                weights_2[
                    node,
                ],
            )
            data[:, node] += np.random.uniform(-0.4 * 1.73, 0.4 * 1.73, size=(n_samples,))
    return data