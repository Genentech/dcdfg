import csv
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data.bipartite_graphs import DagModuleGenerator
from data.sems_vectorized import (init_params, simulate_data_linear, simulate_data_linear_unif,
                                  simulate_data_nn, simulate_data_nn_uniform)


class DatasetLowRankGenerator:
    """Generate datasets using simulations.py. `nb_dag` dags are sampled and
    then data are generated accordingly to the chosen parameters (e.g.
    mechanisms). Can generate dataset with 'hard stochastic' interventions"""

    def __init__(
        self,
        n_features,
        n_modules,
        p_vertex,
        p_module,
        n_samples,
        n_hidden=0,
        rescale=True,
        obs_data=True,
        nb_interventions=10,
        min_nb_target=1,
        max_nb_target=3,
        conservative=False,
        uniform=False,
        cover=False,
        verbose=True,
    ):
        """
        Generate a dataset containing interventions. The setting is similar to the
        one in the DCDI paper, but with a low-rank structure. Save the lists of targets
        in a separate file.

        Args:
            nb_nodes (int): Number of nodes in each DAG
            expected_degree (int): Expected number of edges per node
            nb_points (int): Number of points per interventions (thus the total =
                            nb_interventions * nb_points)
            rescale (bool): if True, rescale each variables

            nb_interventions (int): number of interventional settings
            obs_data (bool): if True, the first setting is generated without any interventions
            min_nb_target (int): minimal number of targets per setting
            max_nb_target (int): maximal number of targets per setting. For a fixed
                                 number of target, one can make min_nb_target==max_nb_target
            conservative (bool): if True, make sure that the intervention family is
                                 conservative: i.e. that all nodes have not been
                                 intervened in at least one setting.
            cover (bool): if True, make sure that all nodes have been
                                 intervened on at least in one setting.
            verbose (bool): if True, print messages to inform users
        """
        self.n_features = n_features
        self.n_modules = n_modules
        self.p_vertex = p_vertex
        self.p_module = p_module
        self.n_samples = n_samples
        self.n_hidden = n_hidden
        self.rescale = rescale
        self.verbose = verbose
        self.uniform = uniform
        if self.uniform:
            self.simulation_function_nn = simulate_data_nn_uniform
            self.simulation_function = simulate_data_linear_unif
        else:
            self.simulation_function_nn = simulate_data_nn
            self.simulation_function = simulate_data_linear          
        self.graph = None

        # attributes related to interventional data
        self.obs_data = obs_data
        self.nb_interventions = nb_interventions
        self.min_nb_target = min_nb_target
        self.max_nb_target = max_nb_target
        self.conservative = conservative
        self.cover = cover

    def generate(self, intervention=False, resample_dag=True):
        # create DAG if does not exist
        if self.graph is None or resample_dag:
            if self.verbose:
                print("Sampling the DAG")
            self.generator = DagModuleGenerator(
                features=self.n_features,
                modules=self.n_modules,
                p_module=self.p_module,
                p_vertex=self.p_vertex,
            )
            self.graph = self.generator()
            self.feature_list = np.where(
                [self.graph.nodes[node]["type"] == "node" for node in self.graph.nodes]
            )[0]
            self.module_list = np.where(
                [
                    self.graph.nodes[node]["type"] == "module"
                    for node in self.graph.nodes
                ]
            )[0]

            if self.verbose:
                print("Init sem")
            self.causal_order, self.weights = init_params(self.graph, self.n_hidden)

        mask_intervention = []
        regimes = []
        # plan intervention scheme, perform them and sample to put together a dataset
        if intervention:
            data = np.zeros((self.n_samples, self.n_features))

            num = self.n_samples
            if self.obs_data:
                div = self.nb_interventions + 1
            else:
                div = self.nb_interventions
            # one-liner taken from https://stackoverflow.com/questions/20348717/algo-for-dividing-a-number-into-almost-equal-whole-numbers/20348992
            points_per_interv = [
                num // div + (1 if x < num % div else 0) for x in range(div)
            ]
            nb_env = self.nb_interventions

            # randomly pick targets
            target_list = self._pick_targets()

            # perform interventions
            for j in tqdm(range(nb_env), desc="interventions"):
                # these interventions are at the feature level, must convert into features
                targets = np.array([self.feature_list[t] for t in target_list[j]])

                # generate the datasets with the given interventions
                if self.n_hidden > 0:
                    dataset = self.simulation_function_nn(
                        points_per_interv[j],
                        self.weights[0],
                        self.weights[1],
                        self.causal_order,
                        targets,
                        # self.module_list,
                    )

                else:
                    dataset = self.simulation_function(
                        points_per_interv[j],
                        self.weights,
                        self.causal_order,
                        targets,
                        self.module_list,
                    )

                # keep only the "feature" nodes
                dataset = dataset[:, self.feature_list]
                # put dataset and targets in arrays
                if j == 0:
                    start = 0
                else:
                    start = np.cumsum(points_per_interv[:j])[-1]
                end = start + points_per_interv[j]
                data[start:end, :] = dataset
                # here add at the feature level, not node
                mask_intervention.extend(
                    [target_list[j] for i in range(points_per_interv[j])]
                )
                regimes.extend([j + 1 for i in range(points_per_interv[j])])

        else:
            # generate the datasets with no intervention
            if self.n_hidden > 0:
                data = self.simulation_function(
                    self.n_samples,
                    self.weights[0],
                    self.weights[1],
                    self.causal_order,
                    np.array([-1]),
                )
            else:
                data = self.simulation_function(
                    self.n_samples,
                    self.weights,
                    self.causal_order,
                    np.array([-1]),
                    self.module_list,
                )
            data = data[:, self.feature_list]

        if self.rescale:
            scaler = StandardScaler()
            scaler.fit_transform(data)

        # dump into class
        self.data = data
        self.mask_intervention = mask_intervention
        self.regimes = regimes

    def _pick_targets(self, nb_max_iteration=100000):
        nodes = np.arange(self.n_features)
        not_correct = True
        i = 0

        if self.max_nb_target == 1:
            intervention = np.random.choice(
                self.n_features, self.nb_interventions, replace=False
            )
            targets = [[i] for i in intervention]

        else:
            while not_correct and i < nb_max_iteration:
                targets = []
                not_correct = False
                i += 1

                # pick targets randomly
                for _ in range(self.nb_interventions):
                    nb_targets = np.random.randint(
                        self.min_nb_target, self.max_nb_target + 1, 1
                    )
                    intervention = np.random.choice(
                        self.n_features, nb_targets, replace=False
                    )
                    targets.append(intervention)

                # apply rejection sampling
                if self.cover and not self._is_covering(nodes, targets):
                    not_correct = True
                if (
                    self.conservative
                    and not self.obs_data
                    and not self._is_conservative(nodes, targets)
                ):
                    not_correct = True

            if i == nb_max_iteration:
                raise ValueError(
                    "Could generate appropriate targets. \
                                 Exceeded the maximal number of iterations"
                )

            for i, t in enumerate(targets):
                targets[i] = np.sort(t)

        return targets

    def _is_conservative(self, elements, lists):
        for e in elements:
            conservative = False

            for list_ in lists:
                if e not in list_:
                    conservative = True
                    break
            if not conservative:
                return False
        return True

    def _is_covering(self, elements, lists):
        return set(elements) == self._union(lists)

    def _union(self, lists):
        union_set = set()

        for l in lists:
            union_set = union_set.union(set(l))
        return union_set

    def save_data(self, folder, i, save_cpdag=False):
        # save da
        print("saving")
        os.makedirs(folder, exist_ok=True)
        dag_path = os.path.join(folder, f"DAG{i}.npy")
        np.save(dag_path, self.generator.bipartite_half)

        # TODO save full dag?
        np.save(os.path.join(folder, f"U{i}.npy"), self.generator.U)
        np.save(os.path.join(folder, f"V{i}.npy"), self.generator.V)
        np.save(os.path.join(folder, f"adj{i}.npy"), self.generator.adjacency_matrix)
        np.save(os.path.join(folder, f"module{i}.npy"), self.generator.module_list)

        # save data
        if len(self.mask_intervention) == 0:
            data_path = os.path.join(folder, f"data{i}.npy")
            np.save(data_path, self.data)
        else:
            data_path = os.path.join(folder, f"data_interv{i}.npy")
            np.save(data_path, self.data)

            data_path = os.path.join(folder, f"intervention{i}.csv")
            with open(data_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.mask_intervention)
        # save regimes
        if self.regimes is not None:
            regime_path = os.path.join(folder, f"regime{i}.csv")
            with open(regime_path, "w", newline="") as f:
                writer = csv.writer(f)
                for regime in self.regimes:
                    writer.writerow([regime])
