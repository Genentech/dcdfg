import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from tqdm import tqdm


class PerturbSeqDataset(Dataset):
    """
    A generic class for simulation data loading and extraction, as well as pre-filtering of interventions
    NOTE: the 0-th regime should always be the observational one
    """

    def __init__(
        self,
        file_path,
        number_genes=None,
        fraction_regimes_to_ignore=None,
        regimes_to_ignore=None,
        load_ignored=False,
    ) -> None:
        """
        :param str file_path: Path to the data and the DAG
        :param list regimes_to_ignore: fractions of regimes that are ignored during training
        """
        super(PerturbSeqDataset, self).__init__()
        self.file_path = file_path
        # load data
        all_data, all_masks, all_regimes, adata = self.load_data(number_genes)
        self.adata = adata
        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes_list = np.unique(all_regimes)
        obs_regime = np.unique(
            all_regimes[np.where([mask == [] for mask in all_masks])[0]]
        )
        assert len(obs_regime) == 1
        obs_regime = obs_regime[0]

        if fraction_regimes_to_ignore is not None or regimes_to_ignore is not None:
            if fraction_regimes_to_ignore is not None and regimes_to_ignore is not None:
                raise ValueError("either fraction or list, not both")
            if fraction_regimes_to_ignore is not None:
                np.random.seed(0)
                # make sure observational regime is in the training, and not in the testing
                sampling_list = self.all_regimes_list[
                    self.all_regimes_list != obs_regime
                ]
                self.regimes_to_ignore = np.random.choice(
                    sampling_list,
                    int(fraction_regimes_to_ignore * len(sampling_list)),
                )
            else:
                self.regimes_to_ignore = regimes_to_ignore
            to_keep = np.array(
                [
                    regime not in self.regimes_to_ignore
                    for regime in np.array(all_regimes)
                ]
            )
            if not load_ignored:
                data = all_data[to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if to_keep[i]]
                )
            else:
                data = all_data[~to_keep]
                masks = [mask for i, mask in enumerate(all_masks) if ~to_keep[i]]
                regimes = np.array(
                    [regime for i, regime in enumerate(all_regimes) if ~to_keep[i]]
                )
        else:
            data = all_data
            masks = all_masks
            regimes = all_regimes

        self.data = data
        self.regimes = regimes
        self.masks = np.array(masks, dtype=object)
        self.intervention = True

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.data.shape[0]
        self.dim = self.data.shape[1]

    def __getitem__(self, idx):
        if self.intervention:
            # binarize mask from list
            masks_list = self.masks[idx]
            masks = np.ones((self.dim,))
            for j in masks_list:
                masks[j] = 0
            return (
                self.data[idx].A[0].astype(np.float32),
                masks.astype(np.float32),
                self.regimes[idx],
            )
        else:
            # put full ones mask
            return (
                self.data[idx].A[0].astype(np.float32),
                np.ones_like(self.regimes[idx]).astype(np.float32),
                self.regimes[idx],
            )

    def __len__(self):
        return self.data.shape[0]

    def load_data(self, number_genes=None, normalized_data=True):
        """
        Load the mask, regimes, and data
        """
        # Load adata file
        adata = sc.read_h5ad(self.file_path)
        if number_genes:
            # filter genes
            targeted_genes = np.where(adata.var.targeted)[0]
            n_targeted = targeted_genes.shape[0]
            if n_targeted > number_genes:
                raise ValueError("add more genes to cover at least perturbations")
            variable_genes = np.where(
                adata.var.highly_variable_rank < number_genes - n_targeted
            )[0]
            gene_indices = np.union1d(targeted_genes, variable_genes)
            # in addition, remove genes that are zero everywhere
            gene_indices = np.intersect1d(
                gene_indices, np.where(np.sum(adata.X.A, 0) > 0)[0]
            )
            gene_set = adata.var.index.values[gene_indices]
            adata = adata[:, gene_set].copy()

        if normalized_data:
            data = adata.X
        else:
            data = adata.layers["counts"]

        # Load intervention masks and regimes
        regimes = adata.obs["regimes"].astype(int)
        masks = []
        unrecognized_genes = []

        # create map gene name -> gene number
        gene_map = {}
        for i, gene in enumerate(adata.var.index):
            gene_map[gene] = i
        for index, row in tqdm(adata.obs.iterrows(), total=adata.n_obs):
            mask = []
            if row["targets"] != "":
                for x in row["targets"].split(","):
                    if x in gene_map:
                        mask += [gene_map[x]]
                    else:
                        unrecognized_genes = np.union1d(unrecognized_genes, [x])
            masks.append(mask)

        Warning("couldn't find genes after filtering:" + str(unrecognized_genes))

        return data, masks, regimes, adata
