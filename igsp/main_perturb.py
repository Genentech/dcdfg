import argparse
import os
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dcdi.callback import CustomProgressBar
from dcdi.linear_baseline.model import LinearGaussianModel
from dcdi.perturbseq_data import PerturbSeqDataset
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dcdi.utils.dag_optim import is_acyclic

from igsp import run_igsp

"""
Example:

python main_perturb.py --data-path cocult --cluster-level 20 --ci-test gaussian
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="../perturb-cite-seq/SCP1064/ready/"
    )
    parser.add_argument("--data-path", type=str, default="ifn")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Threshold for conditional indep tests",
    )
    parser.add_argument(
        "--alpha-inv", type=float, default=1e-3, help="Threshold for invariance tests"
    )
    parser.add_argument(
        "--n-samples", type=int, default=-1, help="Subsample (-1 means all)"
    )
    parser.add_argument(
        "--ci-test",
        type=str,
        default="kci",
        help="Type of conditional independance test to use \
                        (gaussian, hsic, kci)",
    )
    parser.add_argument(
        "--cluster-level", type=int, default=10, help="Clustering used by IGSP"
    )

    arg = parser.parse_args()
    arg.model = "IGSP"

    # load data and make dataset
    folder = arg.data_dir
    file = arg.data_dir + "/" + arg.data_path + "_gene_filtered_adata.h5ad"

    train_dataset = PerturbSeqDataset(
        file, number_genes=1000, fraction_regimes_to_ignore=0.2
    )
    regimes_to_ignore = train_dataset.regimes_to_ignore
    test_dataset = PerturbSeqDataset(
        file, number_genes=1000, regimes_to_ignore=regimes_to_ignore, load_ignored=True
    )

    nb_nodes = test_dataset.dim

    if arg.ci_test == "gaussian":
        # keep only nice interventions (otherwise you get a linalg error)
        nice_regimes = []
        for regime in train_dataset.regimes:
            if len(np.where(train_dataset.regimes == regime)[0]) > 100:
                nice_regimes += [regime]
        nice_regimes = np.unique(nice_regimes)
        filter_ = [reg in nice_regimes for reg in train_dataset.regimes]

    # condense data into partitions
    assignment = pd.read_csv(f"modules_{arg.cluster_level}.csv", index_col=0)
    assignment[assignment == -1] = 0
    n_clusters = len(np.unique(assignment))
    gene_cluster = []
    for gene in train_dataset.adata.var.index:
        if gene in assignment.index:
            gene_cluster.append(assignment.loc[gene].Module)
        else:
            gene_cluster.append(0)
    gene_cluster = np.array(gene_cluster)
    # summarize dataset
    dataset = np.zeros(shape=(train_dataset.data.shape[0], n_clusters))
    for cluster in range(n_clusters):
        dataset[:, cluster] = train_dataset.data.A[
            :, np.where(gene_cluster == cluster)[0]
        ].mean(1)
    # summarize interventions
    interv = []
    for mask in train_dataset.masks:
        cluster_m = []
        for gene in mask:
            cluster_m.append(gene_cluster[gene])
        interv += [list(np.unique(cluster_m))]

    # load data
    dataset_ = dataset - dataset.mean(0)
    dataset_ = dataset_ / (dataset.std(0)[:] + 1e-3)
    dataset_ = np.clip(dataset_, -5, 5)
    if arg.ci_test == "gaussian":
        dataset_ = dataset_[filter_]
        interv = [x for i, x in enumerate(interv) if filter_[i]]
        regimes = np.unique(train_dataset.regimes[filter_], return_inverse=True)[1]

    if arg.n_samples == -1:
        n_samples = dataset.shape[0]
    else:
        n_samples = arg.n_samples

    train_data_pd = pd.DataFrame(dataset_).iloc[:n_samples]
    mask_pd = pd.DataFrame(interv).iloc[:n_samples]
    regimes = np.unique(regimes[:n_samples], return_inverse=True)[1]
    # open logger
    # LOG CONFIG
    logger = WandbLogger(
        project="DCDI-fine-" + arg.data_path, log_model=True, reinit=True
    )
    logger.experiment.config.update(
        {
            "model_name": "IGSP",
            "module_name": "IGSP",
            "lambda_gies": arg.alpha,
            "clusters_gies": arg.cluster_level,
        }
    )

    graph, est_dag, targets_list = run_igsp(
        train_data_pd,
        targets=mask_pd,
        regimes=regimes,
        alpha=arg.alpha,
        alpha_inv=arg.alpha_inv,
        ci_test="kci",
    )
    print("ran IGSP!")
    # check if dag
    acyclic = is_acyclic(graph)

    # now we need to put this back into a full feature dag
    # create attribution matrix
    attr_matrix = np.zeros(shape=(train_dataset.data.shape[1], n_clusters))
    for gene in range(train_dataset.data.shape[1]):
        attr_matrix[gene][gene_cluster[gene]] = 1
    graph_ = np.dot(np.dot(attr_matrix, graph), attr_matrix.T)
    # once we have a dag, must simply train with the gaussian likelihood
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # create model
    model = LinearGaussianModel(
        nb_nodes,
        lr_init=0.01,
        reg_coeff=0,
        constraint_mode="exp",
    )
    # HARDCODE THE DAG, remove acyclicity constraint
    # remove dag constraints: we have a prediction problem now!
    model.gamma = 0.0
    model.mu = 0.0
    model.module.weight_mask.copy_(
        torch.tensor(graph_, device=model.module.weight_mask.device)
    )

    early_stop_2_callback = EarlyStopping(
        monitor="Val/nll", min_delta=1e-6, patience=3, verbose=True, mode="min"
    )
    trainer_fine = pl.Trainer(
        gpus=1,
        max_epochs=300,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[early_stop_2_callback, CustomProgressBar()],
    )
    trainer_fine.fit(
        model,
        DataLoader(train_dataset, batch_size=128),
        DataLoader(val_dataset, num_workers=2, batch_size=256),
    )
    # trainer_fine.save_checkpoint(os.path.join(logger.log_dir, "final.ckpt"))

    # EVAL on held-out data
    pred = trainer_fine.predict(
        ckpt_path="best",
        dataloaders=DataLoader(test_dataset, num_workers=8, batch_size=256),
    )
    held_out_nll = np.mean([x.item() for x in pred])

    # Step 3: score adjacency matrix against groundtruth
    pred_adj = np.array(model.module.weight_mask.detach().cpu().numpy() > 0, dtype=int)
    # check integers
    assert np.equal(np.mod(pred_adj, 1), 0).all()
    print("saved, now evaluating")

    # Step 4: add valid nll and dump metrics
    pred = trainer_fine.predict(
        ckpt_path="best",
        dataloaders=DataLoader(val_dataset, num_workers=8, batch_size=256),
    )
    val_nll = np.mean([x.item() for x in pred])

    acyclic = int(model.module.check_acyclicity())
    wandb.log(
        {
            "interv_nll": held_out_nll,
            "val nll": val_nll,
            "acyclic": acyclic,
            "n_edges": pred_adj.sum(),
        }
    )
