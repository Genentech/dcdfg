import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dcdfg.callback import CustomProgressBar
from dcdfg.linear_baseline.model import LinearGaussianModel
from dcdfg.simulation_data import SimulationDataset
from dcdfg.utils.dag_optim import is_acyclic
from dcdfg.utils.metrics import fdr, shd_metric
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader, random_split

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igsp import run_igsp

"""
Example: 

python main_gauss.py --data-dir data_p100_m10_n50000_linear --i-dataset 0 --cluster-level -1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default=None, help="Path to data files")
    parser.add_argument("--i-dataset", type=str, default=None, help="dataset index")
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
        "--ci-test",
        type=str,
        default="gaussian",
        help="Type of conditional independance test to use \
                        (gaussian, hsic, kci)",
    )
    parser.add_argument(
        "--cluster-level", type=int, default=10, help="Clustering used by IGSP"
    )

    arg = parser.parse_args()

    time0 = time.time()
    arg.model = "gies"

    folder = arg.data_dir
    file = "../data/simulated/" + arg.data_dir

    train_dataset = SimulationDataset(
        file, arg.i_dataset, fraction_regimes_to_ignore=0.2
    )
    regimes_to_ignore = train_dataset.regimes_to_ignore
    test_dataset = SimulationDataset(
        file, arg.i_dataset, regimes_to_ignore=regimes_to_ignore, load_ignored=True
    )
    nb_nodes = test_dataset.dim

    if arg.cluster_level > 0:
        # load data
        sp = SpectralClustering(n_clusters=arg.cluster_level)
        feature_cluster = sp.fit_predict(train_dataset.data.T)
        n_clusters = np.max(feature_cluster) + 1
        # summarize dataset
        dataset = np.zeros(shape=(train_dataset.data.shape[0], n_clusters))
        for cluster in range(n_clusters):
            dataset[:, cluster] = train_dataset.data[
                :, np.where(feature_cluster == cluster)[0]
            ].mean(1)
        # summarize interventions
        interv = []
        for mask in train_dataset.masks:
            cluster_m = []
            for gene in mask:
                cluster_m.append(feature_cluster[gene])
            interv += [list(np.unique(cluster_m))]
    else:
        dataset = train_dataset.data
        interv = [list(x) for x in train_dataset.masks]
    # load data
    train_data_pd = pd.DataFrame(dataset)
    mask_pd = pd.DataFrame(interv)
    regimes = np.unique(train_dataset.regimes, return_inverse=True)[1]
    # open logger
    # LOG CONFIG
    logger = WandbLogger(
        project="DCDI-fine-" + arg.data_dir, log_model=True, reinit=True
    )
    logger.experiment.config.update(
        {
            "model_name": "IGSP",
            "module_name": "IGSP",
            "i-dataset": int(arg.i_dataset),
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
        ci_test=arg.ci_test,
    )
    print("ran IGSP!")
    # check if dag
    acyclic = is_acyclic(graph)
    # now we need to put this back into a full feature dag
    # create attribution matrix
    attr_matrix = np.zeros(shape=(train_dataset.data.shape[1], n_clusters))
    for gene in range(train_dataset.data.shape[1]):
        attr_matrix[gene][feature_cluster[gene]] = 1
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
    file = (
        "../data/simulated/" + arg.data_dir + "/" + "DAG" + str(arg.i_dataset) + ".npy"
    )
    truth = np.load(file)
    shd = shd_metric(pred_adj, truth)
    fdr_score = fdr(pred_adj, truth)
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
            "shd": shd,
            "fdr": fdr_score,
        }
    )
