import argparse
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from dcdfg.callback import (AugLagrangianCallback, ConditionalEarlyStopping,
                            CustomProgressBar)
from dcdfg.linear_baseline.model import LinearGaussianModel
from dcdfg.lowrank_linear_baseline.model import LinearModuleGaussianModel
from dcdfg.lowrank_mlp.model import MLPModuleGaussianModel
from dcdfg.perturbseq_data import PerturbSeqDataset

"""
USAGE:
python -u run_perturbseq_linear.py --data-path control --reg-coeff 0.001 --constraint-mode spectral_radius --lr 0.01 --model linear
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data-path", type=str, default="control", help="Path to data files"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0.8,
        help="Number of samples used for training (default is 80% of the total size)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="number of samples in a minibatch",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=600,
        help="number of meta gradient steps",
    )
    parser.add_argument(
        "--num-fine-epochs", type=int, default=50, help="number of meta gradient steps"
    )
    parser.add_argument("--num-modules", type=int, default=20, help="number of modules")
    # optimization
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for optim"
    )
    parser.add_argument(
        "--reg-coeff",
        type=float,
        default=0.1,
        help="regularization coefficient (lambda)",
    )
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default="exp",
        help="technique for acyclicity constraint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="linear|linearlr|mlplr",
    )
    parser.add_argument(
        "--poly", action="store_true", help="Polynomial on linear model"
    )


    parser.add_argument(
        "--data-dir", type=str, default="../perturb-cite-seq/SCP1064/ready/"
    )
    parser.add_argument("--num-gpus", type=int, default=1)

    arg = parser.parse_args()

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

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    if arg.model == "linear":
        # create model
        model = LinearGaussianModel(
            nb_nodes,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
            poly=arg.poly,
        )
    elif arg.model == "linearlr":
        model = LinearModuleGaussianModel(
            nb_nodes,
            arg.num_modules,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
        )
    elif arg.model == "mlplr":
        model = MLPModuleGaussianModel(
            nb_nodes,
            2,
            arg.num_modules,
            16,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
        )
    else:
        raise ValueError("couldn't find model")

    logger = WandbLogger(project="DCDI-train-" + arg.data_path, log_model=True)
    # LOG CONFIG
    model_name = model.__class__.__name__
    if arg.poly and model_name == "LinearGaussianModel":
        model_name += "_poly"
    logger.experiment.config.update(
        {"model_name": model_name, "module_name": model.module.__class__.__name__}
    )

    # Step 1: augmented lagrangian
    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=arg.num_gpus,
        max_epochs=arg.num_train_epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
    )
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=arg.train_batch_size, num_workers=4),
        DataLoader(val_dataset, num_workers=8, batch_size=256),
    )
    wandb.log({"nll_val": model.nlls_val[-1]})
    wandb.finish()

    # freeze and prune adjacency
    model.module.threshold()
    # WE NEED THIS BECAUSE IF it's exactly a DAG THE POWER ITERATIONS DOESN'T CONVERGE
    # TODO Just refactor and remove constraint at validation time
    model.module.constraint_mode = "exp"
    # remove dag constraints: we have a prediction problem now!
    model.gamma = 0.0
    model.mu = 0.0

    # Step 2:fine tune weights with frozen model
    logger = WandbLogger(project="DCDI-fine-" + arg.data_path, log_model=True)
    model_name = model.__class__.__name__
    if arg.poly and model_name == "LinearGaussianModel":
        model_name += "_poly"
    logger.experiment.config.update(
        {"model_name": model_name, "module_name": model.module.__class__.__name__}
    )

    early_stop_2_callback = EarlyStopping(
        monitor="Val/nll", min_delta=1e-6, patience=5, verbose=True, mode="min"
    )
    trainer_fine = pl.Trainer(
        gpus=arg.num_gpus,
        max_epochs=arg.num_fine_epochs,
        logger=logger,
        val_check_interval=1.0,
        callbacks=[early_stop_2_callback, CustomProgressBar()],
    )
    trainer_fine.fit(
        model,
        DataLoader(train_dataset, batch_size=arg.train_batch_size),
        DataLoader(val_dataset, num_workers=2, batch_size=256),
    )

    # EVAL on held-out data
    pred = trainer_fine.predict(
        ckpt_path="best",
        dataloaders=DataLoader(test_dataset, num_workers=8, batch_size=256),
    )
    held_out_nll = np.mean([x.item() for x in pred])

    # Step 3: score adjacency matrix against groundtruth
    pred_adj = model.module.weight_mask.detach().cpu().numpy()
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
