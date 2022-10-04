import argparse
import os

from data.simulations import DatasetLowRankGenerator

"""
USAGE:
python make_lowrank_dataset.py --folder /home/ubuntu/causal_discovery/data/simulated --rescale --p-module 0.2 --p-vertex 0.1 --n-hidden 1 --n-features 100 --n-modules 10 --nb-dag 10 --n-samples 50000 --nb-interventions 100 --suffix nonlinear --intervention
python make_lowrank_dataset.py --folder /home/ubuntu/causal_discovery/data/simulated --rescale --p-module 0.2 --p-vertex 0.1 --n-hidden 0 --n-features 100 --n-modules 10 --nb-dag 10 --n-samples 50000 --nb-interventions 100 --suffix linear_uniform --intervention --uniform
python make_lowrank_dataset.py --folder /home/ubuntu/causal_discovery/data/simulated --rescale --p-module 0.2 --p-vertex 0.1 --n-hidden 1 --n-features 100 --n-modules 10 --nb-dag 10 --n-samples 50000 --nb-interventions 100 --suffix nonlinear_ninterv
python make_lowrank_dataset.py --folder /home/ubuntu/causal_discovery/data/simulated --rescale --p-module 0.2 --p-vertex 0.1 --n-hidden 0 --n-features 100 --n-modules 10 --nb-dag 10 --n-samples 50000 --nb-interventions 100 --suffix linear_ninterv


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, help="destination folder")
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=0,
        help="number of hidden nodes (0 means linear)",
    )
    parser.add_argument(
        "--n-features", type=int, default=100, help="Number of nodes in the DAGs"
    )
    parser.add_argument(
        "--n-modules", type=int, default=10, help="Number of modules in the DAGs"
    )
    parser.add_argument(
        "--p-module",
        type=float,
        default=0.5,
        help="Expected number of edges per node",
    )
    parser.add_argument(
        "--p-vertex",
        type=float,
        default=0.5,
        help="Expected number of edges per node",
    )
    parser.add_argument(
        "--nb-dag", type=int, default=1, help="Number of DAGs to generate dataset from"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of points per dataset"
    )
    parser.add_argument("--rescale", action="store_true", help="Rescale the variables")
    parser.add_argument(
        "--suffix",
        type=str,
        default="linear",
        help="Suffix that will be added at the \
                        end of the folder name",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="if True, generate data with uniform in place for normal",
    )
    # Arguments related to interventions
    parser.add_argument(
        "--intervention",
        action="store_true",
        help="if True, generate data with interventions",
    )
    parser.add_argument(
        "--nb-interventions",
        type=int,
        default=3,
        help="number of interventional settings",
    )
    parser.add_argument(
        "--obs-data",
        action="store_true",
        help="if True, the first setting is generated without any interventions",
    )
    parser.add_argument(
        "--min-nb-target",
        type=int,
        default=1,
        help="minimal number of targets per setting",
    )
    parser.add_argument(
        "--max-nb-target",
        type=int,
        default=3,
        help="maximal number of targets per setting",
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="if True, make sure that the intervention family is conservative: i.e. that all nodes have not been intervened in at least one setting.",
    )
    parser.add_argument(
        "--cover",
        action="store_true",
        help="if True, make sure that all nodes have been intervened on at least in one setting.",
    )

    arg = parser.parse_args()
    folder = os.path.join(
        arg.folder,
        f"data_p{arg.n_features}_m{arg.n_modules}_n{arg.n_samples}_{arg.suffix}",
    )
    try:
        os.makedirs(folder)
    except OSError:
        print(f"Cannot create the folder: {folder}")

    # create folder
    print(f"Creating folder: {folder}")
    # for loop
    # create generator
    # generate & save

    generator = DatasetLowRankGenerator(
        arg.n_features,
        arg.n_modules,
        arg.p_vertex,
        arg.p_module,
        arg.n_samples,
        arg.n_hidden,
        arg.rescale,
        arg.obs_data,
        arg.nb_interventions,
        arg.min_nb_target,
        arg.max_nb_target,
        arg.conservative,
        arg.uniform,
        arg.cover,
        True,
    )

    for i in range(arg.nb_dag):
        if not arg.intervention:
            # first, generate the observational data
            print("Generating the observational data...")
            generator.generate(False, resample_dag=True)
            generator.save_data(folder, i)
        # then generate interventional data
        if arg.intervention:
            print("Generating the interventional data...")
            generator.generate(True, resample_dag=True)
            generator.save_data(folder, i)
