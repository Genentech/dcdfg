# Differentiable Causal Discovery with Factor Graphs

This repository contains an implementation of the structure learning method described in ["Large-Scale Differentiable Causal Discovery of Factor Graphs"](https://arxiv.org/abs/2206.07824). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{Lopez2022largescale,
  author = {Lopez, Romain and Hütter, Jan-Christian and Pritchard, Jonathan K. and Regev, Aviv}, 
  title = {Large-Scale Differentiable Causal Discovery of Factor Graphs},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
}
```

## Requirements

Python 3.9+ is required. To install the requirements:
```setup
pip install -r requirements.txt
```
wandb is required for now (a PR to make remove this requirement is welcome). Follow the steps [here](https://docs.wandb.ai/quickstart).


## Running DCD-FG

### SEMs simulations (full usage in files)
1. 'python make_lowrank_dataset.py'
2. 'python run_gaussian.py'
### Biological dataset
1. 'perturb-cite-seq/0-data-download.ipynb'
1. 'perturb-cite-seq/1-assignments-vs-variability.ipynb'
2. 'python run_perturbseq_linear.py'

## Acknowledgments
- This repository was originally forked from [DCDI](https://github.com/slachapelle/dcdi). Please refer to the license file for more information.
- Most files in this codebase have been rewritten for:
1. vectorization and scaling to large graphs
2. incorporating the semantic of factor graphs
3. refactoring and implementation in pytorch lightning
4. implementation of DCD-FG, NOTEARS, NOTEARS-LR and NOBEARS
- We are grateful to the authors of the baseline methods for releasing their code.