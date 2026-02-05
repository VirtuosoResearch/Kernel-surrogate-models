# Kernel Surrogate Models

Research code for task-level Hessian and kernel-based surrogate modeling, plus
experiment scripts for modular arithmetic, CIFAR, and in-context learning (ICL).

## Repository layout

- `taskHessian/`: Python package with Hessian utilities, datamodels, solvers, and plotting.
- `src/exp_modular_arithmetic/`: Modular arithmetic experiments (training, influence, TRAK, bounds).
- `src/exp_cifar/`: CIFAR experiments (training, influence functions, TRAK).
- `src/exp_icl/`: ICL prompt generation and inference utilities.
- `src/figrues/`: Notebooks used to generate figures.

## Setup

Create a virtual environment and install in editable mode:

```bash
pip install -e .
```

## Quickstart

Examples (run from the repo root):

```bash
# Train modular arithmetic models
python src/exp_modular_arithmetic/train_all.py

# Train a subset (see script args inside the file)
python src/exp_modular_arithmetic/train_subset.py
```

## Experiments

### Modular arithmetic

Key scripts live in `src/exp_modular_arithmetic/`:

- `train_all.py`, `train_subset.py`: model training.
- `compute_if.py`, `run_compute_if.py`: influence-function style computations.
- `compute_trak.py`, `eval_trak.py`: TRAK-style scores and evaluation.
- `compute_datamodels.py`: datamodel computations.
- `src/configs/`: task and optimizer configs.

Shell helpers:

```bash
bash src/exp_modular_arithmetic/scripts/train_models.sh
bash src/exp_modular_arithmetic/scripts/run_trak.sh
bash src/exp_modular_arithmetic/scripts/run_datamodels.sh
```

### CIFAR

Key scripts in `src/exp_cifar/`:

- `train_cifar.py`: training.
- `influence.py`, `pytorch_influence_functions/`: influence-function utilities.
- `eval_trak.py`, `trak.py`: TRAK evaluation.

### ICL

Key scripts in `src/exp_icl/`:

- `generate_prompts.py`: prompt generation.
- `run_inference.py`: inference runs.

## Figures

Notebooks for plotting and analysis live in `src/figrues/`.

## License

MIT (see `pyproject.toml`).

## Citation

If you use this code in your work, please cite our ICLR paper as follows.

```
@article{zhang2026efficient,
  title={Efficient Estimation of Kernel Surrogate Models for Task Attribution},
  author={Zhang, Zhenshuo and Duan, Minxuan and Zhang, Hongyang R.},
  journal={ICLR},
  year={2026}
}
```
