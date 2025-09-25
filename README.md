## Task modeling and Hessians

We provide the implementation of EFFICIENT ESTIMATION OF KERNEL SURROGATE MODELS FOR TASK ATTRIBUTION (KernelSM).

## Setup

To use `taskHessian`:

```bash
pip install -e .
```

## Run experiments

- Modular reasoning tasks: Use --task krr for KernelSM

  ```bash
  cd ./modular_examples
  python modular_example.py --task addition/quad --solver krr/lstsq
  ```

- In context learning tasks: Use --task krr for KernelSM

  ```bash
  cd ./icl_examples
  python icl_example.py --task sst2/coin_filp --solver krr/lstsq
  ```

