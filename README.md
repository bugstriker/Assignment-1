# Deep Learning Assignment 1

This project trains and evaluates a VGG6-style image classifier on CIFAR-10 with PyTorch. The notebooks contain the assignment experiments; the reusable baseline implementation is split into Python modules under `vgg6/`.

## Project files

| Path | Purpose |
| --- | --- |
| `Q1.ipynb` | Baseline experiment, plots, and notebook results. |
| `Q2.ipynb` | Experiments comparing activation functions, optimizers, and batch sizes. |
| `Q3.ipynb` | W&B random hyperparameter sweep; the agent is limited to five trials. |
| `vgg6/model.py` | VGG6 network definition and weight initialization. |
| `vgg6/data.py` | CIFAR-10 transforms, reproducible train/validation split, and data loaders. |
| `vgg6/engine.py` | Seed setup, training loop, and evaluation routine. |
| `vgg6/train.py` | Command-line entry point that trains, tests, and saves the baseline model. |
| `tests/test_engine.py` | Lightweight model and training/evaluation smoke tests. |
| `vgg6_baseline_best.pth` | Trained baseline model state dictionary. |

The notebooks download CIFAR-10 into `data/` when needed. The archive, extracted dataset, and generated W&B logs are ignored by Git because they can be downloaded or regenerated.

## Verified environment

The commands and pinned package versions below match the environment used to verify this project:

- OS: macOS on Apple Silicon; PyTorch selected the MPS device
- Python: 3.14.4
- PyTorch: 2.14.1
- Torchvision: 0.29.1
- NumPy: 2.5.4
- pandas: 3.1.0
- Matplotlib: 3.11.2
- Weights & Biases: 0.31.0
- ipykernel: 7.4.0
- nbformat: 5.11.1

`requirements.txt` pins these dependency versions. The notebooks also use VS Code's notebook support; a full JupyterLab installation is not required.

## Setup (macOS)

Run these commands from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name assignment1 --display-name "Python (Assignment1)"
```

In VS Code, select the `Python (Assignment1)` kernel, or select the `.venv` interpreter directly. The code chooses MPS when available, then CUDA, and otherwise CPU.

## Train and evaluate the baseline

From the repository root with `.venv` active:

```bash
python -m vgg6.train
```

This downloads CIFAR-10 if it is not already present, trains for 10 epochs with batch size 64 and Adam at learning rate `0.001`, restores the best validation-accuracy weights, evaluates on the test set, and writes `vgg6_baseline_best.pth`.

Optional arguments:

```bash
python -m vgg6.train --epochs 20 --batch-size 128 --learning-rate 0.001
python -m vgg6.train --data-dir ./data --checkpoint ./checkpoints/vgg6.pth --seed 42
```

To run the smoke tests:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

The default seed is **42**. The CLI's `--seed` option seeds Python's `random`, NumPy, and PyTorch generators. It also seeds the training data-loader shuffle and the train/validation split. For CUDA, cuDNN deterministic mode is enabled and benchmarking is disabled. GPU backends and hardware can still produce small differences between runs.

The data pipeline applies random crop and horizontal-flip augmentation to training images, and CIFAR-10 normalization to all splits. It reserves 5,000 of the 50,000 training images for validation; the remaining 45,000 are used for training.

## W&B experiments

`Q2.ipynb` and `Q3.ipynb` use W&B for experiment tracking. Authenticate through the W&B/Forge login flow required by your course before running those cells. Do not put API keys in notebook source or commit them. The Q3 sweep agent is configured for five runs and can be interrupted from the notebook toolbar.
