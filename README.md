# Deep Learning Assignment 1

This project contains three Jupyter notebooks that train and evaluate a VGG6-style image classifier on CIFAR-10 using PyTorch.

## Notebooks

| Notebook | Contents |
| --- | --- |
| `Q1.ipynb` | CIFAR-10 data preparation, VGG6 baseline training, evaluation, and saving the best baseline checkpoint as `vgg6_baseline_best.pth`. |
| `Q2.ipynb` | Experiments comparing activation functions, optimizers, and batch sizes. |
| `Q3.ipynb` | Weights & Biases (W&B) random sweep over activation, optimizer, batch size, epoch count, learning rate, and weight decay. |

The notebooks download CIFAR-10 into `data/` when needed. They reserve 5,000 training images for validation and apply random crop and horizontal-flip augmentation to training images.

## Requirements

- Python
- Jupyter support in VS Code or JupyterLab
- PyTorch and Torchvision
- NumPy, pandas, and Matplotlib
- Weights & Biases (`wandb`) for the tracking and sweep functionality in `Q2.ipynb` and `Q3.ipynb`

## Setup

From the project directory, create and activate a virtual environment on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib torch torchvision wandb jupyter nbformat
```

In VS Code, select `.venv` as the notebook kernel. PyTorch should use Apple's MPS device when available; otherwise, the notebooks use the CPU.

## Running

1. Open a notebook and run its cells from top to bottom.
2. `Q1.ipynb` can be run without W&B authentication.
3. Before running W&B tracking or sweep cells in `Q2.ipynb` or `Q3.ipynb`, authenticate using the W&B/Forge login instructions for your course. Keep API keys out of notebook cells and source control.
4. The sweep agent in `Q3.ipynb` is configured to run five trials. Interrupt its cell in VS Code if you need to stop it early.

Training plots, metrics, and checkpoints are produced by the notebooks. W&B runs and sweep results are available in the configured W&B project.
