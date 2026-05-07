import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    """
    Reproducibility for experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_folders(*folders):
    """
    Create folders if not present.
    """
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_plot(df, output_path="outputs/tradeoff.png"):
    """
    Save Accuracy vs Sparsity tradeoff plot.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Lambda"],
        df["Test Accuracy"],
        marker="o",
        linewidth=2,
        label="Accuracy"
    )

    plt.plot(
        df["Lambda"],
        df["Sparsity %"],
        marker="s",
        linewidth=2,
        label="Sparsity"
    )

    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Value")
    plt.title("Accuracy vs Sparsity Tradeoff")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_gate_histogram(gates, output_path="outputs/gate_histogram.png"):
    """
    Save histogram of learned gate values.
    """
    plt.figure(figsize=(8, 5))

    plt.hist(gates, bins=50)

    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()