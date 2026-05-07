import torch

# Training Configuration
SEED = 42
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lambda Sweep Values
LAMBDAS = [
    1.5e-5,
    2e-5,
    3e-5,
    5e-5,
    7e-5
]

# Output Paths
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"

# Sparsity Threshold
SPARSITY_THRESHOLD = 1e-2