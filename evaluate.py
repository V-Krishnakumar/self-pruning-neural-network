import torch
from config import DEVICE, SPARSITY_THRESHOLD


def evaluate(model, dataloader):
    """
    Compute classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def get_sparsity(model):
    """
    Return model sparsity percentage.
    """
    return model.calculate_sparsity(
        threshold=SPARSITY_THRESHOLD
    )


def collect_gate_values(model):
    """
    Return all gate values for histogram plotting.
    """
    gates = []

    for layer in [model.fc1, model.fc2]:
        values = layer.get_gates().detach().cpu().numpy().flatten()
        gates.extend(values)

    return gates