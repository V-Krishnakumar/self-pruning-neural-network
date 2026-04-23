import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Linear layer with learnable gate scores for dynamic pruning.
    Effective weight = weight * sigmoid(gate_scores / temperature)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

    def forward(self, x, temperature=1.0):
        gates = torch.sigmoid(self.gate_scores / temperature)
        effective_weight = self.weight * gates
        return F.linear(x, effective_weight, self.bias)

    def get_gates(self, temperature=1.0):
        return torch.sigmoid(self.gate_scores / temperature)


class SelfPruningNet(nn.Module):
    """
    CNN backbone + Prunable classifier head
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8 -> 4
        )

        self.fc1 = PrunableLinear(128 * 4 * 4, 256)
        self.fc2 = PrunableLinear(256, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, temperature=1.0):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x, temperature))
        x = self.dropout(x)

        x = self.fc2(x, temperature)

        return x

    def sparsity_loss(self, temperature=1.0):
        return (
            self.fc1.get_gates(temperature).sum() +
            self.fc2.get_gates(temperature).sum()
        )

    def calculate_sparsity(self, threshold=1e-2):
        total = 0
        zeroed = 0

        for layer in [self.fc1, self.fc2]:
            gates = layer.get_gates().detach()

            total += gates.numel()
            zeroed += (gates < threshold).sum().item()

        return 100.0 * zeroed / total