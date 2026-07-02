import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 Eingabewert x -> 16 Werte
        self.fc1 = nn.Linear(1, 16)

        # 16 Werte -> 16 Werte
        self.fc2 = nn.Linear(16, 16)

        # 16 Werte -> 1 Ausgabewert y_hat
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        # Erste versteckte Schicht:
        # (N, 1) -> (N, 16)
        z1 = self.fc1(x)

        # Aktivierung:
        # (N, 16) -> (N, 16)
        h1 = torch.tanh(z1)

        # Zweite versteckte Schicht:
        # (N, 16) -> (N, 16)
        z2 = self.fc2(h1)

        # Aktivierung:
        # (N, 16) -> (N, 16)
        h2 = torch.tanh(z2)

        # Ausgabeschicht:
        # (N, 16) -> (N, 1)
        output = self.fc3(h2)

        return output
