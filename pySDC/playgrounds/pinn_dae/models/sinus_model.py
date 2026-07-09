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

        # Definiert alle Schichten im Netzwerk (inkl. Aktivierungsfunktionen)
        self.layers = nn.ModuleList([self.fc1, nn.Tanh(), self.fc2, nn.Tanh(), self.fc3])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
