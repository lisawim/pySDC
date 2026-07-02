import torch
import torch.nn as nn
import torch.optim as optim

from pySDC.playgrounds.pinn_dae.models import NeuralNet


# Trainingsdaten: y = sin(x)
x = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1)
y = torch.sin(x)

model = NeuralNet()

print(list(model.parameters()))

# Definiere loss function: Wie stark Vorhersage und echte Werte abweichen
loss_fn = nn.MSELoss()

# Optimierer verändert die Gewichte
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(2000):
    y_pred = model(x)              # forward propagation
    loss = loss_fn(y_pred, y)      # Fehler berechnen

    optimizer.zero_grad()          # alte Gradienten löschen
    loss.backward()                # backward propagation
    optimizer.step()               # Gewichte anpassen

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")


m = nn.Linear(20, 30)
input = torch.randn(20, 128)#torch.randn(128, 20)
print(input.size())
output = m(input)
print(output.size())