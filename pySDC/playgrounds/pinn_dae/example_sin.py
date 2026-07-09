import torch
import torch.nn as nn
import torch.optim as optim

from pySDC.playgrounds.pinn_dae.models.sinus_model import NeuralNet


# Trainingsdaten: y = sin(x)
x = torch.linspace(-3.14, 3.14, 200).reshape(-1, 1)
y = torch.sin(x)

model = NeuralNet()

print(len(list(model.parameters())))
for param in list(model.parameters()):
    print(param.size())

# Definiere loss function: Wie stark Vorhersage und echte Werte abweichen
loss_fn = nn.MSELoss()

# Optimierer verändert die Gewichte mit Lernrate lr=0.01
# Optimierer vor dem Training initialisieren, da Informationen aus vorherigen
# Läufen gespeichert werden (z.B. Momentum)
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
input = torch.randn(128, 20)
print(input.size())
output = m(input)
print(output.size())