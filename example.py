import torch
from layer import Detection
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3),
    Detection(16, 5),
)

optimizer = torch.optim.SGD(model.parameters(), 0.1)

optimizer.zero_grad()
x = torch.rand(2, 3, 50, 50)
y, _ = model(x)

loss = y.sum()
loss.backward()
optimizer.step()
