import torch.nn.functional as F
from torch.nn import Linear, Module

class ModelNetwork(Module):
    def __init__(self):
        super().__init__()

        # Intialize model layers
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = F.relu(x)

        return x