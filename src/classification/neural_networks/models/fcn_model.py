"""
Model for a 3-layer Fully-Connected Neural network in PyTorch.
"""

from torch.nn import Module
from torch import nn


class FCN(Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 224 * 224, 1200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1200, 120)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(120, 30)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.flatten(x)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        y = self.softmax(y)

        return y
