import torch.nn as nn

class IDSNet(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def forward(self, x):
        return self.net(x)