import torch.nn as nn

class AMLNet(nn.Module):
    def __init__(self):
        super(AMLNet, self).__init__()
        self.base_module = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU()
        )
        self.head_module = nn.Linear(64, 1)  # client-specific head

    def forward(self, x):
        x = self.base_module(x)
        x = self.head_module(x)
        return x
