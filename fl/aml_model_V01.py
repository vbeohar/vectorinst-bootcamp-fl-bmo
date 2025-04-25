import torch.nn as nn

class AMLNet(nn.Module): 
    def __init__(self, input_dim): 
        super(AMLNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)  # No sigmoid here
        )

    def forward(self, x):
        return self.model(x)

