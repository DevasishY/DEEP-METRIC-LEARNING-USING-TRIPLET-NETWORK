import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.tripletmodel = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=5, stride=1, padding=2
            ),  # Input: 1x28x28, Output: 32x28x28
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),  # Output: 32x14x14
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x14x14
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),  # Output: 64x7x7
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128x7x7
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),  # Output: 128x3x3
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),  # Output: 128x2x2
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.unsqueeze(1)
        return self.tripletmodel(x)
