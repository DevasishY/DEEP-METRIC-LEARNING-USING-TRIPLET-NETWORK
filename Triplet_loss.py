import torch
import torch.nn as nn


class Triplet_loss(nn.Module):
    def __init__(self):
        super(Triplet_loss, self).__init__()

    def forward(self, x, x_plus, x_minus):
        # Compute the distances between embeddings
        d_x_plus = torch.norm(x - x_plus, p=2, dim=1)
        d_x_minus = torch.norm(x - x_minus, p=2, dim=1)

        # Compute the softmax values
        d_plus = torch.exp(d_x_plus) / (torch.exp(d_x_plus) + torch.exp(d_x_minus))
        d_minus = torch.exp(d_x_minus) / (torch.exp(d_x_plus) + torch.exp(d_x_minus))

        # loss
        l = d_plus**2 + (d_minus - 1) ** 2

        return l.mean()
