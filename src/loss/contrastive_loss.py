import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        return torch.mean(
            (1 - labels) * outputs**2
            + labels * torch.clamp(self.margin - outputs, min=0) ** 2
        )
