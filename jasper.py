"""
    Citing from jasper from Nvidia
"""

import torch
import torch.nn as nn
import torch.functional as F


class SubBlock(nn.Module):
    def __init__(self, dropout):
        super(SubBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=256,
                              kernel_size=11, stride=1, padding=5)
        self.batch_norm = nn.BatchNorm1d(num_features=256)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.dropout(x)

class Block():
    def __init__(self):
        pass