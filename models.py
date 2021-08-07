import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class FCDisc(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class FCGen(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class FCGAN:
    def __init__(self, img_dim, z_dim):
        self.disc = FCDisc(img_dim)
        self.gen = FCGen(z_dim, img_dim)

