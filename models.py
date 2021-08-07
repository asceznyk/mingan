import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class FCDisc(nn.Module):
    def __init__(self, img_dim):
        super(FCDisc, self).__init__()
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
        super(FCGen, self).__init__()
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

class DCDisc(nn.Module):
    def __init__(self, img_dim, f_disc):
        super(DCDisc, self).__init__()
        nc, nw, nh = img_dim
        self.disc = nn.Sequential(
            self._block(nc, f_disc * 2, 2, 1, 0), #32x32, 128
            self._block(f_disc * 2, f_disc * 4, 2, 1, 0), #16x16, 256
            self._block(f_disc * 4, f_disc * 8, 2, 1, 0), #8x8, 512
            self._block(f_disc * 8, f_disc * 16, 2, 1, 0), #4x4, 1024
            nn.AdaptiveAvgPool2d(1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=0),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class  DCGAN:
    def __init__(self, img_dim, z_dim, f_disc=64, f_gen=64):
        self.disc = DCDisc(img_dim, f_disc)
        self.gen = DCGen(img_dim, z_dim, f_gen)

