import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

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
    def __init__(self, img_dim, f_disc=64):
        super(DCDisc, self).__init__()
        nc, nw, nh = img_dim
        self.disc = nn.Sequential(
            nn.Conv2d(nc, f_disc, 4, 2, 1), #32x32, 64 if(img_size=64)
            nn.LeakyReLU(0.2),
            self._block(f_disc, f_disc * 2, 4, 2, 1), #16x16, 128
            self._block(f_disc * 2, f_disc * 4, 4, 2, 1), #8x8, 256
            self._block(f_disc * 4, f_disc * 8, 4, 2, 1), #4x4, 512
            nn.Conv2d(f_disc * 8, 1, 4, 2, 0), #1x1, 1
            nn.Sigmoid(),
            nn.Flatten()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)

class DCGen(nn.Module):
    def __init__(self, img_dim, z_dim, f_gen=64):
        super(DCGen, self).__init__()
        nc, _, _ = img_dim
        self.gen = nn.Sequential(
            self._block(z_dim, f_gen * 16, 4, 1, 0), #1x1, 100
            self._block(f_gen * 16, f_gen * 8, 4, 2, 1), #8x8, 512
            self._block(f_gen * 8, f_gen * 4, 4, 2, 1), #16x16, 256
            self._block(f_gen * 4, f_gen * 2, 4, 2, 1), #32x32, 128
            nn.ConvTranspose2d(f_gen * 2, nc, 4, 2, 1), #64x64, nc
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

class  DCGAN:
    def __init__(self, img_dim, z_dim, f_disc=64, f_gen=64):
        self.disc = DCDisc(img_dim, f_disc)
        self.gen = DCGen(img_dim, z_dim, f_gen)


