import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init()
        self.img_dim = img_dim
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyRELU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyRELU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

def train_basic_gan(options):
    basic_tsfm = get_basic_transform()
    if options.dataset is None:
        dataset = datasets.FashionMNIST(root='dataset/', transform=basic_tsfm, dowload=True)
        img_dim,  z_dim = 1*28*28, 64

    disc = Discriminator(img_dim)
    gen = Generator(z_dim, img_dim)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_batch = next(iter(loader))

    px = disc(sample_batch)

    print(px)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='path to image folder you want to train on', default=None)

    options = parser.parse_args()

    print(options)

    train_basic_gan(options)
