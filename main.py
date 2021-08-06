import os
import math
import argparse

import PIL
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

def get_random_noise(dim, batch_size=1): return torch.randn((batch_size, dim))

def get_basic_transform(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize((0.5,), (0.5,))
    ])

class ImageDir(Dataset):
    def __init__(self, dirctry, transform, z_dim=64):
        self.dirctry = dirctry
        self.files = [dirctry+'/'+f for f in os.listdir(dirctry)]
        self.transform = transform

        self.z_dim = z_dim

    def __getitem__(self, i):
        noise = get_random_noise(self.z_dim)[0]
        img = Image.open(self.files[0])
        img = self.transform(img)

        return img, noise

class Discriminator(nn.Module):
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

class Generator(nn.Module):
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

def plt_imgs(imgs, batch_size, save_path):
    rows, cols = 4, batch_size // 4
    fig, axs = plt.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):
            img = imgs[(r * cols + c % cols)]
            axs[r, c].imshow(img, cmap='gray')

    plt.savefig(save_path, dpi=fig.dpi)


def train_basic_gan(options):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if options.dataset is None:
        img_size = 28
        basic_tsfm = get_basic_transform(img_size)
        dataset = datasets.FashionMNIST(root='dataset/', transform=basic_tsfm, download=True)
        img_dim = 1*img_size*img_size

    lr = 3e-4
    batch_size = options.batch_size
    z_dim = options.z_dim

    disc = Discriminator(img_dim)
    gen = Generator(z_dim, img_dim)
    disc, gen = disc.to(device), gen.to(device)

    criterion = nn.BCELoss()
    optim_disc = optim.Adam(disc.parameters(), lr=lr)
    optim_gen = optim.Adam(gen.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for b, (real_imgs, _) in enumerate(loader):
        real_imgs = real_imgs.view(-1, img_dim).to(device)
        noises = get_random_noise(z_dim, batch_size)
        px = disc(imgs)

        if b <= 0:
            fake_imgs = gen(noises)
            fake_imgs = fake_imgs.view(-1, img_size, img_size)
            plt_imgs(fake_imgs.detach().cpu().numpy(), batch_size, 'fakeimgs.png')
            realimgs = real_imgs.view(-1, img_size, img_size)
            plt_imgs(real_imgs.detach().cpu().numpy(), batch_size, 'realimgs.png')

        pg = disc(gen(noises))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='path to image folder you want to train on', default=None)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--z_dim', type=int, help='dimensionality of noise', default=64)

    options = parser.parse_args()

    print(options)

    train_basic_gan(options)
