import os
import math
import argparse

import PIL
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        return self.transform(img), 1

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

    lr = options.lr
    img_size = options.img_size
    z_dim = options.z_dim
    batch_size = options.batch_size
    epochs = options.epochs
    k_steps = options.k_steps
    img_dim = 1*img_size*img_size

    basic_tsfm = get_basic_transform(options.img_size)

    if options.dataset is None:
        dataset = datasets.MNIST(root='dataset/', transform=basic_tsfm, download=True)
    else:
        dataset = ImageDir(options.dataset, basic_tsfm, z_dim)

    disc = Discriminator(img_dim)
    gen = Generator(z_dim, img_dim)
    disc, gen = disc.to(device), gen.to(device)

    criterion = nn.BCELoss()
    optim_disc = optim.Adam(disc.parameters(), lr=lr)
    optim_gen = optim.Adam(gen.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    noise = lambda : get_random_noise(z_dim, batch_size).to(device)

    for e in range(epochs):
        for b, (real_imgs, _) in tqdm(enumerate(loader), total=len(loader)):
            for k in range(k_steps):
                real_imgs = real_imgs.view(-1, img_dim).to(device)
                fake_imgs = gen(noise()).to(device)

                px = disc(real_imgs)
                pg = disc(fake_imgs)

                loss_dx = criterion(px, torch.ones_like(px)) #log(d(x))
                loss_dg = criterion(pg, torch.zeros_like(pg)) #log((1 - d(g(z))))
                loss_d = (loss_dx + loss_dg)

                disc.zero_grad()
                loss_d.backward(retain_graph=True)
                optim_disc.step()

            fake_imgs = gen(noise()).to(device)
            pg = disc(fake_imgs)

            loss_g = criterion(pg, torch.ones_like(pg)) #log(d(g(z))) maximizing
            gen.zero_grad()
            loss_g.backward()
            optim_gen.step()

            if e % 10 == 0 and b == 0:
                real_imgs = real_imgs.view(-1, img_size, img_size)
                real_imgs = real_imgs.detach().cpu().numpy()
                plt_imgs(real_imgs, batch_size, f'realimgs{e}.png')

                with torch.no_grad():
                    fake_imgs = gen(noise())
                    fake_imgs = fake_imgs.view(-1, img_size, img_size).detach().cpu().numpy()
                    plt_imgs(fake_imgs, batch_size, f'fakeimgs{e}.png')

        print(f'discriminator loss at epoch {e} = {loss_d.item()}')
        print(f'generator loss at epoch {e} = {loss_g.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='path to image folder you want to train on', default=None)
    parser.add_argument('--img_size', type=int,  help='image size', default=28)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--z_dim', type=int, help='dimensionality of noise', default=64)
    parser.add_argument('--k_steps', type=int, help='number of steps to train discriminator', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs to train model', default=30)
    parser.add_argument('--lr', type=float, help='learning rate', default=3e-4)

    options = parser.parse_args()

    print(options)

    train_basic_gan(options)

