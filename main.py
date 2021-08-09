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

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_random_noise(dim, batch_size=1, single=True):
    n = torch.randn((batch_size, dim))
    n = n if single else n.view(batch_size, dim, 1, 1)
    return n.to(device)

def get_transform(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize((0.5,), (0.5,))
    ])

class ImageDir(Dataset):
    def __init__(self, dirctry, transform):
        self.dirctry = dirctry
        self.files = [dirctry+'/'+f for f in os.listdir(dirctry)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        return self.transform(img), 1

def plt_imgs(imgs, batch_size, save_path):
    rows, cols = 4, batch_size // 4
    fig, axs = plt.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):
            img = imgs[(r * cols + c % cols)]
            img = np.transpose(img, (1,2,0))
            if img.shape[2] == 1:
                img = img[:,:,0]
            else:
                img = img.clip(min=0)

            axs[r, c].imshow(img)

    plt.savefig(save_path, dpi=fig.dpi)

def init_dataset(mode, tsfm):
    root = 'dataset/'
    if mode == 'dcgan':
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=tsfm)
    else:
        dataset = datasets.FashionMNIST(root=root, transform=tsfm, download=True)
    return dataset

def init_gan(mode, img_size, z_dim, batch_size, nc):
    if mode == 'dcgan':
        lr = 2e-4
        img_dim = (nc, img_size, img_size)
        gan = DCGAN(img_dim, z_dim)
        init_weights(gan.disc)
        init_weights(gan.gen)
        single = False
    else:
        if mode != 'fcgan':
            print('mode is not specified so training an FCGAN')
        lr = 3e-4
        img_dim = (nc * img_size * img_size, )
        gan = FCGAN(img_dim[0], z_dim)
        single = True

    noise = lambda : get_random_noise(z_dim, batch_size, single=single)

    return gan.disc.to(device), gan.gen.to(device), noise, lr, img_dim

def train_gan(options):
    img_size = options.img_size
    batch_size = options.batch_size
    epochs = options.epochs
    k_steps = options.k_steps
    z_dim = options.z_dim
    mode = options.mode

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])

    if options.dataset is None:
        dataset = init_dataset(mode, tsfm)
    else:
        dataset = ImageDir(options.dataset, tsfm)

    nc = dataset[0][0].size(0)
    disc, gen, noise, lr, img_dim = init_gan(mode, img_size, z_dim, batch_size, nc)
    criterion = nn.BCELoss()
    betas = (0.5, 0.999) if mode == 'dcgan' else (0.9, 0.999)
    optim_disc = optim.Adam(disc.parameters(), lr=lr, betas=betas)
    optim_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)

    gen.train()
    disc.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for e in range(1, epochs+1):
        pbar = tqdm(enumerate(loader), total=len(loader))
        for b, (real_imgs, _) in pbar:
            norm_means, norm_stds = ([0.5 for _ in range(nc)]  for _ in range(2))
            real_imgs = transforms.functional.normalize(real_imgs, norm_means, norm_stds)
            for k in range(k_steps):
                real_imgs = real_imgs.view(-1, *img_dim).to(device)
                fake_imgs = gen(noise()).to(device)

                px = disc(real_imgs)
                pg = disc(fake_imgs)

                loss_dx = criterion(px, torch.ones_like(px)) #log(d(x))
                loss_dg = criterion(pg, torch.zeros_like(pg)) #log((1 - d(g(z))))
                loss_d = (loss_dx + loss_dg) / 2

                disc.zero_grad()
                loss_d.backward()
                #loss_d.backward(retain_graph=True)
                optim_disc.step()

            fake_imgs = gen(noise()).to(device)
            pg = disc(fake_imgs)

            loss_g = criterion(pg, torch.ones_like(pg)) #log(d(g(z))) maximizing
            gen.zero_grad()
            loss_g.backward()
            optim_gen.step()

            if b % 100 == 0:
                real_imgs = real_imgs.view(-1, nc, img_size, img_size)
                real_imgs = real_imgs.detach().cpu().numpy()
                plt_imgs(real_imgs, 32, f'realimgs{b}.png')

                with torch.no_grad():
                    fake_imgs = gen(noise())
                    fake_imgs = fake_imgs.view(-1, nc, img_size, img_size).detach().cpu().numpy()
                    plt_imgs(fake_imgs, 32, f'fakeimgs{b}.png')

            pbar.set_description(f'discriminator loss at epoch {e} = {loss_d.item():.4f}; generator loss at epoch {e} = {loss_g.item():.4f};')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='path to image folder you want to train on', default=None)
    parser.add_argument('--mode', type=str, help='you can choose to train a fully connected GAN or a deep convolutional GAN (options : [fcgan, dcgan])', default='fcgan')
    parser.add_argument('--img_size', type=int,  help='image size', default=28)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--z_dim', type=int, help='dimensionality of noise', default=64)
    parser.add_argument('--k_steps', type=int, help='number of steps to train discriminator', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs to train model', default=30)

    options = parser.parse_args()

    print(options)

    train_gan(options)

