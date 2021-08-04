import os
import PIL

from PIL import Image

import torch

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

def get_random_noise(dim, batch_size=1): return torch.randn((batch_size, dim))

def get_basic_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size))
        transforms.Normalize(((0.5,), (0.5,), )),
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



