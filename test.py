import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import *

img_dim = (3, 64, 64)
disc = DCDisc(img_dim)

out = disc(torch.randn(1, *img_dim))
print(out, out.size())

m1 = nn.ConvTranspose2d(100, 1024, 5, 1, 2)
m2 = nn.ConvTranspose2d(1024, 512, 5, 1, 2)

inp = torch.randn(1, 100, 1, 1)

out1 = m1(inp)
print(out1, out1.size())
out2 = m2(out1)
print(out2, out2.size())
