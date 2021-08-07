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

m = nn.ConvTranspose2d(100, 1024, 4, 1, 1)
inp = torch.randn(1, 100, 1, 1)
out = m(inp)

print(out, out.size())
