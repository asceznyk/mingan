import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import *

img_dim = (3, 64, 64)
z_dim = 100

#disc = DCDisc(img_dim)
#out = disc(torch.randn(1, *img_dim))
#print(out, out.size())

gen = DCGen(img_dim, z_dim)
out = gen(torch.randn(1, z_dim, 1, 1))
print(out, out.size())
