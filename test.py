import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

img_dim = (3, 64, 64)
disc = DCDisc(img_dim)

out = disc(torch.randn(1, *img_dim))
print(out)
