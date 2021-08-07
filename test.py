import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


m = nn.AdaptiveAvgPool1d(5)
input = torch.randn(1, 64, 8)
output = m(input)

print(input.size(), output.size())
