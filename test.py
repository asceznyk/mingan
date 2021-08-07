import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


m = nn.AdaptiveAvgPool2d(100)
input = torch.randn(1, 1024, 4, 4)
output = m(input)

print(input.size(), output.size())
