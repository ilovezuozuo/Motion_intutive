from torchviz import make_dot
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

import torch
from differentiable_computation_engine import IK, FK

# 定义机械臂DH参数，以下为UR10e参数
a = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
d = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
alpha = torch.FloatTensor([torch.pi / 2, 0, 0, torch.pi / 2, -torch.pi / 2, 0])  # link twist


tar = torch.FloatTensor([
[0, 0, 0.50, 3.70, 3.60, 0.90]
])
base = torch.FloatTensor([
    [0, 0, 0, 3, 3.2, 0]
])
theta = torch.FloatTensor([
[-2.8402, -1.4138,  1.4180, -1.5750,  1.5708, -1.3722]
])

for i in range(len(tar)):
    print(IK.calculate_IK(IK.shaping(tar)[i], IK.shaping(base)[i], a, d, alpha))
    print(IK.shaping(tar)[i])
for i in range(len(tar)):
    print(FK.FK(theta = theta[i], base = IK.shaping(base)[i], a = a, d = d, alpha = alpha))


