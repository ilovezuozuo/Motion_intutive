import torch
import math
import argparse
from data.old_data import data
from Network import MLP
from differentiable_computation_engine import IK, SAT
from utilities import utilities
from torchviz import make_dot
from utilities import utilities

a = torch.tensor([[0., 0, 0, 0, 0, 0, 6, 6, 20]], requires_grad=True)  # 传入连杆的长方体信息rpyxyzlwh,这里是底面的！
b = torch.tensor([[0, 0, 0], [0., 0., 0.], [2, 2, 2]], requires_grad=True)  # 传入障碍物的长方体信息rpyxyzlwh,这里是中心的！
# .view(3, 3)可以将a变成b的组合形式
for i in range(len(a)):
    print(utilities.calculate_cube_center_from_bottom(a))
    print('b:', b)
    aaaaa = SAT.arm_obs_collision_detect(utilities.calculate_cube_center_from_bottom(a)[i].view(3, 3), b)

# cuboid_1 = torch.tensor([[0, 0, 0], [0., 0., 1.], [2, 2, 30]], requires_grad=True)
# # cuboid_2 = torch.tensor([[-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5]], requires_grad=True)
# # cuboid_2 = torch.tensor([[10, 10, 10], [0, 0, 0.0], [5, 5, 15]], requires_grad=True)
# cuboid_2 = torch.tensor([[5, 5, 0], [0, 0, 0.0], [20, 20, 30]], requires_grad=True)
# # print(SAT.collosion_detect(cuboid_1, cuboid_2))
# aaaaa = SAT.arm_obs_collision_detect(cuboid_1, cuboid_2)

# make_dot(aaaaa).view()
