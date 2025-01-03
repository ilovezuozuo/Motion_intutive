import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义两个长方体的顶点坐标
cube1_vertices = torch.tensor([
    [2.5, 2.5, 1.5], [2.5, -2.5, 1.5], [-2.5, -2.5, 1.5], [-2.5, 2.5, 1.5], [2.5, 2.5, -1.5], [2.5, -2.5, -1.5], [-2.5, -2.5, -1.5], [-2.5, 2.5, -1.5]
], dtype=torch.float32)

cube2_vertices = torch.tensor([
    [-0.30000001192092896, 0.25, -0.25], [-0.30000001192092896, -0.25, -0.25], [-1.2999999523162842, -0.25, -0.25],
    [-1.2999999523162842, 0.25, -0.25], [-0.30000001192092896, 0.25, -0.75], [-0.30000001192092896, -0.25, -0.75],
    [-1.2999999523162842, -0.25, -0.75], [-1.2999999523162842, 0.25, -0.75]
], dtype=torch.float32)

# 计算 Minkowski Difference
minkowski_diff = cube1_vertices[:, None, :] - cube2_vertices[None, :, :]
print('minkowski_diff:', minkowski_diff)
# 判断每个差的每个点的象限位置
quadrant_of_minkowski_diff = torch.sign(minkowski_diff)

# 打印每个点的象限位置
print("每个差的每个点的象限位置：")
print(quadrant_of_minkowski_diff)
#
# # 计算差的点数
# num_points_in_diff = torch.sum(torch.all(minkowski_diff != 0, dim=-1))
#
# print("差的点数：", num_points_in_diff.item())

import torch
import numpy as np
from scipy.spatial import ConvexHull


# 转换为顶点的 NumPy 数组
vertices_np = minkowski_diff.view(-1, 3).numpy()

# 计算凸包
hull = ConvexHull(vertices_np)

# 判断原点是否在凸多面体内
origin_inside = all(hull.equations[:, -1] > 0)

if origin_inside:
    print("凸多面体包含原点。")
else:
    print("凸多面体不包含原点。")
