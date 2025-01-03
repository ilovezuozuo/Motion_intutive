import cv2
import numpy as np
########################################################################################################################################
# # # 2维的包围盒代码示例
# # 创建一个随机的点集
# points = np.random.randint (0, 100, (10, 2))
#
# # 画出点集和它们的凸包
# img = np.zeros ((100, 100, 3), dtype=np.uint8)
# cv2.polylines (img, [points], True, (0, 255, 0), 2)
# cv2.drawContours (img, [points], -1, (255, 0, 0), -1)
#
# # 使用cv2.boundingRect()函数生成bounding box
# x, y, w, h = cv2.boundingRect (points)
#
# # 画出bounding box
# cv2.rectangle (img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# # 显示图像
# cv2.imshow ("Bounding Box", img)
# cv2.waitKey (0)
########################################################################################################################################

########################################################################################################################################
# 3维的包围盒示例（不成功，环境少东西）
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 随机生成一些三维点云
num_points = 100
point_cloud = torch.rand((num_points, 3))

# 计算包围盒
min_coords = point_cloud.min(dim=0).values
max_coords = point_cloud.max(dim=0).values

# 可视化点云和包围盒
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 可视化点云
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', label='Point Cloud')

# 可视化包围盒
bbox_min = min_coords.numpy()
bbox_max = max_coords.numpy()
print(bbox_max)
bbox_dims = bbox_max - bbox_min

ax.set_xlim(bbox_min[0], bbox_max[0])
ax.set_ylim(bbox_min[1], bbox_max[1])
ax.set_zlim(bbox_min[2], bbox_max[2])

# 绘制包围盒
x = [bbox_min[0], bbox_max[0], bbox_max[0], bbox_min[0], bbox_min[0]]
y = [bbox_min[1], bbox_min[1], bbox_max[1], bbox_max[1], bbox_min[1]]
z = [bbox_min[2], bbox_min[2], bbox_max[2], bbox_max[2], bbox_min[2]]
ax.plot(x, y, z, color='r', linestyle='dotted', label='Bounding Box')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()


########################################################################################################################################
