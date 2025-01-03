import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 长方体
length = 2
width = 1
height = 3
ax.bar3d(1, 1, 1, length, width, height, shade=True, color='r', alpha=0.7, edgecolor='black', linewidth=1)
# 长方体
length = 2
width = 1
height = 3
ax.bar3d(1.1, 1.1, 1.1, length, width, height, shade=True, color='r', alpha=0.7, edgecolor='black', linewidth=1)

# 球体
radius = 1
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
x_sphere = radius * np.sin(phi) * np.cos(theta) + 4
y_sphere = radius * np.sin(phi) * np.sin(theta) + 2
z_sphere = radius * np.cos(phi) + 1
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='g', alpha=0.7, edgecolor='black', linewidth=0.5)

# 圆柱体
radius_cylinder = 1
height_cylinder = 2
x_cylinder, y_cylinder = np.linspace(7, 9, 100), np.linspace(4, 6, 100)
x_cylinder, y_cylinder = np.meshgrid(x_cylinder, y_cylinder)
z_cylinder = height_cylinder * np.ones_like(x_cylinder)
ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='b', alpha=0.7, edgecolor='black', linewidth=0.5)

# 设置图形属性
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Visualization')

# 显示图形
plt.show()
