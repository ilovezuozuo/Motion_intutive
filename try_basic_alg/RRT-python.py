from math import sqrt
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 初始化整个空间，定义初始点、终点、采样点数、点与点之间的步长t等信息
x_width = 25  # 空间的长度
y_width = 12  # 空间的宽度
error_list = [[0 for i in range(0, x_width)] for j in range(0, y_width)]
error_list[2][10] = 1
error_list[3][10] = 1
error_list[4][10] = 1
error_list[5][10] = 1
error_list[6][10] = 1
error_list[7][10] = 1
error_list[8][10] = 1

x0 = 6  # 定义初始点的x坐标
y0 = 4  # 定义初始点的y坐标
xn = 17  # 定义终点的x坐标
yn = 5  # 定义终点的y坐标
t = 1  # 点与点之间的步长
error_list[y0][x0] = 4
error_list[yn][xn] = 3
error_list = np.array(error_list)

# print(error_list)
plt.figure()
plt.xlim((-1, x_width))
plt.ylim((-1, y_width))
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(np.arange(x_width))
plt.yticks(np.arange(y_width))
plt.grid()

tree_list = []
tree_list.append([x0, y0, x0, y0])  # 把起点作为树的点放入列表，避免随机点与起点重合
plt.plot(x0, y0, 'ro')
plt.plot(xn, yn, marker='o', color='yellow')
plt.plot([10, 10, 10, 10, 10, 10, 10], [2, 3, 4, 5, 6, 7, 8], 'k-', linewidth='5')
# 在空间中随机产生一个点xrand ->这个点不能是起点
def product_rand(tree_list):
    x_width = 25  # 空间的长度
    y_width = 12  # 空间的宽度
    random_point = list(itertools.product(range(0, x_width), range(0, y_width)))
    xrand = random.sample(random_point, 1)
    xrand = list(xrand[0])  # 将随机点转换成list形式
    tree_list = np.array(tree_list)
    tree = tree_list[:, 0:2]
    while xrand in tree:  # 如果随机点在树的点列表里，重新生成随机点
        xrand = random.sample(random_point, 1)
        xrand = list(xrand[0])  # 将随机点转换成list形式
    return xrand
# 在已知树的点集合中找到距离这个随机点最近的点xnear
def product_near(tree_list, xrand):
    m = np.inf
    for i in range(0, len(tree_list)):
        if abs(tree_list[i][0] - xrand[0]) + abs(tree_list[i][1] - xrand[1]) < m:
            m = abs(tree_list[i][0] - xrand[0]) + abs(tree_list[i][1] - xrand[1])
            xnear = [tree_list[i][0], tree_list[i][1]]
    return xnear
def decide_direction(xrand, xnear, t):
    z_value = sqrt((xnear[0] - xrand[0]) ** 2 + (xnear[1] - xrand[1]) ** 2)  # 斜边长度
    cos_value = (xrand[0] - xnear[0]) / z_value
    sin_value = (xrand[1] - xnear[1]) / z_value
    xnew = [(xnear[0] + t * cos_value), (xnear[1] + t * sin_value)]
    return xnew
xrand = product_rand(tree_list)  # 随机生成点
xnear = product_near(tree_list, xrand)
xnew = decide_direction(xrand, xnear, t)
if xnear[0] != xrand[0]:
    k = (xrand[1] - xnear[1]) / (xrand[0] - xnear[0])
    y = k * (10 - xnear[0]) + xnear[1]
else:
    y = 0

while 10 <= max(xnear[0], xnew[0]) and 10 <= min(xnear[0], xnew[0]) and 2 <= y <= 8:
    xrand = product_rand(tree_list)  # 随机生成点
    xnear = product_near(tree_list, xrand)
    xnew = decide_direction(xrand, xnear, t)
    if xrand[0] - xnear[0] != 0:
        k = (xrand[1] - xnear[1]) / (xrand[0] - xnear[0])
        y = k * (10 - xnear[0]) + xnear[1]
tree_list.append([xnew[0], xnew[1], xnear[0], xnear[1]])
plt.plot(xrand[0], xrand[1], marker='o', color='cyan')
plt.plot(xnew[0], xnew[1], color='violet', marker='o')
# 循环
while ((xnew[0] - xn) ** 2 + (xnew[1] - yn) ** 2) > 1:
    xrand = product_rand(tree_list)  # 随机生成点
    xnear = product_near(tree_list, xrand)
    xnew = decide_direction(xrand, xnear, t)
    if xnear[0] != xrand[0]:
        k = (xrand[1] - xnear[1]) / (xrand[0] - xnear[0])
        y = k * (10 - xnear[0]) + xnear[1]
    else:
        y = 0
    while 10 <= max(xnear[0], xnew[0]) and 10 >= min(xnear[0], xnew[0]) and 2 <= y <= 8:
        xrand = product_rand(tree_list)  # 随机生成点
        xnear = product_near(tree_list, xrand)
        xnew = decide_direction(xrand, xnear, t)
        if xrand[0] - xnear[0] != 0:
            k = (xrand[1] - xnear[1]) / (xrand[0] - xnear[0])
            y = k * (10 - xnear[0]) + xnear[1]
    tree_list.append([xnew[0], xnew[1], xnear[0], xnear[1]])
    plt.plot(xrand[0], xrand[1], marker='o', color='cyan')
    plt.plot(xnew[0], xnew[1], color='violet', marker='o')
tree_list = np.array(tree_list)
routine_list = [[xn,yn]]
n = len(tree_list)-1
x = tree_list[n,0]
y = tree_list[n,1]
f_x = tree_list[n,2]
f_y = tree_list[n,3]
routine_list.append([x,y])
search_list=[]
while [x0,y0] not in routine_list:
    search_list = tree_list[np.where((tree_list[:,0]==f_x) & (tree_list[:,1]==f_y))][0]
    search_list = search_list.tolist()
    routine_list.append([search_list[0],search_list[1]])
    f_x = search_list[2]
    f_y = search_list[3]

print(routine_list)
routine_list = np.array(routine_list)
plt.plot(routine_list[:,0], routine_list[:,1], '-', linewidth='2')
plt.show()
