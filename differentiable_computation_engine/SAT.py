import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utilities import utilities
import torch
from differentiable_computation_engine import IK
from torchviz import make_dot

'''
    Transform the roll pitch yaw to rotation matrix
'''

# 只有当一个立方体的最大值小于同一轴线下的另一个立方体时，立方体才不会在一个轴线上相互碰撞
# 坐标系三方向碰撞损失函数定义在这里
def collide_check(ref_min, ref_max, obsmin, obsmax):

    # print("ref_min, ref_max, obsmin, obsmax", ref_min, ref_max, obsmin, obsmax)
    loss = torch.tensor([0])
    if ref_min > obsmax:
        loss = loss + torch.relu(obsmax - ref_min)
    elif ref_max < obsmin:
        loss = loss + torch.relu(ref_max - obsmin)
    elif ref_min < obsmax:
        loss = loss + (obsmax - ref_min)
    elif ref_max > obsmin:
        loss = loss + (ref_max - obsmin)
    # print("检查点3，loss", loss)
    return loss

# Detect collision in each dimension
def collision_detect(ref_corner, cuboid_corner):

    x_min = torch.min(cuboid_corner[:, 0])
    x_max = torch.max(cuboid_corner[:, 0])
    y_min = torch.min(cuboid_corner[:, 1])
    y_max = torch.max(cuboid_corner[:, 1])
    z_min = torch.min(cuboid_corner[:, 2])
    z_max = torch.max(cuboid_corner[:, 2])

    xref_min = torch.min(ref_corner[:, 0])
    xref_max = torch.max(ref_corner[:, 0])
    yref_min = torch.min(ref_corner[:, 1])
    yref_max = torch.max(ref_corner[:, 1])
    zref_min = torch.min(ref_corner[:, 2])
    zref_max = torch.max(ref_corner[:, 2])



    if xref_min == xref_max == x_min == x_max == 0:
        x_collide = 10000
    else:
        x_collide = collide_check(xref_min, xref_max, x_min, x_max)
    if yref_min==yref_max== y_min== y_max == 0:
        y_collide = 10000
    else:
        y_collide = collide_check(yref_min, yref_max, y_min, y_max)
    if zref_min == zref_max== z_min== z_max== 0:
        z_collide = 10000
    else:
        z_collide = collide_check(zref_min, zref_max, z_min, z_max)
    # print(zref_min, zref_max, z_min, z_max)
    # print('x_collide', x_collide)
    # print('y_collide', y_collide)
    # print('z_collide', z_collide)

    # 任意一个轴上“重叠距离”是0，就表明肯定存在分离平面，直接return
    if x_collide == torch.tensor([0]):
        return x_collide
    elif y_collide == torch.tensor([0]):
        return y_collide
    elif z_collide == torch.tensor([0]):
        return z_collide
    else:
        if x_collide == 10000:
            return (y_collide + z_collide)
        elif y_collide == 10000:
            return (x_collide + z_collide)
        elif z_collide == 10000:
            return (x_collide + y_collide)
        else:
            return (x_collide + y_collide + z_collide)



def Check_Collision(cuboid_ref, cuboid): # 传入参数形式：中心点，rpy， 长宽高 cuboid_1 = torch.tensor([[0, 0, 0], [0, 0, 0.0], [5, 5, 3]], requires_grad=True)
    # print(cuboid_ref, cuboid)

    T_matrix = torch.tensor([[1., 1., 1.], [1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.],
                             [1., 1., -1.], [1., -1., -1.], [-1., -1., -1.], [-1., 1., -1.]])
    Projection_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)

    # 存放潜在分离轴
    Projection_axis = []

    # 欧拉转旋转矩阵，计算两坐标系
    Rotation_ref = IK.euler_to_rotMat(cuboid_ref[1][2], cuboid_ref[1][1], cuboid_ref[1][0])
    Rotation_cub = IK.euler_to_rotMat(cuboid[1][2], cuboid[1][1], cuboid[1][0])

    # 计算6条分离轴（坐标轴）
    # print(type(Projection_matrix))
    # print(type(Rotation_ref))
    # print('Projection_matrix', Projection_matrix)
    # print('Rotation_ref', Rotation_ref)
    # print('Rotation_cub', Rotation_cub)
    Rotation_ref = Rotation_ref.to(torch.float64)

    PA_ref = torch.matmul(Projection_matrix, Rotation_ref)
    Projection_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

    PA_cub = torch.matmul(Projection_matrix, Rotation_cub)
    Projection_axis.append(PA_ref)
    Projection_axis.append(PA_cub)
    # 计算另外9条叉乘分离轴
    for i in range(3):
        base_axis = PA_ref[:,i].reshape(3)
        # print('base_axis:', base_axis)
        PA = torch.zeros((3, 3))
        for j in range(3):

            # print("base_axis", base_axis)
            # print("PA_cub[:,j].reshape(3)", PA_cub[:,j].reshape(3))
            base_axis = base_axis.to(torch.float32)

            a = torch.cross(base_axis, PA_cub[:,j].reshape(3))
            # print('a', base_axis,PA_cub[:,j],a)
            PA[:,j] = a.reshape(3)

        # print('PA:', PA)
        Projection_axis.append(PA)
    # print('Projection_axis:', Projection_axis)
    # print(len(Projection_axis))
    # Rotate each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    cuboid_corner_initial = torch.tensor([cuboid[2][0] / 2, cuboid[2][1] / 2, cuboid[2][2] / 2],
                                         dtype=torch.float32)
    cuboid_corner_dimension = torch.tile(cuboid_corner_initial, (8, 1)) # 给定维度上重复数组
    # print('cuboid_corner_dimension:', cuboid_corner_dimension)
    # print('T_matrix:', T_matrix)
    cuboid_corner = cuboid_corner_dimension * T_matrix
    # print('cuboid_corner:', cuboid_corner)


    # Rotate each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    ref_corner_initial = torch.tensor(
        [cuboid_ref[2][0] / 2, cuboid_ref[2][1] / 2, cuboid_ref[2][2] / 2],
        dtype=torch.float32)
    ref_corner_dimension = torch.tile(ref_corner_initial, (8, 1))
    ref_corner = ref_corner_dimension * T_matrix
    # print('ref_corner:', ref_corner)

    # print('cuboid_ref[0]--------------------------------------------------:', cuboid_ref[0])
    # Add origin to get the absolute cordinates of each corner point
    ref_corner = ref_corner.to(torch.float64)

    ref_corners = torch.matmul(ref_corner, Rotation_ref) + cuboid_ref[0]
    cub_corners = torch.matmul(cuboid_corner, Rotation_cub) + cuboid[0]
    # Uncomment below to plot current position of two cubes
    # print('ref_corners:', ref_corners.tolist())
    # print('cub_corners:', cub_corners.tolist())
    # print('cub_corners:', cub_corners)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # utilities.plot_cube2(np.array(ref_corners.tolist()), np.array(cub_corners.tolist()))

    Collision_or_not = True
    losss = torch.tensor([0])
    for PA in Projection_axis:
        cub_corners = cub_corners.to(torch.float64)
        PA = PA.to(torch.float64)

        # print('cub_corners:',cub_corners)
        # print('PA:', PA)
        cuboid_corner_new = torch.matmul(cub_corners, PA.T)
        ref_corner_new = torch.matmul(ref_corners, PA.T)
        # print('PA.T:', PA.T)
        # print('cuboid_corner_new:', cuboid_corner_new)
        # print('ref_corner_new:', ref_corner_new)

        # 是否碰撞以及相应的损失函数由collision_detect确定
        Collision_Decision = collision_detect(ref_corner_new, cuboid_corner_new)
        # print("当前投影轴上的碰撞情况是（0的话证明有某个方向上不碰撞）：", Collision_Decision)
        if Collision_Decision == 0:
            # print("找到了一个分离轴，这两个物体不碰撞")
            losss = torch.tensor([0])
            losss = losss + Collision_Decision
            break
        else:
            losss = losss + Collision_Decision
    # print("检查点2，losss", losss)

    return losss

def arm_obs_collision_detect(cuboid_1, cuboid_2):   # 传入参数形式：rpy，中心点，长宽高
    # cuboid_1 = torch.tensor([[3, 3, 3], [0, 0, 0.5], [5, 5, 3]])
    # cuboid_2 = torch.tensor([[-0.8, 0, -0.5], [0, 0, 0.2], [1, 0.5, 0.5]])
    cuboid_1 = cuboid_1[[1, 0, 2], :]
    cuboid_2 = cuboid_2[[1, 0, 2], :]
    result = Check_Collision(cuboid_1, cuboid_2)
    # make_dot(result).view()
    # print("这里输出的应该是两个物体是否碰撞的loss了,"
    #       "若是碰撞，各个潜在分离轴上全部惩罚，若不碰撞，这里应该是0：", result)

    return result


