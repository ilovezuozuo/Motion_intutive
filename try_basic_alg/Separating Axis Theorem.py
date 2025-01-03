import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def plot_linear_cube(PS1, PS2, color1='red', color2='blue'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一个正方体
    for i in range(4):
        ax.plot3D([PS1[i, 0], PS1[(i + 1) % 4, 0]],
                  [PS1[i, 1], PS1[(i + 1) % 4, 1]],
                  [PS1[i, 2], PS1[(i + 1) % 4, 2]], color=color1)

        ax.plot3D([PS1[i + 4, 0], PS1[((i + 1) % 4) + 4, 0]],
                  [PS1[i + 4, 1], PS1[((i + 1) % 4) + 4, 1]],
                  [PS1[i + 4, 2], PS1[((i + 1) % 4) + 4, 2]], color=color1)

        ax.plot3D([PS1[i, 0], PS1[i + 4, 0]],
                  [PS1[i, 1], PS1[i + 4, 1]],
                  [PS1[i, 2], PS1[i + 4, 2]], color=color1)

    # 绘制第二个正方体
    for i in range(4):
        ax.plot3D([PS2[i, 0], PS2[(i + 1) % 4, 0]],
                  [PS2[i, 1], PS2[(i + 1) % 4, 1]],
                  [PS2[i, 2], PS2[(i + 1) % 4, 2]], color=color2)

        ax.plot3D([PS2[i + 4, 0], PS2[((i + 1) % 4) + 4, 0]],
                  [PS2[i + 4, 1], PS2[((i + 1) % 4) + 4, 1]],
                  [PS2[i + 4, 2], PS2[((i + 1) % 4) + 4, 2]], color=color2)

        ax.plot3D([PS2[i, 0], PS2[i + 4, 0]],
                  [PS2[i, 1], PS2[i + 4, 1]],
                  [PS2[i, 2], PS2[i + 4, 2]], color=color2)

    plt.title('Cubes')
    plt.show()



'''
    Transform the roll pitch yaw to rotation matrix
'''
def RPY_to_Rotation(RPY_list):
    roll, pitch, yaw = RPY_list[0], RPY_list[1], RPY_list[2]
    yawMatrix = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix @ pitchMatrix @ rollMatrix
    return R

# the cubes do not collide with each other in one axis only when the max of one cube is smaller than another cube
# under the same axis
def collide_check(ref_min, ref_max, _min, _max):
    print("ref_min, ref_max, _min, _max", ref_min, ref_max, _min, _max)
    if ref_min > _max or ref_max < _min:
        return False
    return True

# Detect collision in each dimension
def collision_detect(ref_corner, cuboid_corner):
    x_min = np.min(cuboid_corner[:, 0])
    x_max = np.max(cuboid_corner[:, 0])
    y_min = np.min(cuboid_corner[:, 1])
    y_max = np.max(cuboid_corner[:, 1])
    z_min = np.min(cuboid_corner[:, 2])
    z_max = np.max(cuboid_corner[:, 2])

    xref_min = np.min(ref_corner[:, 0])
    xref_max = np.max(ref_corner[:, 0])
    yref_min = np.min(ref_corner[:, 1])
    yref_max = np.max(ref_corner[:, 1])
    zref_min = np.min(ref_corner[:, 2])
    zref_max = np.max(ref_corner[:, 2])

    x_collide = collide_check(xref_min, xref_max, x_min, x_max)
    y_collide = collide_check(yref_min, yref_max, y_min, y_max)
    z_collide = collide_check(zref_min, zref_max, z_min, z_max)

    return (x_collide and y_collide and z_collide)

def Check_Collision(cuboid_ref, cuboid):
    T_matrix = np.array([[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1],[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1]])
    Projection_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])     # Here stores the information of the projection axis

    # Calculate all possible projection axis for both cubes in respect to the base frame
    Projection_axis = []
    Rotation_ref = RPY_to_Rotation(cuboid_ref["Orientation"])
    Rotation_cub = RPY_to_Rotation(cuboid["Orientation"])

    PA_ref = Projection_matrix @ Rotation_ref
    PA_cub = Projection_matrix @ Rotation_cub
    # print('PA_ref:', PA_ref)
    # print('PA_cub:', PA_cub)
    Projection_axis.append(PA_ref)
    Projection_axis.append(PA_cub)

    for i in range(3):
        base_axis = PA_ref[:,i].reshape(3)
        # print('base_axis:', base_axis)
        PA = np.zeros((3,3))
        for j in range(3):
            a = np.cross(base_axis, PA_cub[:,j].reshape(3))
            PA[:,j] = a.reshape(3)
        Projection_axis.append(PA)

    print(len(Projection_axis))
    # Rotate each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    cuboid_corner_initial = np.array(
        [cuboid["Dimension"][0] / 2, cuboid["Dimension"][1] / 2, cuboid["Dimension"][2] / 2])
    cuboid_corner_dimension = np.tile(cuboid_corner_initial, (8, 1)) # 给定维度上重复数组
    # print('cuboid_corner_dimension:', cuboid_corner_dimension)
    # print('T_matrix:', T_matrix)
    cuboid_corner = cuboid_corner_dimension * T_matrix
    # print('cuboid_corner:', cuboid_corner)


    # Rotate each corner point relative to cube's center (do not consider the position relative to base frame's origin)
    ref_corner_initial = np.array(
        [cuboid_ref["Dimension"][0] / 2, cuboid_ref["Dimension"][1] / 2, cuboid_ref["Dimension"][2] / 2])
    ref_corner_dimension = np.tile(ref_corner_initial, (8, 1))
    ref_corner = ref_corner_dimension * T_matrix

    # Add origin to get the absolute cordinates of each corner point
    ref_corners = ref_corner @ Rotation_ref + np.array(cuboid_ref["Origin"])
    cub_corners = cuboid_corner @ Rotation_cub + np.array(cuboid["Origin"])
    # Uncomment below to plot current position of two cubes
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', ref_corners)

    # print(cub_corners)
    plot_linear_cube(ref_corners, cub_corners)

    Collision_or_not = True
    for PA in Projection_axis:
        cuboid_corner_new = cub_corners @ PA.T
        # print("PA.T:", PA.T)
        ref_corner_new = ref_corners @ PA.T
        Collision_Decision = collision_detect(ref_corner_new, cuboid_corner_new)
        Collision_or_not = Collision_Decision and Collision_or_not

    return Collision_or_not

def collosion_detect(cuboid_1,cuboid_2):
    result = Check_Collision(cuboid_1, cuboid_2)   #  In reference of cuboid1
    return result

def main():
    cuboid_1 = {"Origin": [1, 0, 0], "Orientation": [0, 0, 0.5], "Dimension": [5, 5, 3]}
    cuboid_2 = {"Origin": [-0.8, 0, -0.5], "Orientation": [0, 0, 0.2], "Dimension": [1, 0.5, 0.5]}
    print(collosion_detect(cuboid_1,cuboid_2))

if __name__ == '__main__':
    main()

    # PA.T: tensor([[-0.1987, 0.9801, 0.0000],
    #               [-0.9801, -0.1987, 0.0000],
    #               [0.0000, 0.0000, 0.0000]], grad_fn= < PermuteBackward0 >)