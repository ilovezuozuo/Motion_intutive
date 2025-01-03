import random
import matplotlib.pyplot as plt
import torch

def plot_movement_sequence(relative_sequence):
    # 将相对坐标转换为绝对坐标
    absolute_sequence = [(0, 0)]  # 起点
    current_position = [0, 0]

    for move in relative_sequence:
        current_position[0] += move[0]
        current_position[1] += move[1]
        absolute_sequence.append(tuple(current_position))

    # 提取x和y坐标
    x_coords, y_coords = zip(*absolute_sequence)

    # 绘制路径
    plt.figure(figsize=(6, 6))
    plt.plot(x_coords, y_coords, marker='o')
    plt.title("Movement Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

train_data = torch.tensor([
    [(5, 0), (5, 0), (3, 2), (0, 5), (3, 2), (5, 0), (5, 0), (3, -2), (0, -5), (0, -5)]

])

plot_movement_sequence(  [(5, 0), (5, 0), (3, 2), (0, 5), (3, 2), (5, 0), (5, 0), (0, -5), (-5, 0), (0, -5)])

