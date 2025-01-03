import torch
import random


def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        # 生成机器人末端执行器的rpyxyz
        robot_rpyxyz = [random.uniform(torch.pi / 2 - 0.1, torch.pi / 2),
                        random.uniform(0, 0.1),
                        random.uniform(0, 0.1),
                        random.uniform(0.5, 0.7),
                        random.uniform(-0.5, -0.7),
                        random.uniform(0.2, 0.4)]

        # 生成障碍物的rpyxyzlwh
        if random.choice([True, False]):
            a, b = -1.0, -0.4
        else:
            a, b = 0.4, 1.0
        obstacle_rpyxyzlwh1 = [random.uniform(0, 0.1),
                              random.uniform(0, 0.1),
                              random.uniform(0, 0.1),

                              # random.uniform(a, b),
                              # random.uniform(a, b),
                              # random.uniform(0.5, 1.0),
                               random.uniform(0.8, 0.9),
                               random.uniform(-0.8, -0.9),
                               random.uniform(0.1, 0.5),

                              random.uniform(0.2, 0.3),
                              random.uniform(0.2, 0.3),
                              random.uniform(0.2, 0.3)]
        obstacle_rpyxyzlwh2 = [random.uniform(0, 0.1),
                              random.uniform(0, 0.1),
                              random.uniform(0, 0.1),

                              # random.uniform(a, b),
                              # random.uniform(a, b),
                              # random.uniform(0.5, 1.0),
                               random.uniform(0.8, 0.9),
                               random.uniform(-0.8, -0.9),
                               random.uniform(0.1, 0.5),

                              random.uniform(0.2, 0.3),
                              random.uniform(0.2, 0.3),
                              random.uniform(0.2, 0.3)]

        data = robot_rpyxyz + obstacle_rpyxyzlwh1 + obstacle_rpyxyzlwh2
        dataset.append(data)

    return torch.FloatTensor(dataset)


# 生成10个样本的数据集
data_for_motion_planning = generate_dataset(30)
print(data_for_motion_planning)
