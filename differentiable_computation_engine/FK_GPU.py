import torch
from torchviz import make_dot

def cos(a):
    return torch.cos(a)

def sin(a):
    return torch.sin(a)

def THT(Theta, A, D, Alpha):

    # T = torch.tensor([
    #     [cos(Theta), -sin(Theta)*cos(Alpha), sin(Alpha)*sin(Theta), A*cos(Theta)],
    #     [sin(Theta), cos(Theta)*cos(Alpha), -cos(Theta)*sin(Alpha), A*sin(Theta)],
    #     [0, sin(Alpha), cos(Alpha), D],
    #     [0, 0, 0, 1]
    # ])

    # T = torch.stack([
    #     torch.tensor([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
    #                   A * torch.cos(Theta)]),
    #     torch.tensor([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
    #                   A * torch.sin(Theta)]),
    #     torch.tensor([0, torch.sin(Alpha), torch.cos(Alpha), D]),
    #     torch.tensor([0, 0, 0, 1])
    # ])
    row1 = torch.stack([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
                        A * torch.cos(Theta)])
    row2 = torch.stack([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
                        A * torch.sin(Theta)])
    row3 = torch.stack([torch.tensor(0.0), torch.sin(Alpha), torch.cos(Alpha), D ])  # 将0替换为tensor(0.0)以保持一致性
    row4 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])

    # 使用torch.stack将行合并成一个4x4的矩阵
    T = torch.stack([row1, row2, row3, row4])

    return T
def THT2(Theta, A, D, Alpha):
    # T = torch.tensor([
    #     [cos(Theta), -sin(Theta)*cos(Alpha), sin(Alpha)*sin(Theta), A*cos(Theta)],
    #     [sin(Theta), cos(Theta)*cos(Alpha), -cos(Theta)*sin(Alpha), A*sin(Theta)],
    #     [0, sin(Alpha), cos(Alpha), D+torch.tensor([0.5])],
    #     [0, 0, 0, 1]
    # ])
    # T = torch.stack([
    #     torch.tensor([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
    #                   A * torch.cos(Theta)]),
    #     torch.tensor([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
    #                   A * torch.sin(Theta)]),
    #     torch.tensor([0, torch.sin(Alpha), torch.cos(Alpha), D+torch.tensor([0.5])]),
    #     torch.tensor([0, 0, 0, 1])
    # ])
    row1 = torch.stack([torch.cos(Theta), -torch.sin(Theta) * torch.cos(Alpha), torch.sin(Alpha) * torch.sin(Theta),
                        A * torch.cos(Theta)])
    row2 = torch.stack([torch.sin(Theta), torch.cos(Theta) * torch.cos(Alpha), -torch.cos(Theta) * torch.sin(Alpha),
                        A * torch.sin(Theta)])
    row3 = torch.stack([torch.tensor(0.0), torch.sin(Alpha), torch.cos(Alpha), D + torch.tensor(0.5)])  # 将0替换为tensor(0.0)以保持一致性
    row4 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])

    # 使用torch.stack将行合并成一个4x4的矩阵
    T = torch.stack([row1, row2, row3, row4])

    return T

def FK(theta, base, a, d, alpha):



    T01 = THT(theta[0], a[0], d[0], alpha[0])


    T12 = THT(theta[1], a[1], d[1], alpha[1])
    T23 = THT(theta[2], a[2], d[2], alpha[2])
    T34 = THT(theta[3], a[3], d[3], alpha[3])
    T45 = THT(theta[4], a[4], d[4], alpha[4])
    T56 = THT2(theta[5], a[5], d[5], alpha[5])
    T566 = THT(theta[5], a[5], d[5], alpha[5])
    # T56[2][3] += torch.tensor([0.2])

    T0 = torch.mm(base, T01)
    T1 = torch.mm(T0, T12)
    T2 = torch.mm(T1, T23)
    T3 = torch.mm(T2, T34)
    T4 = torch.mm(T3, T45)
    T5 = torch.mm(T4, T56)
    T55 = torch.mm(T4, T566)
    print('FKT5',T5)
    print('FKT5',T55)
    # T5 = torch.mm(T4, T56)

    # return T5
    # print('T56', T56)
    # return [T01, T12, T23, T34, T45, T56]
    return torch.stack([T0, T1, T2, T3, T4, T5])
