import torch

a = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
b = torch.tensor([4, 4, 4], dtype=torch.float32, requires_grad=True)
c = torch.tensor([100, 2, 0], dtype=torch.float32, requires_grad=True)

t = torch.linspace(0, 1, 100).view(-1, 1)
line_points = a + t * (b - a)

distance = torch.norm(torch.cross(b - a, c - a)) / torch.norm(b - a)

projection_t = torch.sum((c - a) * (b - a)) / torch.sum((b - a) ** 2)
projection_d = a + projection_t * (b - a)

# 输出结果
print("直线上的点：", line_points)
print("c 到直线的距离：", distance.item())
print("投影点 d：", projection_d)
