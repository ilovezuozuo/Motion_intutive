import torch
def interpolate_joint_angles(joint_angles):

    if joint_angles is not None:
        # 将输入的关节角度张量拆分为两组
        group1 = joint_angles[0, :]
        group2 = joint_angles[1, :]

        # 计算插值
        # interpolated_result = group1 + torch.linspace(0, 1, 5).view(5, 1) * (group2 - group1)
        # interpolated_result = group1 + torch.linspace(0, 1, 3).view(3, 1) * (group2 - group1)
        # interpolated_result = group1 + torch.linspace(0, 1, 3).view(3, 1) * (group2 - group1)
        
        interpolated_result = torch.linspace(0, 1, 3).view(3, 1) * (group2 - group1) + group1
        # print('interpolated_result', interpolated_result)
        
        
        return interpolated_result[1:]
    else:
        # print('请优先惩罚是否有解的情况')
        return

def test_for_this():
    # 示例输入
    input_tensor = torch.tensor([[-2.6377, 0.0716, -2.2066, 2.1350, 2.0746, -1.5708],
                                 [-2.3454, 0.3863, -2.2378, 1.8063, 2.3169, -1.6018]],requires_grad=True)

    # 调用函数得到插值结果
    interpolated_result = interpolate_joint_angles(input_tensor)
    # make_dot(interpolated_result).view()

    # 打印插值结果
    print(interpolated_result)

test_for_this()