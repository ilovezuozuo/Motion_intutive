import torch

def angle_solutions_filtering_engine(old_angle_solution, new_angle_solution):
    # 先保证新老情况都有解
    if len(old_angle_solution) == 1 and len(new_angle_solution) == 1:
        # print('这个分支就不该发生，出现了一组新老解都没有解的情况，看下输入数据哪里的问题')
        return [1]
    elif len(old_angle_solution) == 1 and len(new_angle_solution) != 1:
        # print('这个分支就不该发生，出现了一组新有解，老无解的情况，看下输入数据哪里的问题')
        return [1]
    elif len(old_angle_solution) != 1 and len(new_angle_solution) == 1:
        # print('这个情况应该是num_Error1或者num_Error2在记录，在惩罚?')
        # 这个情况应该是num_Error1在记录，在惩罚
        return [1]
    else:
        record_the_index_of_legal_solutions = []
        for solution_index in range(8):
            ls = []
            counter_for_each_joints_legality = 0
            for angle_index in range(6):
                if -torch.pi <= old_angle_solution[solution_index][angle_index] <= torch.pi:
                    counter_for_each_joints_legality += 1
            if counter_for_each_joints_legality == 6:
                ls.append(old_angle_solution[solution_index])
                record_the_index_of_legal_solutions.append(solution_index)
        for solution_index in record_the_index_of_legal_solutions:
            counter_for_each_joints_legality_for_new = 0
            for angle_index in range(6):
                if -torch.pi <= new_angle_solution[solution_index][angle_index] <= torch.pi:
                    counter_for_each_joints_legality_for_new += 1
                if counter_for_each_joints_legality_for_new == 6:
                    # print('[old_angle_solution[solution_index]:', old_angle_solution[solution_index])
                    # print('new_angle_solution[solution_index]:', new_angle_solution[solution_index])
                    # print('new_angle_solution[solution_index].unsqueeze(0):', new_angle_solution[solution_index].unsqueeze(0))

                    new_and_old_solutions_found = torch.cat([old_angle_solution[solution_index].unsqueeze(0),
                                                             new_angle_solution[solution_index].unsqueeze(0)], dim=0)

                    # print('chioced_old_new_solutions:', new_and_old_solutions_found)
                    return new_and_old_solutions_found
                else:
                    pass
        print('丢失的数据在这里丢的')
        return [1]

def test_for_this():
    old_angle_solution = torch.tensor([[-2.6377, -1.9976,  2.2066, -0.2090,  2.0746, -1.5708],
            [-2.6377,  0.0716, -2.2066,  2.1350,  2.0746, -1.5708],
            [-2.6377, -1.7303,  2.5190, -3.9303, -2.0746,  1.5708],
            [-2.6377,  0.5735, -2.5190, -1.1961, -2.0746,  1.5708],
            [-4.7431,  2.5681,  2.5190, -1.9455,  0.0307, -4.7124],
            [-4.7431, -1.4113, -2.5190,  7.0719,  0.0307, -4.7124],
            [-4.7431,  3.0700,  2.2066, -5.2766, -0.0307, -1.5708],
            [-4.7431, -1.1440, -2.2066,  3.3506, -0.0307, -1.5708]])

    new_angle_solution = torch.tensor([[-2.3454, -1.7086,  2.2378, -0.5745,  2.3169, -1.6018],
            [-2.3454,  0.3863, -2.2378,  1.8063,  2.3169, -1.6018],
            [-2.3454, -1.2988,  2.4063,  1.9888, -2.3169,  1.5398],
            [-2.3454,  0.9276, -2.4063,  4.5750, -2.3169,  1.5398],
            [-4.6917,  2.3280,  2.2365, -6.8504,  0.0440,  0.7154],
            [-4.6917, -1.8615, -2.2365,  1.8120,  0.0440,  0.7154],
            [-4.6917,  2.7353,  2.4079, -4.2876, -0.0440,  3.8570],
            [-4.6917, -1.3204, -2.4079,  4.5839, -0.0440,  3.8570]])

    # print(angle_solutions_filtering_engine(old_angle_solution, new_angle_solution))
    return