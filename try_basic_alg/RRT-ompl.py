import sys


import ompl
from ompl import base, geometric

# 创建一个 OMPL 状态空间，这里我们使用2D空间
space = ompl.base.RealVectorStateSpace(2)

# 设置状态空间的边界
bounds = ompl.base.RealVectorBounds(2)
bounds.setLow(0)   # 设置下界
bounds.setHigh(10)  # 设置上界
space.setBounds(bounds)

# 创建一个简单的状态检查函数，用于检查状态是否有效
def isStateValid(state):
    # 在这个示例中，我们假设所有状态都是有效的
    return True

# 创建一个 OMPL 算法，这里我们使用 RRT
simple_setup = ompl.geometric.SimpleSetup(space)
simple_setup.setStateValidityChecker(isStateValid)

# 设置起始状态
start = ompl.base.State(space)
start[0] = 1.0
start[1] = 1.0
simple_setup.setStartAndGoalStates(start)

# 运行 RRT 路径规划
planner = ompl.geometric.RRT(simple_setup.getSpaceInformation())
simple_setup.setPlanner(planner)
simple_setup.setup()
solution = simple_setup.solve(1.0)  # 1.0 是规划时间限制

if solution:
    print("Found solution!")
    path = simple_setup.getSolutionPath()
    path.interpolate()
    path.printAsMatrix()  # 打印路径
else:
    print("No solution found.")
