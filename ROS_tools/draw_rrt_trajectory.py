from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rospy

# 初始化 ROS 节点
rospy.init_node('path_publisher_node', anonymous=True)

# 创建发布器
path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)

# 创建 Path 消息
path = Path()
path.header.frame_id = "base_link"

# 模拟的 RRT 生成点
rrt_points = [[1.,1,1], [2,2,2], [3,3,3]]
for point in rrt_points:  # rrt_points 是您生成的点
    pose = PoseStamped()
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
    path.poses.append(pose)

# 发布路径消息的循环
rate = rospy.Rate(10)  # 设置发布频率为 10 Hz
while not rospy.is_shutdown():
    path_pub.publish(path)  # 发布路径
    rate.sleep()  # 等待直到下一次发布
