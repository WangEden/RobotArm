import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH
from roboticstoolbox import DHRobot

L1 = RevoluteMDH(d=0.08525,   a=0.0,        alpha=0.0,        offset=0.0)   # theta1 = q1
L2 = RevoluteMDH(d=0.0,       a=0.0,        alpha=np.pi/2,    offset=np.pi/2)   # theta2 = q2 + pi/2
L3 = RevoluteMDH(d=0.0,       a=0.12893,    alpha=0.0,        offset=0.0)   # theta3 = q3
L4 = RevoluteMDH(d=0.04039,   a=0.129,      alpha=0.0,        offset=-np.pi/2)   # theta4 = q4 - pi/2
L5 = PrismaticMDH(theta=0.0,  a=0.0,        alpha=-np.pi/2,   offset=0.07403, qlim=[0.0, 0.04])

myarm = DHRobot([L1, L2, L3, L4, L5], name="My5DOF")

# ==== 3. 正运动学测试 ====
if __name__ == "__main__":
    print(myarm) # 打印 DH 表
    print("=== 正运动学测试 ===")

    q_home = np.array([0, 0, 0, 0, 0])  # [q1,q2,q3,q4,d5]
    q_home_deg = np.degrees(q_home)
    T_home = myarm.fkine(q_home)         # 末端 SE3 变换
    T_home_mat = T_home.A
    T_pose = (T_home_mat[0:3, 3][0], T_home_mat[0:3, 3][1], T_home_mat[0:3, 3][2])
    T_home_euler_deg = T_home.rpy(order='zyx', unit='deg')  # 提取欧拉角部分，ZYX 顺序
    T_home_euler_rad = T_home.rpy(order='zyx', unit='rad')
    x, y, z = T_pose
    roll, pitch, yaw = T_home_euler_rad
    pos_home = np.array([x, y, z, roll, pitch, yaw])
    print("零位置: ")
    print(f"关节空间坐标(rad): {np.round(q_home, 4)}")
    print(f"关节空间坐标(deg): {np.round(q_home_deg, 4)}")
    print(f"工作空间坐标: {np.round(pos_home, 4)}")

    # 随便给一组关节量再算一次
    q_test_deg = np.array([0, -30.7, -42.2, 37.1, 0.0])
    q_test = np.array([np.radians(0), np.radians(-30.7), np.radians(-42.2), np.radians(37.1), 0.0])
    T_test = myarm.fkine(q_test)
    T_test_mat = T_test.A
    T_test_pose = (T_test_mat[0:3, 3][0], T_test_mat[0:3, 3][1], T_test_mat[0:3, 3][2])
    T_test_euler_deg = T_test.rpy(order='zyx', unit='deg')
    T_test_euler_rad = T_test.rpy(order='zyx', unit='rad')
    x, y, z = T_test_pose
    roll, pitch, yaw = T_test_euler_rad
    pos_test = np.array([x, y, z, roll, pitch, yaw])
    print("测试位姿: ")
    print(f"关节空间坐标(rad): {np.round(q_test, 4)}")
    print(f"关节空间坐标(deg): {np.round(q_test_deg, 4)}")
    print(f"工作空间坐标: {np.round(pos_test, 4)}")

    # 调用 teach，可以启动带滑块的示教器
    # 初始姿态用 q_home
    myarm.teach(q_home)
