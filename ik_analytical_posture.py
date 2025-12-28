import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH
from roboticstoolbox import DHRobot
import math
from spatialmath import SE3

L1 = RevoluteMDH(d=0.08525,   a=0.0,        alpha=0.0,        offset=0.0)
L2 = RevoluteMDH(d=0.0,       a=0.0,        alpha=np.pi/2,    offset=np.pi/2)
L3 = RevoluteMDH(d=0.0,       a=0.12893,    alpha=0.0,        offset=0.0)
L4 = RevoluteMDH(d=0.04039,   a=0.129,      alpha=0.0,        offset=-np.pi/2)
L5 = PrismaticMDH(theta=0.0,  a=0.0,        alpha=-np.pi/2,   offset=0.07403, qlim=[0.0, 0.04])

myarm = DHRobot([L1, L2, L3, L4, L5], name="My5DOF")

# ==========================================
# 2. 解析法逆运动学函数
# ==========================================
def my_ik_analytical(T_target: np.ndarray, q5_val=0.0, solution_mask=[1, -1]):
    
    # 机械臂参数
    d1 = 0.08525
    d4 = 0.04039
    a2 = 0.12893
    a3 = 0.129
    # L5 长度 = q5 + offset
    d5_total = q5_val + 0.07403
    
    # 提取位置 pos_test = np.array([x, y, z, roll, pitch, yaw])
    px, py, pz = T_target[0], T_target[1], T_target[2]
    roll, pitch, yaw = T_target[3], T_target[4], T_target[5]
    
    phi = 0.0
    if abs(roll) < 1e-6: # 近似为 0, 头朝上
        phi = math.pi / 2 - pitch
    elif abs(roll + math.pi) < 1e-6: # 近似为 -pi, 头朝下
        phi = pitch - math.pi / 2
    print(f"phi(deg): {np.degrees(phi)}")

    # 1) 计算基座 q1
    if math.sqrt(px**2 + py**2) < d4:
        print("Error: Target out of reach.")
        return None

    base_theta_1 = math.atan2(py, px)
    # print(f"base_theta_1(deg): {np.degrees(base_theta_1)}")
    base_theta_2 = math.asin(d4 / math.sqrt(px**2 + py**2))
    # print(f"base_theta_2(deg): {np.degrees(base_theta_2)}")
    q1 = base_theta_1 + base_theta_2
    # print(f"q1: {q1}")

    # 2) 将末端位置投影到机械臂基座竖直平分面
    r = math.sqrt(px**2 + py**2 - d4**2)
    z_plane = pz - d1

    # 3) 求解竖直平面关节变量 q2, q3, q4
    r_wrist = r - d5_total * math.cos(phi)
    z_wrist = z_plane - d5_total * math.sin(phi)
    # r_wrist = r - d5_total * math.sin(pitch)
    # z_wrist = z_plane - d5_total * math.cos(abs(pitch))
    # print(f"r_wrist: {r_wrist}, z_wrist: {z_wrist}")

    cos_theta3 = (a2**2 + a3**2 - r_wrist**2 - z_wrist**2) / (2 * a2 * a3)
    # print(f"cos_theta3: {cos_theta3}")
    if abs(cos_theta3) > 1.0:
        print("Error: Target out of reach.")
        return None
    
    # 取上凸构型
    q3 = solution_mask[1] * (math.pi - math.acos(cos_theta3)) # 乘 -1 取上凸构型
    # print(f"q3: {q3}")

    theta_2 = math.atan2(z_wrist, r_wrist)
    yy = a3 * math.sin(math.acos(cos_theta3))
    xx = a2 - a3 * cos_theta3
    # print(f"xx: {xx}, yy: {yy}")
    theta_1 = math.atan2(yy, xx)
    # print(f"theta_1(deg): {np.degrees(theta_1)}, theta_2(deg): {np.degrees(theta_2)}")
    q2 = theta_2 + theta_1
    q4 = phi - (q2 + q3)

    q2 = q2 - (math.pi / 2) # 减去 pi/2 偏置
    q = np.array([q1, q2, q3, q4, q5_val])
    # q = (q + np.pi) % (2 * np.pi) - np.pi
    return q

# ==========================================
# 3. 验证环节
# ==========================================
if __name__ == "__main__":
    print("--- 逆运动学验证 ---")
    # q_test = np.array([np.radians(-0.1), np.radians(-30.6), np.radians(-42.8), np.radians(-56.1), 0.00])
    # q_test = np.array([np.radians(0), np.radians(-30.6), np.radians(-42.6), np.radians(37.5), 0.00])
    # q_test = np.array([np.radians(49.1), np.radians(-40.9), np.radians(-34.8), np.radians(-120.7), 0.00])
    q_test = np.array([np.radians(-122.7), np.radians(-38.9), np.radians(-45.0), np.radians(118.6), 0.00])
    q_test_deg = np.degrees(q_test)
    T_test = myarm.fkine(q_test)
    T_test_mat = T_test.A
    T_test_pose = (T_test_mat[0:3, 3][0], T_test_mat[0:3, 3][1], T_test_mat[0:3, 3][2])  # 提取位置部分
    T_test_euler = T_test.rpy(order='zyx', unit='deg')
    T_test_euler_rad = T_test.rpy(order='zyx', unit='rad')
    print(f"关节空间坐标(rad): {np.round(q_test, 4)}")
    print(f"关节空间坐标(deg): {np.round(np.degrees(q_test), 4)}")
    x, y, z = T_test_pose
    roll, pitch, yaw = T_test_euler_rad
    # roll 只有 0 和 pi 两种情况
    roll_deg, pitch_deg, yaw_deg = T_test_euler
    pos_test = np.array([x, y, z, roll, pitch, yaw])
    print(f"工作空间坐标(m, rad): {np.round(pos_test, 4)}")
    print(f"工作空间坐标(m, deg): {np.round(np.array([x, y, z, roll_deg, pitch_deg, yaw_deg]), 4)}")
    # 2. 调用逆运动学
    q_sol = my_ik_analytical(pos_test, q5_val=q_test[4])
    if q_sol is not None:
        print(f"逆解→关节空间坐标(rad): {np.round(q_sol, 4)}")
        print(f"逆解→关节空间坐标(deg): {np.round(np.degrees(q_sol), 4)}")
