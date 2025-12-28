import numpy as np
import matplotlib.pyplot as plt

class RobotArmRRRRP:
    def __init__(self):
        # 1. 物理参数定义
        self.g0 = 9.81  # 重力加速度
        
        # 连杆长度 (m)
        self.a2 = 0.1289
        self.a3 = 0.129
        self.d4 = 0.074  # Link 4 的有效偏置/长度
        
        # 修正后的质量 (kg) - 包含电机质量
        # m1 = Link1 + Motor2
        self.m1 = 0.399
        # m2 = Link2 + Motor3
        self.m2 = 0.439
        # m3 = Link3 + Motor4
        self.m3 = 0.439
        # m4 = Link4 + Motor5
        self.m4 = 0.29957 
        # m5 = Link5 (末端)
        self.m5 = 0.00003
        
        # 组合质量
        self.M3_plus = self.m3 + self.m4 + self.m5
        self.M4_plus = self.m4 + self.m5
        
        # 质心位置 (m) - 沿连杆轴线距离
        self.rc2 = 0.094
        self.rc3 = 0.094
        self.rc4 = 0.074 # 假设质心位于 Link 4 的几何中心/偏置处
        
        # 惯性参数 (kg*m^2)
        # 估算值 (基于圆柱体/长方体近似 I = 1/12 * m * L^2)
        self.Izz1 = 0.00015
        self.Izz2 = 0.00028
        self.Izz3 = 0.00028
        self.Izz4 = 0.00003
        
        # 电机参数
        # 等效转子惯量 (k_r^2 * I_rotor)
        self.Im_eff = 0.00202 
        
        # 摩擦系数 (用于仿真模拟真实环境，实际数值需要实验测定)
        self.Fv = np.array([0.05, 0.05, 0.05, 0.01, 0.001]) # 粘滞摩擦
        self.Fs = np.array([0.02, 0.02, 0.02, 0.01, 0.001]) # 库仑摩擦

    def get_dynamics(self, q, dq):
        """
        计算动力学矩阵 B, C, g
        q: [q1, q2, q3, q4, q5]
        dq: [dq1, dq2, dq3, dq4, dq5]
        """
        # 提取状态变量
        q1, q2, q3, q4, q5 = q
        dq1, dq2, dq3, dq4, dq5 = dq
        
        # 预计算三角函数
        s2 = np.sin(q2)
        c2 = np.cos(q2)
        s3 = np.sin(q3)
        c3 = np.cos(q3)
        s4 = np.sin(q4)
        c4 = np.cos(q4)
        
        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)
        
        # 定义 q2 = 0 为竖直向上零位，因此与重力有关的角度是 s234
        # 但 B 矩阵中的几何投影关系通常定义 0 为伸直。
        # 为保持一致性，B 矩阵推导通常基于几何构型。
        # 下面的 B 矩阵公式基于标准 DH 习惯 (q2 = 0 伸直)，与 g 向量的定义需要匹配。
        # 这里假设 q2, q3, q4 均为相对上一级伸直为 0
        
        c234 = np.cos(q2 + q3 + q4)
        s234 = np.sin(q2 + q3 + q4)
        
        # 末端有效长度
        L_eff = self.d4 + q5

        # --- 1. 惯性矩阵 B (5x5) ---
        B = np.zeros((5, 5))
        
        # B11: 绕基座 Z 轴转动惯量 (随臂伸展变化)
        # r_proj 是各质心到转轴的水平距离。
        # q2=0 为竖直向上，因此水平距离用 sin 计算。
        # 因此水平投影半径 R = L * sin(theta)
        r_proj_2 = self.a2 * s2
        r_proj_3 = self.a2 * s2 + self.a3 * s23
        r_proj_4 = self.a2 * s2 + self.a3 * s23 + self.d4 * s234 # 简化的质心位置
        
        # 近似 B11
        B[0, 0] = self.Izz1 + self.Im_eff + \
                  self.m2 * (self.rc2 * s2)**2 + \
                  self.m3 * (self.a2 * s2 + self.rc3 * s23)**2 + \
                  self.m4 * r_proj_4**2 

        # 平面连杆部分 (Link 2, 3, 4)
        # B22
        B[1, 1] = self.Izz2 + self.Im_eff + \
                  self.m2 * self.rc2**2 + \
                  self.m3 * (self.a2**2 + self.rc3**2 + 2 * self.a2 * self.rc3 * c3) + \
                  self.m4 * (self.a2**2 + self.a3**2 + 2 * self.a2 * self.a3 * c3) # 忽略 d4 耦合简化
                  
        # B33
        B[2, 2] = self.Izz3 + self.Im_eff + self.m3 * self.rc3**2 + self.m4 * self.a3**2
        
        # B44
        B[3, 3] = self.Izz4 + self.Im_eff
        
        # B55 (移动关节)
        B[4, 4] = self.m5

        # 耦合项
        # B23 (肩-肘)
        b23_val = self.m3 * (self.rc3**2 + self.a2 * self.rc3 * c3) + \
                  self.m4 * (self.a3**2 + self.a2 * self.a3 * c3)
        B[1, 2] = B[2, 1] = b23_val
        
        # B24 (肩-腕)
        b24_val = self.m4 * (self.rc4**2 + self.a3 * self.rc4 * c4 + self.a2 * self.rc4 * np.cos(q3 + q4))
        B[1, 3] = B[3, 1] = b24_val + self.Im_eff * 0.1 # 假设部分耦合
        
        # B34 (肘-腕)
        b34_val = self.m4 * (self.rc4**2 + self.a3 * self.rc4 * c4)
        B[2, 3] = B[3, 2] = b34_val + self.Im_eff * 0.1
        
        # --- 2. 哥氏力与离心力 C (5x1 向量形式) ---
        # 使用 derived vector form 直接计算 C*dq
        C_vec = np.zeros(5)
        
        # 中间变量 h
        h23 = (self.m3 * self.rc3 + self.m4 * self.a3) * self.a2 * s3 + \
              self.m4 * self.rc4 * self.a2 * np.sin(q3 + q4)
        h34 = self.m4 * self.rc4 * self.a3 * s4
        h24 = self.m4 * self.rc4 * self.a2 * np.sin(q3 + q4)
        
        # 为了简化仿真，这里主要实现平面关节 (2,3,4) 的耦合
        # C2 (Shoulder)
        # -h23 * dq3 * (2*dq2 + dq3) - ...
        C_vec[1] = -h23 * dq3 * (2*dq2 + dq3) - (h24 + h34) * dq4 * (2*dq2 + 2*dq3 + dq4)
        
        # C3 (Elbow)
        C_vec[2] = h23 * dq2**2 - h34 * dq4 * (2*dq2 + 2*dq3 + dq4)
        
        # C4 (Wrist)
        C_vec[3] = (h24 + h34) * dq2**2 + h34 * dq3**2 + 2 * h34 * dq2 * dq3
        
        # C1 (Base) - 简化，主要受 B11 变化影响
        # C1 approx (dB11/dt * dq1 / 2)
        # 这里做简化处理，设为0
        
        # --- 3. 重力向量 g (5x1) ---
        # q=0 竖直向上定义：关节力矩与 sin(theta) 成正比
        g_vec = np.zeros(5)
        
        g_vec[1] = self.g0 * ((self.m2 * self.rc2 + self.M3_plus * self.a2) * s2 + \
                              (self.m3 * self.rc3 + self.M4_plus * self.a3) * s23 + \
                              (self.m4 * self.rc4 + self.m5 * L_eff) * s234)
                              
        g_vec[2] = self.g0 * ((self.m3 * self.rc3 + self.M4_plus * self.a3) * s23 + \
                              (self.m4 * self.rc4 + self.m5 * L_eff) * s234)
                              
        g_vec[3] = self.g0 * ((self.m4 * self.rc4 + self.m5 * L_eff) * s234)
        
        # 移动关节 q5: 竖直向上时 (cos(0)=1) 受最大重力
        g_vec[4] = self.m5 * self.g0 * c234
        
        return B, C_vec, g_vec

    def forward_dynamics(self, q, dq, tau_cmd):
        """
        正动力学求解: ddq = B_inv * (tau - C - g - Fric)
        """
        B, C_vec, g_vec = self.get_dynamics(q, dq)
        
        # 摩擦力
        tau_fric = self.Fv * dq + self.Fs * np.sign(dq)
        
        # 求解加速度
        # tau_net = tau_cmd - C - g - fric
        tau_net = tau_cmd - C_vec - g_vec - tau_fric
        
        # 处理 B 矩阵奇异性（防止数值错误）
        ddq = np.linalg.solve(B + np.eye(5)*1e-6, tau_net)
        return ddq


# 仿真控制器
def computed_torque_control(robot, q, dq, q_des, dq_des, ddq_des):
    """
    计算力矩控制 (CTC)
    """
    # 单个关节电机的 PID 增益
    # Kp: 比例增益
    # Kp = np.array([100, 150, 150, 80, 100]) # test1
    Kp = np.array([50, 75, 75, 40, 50]) # test2
    # Kv: 微分增益
    # Kv = np.array([20,  30,  30,  10, 20]) # test1
    Kv = np.array([14, 17, 17, 12, 14]) # test2
    
    # 获取动力学矩阵 (用于抵消非线性)
    B, C_vec, g_vec = robot.get_dynamics(q, dq)
    
    # 误差
    e = q_des - q
    de = dq_des - dq
    
    # 控制律
    # tau = B * (ddq_d + Kv*de + Kp*e) + C + g
    # 还要补偿预期的摩擦力 (前馈)
    tau_fric_comp = robot.Fv * dq + robot.Fs * np.sign(dq) # 简单补偿
    
    # 计算辅助控制量 u
    u = ddq_des + Kv * de + Kp * e
    
    # 最终力矩
    tau = B @ u + C_vec + g_vec + tau_fric_comp
    
    return tau


if __name__ == "__main__":
    # 初始化机器人
    arm = RobotArmRRRRP()
    
    # 仿真参数
    dt = 0.001
    total_time = 2.0
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)
    
    # 初始状态 (竖直向上 q=0)
    q = np.zeros(5)
    dq = np.zeros(5)
    
    # 期望轨迹 (正弦波跟踪测试)
    # 让 Joint 2, 3, 4 做点头运动，Joint 5 做伸缩
    q_des_traj = np.zeros((steps, 5))
    dq_des_traj = np.zeros((steps, 5))
    ddq_des_traj = np.zeros((steps, 5))
    
    freq = 1.0 # 1 Hz
    for i in range(steps):
        t = time[i]
        # 目标: A * sin(omega * t)
        val = 0.5 * np.sin(2 * np.pi * freq * t)
        dval = 0.5 * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        ddval = -0.5 * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t)
        
        q_des_traj[i, :] = [0, val, val, val*0.5, val*0.1]
        dq_des_traj[i, :] = [0, dval, dval, dval*0.5, dval*0.1]
        ddq_des_traj[i, :] = [0, ddval, ddval, ddval*0.5, ddval*0.1]

    # 数据记录
    q_hist = []
    tau_hist = []
    
    print("开始仿真...")
    for i in range(steps):
        # 1. 控制器计算力矩
        tau = computed_torque_control(arm, q, dq, 
                                      q_des_traj[i], dq_des_traj[i], ddq_des_traj[i])
        
        # 限幅 (模拟电机最大扭矩)
        tau = np.clip(tau, -20, 20)
        
        # 2. 物理引擎步进
        ddq = arm.forward_dynamics(q, dq, tau)
        
        # 欧拉积分
        dq = dq + ddq * dt
        q = q + dq * dt
        
        # 记录
        q_hist.append(q.copy())
        tau_hist.append(tau.copy())
        
        if i % 100 == 0:
            print(f"Time: {time[i]:.2f}s | J2 误差: {q_des_traj[i,1]-q[1]:.4f}，J3 误差: {q_des_traj[i,2]-q[2]:.4f}，J4 误差: {q_des_traj[i,3]-q[3]:.4f}")

    # --- 绘图结果 ---
    q_hist = np.array(q_hist)
    tau_hist = np.array(tau_hist)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    # 1. 位置跟踪图
    axs[0,0].plot(time, q_des_traj[:, 1], 'k--', label='Target J2')
    axs[0,1].plot(time, q_des_traj[:, 2], 'k--.', label='Target J3')
    axs[1,0].plot(time, q_des_traj[:, 3], 'k--', label='Target J4')
    axs[0,0].plot(time, q_hist[:, 1], 'r', label='Actual J2')
    axs[0,1].plot(time, q_hist[:, 2], 'g', label='Actual J3')
    axs[1,0].plot(time, q_hist[:, 3], 'b', label='Actual J4')
    axs[0,0].set_title('Joint Position Tracking (CTC Control)')
    axs[0,1].set_title('Joint Position Tracking (CTC Control)')
    axs[1,0].set_title('Joint Position Tracking (CTC Control)')
    axs[0,0].set_xlabel('Time (s)')
    axs[0,1].set_xlabel('Time (s)')
    axs[1,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Angle (rad)')
    axs[1,0].set_ylabel('Angle (rad)')
    axs[1,1].set_ylabel('Angle (rad)')
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[0,0].grid(True)
    axs[0,1].grid(True)
    axs[1,0].grid(True)
    
    # 2. 力矩输出图
    axs[1,1].plot(time, tau_hist[:, 1], label='Tau J2')
    axs[1,1].plot(time, tau_hist[:, 2], label='Tau J3')
    axs[1,1].plot(time, tau_hist[:, 3], label='Tau J4')
    axs[1,1].plot(time, tau_hist[:, 4], label='Tau J5')
    axs[1,1].set_title('Control Torques')
    axs[1,1].set_xlabel('Time (s)')
    axs[1,1].set_ylabel('Torque (Nm)')
    axs[1,1].legend()
    axs[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()