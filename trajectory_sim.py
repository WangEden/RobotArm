import numpy as np
import matplotlib.pyplot as plt
import time
import roboticstoolbox as rtb

# 引入你提供的模块
from fk_mdh import myarm
# from ik_analytical_posture import my_ik_analytical
from ik_hybrid_posture_all import ik_smart_hybrid

# ==========================================
# 1. 轨迹规划数学核心类
# ==========================================
class TrajectoryPlanner:
    """
    轨迹规划器，基于《机器人控制技术》第4章
    """
    @staticmethod
    def quintic_polynomial(t, t_total, start_val, end_val):
        """
        五次多项式插值 (对应课件 4.2.1 点对点运动 - 指定加速度)
        s(t) = 10(t/T)^3 - 15(t/T)^4 + 6(t/T)^5
        用于生成平滑的归一化因子 s, 范围 [0, 1]
        """
        if t_total <= 1e-6: return end_val
        
        tau = t / t_total
        # 归一化五次多项式，满足位置、速度、加速度边界条件为0
        s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        
        # 映射到实际值
        return start_val + (end_val - start_val) * s

    @staticmethod
    def get_num_steps(start_pos, end_pos, speed, frequency=50):
        """根据平均速度估算需要的步数"""
        dist = np.linalg.norm(end_pos - start_pos)
        duration = dist / speed if speed > 0 else 2.0
        duration = max(duration, 0.5) # 最少0.5秒
        return int(duration * frequency), duration

# ==========================================
# 2. MoveJ: 关节空间运动仿真
# ==========================================
def move_j(q_start, q_end, speed_rad_s=1.0, frequency=50):
    """
    关节空间插值运动
    """
    print(f"\n--- 执行 MoveJ ---")
    
    # 估算时间 (取移动幅度最大的关节计算时间)
    max_diff = np.max(np.abs(q_end - q_start))
    duration = max_diff / speed_rad_s if speed_rad_s > 0 else 2.0
    duration = max(duration, 1.0) # 设定最小时间
    steps = int(duration * frequency)
    
    dt = duration / steps
    trajectory_q = []
    
    print(f"规划时间: {duration:.2f}s, 总步数: {steps}")

    for i in range(steps + 1):
        t = i * dt
        # 对每个关节进行五次多项式插值
        q_curr = TrajectoryPlanner.quintic_polynomial(t, duration, q_start, q_end)
        trajectory_q.append(q_curr)
        
    return np.array(trajectory_q)

# ==========================================
# 3. MoveL: 直线运动仿真
# ==========================================
def move_l(q_current, target_pose, speed_m_s=0.1, frequency=50):
    """
    笛卡尔空间直线运动
    target_pose: [x, y, z, roll, pitch, yaw]
    """
    print(f"\n--- 执行 MoveL ---")
    
    # 1. 获取当前末端位姿
    T_start = myarm.fkine(q_current)
    start_pos = T_start.t
    start_rpy = T_start.rpy(order='zyx', unit='rad') # [roll, pitch, yaw]
    start_pose_vec = np.hstack((start_pos, start_rpy))
    
    target_pose_vec = np.array(target_pose)
    
    # 2. 规划参数
    steps, duration = TrajectoryPlanner.get_num_steps(start_pose_vec[:3], target_pose_vec[:3], speed_m_s, frequency)
    dt = duration / steps
    
    trajectory_q = []
    q_prev = q_current
    
    print(f"直线规划: {start_pose_vec[:3]} -> {target_pose_vec[:3]}")
    print(f"规划时间: {duration:.2f}s, 总步数: {steps}")

    valid_path = True
    
    for i in range(steps + 1):
        t = i * dt
        
        # 3. 笛卡尔空间线性插值 (Lerp)
        # 注意：这里采用了位置和RPY角的线性插值。
        # 对于严格的姿态插值，通常使用四元数SLERP，但在小范围移动或RPY无奇点时线性插值也可接受。
        current_pose_target = TrajectoryPlanner.quintic_polynomial(t, duration, start_pose_vec, target_pose_vec)
        
        # 4. 调用你的解析逆运动学求解器
        # 注意：MoveL需要保证每个插值点都有逆解
        # q5_val 保持不变或根据需要插值，这里假设沿用上一帧的q5或目标q5
        q5_target = q_prev[4] # 或者插值
        
        # 调用 逆解函数
        p_target = current_pose_target[0:3]
        rpy_target = current_pose_target[3:6]
        qlim = [ (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi, np.pi), (0.0, 0.04)]
        q_sol, success, iters, cost = ik_smart_hybrid(p_target, rpy_target, qlim)
        # q_sol = my_ik_analytical(current_pose_target, q5_val=0.0, solution_mask=[1, -1])
        
        if q_sol is None:
            print(f"Error: 路径在 t={t:.2f}s 处无逆解 (不可达或奇异)!")
            valid_path = False
            break
            
        # 简单的解连续性检查 (防止多解跳变)
        # 解析解通常比较稳定，但如果有多组解需选择离 q_prev 最近的
        # 你的解析解目前返回单解，直接使用即可
        
        trajectory_q.append(q_sol)
        q_prev = q_sol
        
    if valid_path:
        return np.array(trajectory_q)
    else:
        return np.array([q_current]) # 返回原点防止报错

# ==========================================
# 4. 仿真主程序
# ==========================================
if __name__ == "__main__":
    # 初始关节角
    q_home = np.array([np.radians(0), np.radians(0), np.radians(0), np.radians(0), 0.0])
    
    # --- 1. 测试 MoveJ ---
    # 定义目标关节角 (随便选一个合法的姿态)
    q_target_j = np.array([np.radians(0), np.radians(-30), np.radians(-40), np.radians(20), 0.02])
    
    traj_j = move_j(q_home, q_target_j, speed_rad_s=0.157)
    
    # 可视化 MoveJ
    print("正在演示 MoveJ ...")
    myarm.plot(traj_j, dt=0.02, limits=[-0.5,0.5,-0.5,0.5,0,0.6], block=False)
    time.sleep(1)
    
    # --- 2. 测试 MoveL ---
    # 从当前位置 (q_target_j) 走直线到新的位置
    # 获取当前末端作为起点
    T_curr = myarm.fkine(q_target_j)
    curr_pos = T_curr.t
    curr_rpy = T_curr.rpy(order='zyx', unit='rad')
    
    # 设定直线运动目标: 保持姿态不变，只改变位置 (例如 Z 轴抬升，Y 轴移动)
    # 注意：5轴机械臂无法在任意位置保持任意姿态。
    # 你的解析逆解中，q4 依赖于 phi (pitch)。如果单纯改变 xyz 而锁死 rpy，可能导致无解。
    # 这里我们设定一个理论上可达的目标（基于你的机械臂结构，在同一平面内移动比较容易保持姿态）
    
    target_pos_l = curr_pos + np.array([0.0, 0, -0.10]) # 向下移动10cm
    target_rpy_l = curr_rpy # 尝试保持姿态
    
    target_pose_l = np.hstack((target_pos_l, target_rpy_l))
    
    traj_l = move_l(q_target_j, target_pose_l, speed_m_s=0.05)
    
    # 可视化 MoveL
    print("正在演示 MoveL ...")
    if len(traj_l) > 1:
        # 为了演示连贯性，接着 MoveJ 的终点画
        myarm.plot(traj_l, dt=0.02, limits=[-0.5,0.5,-0.5,0.5,0,0.6], block=True)
    else:
        print("MoveL 规划失败，请检查目标点是否在工作空间内。")

    # 绘制关节角度变化曲线 (验证平滑性)
    plt.figure()
    plt.title("Joint Trajectories (MoveJ -> MoveL)")
    combined_traj = np.vstack((traj_j, traj_l))
    plt.plot(combined_traj)
    plt.legend([f'q{i+1}' for i in range(5)])
    plt.xlabel('Step')
    plt.ylabel('Rad / m')
    plt.grid(True)
    plt.show()