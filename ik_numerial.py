import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from roboticstoolbox import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH

# ================= 1. 机器人建模 (你的MDH参数) =================
L1 = RevoluteMDH(d=0.08525, a=0.0, alpha=0.0, offset=0.0)
L2 = RevoluteMDH(d=0.0, a=0.0, alpha=np.pi/2, offset=np.pi/2)
L3 = RevoluteMDH(d=0.0, a=0.12893, alpha=0.0, offset=0.0)
L4 = RevoluteMDH(d=0.04039, a=0.129, alpha=0.0, offset=-np.pi/2)
# L5 是移动关节
L5 = PrismaticMDH(theta=0.0, a=0.0, alpha=-np.pi/2, offset=0.07403, qlim=[0.0, 0.04])

myarm = DHRobot([L1, L2, L3, L4, L5], name="My5DOF")

# ================= 2. 阻尼最小二乘法 (DLS) 逆运动学函数 =================

def my_ik_dls(robot, T_target, q_init=None, max_iter=100, tol=1e-4, damping=0.05):
    """
    参数:
    - damping (lambda): 阻尼因子。
      对于5自由度机器人，建议设为 0.01 ~ 0.1 之间。
      值越大越稳定(防止飞车)，但精度可能稍降；值越小越精确，但可能震荡。
    """
    
    # 1. 初始化
    if q_init is None:
        q = np.zeros(robot.n)
    else:
        q = np.copy(q_init)
        
    if not isinstance(T_target, SE3):
        T_target = SE3(T_target)

    # 记录历程
    error_history = []
    
    # 单位矩阵 (5x5)，用于 DLS 公式
    I = np.eye(robot.n)

    for k in range(max_iter):
        # --- A. 正运动学 (反馈) ---
        T_curr = robot.fkine(q)
        
        # --- B. 计算误差 (Error Twist) ---
        # e = [dx, dy, dz, dRx, dRy, dRz] (6x1)
        # 这里的 delta 方法计算的是 T_curr 到 T_target 的瞬时空间速度向量
        e = T_curr.delta(T_target)
        
        # 检查误差范数
        error_norm = np.linalg.norm(e)
        error_history.append(error_norm)
        
        if error_norm < tol:
            print(f"Converged! Iteration: {k}, Error: {error_norm:.6f}")
            return q, True, error_history
            
        # --- C. 计算雅可比矩阵 J (6x5) ---
        J = robot.jacob0(q)
        
        # --- D. DLS 迭代方程 (核心) ---
        # 公式: dq = (J.T * J + λ² * I)⁻¹ * J.T * e
        
        # 1. 计算 H = J^T * J + λ² * I
        H = J.T @ J + (damping**2) * I
        
        # 2. 计算 g = J^T * e (梯度方向)
        g = J.T @ e
        
        # 3. 求解线性方程 H * dq = g
        # 使用 np.linalg.solve 比 inv(H) 更快更准
        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed (Singularity).")
            break
            
        # --- E. 更新关节角 ---
        q = q + dq
        
        # --- F. 关节限位 (特别是 L5) ---
        q[4] = np.clip(q[4], 0.0, 0.04)
        
        # (可选) 动态阻尼调整：随着误差变小，减小阻尼以提高最终精度
        if error_norm < 0.1:
            damping = max(0.001, damping * 0.95)

    print(f"Iteration limit reached. Final Error: {error_norm:.6f}")
    return q, False, error_history

# ================= 3. 验证 =================
print("\n--- DLS Inverse Kinematics Verification ---")

# 1. 设定目标 (Target)
q_expect = np.array([-2.1415, -0.6789, -0.7854, 2.07, 0.0])
T_goal = myarm.fkine(q_expect)
print(f"目标关节角: {np.round(q_expect, 4)}")
print("目标位姿 (Position):\n", T_goal.t)

# 2. 设定初始猜测 (Initial Guess) - 依然使用你的“坏猜测”
q_guess = np.array([np.radians(-110.1), np.radians(-40.9), np.radians(-34.8), np.radians(-120.7), 0.00])
print(f"初始猜测: {np.round(q_guess, 4)}")

# 3. 运行 DLS 求解
q_sol, success, err_hist = my_ik_dls(myarm, T_goal, q_init=q_guess, max_iter=200, damping=0.1)

# 4. 结果分析
print("\n--- 求解结果 ---")
print(f"计算出的关节角: {np.round(q_sol, 4)}")

# 计算笛卡尔空间位置误差
T_final = myarm.fkine(q_sol)
pos_error = np.linalg.norm(T_goal.t - T_final.t)
print(f"最终位置误差 (m): {pos_error:.6f}")

# 解释 5-DOF 的局限性
if pos_error < 1e-3:
    print(">> 位置求解成功！(位置误差极小)")
else:
    print(">> 位置误差较大，可能陷入局部极小值。")

# 检查姿态误差 (5DOF 必定会有姿态误差，因为它不能满足任意6D姿态)
rpy_goal = T_goal.rpy(unit='deg')
rpy_final = T_final.rpy(unit='deg')
print(f"目标 RPY (deg): {np.round(rpy_goal, 2)}")
print(f"结果 RPY (deg): {np.round(rpy_final, 2)}")
print(">> 注意：由于是5自由度机器人，RPY角（特别是Roll/Yaw）不一致是正常的数学现象。")