"""
混合运动学逆解，利用几何特性简化雅可比迭代，
由于机械臂是 5-DOF，无法完全任意定位和定姿，因此标准的雅可比迭代法无法直接应用，
需要利用几何特性进行降维，只在部分自由度上进行迭代
"""
import numpy as np
from scipy.optimize import least_squares

d1 = 0.08525
a2 = 0.12893
a3 = 0.129
d4 = 0.04039
d50 = 0.07403

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def mdh_T(theta, d, a, alpha):
    cth, sth = np.cos(theta), np.sin(theta)
    ca,  sa  = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ cth,       -sth,        0,        a],
        [ sth*ca,  cth*ca,     -sa,   -sa*d],
        [ sth*sa,  cth*sa,      ca,    ca*d],
        [ 0,          0,         0,        1],
    ], dtype=float)

def fk_all(q):
    q1,q2,q3,q4,q5 = q
    A1 = mdh_T(theta=q1 + 0.0,        d=0.08525,      a=0.0,     alpha=0.0)
    A2 = mdh_T(theta=q2 + np.pi/2,    d=0.0,          a=0.0,     alpha=np.pi/2)
    A3 = mdh_T(theta=q3 + 0.0,        d=0.0,          a=0.12893, alpha=0.0)
    A4 = mdh_T(theta=q4 - np.pi/2,    d=0.04039,      a=0.129,   alpha=0.0)
    A5 = mdh_T(theta=0.0,             d=q5 + 0.07403, a=0.0,     alpha=-np.pi/2)
    
    T0 = np.eye(4)
    T1 = T0 @ A1; T2 = T1 @ A2; T3 = T2 @ A3; T4 = T3 @ A4; T5 = T4 @ A5
    return [T0,T1,T2,T3,T4,T5]

def fk_pos_mdh(q):
    return fk_all(q)[-1][:3,3]

# 利用几何特性的混合迭代逆解函数

def ik_smart_hybrid(p_des, rpy_des, qlim=None):
    """
    利用 5-DOF 几何约束的混合雅可比迭代求解器
    思路：
    1. q1 由 rpy 姿态中的 Yaw 直接决定
    2. q4 不是独立变量，由目标 Pitch 和 q2, q3 决定 (q2+q3+q4 = const)
    3. 于是问题降维为寻找最佳的 [q2, q3, q5] 来匹配位置 (x,y,z)
    """
    p_des = np.array(p_des)
    r_des, p_des_ang, y_des = rpy_des
    
    # --- 步骤 1: 确定构型 (头朝上 vs 头朝下) ---
    # 根据 Roll 判断：接近 0 为上，接近 +/-pi 为下
    is_head_up = True
    if abs(wrap_to_pi(r_des + np.pi)) < 1e-3:
        is_head_up = False
    # if abs(wrap_to_pi(r_des)) > np.pi/2:
        # is_head_up = False
        
    # --- 步骤 2: 解析计算 q1 (Base Yaw) ---
    # 机械臂必须旋转到底座面向目标的位置，即 q1 总是让平面正对目标 (或目标在平面内)
    # rpy中的y，在头朝上时，其值和底部基座旋转角度完全一致，头朝下时，为-180°加上底部基座旋转角度
    if is_head_up:
        q1_fixed = wrap_to_pi(y_des)
    else:
        q1_fixed = wrap_to_pi(y_des - np.pi)

    # --- 步骤 3: 确定平面关节和 (Phi) ---
    # 头朝上: sum(q2,q3,q4) = -Pitch
    # 头朝下: sum(q2,q3,q4) = Pitch + pi
    if is_head_up:
        phi_target = -p_des_ang
    else:
        phi_target = p_des_ang + np.pi
        
    # --- 步骤 4: 构建降维优化问题 ---
    # 我们只优化 x = [q2, q3, q5]
    # q1 固定，q4 动态计算
    
    def error_func(x_optim):
        q2_curr, q3_curr, q5_curr = x_optim
        
        # 根据约束计算 q4
        # q2 + q3 + q4 = phi_target  =>  q4 = phi_target - q2 - q3
        q4_curr = phi_target - q2_curr - q3_curr
        
        # 组装完整关节向量
        q_full = [q1_fixed, q2_curr, q3_curr, q4_curr, q5_curr]
        
        # 计算 FK
        pos_curr = fk_pos_mdh(q_full)
        
        # 误差仅为位置误差 (姿态通过上面的约束天然满足)
        return pos_curr - p_des

    # 初始猜测
    # q2, q3 给个常用弯曲角度，q5 给中值
    x0 = [0.0, -np.pi/2, 0.02] 
    
    # 提取边界
    # qlim 格式: [(min,max), ...]
    if qlim is None:
        bounds = (-np.inf, np.inf)
    else:
        # 提取 q2, q3, q5 的边界
        lb = [qlim[1][0], qlim[2][0], qlim[4][0]]
        ub = [qlim[1][1], qlim[2][1], qlim[4][1]]
        bounds = (lb, ub)

    # --- 步骤 5: 求解 ---
    # 使用 Trust Region Reflective 算法 (适合有边界的最小二乘)
    res = least_squares(error_func, x0, bounds=bounds, ftol=1e-4, xtol=1e-4, max_nfev=100)
    
    # --- 步骤 6: 组装最终结果 ---
    q2_sol, q3_sol, q5_sol = res.x
    q4_sol = phi_target - q2_sol - q3_sol
    
    # 对 q4 进行 wrap 和限位检查
    q4_sol = wrap_to_pi(q4_sol)
    # 如果 q4 超限，说明该姿态不可达
    if qlim and (q4_sol < qlim[3][0] or q4_sol > qlim[3][1]):
         print(f"[Warning] 计算出的 q4 ({np.degrees(q4_sol):.2f}) 超出限位!")
    
    q_final = np.array([q1_fixed, q2_sol, q3_sol, q4_sol, q5_sol])
    
    success = res.success and (res.cost < 1e-3) # cost是残差平方和的一半
    return q_final, success, res.nfev, res.cost

# ================= 测试代码 =================
if __name__ == "__main__":
    from fk_mdh import myarm
    
    # 设定测试目标位姿
    # q_init = np.array([np.radians(20.0), np.radians(-30.7), np.radians(-42.2), np.radians(37.1), 0.0], dtype=float) # 测试 1 ✅
    q_init = np.array([np.radians(46.0), np.radians(-30.7), np.radians(-33.2), np.radians(-76.7), 0.02], dtype=float) # 测试 2 ✅
    # q_init = np.array([np.radians(46.0), np.radians(20.5), np.radians(-28.1), np.radians(-56.2), 0.0], dtype=float) # 测试 3 ✅
    # q_init = np.array([np.radians(15.3), np.radians(15.3), np.radians(28.1), np.radians(79.3), 0.0], dtype=float) # 测试 4 ⭕️
    # q_init = np.array([np.radians(46.5), np.radians(71.6), np.radians(43.5), np.radians(2.6), 0.02], dtype=float) # 测试 5
    print("目标位姿可视化")
    myarm.teach(q_init)
    
    T_init = myarm.fkine(q_init)
    T_init_mat = T_init.A
    T_pose = (T_init_mat[0:3, 3][0], T_init_mat[0:3, 3][1], T_init_mat[0:3, 3][2])
    T_init_euler_deg = T_init.rpy(order='zyx', unit='deg')
    T_init_euler_rad = T_init.rpy(order='zyx', unit='rad')
    x, y, z = T_pose
    roll, pitch, yaw = T_init_euler_rad
    p_init = np.array([x, y, z, roll, pitch, yaw])

    # p_target = np.array([0.2322232, 0.04154019, 0.29408501])
    # rpy_target = np.array([0.0, 0.624827872, 0.34906585]) # Roll=0 (Head Up)
    p_target = p_init[0:3]
    rpy_target = p_init[3:6]
    
    # 关节限位
    qlim = [ (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi, np.pi), (0.0, 0.04)]

    print("=== 开始 Hybrid IK 计算 ===")
    print(f"目标位置: {p_target}")
    print(f"目标 RPY: {rpy_target}")

    q_sol, success, iters, cost = ik_smart_hybrid(p_target, rpy_target, qlim)

    print("-" * 30)
    print(f"计算结束: {'成功' if success else '失败'}")
    print(f"优化迭代次数: {iters}")
    print(f"残差 (Cost): {cost:.6f}")
    
    print(f"解关节角 (rad): {q_sol}")
    print(f"解关节角 (deg): {np.degrees(q_sol)}")

    # 验证
    T_final = fk_all(q_sol)[-1]
    pos_final = T_final[:3, 3]
    
    # 计算实际 RPY
    import math
    sy = math.sqrt(T_final[0,0]**2 + T_final[1,0]**2)
    r_act = math.atan2(T_final[2,1], T_final[2,2])
    p_act = math.atan2(-T_final[2,0], sy)
    y_act = math.atan2(T_final[1,0], T_final[0,0])
    
    print(f"实际位置: {pos_final}")
    print(f"位置误差: {np.linalg.norm(p_target - pos_final) * 1000:.4f} mm")
    print(f"实际 RPY: {np.array([r_act, p_act, y_act])}")
    
    # 验证约束规律
    q2, q3, q4 = q_sol[1], q_sol[2], q_sol[3]
    q_sum = q2+q3+q4
    print(f"验证 Pitch 约束: q2+q3+q4 = {np.degrees(q_sum):.2f}°, -Pitch目标 = {np.degrees(-rpy_target[1]):.2f}°")

    # 可视化
    print("正在可视化...")
    myarm.teach(q_sol)