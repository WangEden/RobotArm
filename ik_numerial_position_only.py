import numpy as np

# ===== 常量（你的参数） =====
d1 = 0.08525
a2 = 0.12893
a3 = 0.129
d4 = 0.04039
d50 = 0.07403

def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], dtype=float)

def Rx(al):
    c, s = np.cos(al), np.sin(al)
    return np.array([[1,0,0],
                     [0,c,-s],
                     [0,s, c]], dtype=float)

def mdh_T(theta, d, a, alpha):
    """
    RTB / Corke Modified DH:
      T = Tx(a) Rx(alpha) Rz(theta) Tz(d)
    """
    cth, sth = np.cos(theta), np.sin(theta)
    ca,  sa  = np.cos(alpha), np.sin(alpha)

    T = np.array([
        [ cth,       -sth,        0,        a],
        [ sth*ca,  cth*ca,     -sa,   -sa*d],
        [ sth*sa,  cth*sa,      ca,    ca*d],
        [ 0,          0,         0,        1],
    ], dtype=float)
    return T


def fk_all(q):
    """
    q = [q1,q2,q3,q4,q5], q1~q4 rad, q5 m
    返回 T_list: [T0, T1, ..., T5]  (T0=I)
    """
    q1,q2,q3,q4,q5 = q

    A1 = mdh_T(theta=q1 + 0.0,        d=0.08525,      a=0.0,     alpha=0.0)
    A2 = mdh_T(theta=q2 + np.pi/2,    d=0.0,          a=0.0,     alpha=np.pi/2)
    A3 = mdh_T(theta=q3 + 0.0,        d=0.0,          a=0.12893, alpha=0.0)
    A4 = mdh_T(theta=q4 - np.pi/2,    d=0.04039,      a=0.129,   alpha=0.0)
    A5 = mdh_T(theta=0.0,             d=q5 + 0.07403, a=0.0,     alpha=-np.pi/2)  # prismatic: d 变

    T0 = np.eye(4)
    T1 = T0 @ A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5

    return [T0,T1,T2,T3,T4,T5]

def fk_pos_mdh(q):
    T5 = fk_all(q)[-1]
    return T5[:3,3].copy()

def jacobian_pos_geom(q):
    """
    几何法位置雅可比 Jv (3x5)
    """
    Ts = fk_all(q)
    on = Ts[-1][:3,3]     # o_n

    Jv = np.zeros((3,5), dtype=float)

    # 关节类型：前4个R，第5个P
    joint_type = ['R','R','R','R','P']

    for i in range(1,6):  # i=1..5，对应关节i
        Ti_1 = Ts[i-1]        # ^0T_{i-1}
        zi_1 = Ti_1[:3,2]     # z_{i-1} in base
        oi_1 = Ti_1[:3,3]     # o_{i-1} in base

        if joint_type[i-1] == 'R':
            Jv[:, i-1] = np.cross(zi_1, (on - oi_1))
        else:  # 'P'
            Jv[:, i-1] = zi_1

    return Jv

def ik_position_dls(p_des, q0, qlim=None, max_iter=500, tol=1e-6,
                    lam=1e-3, k=0.5, step_clip=0.2):
    """
    DLS: dq = J^T (J J^T + lam^2 I)^-1 * (k e)
    step_clip: 限制每步最大关节增量（防止发散）
    """
    q = np.array(q0, dtype=float).copy()

    for it in range(max_iter):
        p = fk_pos_mdh(q)
        e = p_des - p
        err = np.linalg.norm(e)
        if err < tol:
            return q, True, it, err

        J = jacobian_pos_geom(q)          # 3x5
        A = J @ J.T + (lam**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, k * e)   # 5,

        # 防发散：限制单步幅度
        dq = np.clip(dq, -step_clip, step_clip)

        q = q + dq

        # 限位
        if qlim is not None:
            for i in range(4):
                q[i] = (q[i] + np.pi) % (2*np.pi) - np.pi   # wrap 到 [-pi,pi] 更合理
                q[i] = np.clip(q[i], qlim[i][0], qlim[i][1])
            q[4] = np.clip(q[4], qlim[4][0], qlim[4][1])

    return q, False, max_iter, np.linalg.norm(p_des - fk_pos_mdh(q))


if __name__ == "__main__":
    from fk_mdh import myarm

    # q_init = np.array([np.radians(20.0), np.radians(-30.7), np.radians(-42.2), np.radians(37.1), 0.0], dtype=float) # 测试 1
    q_init = np.array([np.radians(10.0), np.radians(-46.0), np.radians(-84.4), np.radians(40.9), 0.02], dtype=float) # 测试 2
    # print("初始关节量(rad):", q_init)
    # [ 0.         -0.53581608 -0.73652894  0.64751715  0.        ]
    # myarm.teach(q_init)

    q_init_deg = np.degrees(q_init)

    T_init = myarm.fkine(q_init)
    T_init_mat = T_init.A
    T_pose = (T_init_mat[0:3, 3][0], T_init_mat[0:3, 3][1], T_init_mat[0:3, 3][2])
    T_init_euler_deg = T_init.rpy(order='zyx', unit='deg')  # 提取欧拉角部分，ZYX 顺序
    T_init_euler_rad = T_init.rpy(order='zyx', unit='rad')
    x, y, z = T_pose
    roll, pitch, yaw = T_init_euler_rad
    p_init = np.array([x, y, z, roll, pitch, yaw])
    # print("RTB_MDH计算结果: ")
    # print(f"关节空间坐标(rad): {np.round(q_init, 4)}")
    # print(f"关节空间坐标(deg): {np.round(q_init_deg, 4)}")
    # print(f"工作空间坐标: {np.round(p_init, 4)}")

    # # 用 RTB 算
    # p_rtb = T_init_mat

    # # 用手写 FK 算
    # T = fk_all(q_init)[-1]
    # p_mine = T

    # print("rtb:", p_rtb)
    # print("mine:", p_mine)
    # print("diff:", p_mine - p_rtb)

    # 目标位置
    p_target = p_init[0:3]
    print("目标位置:", p_target)
    
    q_iter_start = np.array([np.radians(10.0), np.radians(0.0), np.radians(0.0), np.radians(0.0), 0.02], dtype=float)
    qlim = [ (-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi, np.pi), (0.0, 0.04),]
    q_sol, status, iters, err = ik_position_dls(
        p_target, q_iter_start, qlim=qlim, max_iter=2000, tol=1e-4, # 误差容许值：0.1mm
        lam=1e-3, k=0.5, step_clip=0.1
    )

    print("迭代结果:", status, "迭代轮次:", iters, "末端位置误差:", err)
    print("迭代结果位置:", fk_pos_mdh(q_sol), "\n迭代结果关节量:", q_sol)
    # print("q_sol_deg:", np.degrees(q_sol[:4]), q_sol[4])
    print("迭代前位姿可视化")
    myarm.teach(q_iter_start)
    print("迭代后位姿可视化")
    myarm.teach(q_sol)