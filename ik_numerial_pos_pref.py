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
                    lam=1e-3, k=0.5, step_clip=0.2,
                    q_pref=None, k_pref=0.2):
    q = np.array(q0, dtype=float).copy()

    for it in range(max_iter):
        p = fk_pos_mdh(q)
        e = p_des - p
        err = np.linalg.norm(e)
        if err < tol:
            return q, True, it, err

        J = jacobian_pos_geom(q)              # 3x5
        A = J @ J.T + (lam**2) * np.eye(3)
        Jsharp = J.T @ np.linalg.solve(A, np.eye(3))  # 5x3

        dq_task = Jsharp @ (k * e)

        dq = dq_task

        # ======= 零空间偏好项：把解拉向你想要的“上凸”构型 =======
        if q_pref is not None:
            N = np.eye(5) - Jsharp @ J               # 5x5
            dq0 = k_pref * (q_pref - q)              # 5,
            dq = dq_task + N @ dq0

        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq

        # 限位
        if qlim is not None:
            for i in range(4):
                q[i] = (q[i] + np.pi) % (2*np.pi) - np.pi
                q[i] = np.clip(q[i], qlim[i][0], qlim[i][1])
            q[4] = np.clip(q[4], qlim[4][0], qlim[4][1])

    return q, False, max_iter, np.linalg.norm(p_des - fk_pos_mdh(q))


if __name__ == "__main__":
    from fk_mdh import myarm

    q_init = np.array([np.radians(0.0),
                np.radians(-30.7),
                np.radians(-42.2),
                np.radians(37.1),
                0.0], dtype=float)
    
    q_init_deg = np.degrees(q_init)
    T_home = myarm.fkine(q_init)
    T_home_mat = T_home.A
    T_pose = (T_home_mat[0:3, 3][0], T_home_mat[0:3, 3][1], T_home_mat[0:3, 3][2])
    T_home_euler_deg = T_home.rpy(order='zyx', unit='deg')  # 提取欧拉角部分，ZYX 顺序
    T_home_euler_rad = T_home.rpy(order='zyx', unit='rad')
    x, y, z = T_pose
    roll, pitch, yaw = T_home_euler_rad
    pos_home = np.array([x, y, z, roll, pitch, yaw])
    # print("RTB_MDH计算结果: ")
    # print(f"关节空间坐标(rad): {np.round(q_init, 4)}")
    # print(f"关节空间坐标(deg): {np.round(q_init_deg, 4)}")
    # print(f"工作空间坐标: {np.round(pos_home, 4)}")

    # 用 RTB 算
    p_rtb = T_home_mat

    # 用你手写 FK 算
    T = fk_all(q_init)[-1]
    p_mine = T

    print("rtb:", p_rtb)
    print("mine:", p_mine)
    print("diff:", p_mine - p_rtb)
    """结果：
    rtb: [[ 8.11063819e-01  3.58183272e-17  5.84957675e-01  2.32426014e-01]
    [-3.58183272e-17  1.00000000e+00 -1.15690045e-17 -4.03900000e-02]
    [-5.84957675e-01 -1.15690045e-17  8.11063819e-01  2.94085010e-01]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    mine: [[ 8.11063819e-01  3.58183272e-17  5.84957675e-01  2.32426014e-01]
    [-3.58183272e-17  1.00000000e+00 -1.15690045e-17 -4.03900000e-02]
    [-5.84957675e-01 -1.15690045e-17  8.11063819e-01  2.94085010e-01]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    diff: [[0. 0. 0. 0.]
    [0. 0. 0. 0.]
    [0. 0. 0. 0.]
    [0. 0. 0. 0.]]
    """

    # 目标位置
    p_target = np.array([0.232, -0.04, 0.294], dtype=float)

    qlim = [
        (-np.pi, np.pi),
        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),
        (-np.pi, np.pi),
        (0.0, 0.04),
    ]

    q_pref = np.array([np.radians(0.0),
                   np.radians(-30.),
                   np.radians(-45.),
                   np.radians(38.),
                   0.0])

    q_sol, ok, iters, err = ik_position_dls(
        p_target, q_init, qlim=qlim, max_iter=2000, tol=1e-6,
        lam=1e-3, k=0.5, step_clip=0.1,
        q_pref=q_pref, k_pref=0.15
    )

    print("ok:", ok, "iters:", iters, "final_err:", err)
    print("q_sol_rad:", q_sol)
    print("q_sol_deg:", np.degrees(q_sol[:4]), q_sol[4])
    print("p_check:", fk_pos_mdh(q_sol))
    """结果：对初值敏感，有的会失败
    ok: True iters: 49 final_err: 9.732083595489646e-07
    q_sol_rad: [ 0.00168192 -0.52192111 -0.77078301  0.67644881  0.00142215]
    q_sol_deg: [  0.09636683 -29.90387674 -44.16261323  38.75766171] 0.0014221464506117523
    p_check: [ 0.23199922 -0.03999985  0.29400056]
    """
