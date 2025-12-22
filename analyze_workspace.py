import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb

# 引入你的机械臂定义
from fk_mdh import myarm

def calculate_manipulability(robot, q):
    """
    计算可操作度 (Manipulability)
    对于 5自由度机械臂，雅可比矩阵 J 是 6x5 的。
    我们使用 Yoshikawa 可操作度指数: w = sqrt(det(J^T * J))
    w 越大，机械臂越灵活；w 接近 0，表示接近奇异点。
    """
    # 计算基础坐标系下的几何雅可比矩阵 (6x5)
    J = robot.jacob0(q)
    
    # 计算可操作度 w = sqrt(det(J.T @ J))
    # 对于冗余或欠驱动机械臂，这是衡量奇异性的标准方法
    w = np.sqrt(np.linalg.det(J.T @ J))
    return w

def generate_workspace_cloud(robot, n_samples=3000):
    """
    使用蒙特卡洛法生成工作空间点云
    """
    print(f"正在采样 {n_samples} 个点以生成工作空间...")
    
    # 1. 定义关节限位 (根据 ik_hybrid_posture_all.py 中的定义)
    # q1: -pi ~ pi
    # q2: -pi/2 ~ pi/2
    # q3: -pi/2 ~ pi/2
    # q4: -pi ~ pi
    # q5: 0.0 ~ 0.04 (移动副)
    q_limits = np.array([
        [-np.pi, np.pi],
        [-np.pi/2, np.pi/2],
        [-np.pi/2, np.pi/2],
        [-np.pi, np.pi],
        [0.0, 0.04]
    ])
    
    points = []
    manipulabilities = []
    
    for _ in range(n_samples):
        # 在限位内随机采样关节角
        q_rand = np.random.uniform(q_limits[:, 0], q_limits[:, 1])
        
        # 正运动学计算末端位置
        T = robot.fkine(q_rand)
        pos = T.t # [x, y, z]
        
        # 计算奇异度
        w = calculate_manipulability(robot, q_rand)
        
        points.append(pos)
        manipulabilities.append(w)
        
    return np.array(points), np.array(manipulabilities)

def plot_workspace_analysis(points, w_scores):
    """
    绘制3D散点图，颜色代表奇异度
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用散点图，c=w_scores 设置颜色映射
    # cmap='jet_r' : 红色代表低可操作度（奇异），蓝色代表高可操作度
    # 或者 'viridis': 黄色高，紫色低
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=w_scores, cmap='viridis', s=5, alpha=0.6)
    
    # 添加颜色条
    cbar = fig.colorbar(p, ax=ax, shrink=0.6)
    cbar.set_label('Manipulability Index (Close to 0 is Singular)')
    
    # 设置坐标轴标签
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Robot Workspace & Singularity Analysis\n(Yellow=Flexible, Purple=Singular)')
    
    # 设置轴比例相等 (这是观察真实形状的关键)
    # matplotlib 默认不支持 set_aspect('equal') for 3D，使用自定义范围
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

if __name__ == "__main__":
    # 生成数据
    pts, w = generate_workspace_cloud(myarm, n_samples=5000)
    
    # 统计信息
    print(f"最大可操作度: {np.max(w):.4f}")
    print(f"最小可操作度: {np.min(w):.4f} (接近0即为奇异)")
    
    # 绘图
    plot_workspace_analysis(pts, w)
    
    print("\n分析建议:")
    print("1. 观察颜色较深（紫色/深蓝）的区域，这些是奇异区。")
    print("2. MoveL 直线运动时，请确保路径完全落在亮黄色/绿色区域内。")
    print("3. 如果目标点在边缘或颜色很深的地方，逆解数值法(ik_smart_hybrid)容易计算发散或产生跳变。")