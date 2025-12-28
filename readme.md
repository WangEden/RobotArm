环境：
```bash
conda create -n "robotics" python=3.9
conda activate robotics
conda install "numpy<2" "matplotlib<3.8" -y
conda install roboticstoolbox-python spatialmath-python swift-sim -y
conda install pinocchio -c conda-forge
# or
pip install "numpy<2" -y
pip install "matplotlib<3.8" -y
pip install roboticstoolbox-python spatialmath-python swift-sim -y
pip install pin #(仅Linux可用)
```

文件说明
正运动学：fk_mdh.py
逆运动学-几何解析法：ik_analytical_posture.py
逆运动学-混合雅可比迭代法：ik_hybrid_posture_all.py
轨迹规划：path_trajectory_sim.py
运动控制：move_control_sim.py