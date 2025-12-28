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
