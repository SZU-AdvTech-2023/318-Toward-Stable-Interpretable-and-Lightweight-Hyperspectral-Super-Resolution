import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from PIL import Image


# 加载MATLAB的.mat文件
mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/results/19.mat')


# 从MATLAB数据中提取高光谱图像立方体
data = mat['data']

# 获取图像的维度信息
num_rows, num_cols, num_bands = data.shape

# 创建一个三维绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y, Z = np.meshgrid(np.arange(num_cols), np.arange(num_rows), np.arange(num_bands))

# 使用scatter绘制立方体
ax.scatter(X, Y, Z, c=data, marker='.', cmap='jet')# jet

# 设置坐标轴标签
ax.set_xlabel('Row')
ax.set_ylabel('Column')
ax.set_zlabel('Band')

# ax.view_init(elev=20, azim=60)  # 调整视角以旋转图像
# ax.view_init(elev=-90, azim=0)  # 仰角设置为负值以使正上方朝着
ax.view_init(elev=110, azim=90)  # 仰角设置为20°，方位角设置为135° 5
# ax.view_init(elev=75, azim=135)  # 仰角设置为75°，方位角设置为135° 4
# ax.view_init(elev=75, azim=90)  # 仰角设置为75°，方位角设置为90° 3
# ax.view_init(elev=110, azim=90)  # 仰角设置为75°，方位角设置为90° 2

ax.grid(False)
# 隐藏坐标轴
ax.set_axis_off()

plt.savefig('./figs/high5.png')

# # 显示图像
# plt.show()


