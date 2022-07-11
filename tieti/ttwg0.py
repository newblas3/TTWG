import numpy as np
import matplotlib.pyplot as plt
import SOR
from matplotlib.font_manager import FontProperties
'创建数组 20×20'
# 误差
Errorx = np.zeros((20, 20))
Errory = np.zeros((20, 20))
# 系数
Xksi = np.zeros((20, 20))
Yksi = np.zeros((20, 20))
Xeta = np.zeros((20, 20))
Yeta = np.zeros((20, 20))
Alpha = np.zeros((20, 20))
Beta = np.zeros((20, 20))
Gama = np.zeros((20, 20))
# 正交源项
P = np.zeros((20, 20))
Q = np.zeros((20, 20))
Kksi = np.zeros((20, 20))
Keta = np.zeros((20, 20))
Psi = np.zeros((20, 20))
Phi = np.zeros((20, 20))
J = np.zeros((20, 20))
# 初始化数组
x = np.zeros((20, 20))
y = np.zeros((20, 20))
'边界初值'
for i in range(0, 20):
    ii = i
    x[i, 0] = ii  # 下边界点初值
    y[i, 0] = 0
    x[i, 19] = 9 + ii * 10 / 19  # 上边界点初值
    y[i, 19] = 19 * 0.75
for j in range(0, 10):
    jj = j
    x[19, j] = 19  # 下右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = jj  # 下左边界点初值
    y[0, j] = jj * 0.75
for j in range(10, 20):
    jj = j
    x[19, j] = 19  # 上右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = 19 / 2  # 上右边界点初值
    y[0, j] = jj * 0.75
'所有内部点初值'
for i in range(1, 19):
    for j in range(1, 19):
        x[i, j] = 5
        y[i, j] = 5
Relax = 0.8  # 松弛系数
# 下面是绘制的初始状态的网格图
# p, q =[], []
# for j in x:
#     p.append(j)
# for k in y:
#     q.append(k)
# plt.scatter(p, q, s=3, c='red')

# 下面是迭代求解后的图，给A和B分别赋返回值，A为x坐标值，B为y坐标值, C和D分别为历次迭代的值
A, B, C, D = SOR.s(x, y, Errorx, Errory, Relax, Alpha, Beta, Gama,Xksi,Xeta,Yksi,Yeta, 500, 0.0001)
p, q = [], []
for x in A:
    p.append(x)
for y in B:
    q.append(y)
plt.scatter(p, q, s=3, c='red')
plt.title('TietiWangge')
plt.legend()
plt.show()
