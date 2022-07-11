import matplotlib.pyplot as plt
import numpy as np
t = [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]
v = [[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]]
t3 = np.array(t)
v3 = np.array(v)
for i in range(0, 4):  # 画网格竖线
    ti = [[t3[i, 0], t3[i, 1], t3[i, 2], t3[i, 3]]]
    vi = [[v3[i, 0], v3[i, 1], v3[i, 2], v3[i, 3]]]
    plt.plot(ti[0], vi[0], linewidth=0.5, color='r')
plt.xlim(-1, 20)
plt.ylim(-1, 18)
plt.scatter(t3, v3, s=5, c='b')  # 画散点
plt.title('Drawing Picture')
plt.show()