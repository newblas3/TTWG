import numpy as np
import matplotlib.pyplot as plt

"初边值条件部分"
'创建数组 20×20'
# 正交源项
P = np.zeros((20, 20))
Q = np.zeros((20, 20))
# 坐标初始化为数组
x = np.zeros((20, 20))
y = np.zeros((20, 20))
'边界点的初值'
for i in range(0, 20):
    ii = i
    x[i, 0] = ii  # 下边界点初值
    y[i, 0] = 0
    x[i, 19] = 19 / 2 + ii / 2  # 上边界点初值
    y[i, 19] = 19 * 0.75
for j in range(0, 10):
    jj = j
    x[19, j] = 19  # 下右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = jj * 19 / 18 # 下左边界点初值
    y[0, j] = jj * 0.75
for j in range(10, 20):
    jj = j
    x[19, j] = 19  # 上右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = 19 / 2  # 上左边界点初值
    y[0, j] = jj * 0.75
print("=" * 100)
# a, b = map(int, input("输入内部点初始坐标(x y空格分开):").split())
a = 15
b = 5
print("初始化网格……")
'所有内部点的初值'
for i in range(1, 19):
    for j in range(1, 19):
        x[i, j] = a
        y[i, j] = b
# 下面是绘制的初始状态的网格图
p, q = [], []
for j in x:
    p.append(j)
for k in y:
    q.append(k)
p1 = np.asarray(p).squeeze()
p2 = np.array(p1.flatten()).reshape((20, 20))   # 将嵌套的数组重组为普通的数组，便于索引取值
q1 = np.asarray(q).squeeze()
q2 = np.array(q1.flatten()).reshape((20, 20))   # 将嵌套的数组重组为普通的数组，便于索引取值
# 54-59行为画网格线
for i in range(1, 19):
    for j in range(1, 19):
        pi = [[p2[i, j], p2[i + 1, j]], [p2[i, j], p2[i, j + 1]], [p2[i, j], p2[i - 1, j]],
              [p2[i, j], p2[i, j - 1]], [p2[i + 1, j - 1], p2[i, j - 1]], [p2[i + 1, j - 1], p2[i + 1, j]],
              [p2[i + 1, j + 1], p2[i, j + 1]], [p2[i + 1, j + 1], p2[i + 1, j]],
              [p2[i - 1, j - 1], p2[i, j - 1]], [p2[i - 1, j - 1], p2[i - 1, j]],
              [p2[i - 1, j + 1], p2[i - 1, j]], [p2[i - 1, j + 1], p2[i, j + 1]]]
        qi = [[q2[i, j], q2[i + 1, j]], [q2[i, j], q2[i, j + 1]], [q2[i, j], q2[i - 1, j]],
              [q2[i, j], q2[i, j - 1]], [q2[i + 1, j - 1], q2[i, j - 1]], [q2[i + 1, j - 1], q2[i + 1, j]],
              [q2[i + 1, j + 1], q2[i, j + 1]], [q2[i + 1, j + 1], q2[i + 1, j]],
              [q2[i - 1, j - 1], q2[i, j - 1]], [q2[i - 1, j - 1], q2[i - 1, j]],
              [q2[i - 1, j + 1], q2[i - 1, j]], [q2[i - 1, j + 1], q2[i, j + 1]]]
        for m in range(0, 12):
            plt.plot(pi[m], qi[m], linewidth=0.5, color='r')
plt.scatter(p2, q2, s=5, color='b')
plt.xlim(-1, 20)
plt.ylim(-1, 18)
plt.title('Initial Fig')
plt.show()

"迭代求解部分"
# 迭代法 初值矩阵mx、my，relax松弛系数，迭代次数n、误差ac(以数组模拟矩阵 行优先)
def SOR(mx, my, relax, tp, tq, n, ac):
    xksi = np.zeros((20, 20))
    yksi = np.zeros((20, 20))
    xeta = np.zeros((20, 20))
    yeta = np.zeros((20, 20))
    alpha = np.zeros((20, 20))
    beta = np.zeros((20, 20))
    gama = np.zeros((20, 20))
    tj = np.zeros((20, 20))
    x = mx
    y = my  # 迭代初值
    xm = {}    # 创建字典用于保存每次迭代的结果
    ym = {}
    count = 0  # 迭代次数计数
    while count < n:
        nx = x
        ny = y  # 保存单次迭代后的值的集合
        '内部点系数'
        for i in range(1, 19):
            for j in range(1, 19):
                xksi[i, j] = nx[i + 1, j] - nx[i, j]
                yksi[i, j] = ny[i + 1, j] - ny[i, j]
                xeta[i, j] = nx[i, j + 1] - nx[i, j]
                yeta[i, j] = ny[i, j + 1] - ny[i, j]
        '边界点系数，向前或向后差分，步长Δ为1，精度为一阶精度'
        # 左下角点
        i = 0
        j = 0
        xksi[i, j] = nx[i + 1, j] - nx[i, j]
        yksi[i, j] = ny[i + 1, j] - ny[i, j]
        xeta[i, j] = nx[i, j + 1] - nx[i, j]
        yeta[i, j] = ny[i, j + 1] - ny[i, j]
        i = 19
        j = 0
        xksi[i, j] = nx[i, j] - nx[i - 1, j]
        yksi[i, j] = ny[i, j] - ny[i - 1, j]
        xeta[i, j] = nx[i, j + 1] - nx[i, j]
        yeta[i, j] = ny[i, j + 1] - ny[i, j]
        # 左上角点
        i = 0
        j = 19
        xksi[i, j] = nx[i + 1, j] - nx[i, j]
        yksi[i, j] = ny[i + 1, j] - ny[i, j]
        xeta[i, j] = nx[i, j] - nx[i, j - 1]
        yeta[i, j] = ny[i, j] - ny[i, j - 1]
        # 右上角点
        i = 19
        j = 19
        xksi[i, j] = nx[i, j] - nx[i - 1, j]
        yksi[i, j] = ny[i, j] - ny[i - 1, j]
        xeta[i, j] = nx[i, j] - nx[i, j - 1]
        yeta[i, j] = ny[i, j] - ny[i, j - 1]
        # 左右边界
        for j in range(1, 19):
            i = 0  # 左边界
            xksi[i, j] = nx[i + 1, j] - nx[i, j]
            yksi[i, j] = ny[i + 1, j] - ny[i, j]
            xeta[i, j] = nx[i, j + 1] - nx[i, j]
            yeta[i, j] = ny[i, j + 1] - ny[i, j]
            i = 19  # 右边界
            xksi[i, j] = nx[i, j] - nx[i - 1, j]
            yksi[i, j] = ny[i, j] - ny[i - 1, j]
            xeta[i, j] = nx[i, j + 1] - nx[i, j]
            yeta[i, j] = ny[i, j + 1] - ny[i, j]
        # 上下边界
        for i in range(1, 19):
            j = 0  # 下边界
            xksi[i, j] = nx[i + 1, j] - nx[i, j]
            yksi[i, j] = ny[i + 1, j] - ny[i, j]
            xeta[i, j] = nx[i, j + 1] - nx[i, j]
            yeta[i, j] = ny[i, j + 1] - ny[i, j]
            j = 19  # 上边界
            xksi[i, j] = nx[i, j] - nx[i - 1, j]
            yksi[i, j] = ny[i, j] - ny[i - 1, j]
            xeta[i, j] = nx[i, j] - nx[i, j - 1]
            yeta[i, j] = ny[i, j] - ny[i, j - 1]
        for i in range(1, 20):  # α，β，γ系数
            for j in range(1, 20):
                alpha[i, j] = xeta[i, j] * xeta[i, j] + yeta[i, j] * yeta[i, j]
                beta[i, j] = xksi[i, j] * xeta[i, j] + yksi[i, j] * yeta[i, j]
                gama[i, j] = xksi[i, j] * xksi[i, j] + yksi[i, j] * yksi[i, j]
                tj[i, j] = xksi[i, j] * yeta[i, j] - xeta[i, j] * yksi[i, j]
        abserrorx = np.zeros((20, 20))
        abserrory = np.zeros((20, 20))
        for i in range(1, 19):
            for j in range(1, 19):  # 计算误差并迭代得到下一个值
                abserrorx[i, j] = mx[i, j]
                abserrory[i, j] = my[i, j]
                nx[i, j] = nx[i, j] + relax * ((alpha[i, j] * (nx[i + 1, j] + nx[i - 1, j]) - 0.5 * beta[i, j] * (
                            nx[i + 1, j + 1] - nx[i + 1, j - 1] - nx[i - 1, j + 1] + nx[i - 1, j - 1]) + gama[i, j] * (
                                                            nx[i, j + 1] + nx[i, j - 1]) + tj[i, j] * tj[i, j] * (
                                                            xksi[i, j] * tp[i, j] + xeta[i, j] * tq[i, j])) / (
                                                           0.00000001 + 2 * (alpha[i, j] + gama[i, j])) - nx[i, j])
                ny[i, j] = ny[i, j] + relax * ((alpha[i, j] * (ny[i + 1, j] + ny[i - 1, j]) - 0.5 * beta[i, j] * (
                            ny[i + 1, j + 1] - ny[i + 1, j - 1] - ny[i - 1, j + 1] + ny[i - 1, j - 1]) + gama[i, j] * (
                                                            ny[i, j + 1] + ny[i, j - 1]) + tj[i, j] * tj[i, j] * (
                                                            yksi[i, j] * tp[i, j] + yeta[i, j] * tq[i, j])) / (
                                                           0.00000001 + 2 * (alpha[i, j] + gama[i, j])) - ny[i, j])
                abserrorx[i, j] = abs(abserrorx[i, j] - nx[i, j])
                abserrory[i, j] = abs(abserrory[i, j] - ny[i, j])
        v, t = [], []   # 创建并保存坐标的集合
        for xi in nx:
            t.append(xi)
        for yi in ny:
            v.append(yi)
        # plt.cla()
        t1 = np.asarray(t).squeeze()
        t2 = t1.faltten()
        t3 = np.array(t2).reshape((20, 20))
        v1 = np.asarray(v).squeeze()
        v2 = v1.faltten()
        v3 = np.array(v2).reshape((20, 20))
        # xm.update({count: t3})
        # ym.update({count: v3})
        # 判断误差
        if (np.max(abserrorx) < ac) and (np.max(abserrory) < ac):
            return nx, ny  # 当误差满足要求时 返回计算结果
        x = nx
        y = ny
        count += 1
    return False  # 若达到设定的迭代结果仍不满足精度要求 则方程无解

'''
    输入参数并计算
    当松弛系数太小，迭代次数太少时，迭代结果不满足精，因为迭代速度慢，太大也不行
'''
print('=' * 100)
Relax = float(input("输入松弛系数："))  # 松弛系数
n = 50000  # 迭代次数
ac = float(input("输入误差："))
tmp = input("输入：正交方式(上[u];下[d];左[l];右[r]);右上[ur];无[n]:")
print("计算绘图中……")
for i in range(1, 19):
    for j in range(1, 19):
        if tmp == 'u':
            P[i, j] = 0
            Q[i, j] = 0.8
        elif tmp == 'd':
            P[i, j] = 0
            Q[i, j] = - 0.4
        elif tmp == 'l':
            P[i, j] = - 0.3
            Q[i, j] = 0
        elif tmp == 'r':
            P[i, j] = 1
            Q[i, j] = 0
        elif tmp == 'ur':
            P[i, j] = 1
            Q[i, j] = 1
        else:
            P[i, j] = 0
            Q[i, j] = 0
# 下面是迭代求解后的图，给A和B分别赋返回值，A为x坐标值，B为y坐标值
A, B = SOR(x, y, Relax, P, Q, n, ac)
# plt.ion()
# count = 0
# while count < n:
#     Ax = A[count]
#     By = B[count]
#     v, t = [], []  # 创建并保存坐标的集合
#     for xi in Ax:
#         t.append(xi)
#     for yi in By:
#         v.append(yi)
#     plt.cla()
#     t1 = np.asarray(t).squeeze()
#     t2 = t1.flatten()
#     t3 = np.array(t2).reshape((20, 20))
#     v1 = np.asarray(v).squeeze()
#     v2 = v1.flatten()
#     v3 = np.array(v2).reshape((20, 20))
#     # 180-193行为画网格
#     for i in range(1, 19):
#         for j in range(1, 19):
#             ti = [[t3[i, j], t3[i + 1, j]], [t3[i, j], t3[i, j + 1]], [t3[i, j], t3[i - 1, j]],
#                   [t3[i, j], t3[i, j - 1]], [t3[i + 1, j - 1], t3[i, j - 1]], [t3[i + 1, j - 1], t3[i + 1, j]],
#                   [t3[i + 1, j + 1], t3[i, j + 1]], [t3[i + 1, j + 1], t3[i + 1, j]],
#                   [t3[i - 1, j - 1], t3[i, j - 1]], [t3[i - 1, j - 1], t3[i - 1, j]],
#                   [t3[i - 1, j + 1], t3[i - 1, j]], [t3[i - 1, j + 1], t3[i, j + 1]]]
#             vi = [[v3[i, j], v3[i + 1, j]], [v3[i, j], v3[i, j + 1]], [v3[i, j], v3[i - 1, j]],
#                   [v3[i, j], v3[i, j - 1]], [v3[i + 1, j - 1], v3[i, j - 1]], [v3[i + 1, j - 1], v3[i + 1, j]],
#                   [v3[i + 1, j + 1], v3[i, j + 1]], [v3[i + 1, j + 1], v3[i + 1, j]],
#                   [v3[i - 1, j - 1], v3[i, j - 1]], [v3[i - 1, j - 1], v3[i - 1, j]],
#                   [v3[i - 1, j + 1], v3[i - 1, j]], [v3[i - 1, j + 1], v3[i, j + 1]]]
#             for m in range(0, 12):
#                 plt.plot(ti[m], vi[m], linewidth=0.5, color='r')
#     plt.xlim(-1, 20)
#     plt.ylim(-1, 18)
#     plt.scatter(t3, v3, s=8, c='b')
#     plt.title('Computing Fig')
#     plt.pause(0.000001)