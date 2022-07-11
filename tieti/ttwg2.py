import numpy as np
import matplotlib.pyplot as plt

'''
初边值条件部分
'''
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
    x[i, 19] = 9 + ii * 10 / 19  # 上边界点初值
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
a = 0
b = 0
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
plt.xlim(-1, 20)
plt.ylim(-1, 18)
plt.scatter(p, q, s=3, c='red')
plt.title('Initial Fig')
plt.show()

'''
迭代求解部分
'''
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
    count = 0  # 迭代次数计数
    plt.ion()  # 开始制图
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
        plt.cla()
        plt.xlim(-1, 20)
        plt.ylim(-1, 18)
        plt.scatter(t, v, s=1, c='red')
        plt.title('Computing Fig')
        plt.pause(0.000001)
        # 判断误差
        if (np.max(abserrorx) < ac) and (np.max(abserrory) < ac):
            return nx, ny  # 当误差满足要求时 返回计算结果
        x = nx
        y = ny
        count += 1
    plt.ioff()
    plt.show()
    return False  # 若达到设定的迭代结果仍不满足精度要求 则方程无解
'''
    输入参数并计算
    当松弛系数太小，迭代次数太少时，迭代结果不满足精，因为迭代速度慢，太大也不行
'''
print('=' * 100)
for i in range(1):
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
    SOR(x, y, Relax, P, Q, n, ac)
