import numpy as np
import matplotlib.pyplot as plt
"初边值条件部分"
'创建数组 20×20'
P = np.zeros((20, 20))  # 正交源项
Q = np.zeros((20, 20))
x = np.zeros((20, 20))  # 坐标初始化为数组
y = np.zeros((20, 20))
'边界点的初值,所有点，横着从左往右数是i，竖着从下往上数是j'
for i in range(20):
    ii = i
    x[i, 0] = ii  # 下边界点初值
    y[i, 0] = 0
    x[i, 19] = 19 / 2 + ii / 2  # 上边界点初值
    y[i, 19] = 19 * 0.75
for j in range(10):
    jj = j
    x[19, j] = 19  # 下右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = jj * 19 / 18  # 下左边界点初值
    y[0, j] = jj * 0.75
for j in range(10, 20):
    jj = j
    x[19, j] = 19  # 上右边界点初值
    y[19, j] = jj * 0.75
    x[0, j] = 19 / 2  # 上左边界点初值
    y[0, j] = jj * 0.75
print("Created By Hcs")
print("=" * 100)
'所有内部点的初值'
a = 0   # 设置所有内部点初始位置
b = 0
for i in range(1, 19):
    for j in range(1, 19):
        x[i, j] = a
        y[i, j] = b
print("初始化网格……")
print("所有内部点初始位置为原点(0, 0)")
print("关闭Initial Picture窗口以继续")
"下面是绘制的初始状态的网格图"
p, q = [], []
for j in x:
    p.append(j)
for k in y:
    q.append(k)
p1 = np.asarray(p).squeeze()    # 去除多余的嵌套格式
q1 = np.asarray(q).squeeze()
plt.plot(p1, q1, linewidth=0.5, color='r')    # 画网格横线
for i in range(20):  # 画网格竖线
    plt.plot(p1[i], q1[i], linewidth=0.5, color='r')
plt.scatter(p1, q1, s=5, color='b')    # 画散点
plt.xlim(-1, 20)    # 图表显示范围，下同
plt.ylim(-1, 18)
plt.title('Initial Picture')
plt.show()
"迭代求解部分"
def SOR(mx, my, relax, tp, tq, n, ac):  # 迭代法 初值矩阵mx、my，relax松弛系数，迭代次数n、误差ac(以数组模拟矩阵 行优先)
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
    xm, ym = {}, {}  # 保存迭代的集合
    for count in range(n):
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
        i = 0   # 左下角点
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
        i = 0   # 左上角点
        j = 19
        xksi[i, j] = nx[i + 1, j] - nx[i, j]
        yksi[i, j] = ny[i + 1, j] - ny[i, j]
        xeta[i, j] = nx[i, j] - nx[i, j - 1]
        yeta[i, j] = ny[i, j] - ny[i, j - 1]
        i = 19  # 右上角点
        j = 19
        xksi[i, j] = nx[i, j] - nx[i - 1, j]
        yksi[i, j] = ny[i, j] - ny[i - 1, j]
        xeta[i, j] = nx[i, j] - nx[i, j - 1]
        yeta[i, j] = ny[i, j] - ny[i, j - 1]
        for j in range(1, 19):  # 左右边界
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
        for i in range(1, 19):  # 上下边界
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
        v, t = [], []  # 创建并保存坐标的集合
        for xi in nx:
            t.append(xi)
        for yi in ny:
            v.append(yi)
        t1 = np.asarray(t).squeeze()
        v1 = np.asarray(v).squeeze()
        xm.update({count: t1})  # 将每次迭代的值的坐标更新到字典里
        ym.update({count: v1})
        if (np.max(abserrorx) < ac) and (np.max(abserrory) < ac):   # 判断误差
            return xm, ym, count  # 当误差满足要求时 返回计算结果
        x = nx
        y = ny
    return 0, 0  # 若达到设定的迭代结果仍不满足精度要求 则方程无解
'''输入参数并计算，当松弛系数太小，迭代次数太少时，迭代结果不满足精，因为迭代速度慢，太大也不行'''
print('=' * 100)
Relax = float(input("输入松弛系数(默认0.9)：") or 0.9)  # 松弛系数
n = 500000  # 迭代次数
ac = float(input("输入误差(默认0.001)：") or 0.001)
while True:
    tmp = input("输入：正交方式(上[u];下[d];左[l];右[r]);右上[ur];无[n];默认无:")
    print("贴体网格生成中(没有输出图形表示发散了，需要更改松弛系数重新运行)……")
    for i in range(1, 19):  # 通过输入的正交方式tmp来判断贴近方式
        for j in range(1, 19):
            if tmp == 'u':  # 向上贴近
                P[i, j] = 0
                Q[i, j] = 0.8
            elif tmp == 'd':  # 向下贴近
                P[i, j] = 0
                Q[i, j] = - 0.4
            elif tmp == 'l':  # 向左贴近
                P[i, j] = - 0.3
                Q[i, j] = 0
            elif tmp == 'r':  # 向右贴近
                P[i, j] = 1
                Q[i, j] = 0
            elif tmp == 'ur':  # 向右上贴近
                P[i, j] = 1
                Q[i, j] = 1
            else:  # 居中贴近
                P[i, j] = 0
                Q[i, j] = 0
    A, B, count1 = SOR(x, y, Relax, P, Q, n, ac)    # 下面是迭代求解后画图，给A和B分别赋返回值，A为x坐标值，B为y坐标值，count1为达到精度时的迭代次数
    for count in range(count1 + 1):  # 使网格线绘制次数与散点绘制次数一致
        Ax = A[count]  # 从保存了所有迭代数据的字典里索引某次迭代的横纵坐标用于绘图
        By = B[count]
        v, t = [], []  # 创建并保存某次迭代结果的坐标的集合
        for xi in Ax:
            t.append(xi)
        for yi in By:
            v.append(yi)
        plt.cla()  # 清除上一次画的图
        t1 = np.asarray(t).squeeze()    # 去除多余的嵌套格式
        v1 = np.asarray(v).squeeze()
        plt.plot(t1, v1, linewidth=0.5, color='r')  # 画网格横线
        for i in range(20):
            plt.plot(t1[i], v1[i], linewidth=0.5, color='r')    # 画网格竖线
        if count == count1:
            print('-' * 100)
            print("绘制完成!继续输入正交方式可重新计算出图")
        plt.xlim(-1, 20)
        plt.ylim(-1, 18)
        plt.scatter(t1, v1, s=5, c='b')    # 画散点
        plt.title('Drawing Picture')
        plt.pause(0.0001)
    plt.show()
