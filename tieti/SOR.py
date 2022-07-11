# -*- coding: utf-8 -*-
import numpy as np

# 迭代法 输入系数矩阵mx、值矩阵mr、迭代次数n、误差c(以数组模拟矩阵 行优先)
def s(mx,my, errorx, errory, relax, alpha, beta, gama, xksi, xeta, yksi, yeta, n, ac):
    x = mx
    y = my        # 迭代初值
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
        '边界点系数'
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
        for i in range(1, 20):
            for j in range(1, 20):
                alpha[i, j] = xeta[i, j] * xeta[i, j] + yeta[i, j] * yeta[i, j]
                beta[i, j] = xksi[i, j] * xeta[i, j] + yksi[i, j] * yeta[i, j]
                gama[i, j] = xksi[i, j] * xksi[i, j] + yksi[i, j] * yksi[i, j]
        abserrorx = np.zeros((20,20))
        abserrory = np.zeros((20,20))
        for i in range(1, 19):
            for j in range(1, 19):
                abserrorx[i, j] = mx[i, j]
                abserrory[i, j] = my[i, j]
                nx[i, j] = nx[i, j] + relax * ((alpha[i, j] * (nx[i + 1, j] + nx[i - 1, j]) - 0.5 * beta[i, j] * (nx[i + 1, j + 1] - nx[i + 1, j - 1] - nx[i - 1, j + 1] + nx[i - 1, j - 1]) + gama[i, j] * (nx[i, j + 1] + nx[i, j - 1])) / (0.000001 + 2 * (alpha[i, j] + gama[i, j])) - nx[i, j])
                ny[i, j] = ny[i, j] + relax * ((alpha[i, j] * (ny[i + 1, j] + ny[i - 1, j]) - 0.5 * beta[i, j] * (ny[i + 1, j + 1] - ny[i + 1, j - 1] - ny[i - 1, j + 1] + ny[i - 1, j - 1]) + gama[i, j] * (ny[i, j + 1] + ny[i, j - 1])) / (0.000001 + 2 * (alpha[i, j] + gama[i, j])) - ny[i, j])
                abserrorx[i, j] = abs(abserrorx[i, j]-nx[i, j])
                abserrory[i, j] = abs(abserrory[i, j] - ny[i, j])
        v, t = [], []
        v.append(nx[:])
        t.append(ny[:])
        # 计算误差
        if (np.max(abserrorx) < ac) and (np.max(abserrory) < ac):
            return nx, ny, v, t  # 当误差满足要求时 返回计算结果
        x = nx
        y = ny
        count += 1
    return False  # 若达到设定的迭代结果仍不满足精度要求 则方程无解