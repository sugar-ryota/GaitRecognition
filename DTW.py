#%%
import matplotlib.gridspec as gridspec
import numpy as np
import pylab as plt
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import *
#%%
T = 150
t = .4

# A = np.sin(np.array(range(T))/10)
# B = np.sin((np.array(range(T))/10 + t*np.pi))
# C = np.zeros((T))
seed(100)
rand()

A = list(randint(0, 15, 22))
# bはaを少しだけずらしたデータにする
B = [0, 2, 7, 3] + A

plt.plot(A)
plt.plot(B)
# plt.plot(C)
plt.show()
#%%
def δ(a, b): return (a - b)**2
def first(x): return x[0]
def second(x): return x[1]


def minVal(v1, v2, v3):
    if first(v1) <= min(first(v2), first(v3)):
        return v1, 0
    elif first(v2) <= first(v3):
        return v2, 1
    else:
        return v3, 2


def calc_dtw(A, B):
    S = len(A)
    T = len(B)

    m = [[0 for j in range(T)] for i in range(S)]
    m[0][0] = (δ(A[0], B[0]), (-1, -1))
    for i in range(1, S):
        m[i][0] = (m[i-1][0][0] + δ(A[i], B[0]), (i-1, 0))
    for j in range(1, T):
        m[0][j] = (m[0][j-1][0] + δ(A[0], B[j]), (0, j-1))

    for i in range(1, S):
        for j in range(1, T):
            minimum, index = minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
            indexes = [(i-1, j), (i, j-1), (i-1, j-1)]
            m[i][j] = (first(minimum)+δ(A[i], B[j]), indexes[index])
    return m


print("A-B: ", calc_dtw(A, B)[-1][-1][0])
# print("A-C: ", calc_dtw(A, C)[-1][-1][0])


# %%
def backward(m):
    path = []
    path.append([len(m)-1, len(m[0])-1])
    while True:
        path.append(m[path[-1][0]][path[-1][1]][1])
        if path[-1] == (0, 0):
            break
    path = np.array(path)
    return path


def plot_path(path, A, B):
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 5],
                           height_ratios=[5, 1]
                           )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])

    list_δ = [[t[0] for t in row] for row in m]
    list_δ = np.array(list_δ)
    # ax2.pcolor(list_δ, cmap=plt.cm.Greys)
    ax2.plot(path[:, 1], path[:, 0], c="C3")

    ax1.plot(A, range(len(A)))
    ax1.invert_xaxis()
    ax4.plot(B, c="C1")
    plt.show()

    for line in path:
        plt.plot(line, [A[line[0]], B[line[1]]], linewidth=0.2, c="black")
    plt.plot(A,label="sequence1")
    plt.plot(B,label="sequence2")
    plt.xlabel("time")
    plt.ylabel("data value")
    plt.legend()
    plt.show()


# %%
m = calc_dtw(A, B)
path = backward(m)
plot_path(path, A, B)


# %%
