import numpy as np
import matplotlib.pyplot as plt

# ステップ関数
def step_function(x):
    y = x > 0
    return y.astype(np.int)
# # グラフ出力する
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()


# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# # グラフ出力する
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()


# ReLU関数
def relu(x):
    return np.maximum(0, x)
# print(relu(-0.3))
# print(relu(0))
# print(relu(0.3))
# print(relu(1))
