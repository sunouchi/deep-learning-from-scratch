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


# 恒等関数
def identity_function(x):
    return x


# 重みとバイアスの初期化
def init_network():
    network = {
        'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        'b1': np.array([0.1, 0.2, 0.3]),
        'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        'b2': np.array([0.1, 0.2]),
        'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
        'b3': np.array([0.1, 0.2]),
    }
    return network


# 入力信号を出力信号に変換する
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y
# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)


# ソフトマックス関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
# a = np.array([0.3, 2.9, 4.0])
# print(softmax(a))

# a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a)))

# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(a)
# print(y)
# print(np.sum(y))





