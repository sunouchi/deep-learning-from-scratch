import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient
from two_layer_net import TwoLayerNet


# 二条和誤差
def mean_squared_error(y, t):
    e = 0.5 * np.sum((y-t)**2)
    return e


# # 交差エントロピー誤差
# def cross_entropy_error(y, t):
#     delta = 1e-7 # 0.0000001
#     return -np.sum(t * np.log(y + delta))
# # バッチ対応版クロスエントロピー誤差
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         # t = t.reshape(28, 28)
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     # return -np.sum(t * np.log(y)) / batch_size
#     return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


# # t=教師データ
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# # y=ニューラルネットワークの出力（ソフトマックス関数。確率）
# # 正解は「2」
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# # print(mean_squared_error(np.array(y), np.array(t)))
# print(cross_entropy_error(np.array(y), np.array(t)))

# # 正解は「7」
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# # print(mean_squared_error(np.array(y), np.array(t)))
# print(cross_entropy_error(np.array(y), np.array(t)))



# # 4.2.3 ミニバッチ学習をやる
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape)
# print(t_train.shape)
# # 訓練データからランダムに10を抜き出す
# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# print(batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]
# print(x_batch.shape)
# print(t_batch.shape)


# 数値微分(numerical differentiation)
def numerical_diff(f, x):
    h = 1e-4
    # return (f(x+h) - f(x)) / (2*h)
    return (f(x+h) - f(x-h)) / (2*h)

# 数値微分の例
def function_1(x):
    return 0.01*x**2 + 0.1*x

# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.show()


# 偏微分
def function_2(x):
    return x[0]**2 + x[1]**2 #np.sum(x**2)と同じ意味
    # return np.sum(x**2)

# print(numerical_diff(function_2, [3.0, 4.0]))
# x = np.array([3,4])
# print(numerical_diff(function_2(x)), 3)

def function_tmp1(x0):
    return x0**2 + 4.0**2

def function_tmp2(x1):
    return 3.0**2 + x1**2

# print(numerical_diff(function_tmp1, 3.0))
# print(numerical_diff(function_tmp2, 4.0))


# # 数値勾配？
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     for idx in range(x.size):
#         tmp_val = x[idx]
#         # f(x+h)を求める
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         # f(x-h)を求める
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2) / (2*h)
#         x[idx] = tmp_val
#     return grad
# # print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# # print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# # print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# 勾配降下法 p107
# xから学習率を差し引く。step_numの回数だけ更新を繰り返す
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# f(x0, x1) = x0**2 + x1**2 の最小値を勾配法で求めよ
def function_2(x):
    return x[0]**2 + x[1]**2 
# init_x = np.array([-3.0, 4.0])
# print(gradient_descent(function_2, init_x=init_x, lr=0.1))



# ニューラルネットワークにおける勾配を求める
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
# net = simpleNet()
# print(net.W)
# x = np.array([0.6, 0.9])
# p = net.predict(x)
# print(p)
# print(np.argmax(p))
# t = np.array([0, 0, 1])
# print(net.loss(x, t))
# # def f(W):
# #     return net.loss(x, t)
# f = lambda w: net.loss(x, t)
# dW = numerical_gradient(f, net.W)
# print(dW)



# 4.5 学習アルゴリズムの実装
# 2層のニューラルネットを実装する
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
#         # 重みの初期化
#         self.params = {}
#         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)

#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']

#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2) #sigmoid()とsoftmaxの違いってなんだっけ？最後にやるのがsoftmaxというぐらいは覚えているけども。原理的に何をやっているのか知りたい
#         return y

#     # x:入力データ, y:教師データ
#     def loss(self, x, t):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)

#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy

#     # x:入力データ, y:教師データ
#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#         return grads

# net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# # print(net.params['W1'].shape)
# # print(net.params['b1'].shape)
# # print(net.params['W2'].shape)
# # print(net.params['b2'].shape)

# x = np.random.rand(100, 784)
# t = np.random.rand(100, 10)
# grads = net.numerical_gradient(x, t)
# # y = net.predict(x)
# # print(grads)

# print(net.params['W1'].shape)
# print(net.params['b1'].shape)
# print(net.params['W2'].shape)
# print(net.params['b2'].shape)



# 4.5.2 ミニバッチ学習の実装
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size = 784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #高速版！

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | ' + str(train_acc) + ', ' + str(test_acc))


