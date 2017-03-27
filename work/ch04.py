import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


# 二条和誤差
def mean_squared_error(y, t):
    e = 0.5 * np.sum((y-t)**2)
    return e


# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7 # 0.0000001
    return -np.sum(t * np.log(y + delta))
# バッチ対応版クロスエントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        # t = t.reshape(28, 28)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # return -np.sum(t * np.log(y)) / batch_size
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


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



# 4.2.3 ミニバッチ学習をやる
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

# 訓練データからランダムに10を抜き出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch.shape)


# 数値微分(numerical differentiation)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

