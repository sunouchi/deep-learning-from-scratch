import numpy as np


# 二条和誤差
def mean_squared_error(y, t):
    e = 0.5 * np.sum((y-t)**2)
    return e


# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7 # これなんだろう？微小な値らしいけど
    return -np.sum(t * np.log(y + delta))


# t=教師データ
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# y=ニューラルネットワークの出力（ソフトマックス関数。確率）
# 正解は「2」
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))

# 正解は「7」
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))


# TODO: 4.2.3 ミニバッチ学習をやる
