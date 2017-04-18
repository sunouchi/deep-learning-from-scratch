import numpy as np

# 5.4.1 乗算レイヤの実装
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        # 上流から返ってきた微分(dout)に対して、
        # 順伝播のひっくり返した値を乗算して下流に流す
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# apple = 100
# apple_num = 2
# tax = 1.1

# # layer
# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()

# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_tax_layer.forward(apple_price, tax)
# print(price)

# # backward
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple, dapple_num, dtax)


# 5.4.2 加算レイヤの実装
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# apple = 100
# apple_num = 2
# orange = 150
# orange_num = 3
# tax = 1.1

# # layer
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# add_apple_orange_layer = AddLayer()
# mul_tax_layer = MulLayer()

# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)
# orange_price = mul_orange_layer.forward(orange, orange_num)
# all_price = add_apple_orange_layer.forward(apple_price, orange_price)
# price = mul_tax_layer.forward(all_price, tax)

# # backward
# dprice = 1
# dall_price, dtax = mul_tax_layer.backward(dprice)
# dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print(price)
# print(dapple_num, dapple, dorange, dorange_num, dtax)


# 5.5.1
# ReLU: Rectified Linear Unit
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# print(x)
# mask = (x <= 0)
# print(mask)

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# 5.6.1 Affineレイヤ
# X_dot_W = np.array([[0,0,0], [10,10,10]])
# B = np.array([1, 2, 3])
# # print(X_dot_W)
# # print(X_dot_W + B)
# dY = np.array([[1,2,3,], [4,5,6]])
# print(dY)
# dB = np.sum(dY, axis=0)
# print(dB)

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

    def backward(self, dout)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None    # softmaxの出力
        self.t = None    # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx






