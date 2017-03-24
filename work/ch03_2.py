# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


'''
MNISTデータを使う
'''
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# img = x_train[0]
# label = t_train[0]
# print(label)

# print(img)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)


# MNISTテストデータの取得
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# x, t = get_data()
# network = init_network()
# print(x.shape)
# print(x)
# print(t.shape)
# print(t)

# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1

# print('Accuracy:' + str(float(accuracy_cnt / len(x))))


x, t = get_data()
network = init_network()
# W1, W2, W3 = network['W1'], network['W2'], network['W3']

# print(x.shape)
# print(x[0].shape)
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # print(x_batch)
    # print(y_batch)
    p = np.argmax(y_batch, axis=1)
    # accuracy_cnt += 1 if p[i] == 
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print(accuracy_cnt)



