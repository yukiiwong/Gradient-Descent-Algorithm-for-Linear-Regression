import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
#获取当前脚本工作路径
projectDir = os.path.dirname(os.path.realpath(__file__))

# TRAFFIC
# df.dropna(inplace=True)
# learningRate = 0.001
# numEpochs = 5000

# BURGLARIESí
df = genfromtxt("data.csv", delimiter=",") #筛选出 Burglary 和 Murder and\nnonnegligent\nmanslaughter 两列数据
learningRate = 0.0001
numEpochs = 1000

x = tf.placeholder(tf.float32, shape=[None, ])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.zeros(shape=[1]))
b = tf.Variable(tf.zeros(shape=[1]))

y_pred = tf.add(tf.multiply(x, w), b)

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_pred)))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    input, output = df[:, 0], df[:, 1]
    for epoch in range(numEpochs):
        _, sampleLoss, pred = session.run([optimiser, loss, y_pred], {x: input, y: output})
        print("EPOCH:", epoch + 1)
        print("LOSS: ", sampleLoss)
        print("")

    y_preds = session.run(y_pred, {x: input, y: output})

for i in range(df.shape[0]):
    x, y = df[i, 0], df[i, 1]
    plt.scatter(x, y, c='b')

plt.plot(input, y_preds, c='r')

plt.show()
