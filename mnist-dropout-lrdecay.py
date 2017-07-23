import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import math

mnist = mnist_data.read_data_sets("data", one_hot = True, reshape = False, validation_size = 0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

L = 200
M = 100
N = 30

w1 = tf.Variable(tf.truncated_normal([784, L], stddev = 0.1))
b1 = tf.Variable(tf.ones([L]) / 10)
w2 = tf.Variable(tf.truncated_normal([L, M], stddev = 0.1))
b2 = tf.Variable(tf.ones([M]) / 10)
w3 = tf.Variable(tf.truncated_normal([M, N], stddev = 0.1))
b3 = tf.Variable(tf.ones([N]) / 10)
w4 = tf.Variable(tf.truncated_normal([N, 10], stddev = 0.1))
b4 = tf.Variable(tf.ones([10]))

XX = tf.reshape(X, [-1, 28 * 28])
Y1 = tf.nn.relu(tf.matmul(XX, w1) + b1)
y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(y1d, w2) + b2)
y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(y2d, w3) + b3)
y3d = tf.nn.dropout(Y3, pkeep)

Ylogits = tf.matmul(y3d, w4) + b4
Y = tf.nn.softmax(Ylogits)

xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)   #Y_ is actual
xentropy = tf.reduce_mean(xentropy) * 100

cp = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
acc = tf.reduce_mean(tf.cast(cp, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(xentropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, upd_test, upd_train):
    batch_x, batch_y = mnist.train.next_batch(100)

    maxlr = 0.003
    minlr = 0.0001
    speed = 2000.0
    learning_rate = minlr + (maxlr - minlr) * math.exp(-i / speed)

    if upd_train:
        a, c = sess.run([acc, xentropy], {X : batch_x, Y_ : batch_y, pkeep : 0.75})
        print(str(i) + " " + "acc = " + str(a) + " learning_rate = " + str(learning_rate) + "loss = " + str(c))

    if upd_test:
        a, c = sess.run([acc, xentropy], {X: mnist.test.images, Y_ : mnist.test.labels, pkeep: 1.0})
        print(str(i) + "test_acc : " + str(a) + " test_loss = " + str(c))

    sess.run(train_step, {X: batch_x, Y_ : batch_y, pkeep : 0.75, lr: learning_rate})

for i in range(5000):
    training_step(i, i % 100 == 0, i % 20 == 0)
















