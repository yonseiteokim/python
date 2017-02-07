# Use 'Deep' neural network to solve XOR problem.

import numpy as np
import tensorflow as tf

xy = np.loadtxt('D:\\ppy\\07train.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

# Deep network configuration.: Use more layers.
W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight3')
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight4')
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight5')
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight6')
W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight7')
W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight8')
W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight9')
W10 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight10')
W11 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight11')

b1 = tf.Variable(tf.zeros([5]), name="bias1")
b2 = tf.Variable(tf.zeros([5]), name="bias2")
b3 = tf.Variable(tf.zeros([5]), name="bias3")
b4 = tf.Variable(tf.zeros([5]), name="bias4")
b5 = tf.Variable(tf.zeros([5]), name="bias5")
b6 = tf.Variable(tf.zeros([5]), name="bias6")
b7 = tf.Variable(tf.zeros([5]), name="bias7")
b8 = tf.Variable(tf.zeros([5]), name="bias8")
b9 = tf.Variable(tf.zeros([5]), name="bias9")
b10 = tf.Variable(tf.zeros([5]), name="bias10")
b11 = tf.Variable(tf.zeros([1]), name="bias11")

# Hypotheses
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("layer4") as scope:
    L4 = tf.sigmoid(tf.matmul(L3, W3) + b3)

with tf.name_scope("layer5") as scope:
    L5 = tf.sigmoid(tf.matmul(L4, W4) + b4)

with tf.name_scope("layer6") as scope:
    L6 = tf.sigmoid(tf.matmul(L5, W5) + b5)

with tf.name_scope("layer7") as scope:
    L7 = tf.sigmoid(tf.matmul(L6, W6) + b6)

with tf.name_scope("layer8") as scope:
    L8 = tf.sigmoid(tf.matmul(L7, W7) + b7)

with tf.name_scope("layer9") as scope:
    L9 = tf.sigmoid(tf.matmul(L8, W8) + b8)

with tf.name_scope("layer10") as scope:
    L10 = tf.sigmoid(tf.matmul(L9, W9) + b9)

with tf.name_scope("layer11") as scope:
    L11 = tf.sigmoid(tf.matmul(L10, W10) + b10)

with tf.name_scope("last") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L11, W11) + b11)

# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

# Minimize cost.
a = tf.Variable(0.1)
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # tensorboard merge
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

    sess.run(init)

    # Run graph.
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))