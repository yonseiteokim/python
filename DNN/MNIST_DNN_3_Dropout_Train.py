"""
4-Layer Neural Networks
-Initialization: Xavier Initialization
-Dropout (Rate: 0.7) Used
-Performance: 98.48% ~ 98.59%
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')
dropout_rate = tf.placeholder(tf.float32)

# 4-Layer Neural Networks
W1 = tf.get_variable("W1", shape=[784, 500], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[500, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([500]), name='Bias1')
b2 = tf.Variable(tf.zeros([256]), name='Bias2')
b3 = tf.Variable(tf.zeros([128]), name='Bias3')
b4 = tf.Variable(tf.zeros([10]), name='Bias4')

# ReLU & Dropout
with tf.name_scope('Layer1'):
    _L2 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
    L2 = tf.nn.dropout(_L2, dropout_rate)
with tf.name_scope('Layer2'):
    _L3 = tf.nn.relu(tf.add(tf.matmul(L2, W2),b2))
    L3 = tf.nn.dropout(_L3, dropout_rate)
with tf.name_scope('Layer3'):
    _L4 = tf.nn.relu(tf.add(tf.matmul(L3, W3),b3))
    L4 = tf.nn.dropout(_L4, dropout_rate)
with tf.name_scope('Layer4'):
    hypothesis = tf.add(tf.matmul(L4, W4), b4)

with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
    tf.summary.scalar('Cost', cost)

with tf.name_scope('Train'):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

ckpt_dir = "./ckpt_dir_dnn_3_Dropout"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()
start_time = time.time()

init = tf.global_variables_initializer()

training_epoch = 50
display_step = 1
batch_size = 100

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/DNN3_Dropout/", session.graph)

    session.run(init)

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)  # restore all variables

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, summary = session.run([optimizer, merged], feed_dict={X: batch_xs, Y: batch_ys, dropout_rate:0.7})
            writer.add_summary(summary, epoch)

        # show logs per epoch step
        avg_cost += session.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate:0.7}) / total_batch
        if epoch % display_step == 0:  # Softmax
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        global_step.assign(epoch).eval()  # set and update(eval) global_step with index, i
        saver.save(session, ckpt_dir + "/model.ckpt", global_step=global_step)

    writer.close()
    print ("Optimization Finished!")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

    # Test Phase
    batch_size_test = 100
    total_batch_test = int(mnist.test.num_examples / batch_size_test)
    corrects = 0
    nums = 0
    for i in range(total_batch_test):
        testSet = mnist.test.next_batch(batch_size_test)
        corrects += accuracy.eval({X: testSet[0], Y: testSet[1], dropout_rate:1.0})
        nums += testSet[0].shape[0]
    print("Correct Ones: %g" % (corrects))
    print("Total Ones: %g" % (nums))
    print("Accuracy: %g" % (corrects / nums))


