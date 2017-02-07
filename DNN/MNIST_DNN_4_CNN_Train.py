"""
5-Layer Convolutional Neural Networks

-Initialization: Xavier / Random / He's
-Dropout (CNNRate: 0.7 / FCRate: 0.5) Used
-Optimization: Adam / RMSPropOptimizer(0.9)

-Speed: GTX 750, About 660 seconds
-Performance: About 99.4%
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')
dropout_rate_cnn = tf.placeholder(tf.float32)
dropout_rate_fcc = tf.placeholder(tf.float32)

# 4-Layer Neural Networks
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
W4 = tf.Variable(tf.random_normal([2048, 625], stddev=0.01))
W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

# tf.contrib.layers.xavier_initializer()
#W1 = tf.get_variable("W1", shape=[3,3,1,32], initializer=tf.contrib.layers.variance_scaling_initializer())
#W2 = tf.get_variable("W2", shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer())
#W3 = tf.get_variable("W3", shape=[3,3,64,128], initializer=tf.contrib.layers.variance_scaling_initializer())
#W4 = tf.get_variable("W4", shape=[2048,625], initializer=tf.contrib.layers.variance_scaling_initializer())
#W5 = tf.get_variable("W5", shape=[625,10], initializer=tf.contrib.layers.variance_scaling_initializer())

# Networks
X_image = tf.reshape(X, [-1, 28, 28, 1], name='X-input-reshape')

with tf.name_scope('Layer1'):
    l1a = tf.nn.relu(tf.nn.conv2d(X_image, W1, strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a)  # l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l1)  # l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, dropout_rate_cnn)

with tf.name_scope('Layer2'):
    l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a)  # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l2)  # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, dropout_rate_cnn)

with tf.name_scope('Layer3'):
    l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a)  # l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l3)  # l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
    print(l3)  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, dropout_rate_cnn)

with tf.name_scope('Layer4'):
    l4 = tf.nn.relu(tf.matmul(l3, W4))
    l4 = tf.nn.dropout(l4, dropout_rate_fcc)

with tf.name_scope('Layer5'):
    hypothesis = tf.matmul(l4, W5)


with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
    tf.summary.scalar('Cost', cost)

with tf.name_scope('Train'):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

init = tf.global_variables_initializer()

training_epoch = 50
display_step = 1
batch_size = 100

ckpt_dir = "./ckpt_dir_cnn_mnist_he_RMSProp"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()
start_time = time.time()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/DNN4_CNN/", session.graph)

    session.run(init)
    avg_cost = 0.

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)  # restore all variables
    #start = global_step.eval()

    for epoch in range(training_epoch):

        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, summary = session.run([optimizer, merged], feed_dict={X: batch_xs, Y: batch_ys, dropout_rate_cnn:0.7, dropout_rate_fcc:0.5})
            writer.add_summary(summary, epoch)

        # show logs per epoch step
        avg_cost += session.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate_cnn:0.7, dropout_rate_fcc:0.5}) / total_batch
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
        corrects += accuracy.eval({X: testSet[0], Y: testSet[1], dropout_rate_cnn: 1.0, dropout_rate_fcc: 1.0})
        nums += testSet[0].shape[0]
    print("Correct Ones: %g" % (corrects))
    print("Total Ones: %g" % (nums))
    print("Accuracy: %g" % (corrects / nums))




