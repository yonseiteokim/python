from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')
dropout_rate_cnn = tf.placeholder(tf.float32)
dropout_rate_fcc = tf.placeholder(tf.float32)

# xavier_initializer / variance_scaling_initializer
W1 = tf.get_variable("W1", shape=[3,3,1,32], initializer=tf.contrib.layers.variance_scaling_initializer())
W2 = tf.get_variable("W2", shape=[3,3,32,64], initializer=tf.contrib.layers.variance_scaling_initializer())
W3 = tf.get_variable("W3", shape=[3,3,64,128], initializer=tf.contrib.layers.variance_scaling_initializer())
W4 = tf.get_variable("W4", shape=[2048,625], initializer=tf.contrib.layers.variance_scaling_initializer())
W5 = tf.get_variable("W5", shape=[625,10], initializer=tf.contrib.layers.variance_scaling_initializer())

#W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#W4 = tf.Variable(tf.random_normal([2048, 625], stddev=0.01))
#W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

# Networks
X_image = tf.reshape(X, [-1, 28, 28, 1], name='X-input-reshape')
l1a = tf.nn.relu(tf.nn.conv2d(X_image, W1, strides=[1, 1, 1, 1], padding='SAME'))
l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, dropout_rate_cnn)
l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME'))
l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, dropout_rate_cnn)
l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 1, 1, 1], padding='SAME'))
l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, dropout_rate_cnn)
l4 = tf.nn.relu(tf.matmul(l3, W4))
l4 = tf.nn.dropout(l4, dropout_rate_fcc)
hypothesis = tf.matmul(l4, W5)

init = tf.global_variables_initializer()

"""
ckpt_dir = 여기에 저장된 모델의 폴더를 명시하라
"""
ckpt_dir = "./ckpt_dir_cnn_mnist_he_RMSProp"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)  # restore all variables
    start = global_step.eval()

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