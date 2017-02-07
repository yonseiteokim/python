from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')
W1 = tf.get_variable("W1", shape=[784, 500], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[500, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([500]), name='Bias1')
b2 = tf.Variable(tf.zeros([256]), name='Bias2')
b3 = tf.Variable(tf.zeros([128]), name='Bias3')
b4 = tf.Variable(tf.zeros([10]), name='Bias4')

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
hypothesis = tf.add(tf.matmul(L3, W4), b4)

ckpt_dir = "./ckpt_dir_dnn_2_Xavier"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
global_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

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
        corrects += accuracy.eval({X: testSet[0], Y: testSet[1]})
        nums += testSet[0].shape[0]
    print("Correct Ones: %g" % (corrects))
    print("Total Ones: %g" % (nums))
    print("Accuracy: %g" % (corrects / nums))

