from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
X = tf.placeholder(tf.float32, [None, 784], name = 'X-input')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y-input')
W1 = tf.Variable(tf.random_normal([784, 256]), name = 'Weight1')
W2 = tf.Variable(tf.random_normal([256, 256]), name = 'Weight2')
W3 = tf.Variable(tf.random_normal([256, 10]), name = 'Weight3')
b1 = tf.Variable(tf.random_normal([256]), name = 'Bias1')
b2 = tf.Variable(tf.random_normal([256]), name = 'Bias2')
b3 = tf.Variable(tf.random_normal([10]), name = 'Bias3')

# ReLU Function
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
hypothesis = tf.add(tf.matmul(L2, W3), b3)
init = tf.global_variables_initializer()

ckpt_dir = "./ckpt_dir_dnn_1_ReLU"
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
        corrects += accuracy.eval({X: testSet[0], Y: testSet[1]})
        nums += testSet[0].shape[0]
    print("Correct Ones: %g" % (corrects))
    print("Total Ones: %g" % (nums))
    print("Accuracy: %g" % (corrects / nums))
