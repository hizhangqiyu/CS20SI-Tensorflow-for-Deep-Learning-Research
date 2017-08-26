from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("./data/mnist", one_hot=True)

learning_rate = 0.01
batch_size = 128
n_epochs = 25


# input shape defaults to NHWC(batch_size, height, width, channel)
# kernel shape defaults to (height, width, input_channel, output_channel)
def generate_filter_layer(layer_name, input_layer, kernel_shape, bias_shape, stride_shape=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
        k = tf.get_variable('kernel', kernel_shape)
        b = tf.get_variable('biases', bias_shape, initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(input_layer, k, strides=stride_shape, padding=padding)
        return tf.nn.relu(conv + b, name=scope.name)

def generate_pool_layer(layer_name, input_layer, ksize, stride_shape=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
        return tf.nn.max_pool(input_layer, ksize=ksize, strides=stride_shape, padding=padding)

def generate_fc_layer(layer_name, input_layer, input_features):
    with tf.variable_scope(layer_name) as scope:
        w = tf.get_variable('weights', [input_features, 1024],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [1024],
                            initializer=tf.constant_initializer(0.0))

        # reshape pool2 to 2 dimensional
        input_layer = tf.reshape(input_layer, [-1, input_features])
        return tf.nn.relu(tf.matmul(input_layer, w) + b, name='relu')

with tf.name_scope("data"):
    X = tf.placeholder(tf.float32, [batch_size, 784], name="input")
    Y = tf.placeholder(tf.float32, [batch_size, 10], name="lables")

conv1 = generate_filter_layer(layer_name='conv1',
                            input_layer=tf.reshape(X, shape=[-1, 28, 28, 1]), # -1: dynamically decided 
                            kernel_shape=[5, 5, 1, 32],
                            bias_shape=[32])
pool1 = generate_pool_layer(layer_name='pool1', input_layer=conv1, ksize=[1, 2, 2, 1])

conv2 = generate_filter_layer(layer_name='conv2',
                            input_layer=pool1,
                            kernel_shape=[5, 5, 32, 64],
                            bias_shape=[64])
pool2 = generate_pool_layer(layer_name='pool2', input_layer=conv2, ksize=[1, 2, 2, 1])
fc = generate_fc_layer(layer_name='fc', input_layer=pool2, input_features = 7 * 7 * 64)

with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('weights', [1024, 10],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [10],
                        initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    print(fc.eval())
    print(logits.eval())
    sess.run(init)
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, loss_r = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})

    print("n_batches=%r test_num=%r loss=%r" % (n_batches, MNIST.train.num_examples, loss_r))
    
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy, feed_dict={X: X_batch, Y:Y_batch})
    print("total_correct_preds=%r Accuracy=%r" % (total_correct_preds, total_correct_preds/MNIST.test.num_examples))
    