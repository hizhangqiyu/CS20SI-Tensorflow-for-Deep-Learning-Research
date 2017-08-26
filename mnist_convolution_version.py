from __future__ import division
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

WORK_DIR = os.path.abspath(os.curdir)
LEARNING_RATE = 0.01
BATCH_SIZE = 128
N_EPOCHS = 20000

class MnistConvnet:
    def __init__(self, learning_rate, batch_size, n_epochs, data_path="./data/mnist"):
        self.learning_rate = 0.01
        self.batch_size = 128
        self.n_epochs = 25
        self.data = input_data.read_data_sets(os.path.join(WORK_DIR, data_path), one_hot=True)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholder(self, batch_size):
        X = tf.placeholder(tf.float32, [batch_size, 784], name="input")
        Y = tf.placeholder(tf.float32, [batch_size, 10], name="lables")
        return X, Y 

    # input shape defaults to NHWC(batch_size, height, width, channel)
    # kernel shape defaults to (height, width, input_channel, output_channel)
    def _create_filter_layer(self, layer_name, input_layer, kernel_shape, bias_shape, stride_shape=[1, 1, 1, 1], padding='SAME'):
        with tf.variable_scope(layer_name) as scope:
            k = tf.get_variable('kernel', kernel_shape)
            b = tf.get_variable('biases', bias_shape, initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(input_layer, k, strides=stride_shape, padding=padding)
            return tf.nn.relu(conv + b, name=scope.name)

    def _create_pool_layer(self, layer_name, input_layer, ksize, stride_shape=[1, 2, 2, 1], padding='SAME'):
        with tf.variable_scope(layer_name) as scope:
            return tf.nn.max_pool(input_layer, ksize=ksize, strides=stride_shape, padding=padding)

    def _create_fc_layer(self, layer_name, input_layer, weight_shape, bias_shape):
        with tf.variable_scope(layer_name) as scope:
            w = tf.get_variable('weights', weight_shape, initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', bias_shape, initializer=tf.constant_initializer(0.0))

            return tf.nn.relu(tf.matmul(input_layer, w) + b, name='relu')

    def _create_dropout(self, input_layer):
        keep_prob = tf.placeholder(tf.float32)
        dropout = tf.nn.dropout(input_layer, keep_prob)
        return keep_prob, dropout

    def _create_softmax(self, input_layer, layer_name='softmax'):
        with tf.variable_scope(layer_name) as scope:
            w = tf.get_variable('weights', [1024, 10], initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [10], initializer=tf.random_normal_initializer())
            return tf.matmul(input_layer, w) + b

    def _create_loss(self, logits, labels):
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(entropy)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("learning_rate", self.learning_rate)
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()


    def build_graph(self):
        with tf.name_scope('data'):
            self.X, self.Y = self._create_placeholder(self.batch_size)
        
        with tf.name_scope('conv1'):
            conv1 = self._create_filter_layer(layer_name='conv1',
                                    input_layer=tf.reshape(self.X, shape=[-1, 28, 28, 1]),
                                    kernel_shape=[5, 5, 1, 32],
                                    bias_shape=[32])
        
        with tf.name_scope('pool1'):
            pool1 = self._create_pool_layer(layer_name='pool1', input_layer=conv1, ksize=[1, 2, 2, 1])

        with tf.name_scope('conv2'):
            conv2 = self._create_filter_layer(layer_name='conv2',
                                    input_layer=pool1,
                                    kernel_shape=[5, 5, 32, 64],
                                    bias_shape=[64])

        with tf.name_scope('pool2'):
            pool2 = self._create_pool_layer(layer_name='pool2', input_layer=conv2, ksize=[1, 2, 2, 1])
        
        with tf.name_scope('fc1'):
            #fc1 = self._create_fc_layer(layer_name='fc1', input_layer=tf.reshape(pool3, [-1, 7*7*64]), weight_shape=[7*7*64,1024], bias_shape=[1024])
            fc1 = self._create_fc_layer('fc1', tf.reshape(pool3, [-1, 7*7*64]), [7*7*64,1024], [1024])

        with tf.name_scope('dropout'):
            dp = self._create_dropout('dropout', fc1)

        with tf.name_scope('fc2')
            fc2 = self._create_fc_layer(layer_name='fc2', input_layer=dp, weight_shape=[1024,10], bias_shape=[10])

        self.logits = self._create_softmax(input_layer=fc)
        self.loss = self._create_loss(logits=self.logits, labels=self.Y)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train_model(self):
        saver = tf.train.Saver() # defaults to saving all variables

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            train_writer = tf.summary.FileWriter("./graph/convnet_mnist", sess.graph)
            initial_step = model.global_step.eval()

            n_batches = int(self.data.train.num_examples/self.batch_size)
            for i in range(initial_step, initial_step + self.n_epochs):
                total_correct_preds = 0
                for _ in range(n_batches):
                    batch = self.data.train.next_batch(self.batch_size)

                    correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(batch[1], 1))
                    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

                    _, loss_r, correct_preds_r = sess.run([self.optimizer, self.loss, accuracy], feed_dict={self.X: batch[0], self.Y:batch[1]})
                    total_correct_preds += correct_preds_r
                writer.add_summary(summary, global_step=i)
                print("epoch %r loss_r= %r total_correct_preds=%r accuracy=%r" % (i, loss_r, total_correct_preds, self.data.test.num_examples))
            

            # generate data for tensorboard.
            final_embed_matrix = sess.run(self.embed_matrix)
            embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
            sess.run(embedding_var.initializer)
            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter('./graph/mnist')
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = os.path.join(WORK_DIR,'./graph/mnist/image_500.tsv')
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, './graph/mnist/mnist_convnet.ckpt', 1)

            summary_writer.close()
            train_writer.close()


if __name__ == "__main__":
    model = MnistConvnet(LEARNING_RATE, BATCH_SIZE, N_EPOCHS)
    model.build_graph()
    model.train_model()
