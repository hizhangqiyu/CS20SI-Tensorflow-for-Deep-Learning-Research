from __future__ import division
import os
import zipfile
import collections
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


# global variables
data_name = "data/text/text8.zip"
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
EPOCH = 10000
SKIP_WINDOW = 1 
NUM_SKIPS = 2
NUM_SAMPLED = 64
LEARNNING_RATE = 1.0
VALID_SIZE = 16
VALID_WINDOW = 100

class SkipGramModel:
    ''' Build the graph for word2vec model'''
    def __init__(self, vocab_size, batch_size, embed_size, epoch, skip_window, num_skips, num_sampled, learning_rate=1.0):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.epoch = epoch
        self.skip_window = skip_window # the number of context words from left/right of input word
        self.num_skips = num_skips    # the number of labels used for one input
        self.num_sampled = num_sampled
        self.index = 0
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _read_data(self):
        with zipfile.ZipFile(data_name) as zf:
            self.data = tf.compat.as_str(zf.read(zf.namelist()[0])).split()

    def _build_dataset(self):
        count = [['UNK', -1]]
        count.extend(collections.Counter(self.data).most_common(self.vocab_size - 1))
        vocabulary = dict()

        for word, _ in count:
            vocabulary[word] = len(vocabulary) # index

        self.indices = list()
        unk_count = 0
        for word in self.data:
            if word in vocabulary:
                index = vocabulary[word]
            else:
                index = 0
                unk_count += 1
            self.indices.append(index)

        with open('./graph/word2vec/vocab_500.tsv', "w") as f:
            index = 0
            for word, _ in count:
                vocabulary[word] = index
                if index < 500:
                    f.write(word + "\n")
                index += 1

        count[0][1] = unk_count
        self.reversed_vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))

    def _generate_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= (2 * self.skip_window)
        self.batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        self.labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1
        buf = collections.deque(maxlen=span)

        # round back
        if self.index + span > len(self.indices):
            self.index = 0

        buf.extend(self.indices[self.index:self.index + span])
        self.index += span

        for i in range(self.batch_size // self.num_skips): # for each span
            target = self.skip_window # center words as target
            targets_to_avoid = [self.skip_window]
            
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                self.batch[i * self.num_skips + j] = buf[self.skip_window]
                self.labels[i * self.num_skips + j, 0] = buf[target]
            
            if self.index == len(self.indices):
                buf[:] = self.indices[:span]
                self.index = span
            else:
                buf.append(self.indices[self.index])
                self.index += 1

        self.index = (self.index + len(self.indices) - span) % len(self.indices)

    def _create_placeholder(self):
        """ define placeholder for input and output """
        with tf.name_scope("data"):
            self.train_inputs = tf.placeholder(tf.int32, [self.batch_size])
            self.train_labels = tf.placeholder(tf.int32,[self.batch_size, 1])

    def _create_embedding(self):
        ''' define the weight '''
        with tf.name_scope("embedding"):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))

    def _create_loss(self):
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.train_inputs)

            # define the loss function
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0 / self.embed_size ** 0.5))
            nce_bias = tf.Variable(tf.zeros([self.vocab_size]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                    biases=nce_bias, 
                                    labels=self.train_labels, 
                                    inputs=embed,
                                    num_sampled=self.num_sampled,
                                    num_classes=self.vocab_size))
    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_validation(self, valid_size, valid_window):
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embed_matrix), 1, keep_dims=True))
        normalized_embeddings = self.embed_matrix / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    def build_graph(self, valid_size=0, valid_window=0):
        self._read_data()
        self._build_dataset()
        self._create_placeholder()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        if valid_size > 0 and valid_window > 0:
            self._create_validation(valid_size=valid_size, valid_window=valid_window)

    def train_word2vec(self, validation=False):
        saver = tf.train.Saver() # defaults to saving all variables

        initial_step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            ckpt = tf.train.get_checkpoint_state(os.path.dirname("./graph/word2vec/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            writer = tf.summary.FileWriter("./graph/word2vec", sess.graph)
            initial_step = model.global_step.eval()

            average_loss = 0.0
            for step in range(initial_step, initial_step + self.epoch):
                self._generate_batch()
                feed_dict = {self.train_inputs:self.batch, self.train_labels:self.labels}
                _, batch_loss, summary = sess.run([self.optimizer, self.loss, self.summary_op], feed_dict)
                
                writer.add_summary(summary, global_step=step)

                average_loss += batch_loss


                if (step + 1) % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000  
                    print("average loss=%r" % average_loss)
                    average_loss = 0
                    saver.save(sess, "./graph/word2vec/checkpoint", step)    

                if validation:
                    if step % 4000 == 0:
                        sim = self.similarity.eval()
                        for i in range(self.valid_size):
                            valid_word = self.reversed_vocabulary[self.valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = "Nearest to %s:" % valid_word
        
                            for k in range(top_k):
                                close_word = self.reversed_vocabulary[nearest[k]]
                                log_str = "%s %s," % (log_str, close_word)
                            print(log_str)

            final_embed_matrix = sess.run(self.embed_matrix)

            embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
            sess.run(embedding_var.initializer)
            
            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter('./graph/word2vec')

            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            #embedding.metadata_path = './graph/word2vec/vocab_500.tsv'
            embedding.metadata_path = './vocab_500.tsv'

            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, './graph/word2vec/skip-gram.ckpt', 1)

            summary_writer.close()
            writer.close()


if __name__ == "__main__":
    model = SkipGramModel(VOCAB_SIZE, BATCH_SIZE, EMBED_SIZE, EPOCH, SKIP_WINDOW, NUM_SKIPS, NUM_SAMPLED)
    #model.build_graph(valid_size=VALID_SIZE, valid_window=VALID_WINDOW)
    model.build_graph()
    #model.train_word2vec(validation=True)
    model.train_word2vec()


