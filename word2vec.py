from __future__ import division

import tensorflow as tf
import zipfile
import collections
import random
import numpy as np

# MACRO
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128
EPOCH = 100001
SKIP_WINDOW = 1 # the number of context words from left/right of input word
NUM_SKIPS = 2    # the number of labels used for one input
NUM_SAMPLED = 64

# data
data_name = "data/text/text8.zip"

def read_data():
    with zipfile.ZipFile(data_name)    as zf:
        data = tf.compat.as_str(zf.read(zf.namelist()[0])).split()
    return data

words = read_data()
print("data size=%r" % len(words))


def build_dataset(words):
    count = [['UNK', -1]]
    #temp = collections.Counter(words))
    count.extend(collections.Counter(words).most_common(VOCAB_SIZE - 1))
    vocabulary = dict()

    for word, _ in count:
        vocabulary[word] = len(vocabulary) # index

    indices = list()
    unk_count = 0
    for word in words:
        if word in vocabulary:
            index = vocabulary[word]
        else:
            index = 0
            unk_count += 1
        indices.append(index)

    count[0][1] = unk_count
    reversed_vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))
    return indices, count, vocabulary, reversed_vocabulary

indices, count, vocabulary, reversed_vocabulary = build_dataset(words)

del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', indices[:10], [reversed_vocabulary[i] for i in indices[:10]])

index = 0
def generate_batch():
    assert BATCH_SIZE % NUM_SKIPS == 0
    assert NUM_SKIPS <=    (2 * SKIP_WINDOW)
    batch = np.ndarray(shape=(BATCH_SIZE), dtype=np.int32)
    labels = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)
    span = 2 * SKIP_WINDOW + 1
    buf = collections.deque(maxlen=span)

    global index
    # round back
    if index + span > len(indices):
        index = 0

    buf.extend(indices[index:index + span])
    index += span

    for i in range(BATCH_SIZE // NUM_SKIPS): # for each span
        target = SKIP_WINDOW # center words as target
        targets_to_avoid = [SKIP_WINDOW]
        
        for j in range(NUM_SKIPS):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * NUM_SKIPS + j] = buf[SKIP_WINDOW]
            labels[i * NUM_SKIPS + j, 0] = buf[target]
        
        if index == len(indices):
            buf[:] = indices[:span]
            index = span
        else:
            buf.append(indices[index])
            index += 1

    index = (index + len(indices) - span) % len(indices)
    return batch, labels


# skip-gram model
# define placeholder for input and output
train_inputs = tf.placeholder(tf.int32, [BATCH_SIZE])
train_labels = tf.placeholder(tf.int32,[BATCH_SIZE, 1])

# define the weight
embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0))

# inference
embed = tf.nn.embedding_lookup(embed_matrix, train_inputs)

# define the loss function
nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / EMBED_SIZE ** 0.5))
nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                    biases=nce_bias, 
                                    labels=train_labels, 
                                    inputs=embed,
                                    num_sampled=NUM_SAMPLED,
                                    num_classes=VOCAB_SIZE))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0.0
    for step in range(EPOCH):
        batch_inputs, batch_labels = generate_batch()
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
        _, batch_loss = sess.run([optimizer, loss], feed_dict)
        average_loss += batch_loss

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print("average loss=%r" % average_loss)
            average_loss = 0    


