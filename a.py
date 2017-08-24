import tensorflow as tf

tf.set_random_seed(2)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))