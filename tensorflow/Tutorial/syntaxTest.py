import tensorflow as tf
import numpy as np

# embeddings=np.array([[1,1],[2,2],[3,3],[4,4]])
# word_ids = tf.placeholder(tf.int32, shape=embeddings.shape)
# L = tf.Variable(embeddings, dtype=tf.float32, trainable=False, name="L")
# pretrained_embeddings = tf.nn.embedding_lookup(L, word_ids, name="pretrained_embeddings")
#
#
# b = tf.nn.embedding_lookup(embeddings, [1, 3])
#
# with tf.Session() as sess:
#     init=tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(b))
#     print(embeddings)

"""
somevalues = [1, 2, 3, 4, 5]
a = tf.Variable(somevalues, tf.int32)
with tf.Session() as sess1:
    init1 = tf.global_variables_initializer()
    sess1.run(init1)
    print(sess1.run(a))
"""

embeddings=np.array([[1,2],[3,4],[5,6],[7,8]])
lookup=tf.nn.embedding_lookup(embeddings,[1,2])

with tf.Session() as sess:
    print(sess.run(lookup))

