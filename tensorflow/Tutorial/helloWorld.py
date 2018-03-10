import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(10)
b = tf.constant(32)
c = a+b
sess=tf.Session()
fileWrite=tf.summary.FileWriter("/home/konroy/PycharmProjects/MachineLearning/tensorflow/Tutorial/graph",sess.graph)


print(sess.run(a + b))
sess.close()