import tensorflow as tf

n1=tf.constant(5.0,tf.float32)
n2=tf.constant(7.0,tf.float32)

c=n1*n2

sess=tf.Session()

fileWrite=tf.summary.FileWriter("/home/konroy/PycharmProjects/MachineLearning/tensorflow/Tutorial/graph",sess.graph)

print(sess.run(c))
sess.close()

# /home/konroy/PycharmProjects/MachineLearning/tensorflow/Tutorial