from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class Softmax_Model:
    def __init__(self):
        self.mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

        self.image_size = 784  # image 28*28=784
        self.label_number = 10  # hand written number from 0-9

        self.y_labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size])
        # self.y_ = tf.placeholder(tf.float32, [None, 10])

    def train(self):
        # softmax_model
        W = tf.Variable(tf.zeros([self.image_size, self.label_number]))  # must be a variable rather than placeholder
        b = tf.Variable(tf.zeros([self.label_number]))  # must be a variable rather than placeholder
        y = tf.matmul(self.x, W) + b

        # loss
        # calculate tf.nn.softmax(y) automatically and apply cross entropy
        predict = tf.nn.softmax(y)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_labels * tf.log(predict),
                                                      reduction_indices=[1]))
        # train
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Training
            for _ in range(1000):
                batch_xs, batch_ys = self.mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={self.x: batch_xs, self.y_labels: batch_ys})

            # Testing
            predictions = sess.run(y, feed_dict={self.x: self.mnist.test.images})
            labels = sess.run(self.y_labels, feed_dict={self.y_labels: self.mnist.test.labels})
            p = tf.argmax(predictions, 1)
            l = tf.argmax(labels, 1)
            correct_prediction = tf.equal(p, l)
            print(sess.run(p))
            print(sess.run(l))
            print(sess.run(correct_prediction))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy))


if __name__ == '__main__':
    model = Softmax_Model()
    model.train()
