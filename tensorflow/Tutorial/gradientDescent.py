import tensorflow as tf

# softmax_model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

# inputs and outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)



# linear softmax_model
linear_model = W * x + b

# Loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# for i in range(1000):
#     sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#
# print(sess.run([W, b]))
