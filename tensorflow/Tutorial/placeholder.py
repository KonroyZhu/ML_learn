import tensorflow as tf

a=tf.placeholder(tf.float32,shape=[2,2])
b=tf.placeholder(tf.float32)

c=a+b

sess=tf.Session()
print(sess.run(a,{a:[[2,2],[3,3]]}))

