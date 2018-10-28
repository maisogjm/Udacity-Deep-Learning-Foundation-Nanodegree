# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.subtract(tf.divide(x, y),1)

with tf.Session() as sess:
    output = sess.run(z,feed_dict={x: 10, y: 2})
    print(output)

# TODO: Print z from a session
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y),tf.constant(1))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)

# TODO: Print z from a session
z = tf.subtract(tf.divide(tf.constant(10.0), tf.constant(2.0)),tf.constant(1.0))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)