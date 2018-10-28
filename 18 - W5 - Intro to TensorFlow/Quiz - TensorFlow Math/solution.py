# Quiz Solution
# Note: You can't run code in this tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

# Note:TensorFlow has multiple ways to divide.
#   tf.divide(x,y) uses Python 3 division semantics and will return a float here
#          It would be the best choice if all the other values had been floats
#   tf.div(x,y) uses Python 2 division semantics and will return an integer here
#          TensorFlow documentation suggests we should prefer tf.divide
#   tf.floordiv(x,y) will do floating point division and then round down to the nearest
#          integer (but the documentation says it may still represent
#          its result as a floating point value)
#   tf.cast(tf.divide(x,y), tf.int32)
#          This lets us do floating point division and then cast it to an integer
#          to match the 1 passed to subtract
#   tf.cast(tf.constant(1), tf.float64)
#          This lets us do floating point division and not lose any precision that
#          would be lost by casting to int if the numbers were not evenly divisible
#   Another option, change the constants to have floats so no casting is required,
#          like this:
#            x = tf.constant(10.0)
#            y = tf.constant(2.0)
#            z = tf.subtract(tf.divide(x,y),tf.constant(1.0))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
