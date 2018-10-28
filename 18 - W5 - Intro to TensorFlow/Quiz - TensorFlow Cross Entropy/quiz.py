# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax            = tf.placeholder(tf.float32)
one_hot            = tf.placeholder(tf.float32)
log_softmax        = tf.log(softmax)
dot_onehot_logsf   = tf.multiply(one_hot,log_softmax)
sum_dot_oh_lsf     = tf.reduce_sum(dot_onehot_logsf)
neg_sum_dot_oh_lsf = tf.multiply(sum_dot_oh_lsf,-1.0)

# TODO: Print cross entropy from session
with tf.Session() as sess:
    output = sess.run(neg_sum_dot_oh_lsf, feed_dict={softmax:softmax_data, one_hot:one_hot_data})
    print(output)