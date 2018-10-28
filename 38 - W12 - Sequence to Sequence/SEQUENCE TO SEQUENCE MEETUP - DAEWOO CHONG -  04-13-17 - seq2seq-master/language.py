import tensorflow as tf
import numpy as np
import os
import shutil

flags = tf.flags

flags.DEFINE_boolean("reset", True, "")

flags.DEFINE_string("summary_dir", "language-summary", "")
flags.DEFINE_string("model_dir", "language-model", "")
flags.DEFINE_string("model_path", "language-model/model", "")

flags.DEFINE_string("data", "taylor.txt", "")

flags.DEFINE_float("learning_rate", 1e-2, "")
flags.DEFINE_integer("epochs", 500, "")
flags.DEFINE_integer("batch_size", 1, "")
flags.DEFINE_integer("clip_norm", 5, "")
flags.DEFINE_integer("num_units", 500, "")
flags.DEFINE_integer("num_steps", 50, "")
flags.DEFINE_integer("num_layers", 1, "")
flags.DEFINE_integer("keep_prob", 0.5, "")

flags.DEFINE_integer("generate_after", 50, "")
flags.DEFINE_integer("log_after", 50, "")
flags.DEFINE_integer("save_after", 100, "")

FLAGS = flags.FLAGS

class LanguageModelData(object):

    def __init__(self, batch_size, num_steps, txt_file):

        with open(txt_file) as f:
            txt = f.read()

        chars = set(txt)
        self.N = len(chars)

        self.char2idx = { char:i for i, char in enumerate(chars) }
        self.idx2char = { i:char for i, char in enumerate(chars) }

        indexes = [self.char2idx[c] for c in txt]
        self.X = indexes[:-1]
        self.Y = indexes[1:]

        self.batch_size = batch_size
        self.num_steps = num_steps
        batch_len =  len(self.X) // self.batch_size
        self.iters_in_epoch = (batch_len - 1) // self.num_steps
        self.current_iter = 0

        self.X = np.array(self.X[0:self.batch_size*batch_len]).reshape((self.batch_size, batch_len))
        self.Y = np.array(self.Y[0:self.batch_size*batch_len]).reshape((self.batch_size, batch_len))

    def next_batch(self):
        xs = self.X[:,self.current_iter*self.num_steps:(self.current_iter+1)*self.num_steps]
        ys = self.Y[:,self.current_iter*self.num_steps:(self.current_iter+1)*self.num_steps]

        self.current_iter += 1

        if self.current_iter >= self.iters_in_epoch:
            self.current_iter = 0

        return xs, ys

class LanguageModel(object):

    def __init__(self, N, config):
        self.global_step = tf.get_variable("global_step", initializer=1, trainable=False)

        self.N = N

        self.num_steps = config.num_steps
        self.num_units = config.num_units
        self.num_layers = config.num_layers
        self.clip_norm = config.clip_norm
        self.learning_rate = config.learning_rate

        self.prepare_data()

        self.init_rnn()

        self.propagate_forward()

        self.calculate_loss()
            
        self.propagate_backward()
            
        self.calculate_accuracy()
        
        self.register_summaries()

    def prepare_data(self):
        self.batch_size = tf.placeholder(tf.int32)

        self.X = tf.placeholder(tf.int32, shape=(None, self.num_steps), name="X")
        one_hots = np.eye(self.N)
        embeddings = tf.constant(one_hots, dtype=tf.float32, name="embeddings")
        
        self.X_hots = tf.nn.embedding_lookup(embeddings, self.X)
        self.Y = tf.placeholder(tf.int64, shape=(None, self.num_steps), name="Y")

        self.keep_prob = tf.placeholder(tf.float32)

    def init_rnn(self, rnn=tf.contrib.rnn.LSTMCell):
        self.num_units = self.num_units
                
        self.rnn = rnn(self.num_units)
        
        self.layers = [
            tf.contrib.rnn.DropoutWrapper(rnn(self.num_units), output_keep_prob=self.keep_prob) 
            for _ in xrange(self.num_layers)
        ]
                
        self.rnn = tf.contrib.rnn.MultiRNNCell(self.layers)

        self.state = self.init_state = self.rnn.zero_state(self.batch_size, tf.float32)

        self.sequence_length = tf.placeholder(tf.int32, shape=(None))

    def propagate_forward(self):

        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.rnn, self.X_hots, 
                                            sequence_length=self.sequence_length, initial_state=self.state)

        self.W_final = tf.get_variable("W_final", shape=(self.num_units, self.N), 
                initializer=tf.contrib.layers.xavier_initializer())
        
        self.b_final = tf.get_variable("b_final", shape=(self.N), initializer=tf.zeros_initializer())
        
        stacked_outputs = tf.reshape(self.outputs, shape=(-1, self.num_units))

        self.logits = tf.matmul(stacked_outputs, self.W_final) + self.b_final

        self.softmax = tf.nn.softmax(self.logits)

    def calculate_loss(self):
        self.stacked_Y = tf.reshape(self.Y, shape=(-1,))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.stacked_Y)
        self.loss = tf.reduce_mean(cross_entropy)

    def propagate_backward(self, optimizer=tf.train.AdamOptimizer):
        grads = tf.gradients(self.loss, tf.trainable_variables())
        self.clipped_grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        
        optimizer = optimizer(self.learning_rate)

        self.train_op = optimizer.apply_gradients(
                            zip(self.clipped_grads, tf.trainable_variables()),
                            global_step=self.global_step)

    def calculate_accuracy(self):
        self.predictions = tf.argmax(self.logits, dimension=1)
        self.bools = tf.equal(self.predictions, self.stacked_Y)
        self.correct_predictions = tf.cast(self.bools, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_predictions)

    def register_summaries(self):
        self.variable_summaries(self.W_final, "W_final")
        self.variable_summaries(self.b_final, "b_final")
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        for i, grad in enumerate(self.clipped_grads):
            self.variable_summaries(grad, "grad-%d" % i)
            
        self.merged = tf.summary.merge_all()      

    def variable_summaries(self, var, name_scope):
        with tf.name_scope(name_scope):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))

def prep_directories():
    if os.path.isdir(FLAGS.summary_dir): shutil.rmtree(FLAGS.summary_dir)
    if os.path.isdir(FLAGS.model_dir) and FLAGS.reset: shutil.rmtree(FLAGS.model_dir)
        
    if not os.path.isdir(FLAGS.summary_dir): os.mkdir(FLAGS.summary_dir)
    if not os.path.isdir(FLAGS.model_dir): os.mkdir(FLAGS.model_dir)

def log(fetched, train_data, epoch, ys):
    prediction = ""
    target = ""

    ys = ys.ravel()

    for target_idx, prediction_idx in enumerate(fetched["predictions"]):
        prediction += train_data.idx2char[prediction_idx].encode("string_escape") 
        target += train_data.idx2char[ys[target_idx]].encode("string_escape")
        
    print "\nprediction: %s\n    target: %s" % (prediction, target)

    summary = "\nglobal step: %d, epoch: %d, loss: %f, accuracy: %f"
    print summary % (fetched["global_step"], epoch, fetched["loss"], fetched["accuracy"])

def generate_text(sess, model, train_data, num_samples=250):
    
    chars = []

    idx = np.zeros(model.num_steps)
    idx[0] = train_data.char2idx["\n"]

    sample_state = sess.run(model.init_state, {model.batch_size: 1})

    for t in xrange(num_samples):

        feed_dict = { 
            model.X: [idx], 
            model.state: sample_state,
            model.sequence_length: [1],
            model.keep_prob: 1.0
        }

        fetch = {
            "softmax": model.softmax,
            "state": model.final_state
        }

        fetched = sess.run(fetch, feed_dict=feed_dict)
        pred = np.random.choice(range(train_data.N), p=fetched["softmax"][0].ravel())

        idx = np.zeros(model.num_steps)
        idx[0] = pred

        chars.append(train_data.idx2char[pred])

        sample_state = fetched["state"]
 
    print "\n*****SAMPLE TIME*****\n\n%s\n\n*****END SAMPLE*****" % ''.join(chars)


def main(_):

    prep_directories()

    with tf.Graph().as_default():

        train_data = LanguageModelData(FLAGS.batch_size, FLAGS.num_steps, FLAGS.data)
        model = LanguageModel(train_data.N, FLAGS)
            
        with tf.Session() as sess:

            train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            saver = tf.train.Saver()
                        
            if FLAGS.reset: 
                sess.run(tf.global_variables_initializer()) 
            else:
                saver.restore(sess, FLAGS.model_path)

            state = sess.run(model.init_state, feed_dict={model.batch_size: FLAGS.batch_size})

            for epoch in xrange(FLAGS.epochs):

                for minibatch_num in xrange(train_data.iters_in_epoch):

                    xs, ys = train_data.next_batch()

                    feed_dict = { 
                        model.X: xs,  
                        model.Y: ys, 
                        model.state: state,
                        model.sequence_length: [FLAGS.num_steps for _ in xrange(FLAGS.batch_size)],
                        model.keep_prob: FLAGS.keep_prob
                    }

                    fetch = {
                        "merged": model.merged, 
                        "loss": model.loss, 
                        "train_op": model.train_op, 
                        "accuracy": model.accuracy,
                        "final_state": model.final_state, 
                        "global_step": model.global_step, 
                        "predictions": model.predictions
                    }

                    fetched = sess.run(fetch, feed_dict)

                    state = fetched["final_state"]
                                    
                    if fetched["global_step"] % FLAGS.log_after == 0:
                        log(fetched, train_data, epoch, ys)

                    if fetched["global_step"] % FLAGS.save_after == 0:
                        train_writer.add_summary(fetched["merged"], fetched["global_step"])
                        saver.save(sess, FLAGS.model_path)

                    if fetched["global_step"] % FLAGS.generate_after == 0:  
                        generate_text(sess, model, train_data)

if __name__ == "__main__":
    tf.app.run()


