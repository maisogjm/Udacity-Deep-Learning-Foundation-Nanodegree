import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import shutil

flags = tf.flags

flags.DEFINE_boolean("reset", True, "")
flags.DEFINE_boolean("chat", False, "")

flags.DEFINE_string("summary_dir", "seq2seq-summary", "")
flags.DEFINE_string("model_dir", "seq2seq-model", "")
flags.DEFINE_string("model_path", "seq2seq-model/model", "")

flags.DEFINE_string("data", "dialogue_1000.txt", "")

flags.DEFINE_string("attention_type", "bahdanau", "luong or bahdanau?")
flags.DEFINE_float("learning_rate", 1e-3, "")
flags.DEFINE_integer("epochs", 500, "")
flags.DEFINE_integer("batch_size", 1, "")
flags.DEFINE_integer("clip_norm", 5, "")
flags.DEFINE_integer("num_units", 512, "")
flags.DEFINE_integer("num_layers", 2, "")
flags.DEFINE_integer("keep_prob", 0.75, "")
flags.DEFINE_integer("max_response_len", 100, "")


flags.DEFINE_integer("sample_after", 50, "")
flags.DEFINE_integer("log_after", 25, "")
flags.DEFINE_integer("save_after", 100, "")

FLAGS = flags.FLAGS

class Seq2SeqData(object):

    def __init__(self, batch_size, txt_file):

        with open(txt_file) as f:
            txt = f.read()

        DECODER_START_CHAR = "\0"
        DECODER_END_CHAR = "\1"

        chars = set(txt).union([DECODER_START_CHAR, DECODER_END_CHAR])
        self.N = len(chars)

        lines = txt.split("\n")

        self.char2idx = { char:i for i, char in enumerate(chars) }
        self.idx2char = { i:char for i, char in enumerate(chars) }

        self.decoder_start_idx = self.char2idx[DECODER_START_CHAR]
        self.decoder_end_idx = self.char2idx[DECODER_END_CHAR]

        indexes = []

        for line in lines:
            processed = [self.char2idx[char] for char in line]
            indexes.append(processed)

        self.X = indexes[:-1]
        self.Y = []
        self.Y_decoder = []

        for line in indexes[1:]:
            self.Y.append(line + [self.char2idx[DECODER_END_CHAR]])
            self.Y_decoder.append([self.char2idx[DECODER_START_CHAR]] + line)

        self.batch_size = batch_size
        self.iters_in_epoch = len(self.X) // self.batch_size
        self.current_iter = 0

    def next_batch(self):

        to = self.current_iter * self.batch_size + self.batch_size

        xs = self.X[self.current_iter:to]
        xs_len = [len(x) for x in xs]
        ys = self.Y[self.current_iter:to]
        ysd = self.Y_decoder[self.current_iter:to]
        ysd_len = [len(y) for y in ysd]

        self.current_iter += 1

        if self.current_iter >= self.iters_in_epoch:
            self.current_iter = 0

        return xs, ys, ysd, xs_len, ysd_len

class Seq2SeqModel(object):

    def __init__(self, N, decoder_start_idx, decoder_end_idx, config):
        
        self.global_step = tf.get_variable("global_step", initializer=1, trainable=False)

        self.N = N
        self.decoder_start_idx = decoder_start_idx
        self.decoder_end_idx = decoder_end_idx

        self.batch_size = config.batch_size
        self.num_units = config.num_units
        self.num_layers = config.num_layers
        self.max_response_len = config.max_response_len
        self.clip_norm = config.clip_norm
        self.learning_rate = config.learning_rate
        self.attention_type = config.attention_type

        self.prepare_data()

        self.init_rnn()

        self.propagate_forward()

        self.infer()

        self.calculate_loss()
            
        self.propagate_backward()
            
        self.calculate_accuracy()
        
        self.register_summaries()

    def prepare_data(self):        
        self.X = tf.placeholder(tf.int32, shape=(self.batch_size, None), name="X")
        one_hots = np.eye(self.N)
        self.embeddings = tf.constant(one_hots, dtype=tf.float32, name="embeddings")
        self.X_hots = tf.nn.embedding_lookup(self.embeddings, self.X)

        self.Y_decoder = tf.placeholder(tf.int32, shape=(self.batch_size, None), name="Y_decoder")
        self.Y_decoder_hots = tf.nn.embedding_lookup(self.embeddings, self.Y_decoder)      

        self.Y = tf.placeholder(tf.int64, shape=(self.batch_size, None), name="Y")

        self.keep_prob = tf.placeholder(tf.float32)

    def init_rnn(self, rnn=tf.contrib.rnn.GRUCell):
        self.num_units = self.num_units

        self.encoder_sequence_length = tf.placeholder(tf.int32, shape=(None))
        self.decoder_sequence_length = tf.placeholder(tf.int32, shape=(None))

        self.encoder_layers = [
            tf.contrib.rnn.DropoutWrapper(rnn(self.num_units), output_keep_prob=self.keep_prob) 
            for _ in xrange(self.num_layers)
        ]
                
        self.encoder_rnn = tf.contrib.rnn.MultiRNNCell(self.encoder_layers)

        self.state = self.init_state = self.encoder_rnn.zero_state(self.batch_size, tf.float32)

        self.decoder_layers = [
            tf.contrib.rnn.DropoutWrapper(rnn(self.num_units), output_keep_prob=self.keep_prob) 
            for _ in xrange(self.num_layers)
        ]
                
        self.decoder_rnn = tf.contrib.rnn.MultiRNNCell(self.decoder_layers)

    def propagate_forward(self):

        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.encoder_rnn, self.X_hots, 
                                    sequence_length=self.encoder_sequence_length, dtype=tf.float32)

        attention = tf.contrib.seq2seq.prepare_attention(self.encoder_outputs, self.attention_type, self.num_units)

        self.attention_keys, self.attention_values, self.attention_score_fn, self.attention_construct_fn = attention


        fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(self.encoder_final_state, 
                                                                self.attention_keys, 
                                                                self.attention_values, 
                                                                self.attention_score_fn, 
                                                                self.attention_construct_fn)

        reuse = None

        with tf.variable_scope("decoder_rnn", reuse=reuse):
            decoder_output = tf.contrib.seq2seq.dynamic_rnn_decoder(self.decoder_rnn, 
                                                                    fn_train, 
                                                                    self.Y_decoder_hots, 
                                                                    sequence_length=self.decoder_sequence_length)
        reuse = True

        self.decoder_outputs, self.decoder_final_state, c = decoder_output

        self.W_final = tf.get_variable("W_final", shape=(self.num_units, self.N), 
                initializer=tf.contrib.layers.xavier_initializer())
        
        self.b_final = tf.get_variable("b_final", shape=(self.N), initializer=tf.zeros_initializer())
        
        stacked_outputs = tf.reshape(self.decoder_outputs, shape=(-1, self.num_units))

        self.logits = tf.matmul(stacked_outputs, self.W_final) + self.b_final

        self.softmax = tf.nn.softmax(self.logits)

    def infer(self):
        fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(lambda x: tf.matmul(tf.reshape(x, shape=(-1, self.num_units)), self.W_final) + self.b_final, 
                                                                    self.encoder_final_state, 
                                                                    self.attention_keys, 
                                                                    self.attention_values, 
                                                                    self.attention_score_fn, 
                                                                    self.attention_construct_fn, 
                                                                    self.embeddings, 
                                                                    self.decoder_start_idx, 
                                                                    self.decoder_end_idx, 
                                                                    self.max_response_len, 
                                                                    self.N)
        
        with tf.variable_scope("decoder_rnn", reuse=True):
            decoder_output = tf.contrib.seq2seq.dynamic_rnn_decoder(self.decoder_rnn, fn_inference)

        decoder_outputs_inference, decoder_final_state_inference, c = decoder_output

        self.inferred_indexes = tf.argmax(decoder_outputs_inference, axis=2)


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
            self.variable_summaries(grad, "grad-" + str(i))
            
        self.merged = tf.summary.merge_all()      


    def variable_summaries(self, var, name_scope):
        with tf.name_scope(name_scope):
            with tf.name_scope("summaries"):
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

    for target_idx, prediction_idx in enumerate(fetched["predictions"]):
        prediction += train_data.idx2char[prediction_idx].encode("string_escape") 
        target += train_data.idx2char[ys[0][target_idx]].encode("string_escape")
        
    print "\nprediction: %s\n    target: %s" % (prediction, target)

    summary = "\nglobal step: %d, epoch: %d, loss: %f, accuracy: %f"
    print summary % (fetched["global_step"], epoch, fetched["loss"], fetched["accuracy"])

def chat(me, sess, model, train_data):

    feed_dict = { 
        model.X: [[train_data.char2idx[char] for char in me]],
        model.encoder_sequence_length: [len(me)],
        model.keep_prob: 1.0
    }

    inferred = sess.run(model.inferred_indexes, feed_dict)

    inferred = inferred.ravel()

    response = [train_data.idx2char[idx] for idx in inferred]

    return ''.join(response)

def chat_with_chathy(sess, model, train_data):

    while True:
        me = raw_input("me: ")

        if me:
            chathy = chat(me, sess, model, train_data)

            print "\nchathy: %s\n" % chathy

def main(_):

    prep_directories()

    with tf.Graph().as_default():

        train_data = Seq2SeqData(FLAGS.batch_size, FLAGS.data)
        model = Seq2SeqModel(train_data.N, train_data.decoder_start_idx, train_data.decoder_end_idx, FLAGS)
            
        with tf.Session() as sess:

            train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            saver = tf.train.Saver()
                        
            if FLAGS.reset: 
                sess.run(tf.global_variables_initializer()) 
            else:
                saver.restore(sess, FLAGS.model_path)

            if FLAGS.chat: 
                chat_with_chathy(sess, model, train_data)

            for epoch in xrange(FLAGS.epochs):

                for minibatch_num in xrange(train_data.iters_in_epoch):

                    xs, ys, ysd, xs_len, ysd_len = train_data.next_batch()

                    feed_dict = { 
                        model.X: xs,  
                        model.Y: ys,
                        model.Y_decoder: ysd,
                        model.encoder_sequence_length: xs_len,
                        model.decoder_sequence_length: ysd_len,
                        model.keep_prob: FLAGS.keep_prob
                    }

                    fetch = {
                        "merged": model.merged, 
                        "loss": model.loss, 
                        "train_op": model.train_op, 
                        "accuracy": model.accuracy,
                        "global_step": model.global_step, 
                        "predictions": model.predictions
                    }

                    fetched = sess.run(fetch, feed_dict)
                                    
                    if fetched["global_step"] % FLAGS.log_after == 0:
                        log(fetched, train_data, epoch, ys)

                    if fetched["global_step"] % FLAGS.save_after == 0:
                        train_writer.add_summary(fetched["merged"], fetched["global_step"])
                        saver.save(sess, FLAGS.model_path)

                    if fetched["global_step"] % FLAGS.sample_after == 0:  
                        questions = [
                            "Hi", "Let's go.", "Have fun?", "What's your name?", 
                            "What do I mix with brown turkey?", 
                            "What is the meaning of life?"
                        ]
                        me = np.random.choice(questions)
                        chathy = chat(me, sess, model, train_data)
                        print "\n*****SAMPLE TIME*****\n\nme: %s\n\nchathy: %s\n\n*****END SAMPLE*****" % (me, chathy)

if __name__ == "__main__":
    tf.app.run()


