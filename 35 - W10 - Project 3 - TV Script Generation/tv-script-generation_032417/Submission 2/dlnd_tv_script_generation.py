
# coding: utf-8

# # TV Script Generation
# In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
# ## Get the Data
# The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..

# In[2]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]


# ## Explore the Data
# Play around with `view_sentence_range` to view different parts of the data.

# In[3]:

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Functions
# The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
# - Lookup Table
# - Tokenize Punctuation
# 
# ### Lookup Table
# To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
# - Dictionary to go from the words to an id, we'll call `vocab_to_int`
# - Dictionary to go from the id to word, we'll call `int_to_vocab`
# 
# Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`

# In[4]:

import numpy as np
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # It is convenient to sort the words by frequency, so that,
    # e.g., the first word in the two dictionaries we construct
    # is the most frequent word.
    counts = Counter(text)
    vocab  = sorted(counts, key=counts.get, reverse=True)
    
    # Dictionary to go from the words to an id, we'll call vocab_to_int.
    # Dictionary to go from the id to word, we'll call int_to_vocab.
    vocab_to_int = { word :   ii for ii, word in enumerate(vocab,1) }
    int_to_vocab = { ii   : word for ii, word in enumerate(vocab,1) }
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


# ### Tokenize Punctuation
# We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".
# 
# Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
# - Period ( . )
# - Comma ( , )
# - Quotation Mark ( " )
# - Semicolon ( ; )
# - Exclamation mark ( ! )
# - Question mark ( ? )
# - Left Parentheses ( ( )
# - Right Parentheses ( ) )
# - Dash ( -- )
# - Return ( \n )
# 
# This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".

# In[5]:

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # Create a dictionary to tokenize punctuation marks.
    punctuationDictionary = { '.'  : '||period||',
                              ','  : '||comma||',
                              '"'  : '|||doublequote|',
                              ';'  : '||semicolon||',
                              '!'  : '||exclamationmark||',
                              '?'  : '||questionmark||',
                              '('  : '||leftparentheses||',
                              ')'  : '|||rightparentheses|',
                              '--' : '||dash||',
                              '\n' : '||return||'}
    return punctuationDictionary

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


# ## Preprocess all the data and save it
# Running the code cell below will preprocess all the data and save it to file.

# In[6]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# # Check Point
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

# In[7]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## Build the Neural Network
# You'll build the components necessary to build a RNN by implementing the following functions below:
# - get_inputs
# - get_init_cell
# - get_embed
# - build_rnn
# - build_nn
# - get_batches
# 
# ### Check the Version of TensorFlow and Access to GPU

# In[8]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
# - Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
# - Targets placeholder
# - Learning Rate placeholder
# 
# Return the placeholders in the following the tuple `(Input, Targets, LearingRate)`

# In[9]:

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # Create TF Placeholders for the Neural Network.
    # Submission #2: use 'inputs_data' rather than 'input' for the
    # variable name. I tried to also change the TensorFlow label to
    # 'inputs_data', but the unit test was specifically
    # looking for the name 'input:0'. I hacked the unit test to
    # allow the change in the TensorFlow label, but this resulted
    # in obtaining training errors of NaN, and I could not figure
    # out why!
    inputs_data = tf.placeholder(tf.int32,   [None,None], name='input')
    targets     = tf.placeholder(tf.int32,   [None,None], name='targets')
    learn_rate  = tf.placeholder(tf.float32,              name='learn_rate')
    return (inputs_data, targets, learn_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


# ### Build RNN Cell and Initialize
# Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
# - The Rnn size should be set using `rnn_size`
# - Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
#     - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the cell and initial state in the following tuple `(Cell, InitialState)`

# In[10]:

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # Set up basic LSTM cell.
    # The Rnn size should be set using rnn_size.
    # Submission #2: rnn-size should be used for the number of units in the
    # LSTM cell. For the number of layers, assign a separate variable to hold
    # this information.
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # It wasn't explicitly mentioned in the instructions,
    # but let's try some dropout to help prevent overfitting.
    # Note that I have hard-coded the parameter output_keep_prob with a "magic number", 0.9.
    # This is a hyperparameter that could be tweaked.
    # N.b.: dropout didn't seem to help the testing loss or the final results,
    # so I am removing it.
    #drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.9)
    
    # Stack up multiple LSTM layers.
    # Submission #2: for the number of layers, assign a separate variable to hold
    # this information. I started with only 1 layer and seemed to get a decent
    # training loss.
    lstm_layers = 1
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
    
    # Initalize Cell State using the MultiRNNCell's zero_state() function.
    # Apply the name "initial_state" to the initial state using tf.identity()
    initial_state = tf.identity( cell.zero_state(batch_size, tf.float32) , name='initial_state')
    return (cell, initial_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


# ### Word Embedding
# Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.

# In[11]:

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensionsembeddding,
    :return: Embedded input.
    """
    # Apply embedding to input_data using TensorFlow. Return the embedded sequence.
    embedding = tf.Variable( tf.random_uniform((vocab_size, embed_dim), -1, 1) )
    embed     = tf.nn.embedding_lookup(embedding,input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


# ### Build RNN
# You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
# - Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
#  - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 

# In[12]:

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # Build the RNN using the tf.nn.dynamic_rnn().
    # If no initial_state is provided, dtype must be.
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    
    # Apply the name "final_state" to the final state using tf.identity().
    named_final_state = tf.identity( final_state , name='final_state')
    return (outputs, named_final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


# ### Build the Neural Network
# Apply the functions you implemented above to:
# - Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
# - Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
# - Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.
# 
# Return the logits and final state in the following tuple (Logits, FinalState) 

# In[13]:

def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    # Apply embedding to input_data using your get_embed(input_data, vocab_size, embed_dim) function.
    # Note that I have hard-coded the parameter embed_size with a "magic number", 300.
    # This is a hyperparameter that could be tweaked.
    embed = get_embed(input_data, vocab_size, 300)

    # Build RNN using cell and your build_rnn(cell, inputs) function.
    outputs, final_state = build_rnn(cell, embed)
    
    # Apply a fully connected layer with a linear activation and vocab_size as the number of outputs.
    # Set activation_fn to skip it and maintain a linear activation.
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    return (logits, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


# ### Batches
# Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
# - The first element is a single batch of **input** with the shape `[batch size, sequence length]`
# - The second element is a single batch of **targets** with the shape `[batch size, sequence length]`
# 
# If you can't fill the last batch with enough data, drop the last batch.
# 
# For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
# ```
# [
#   # First Batch
#   [
#     # Batch of Input
#     [[ 1  2  3], [ 7  8  9]],
#     # Batch of targets
#     [[ 2  3  4], [ 8  9 10]]
#   ],
#  
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 4  5  6], [10 11 12]],
#     # Batch of targets
#     [[ 5  6  7], [11 12 13]]
#   ]
# ]
# ```

# In[14]:

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """

    # Compute number of batches.
    n_batches = len(int_text)//(batch_size*seq_length)

    # The target is lagged by an offset of 1. So, if
    # len(int_text) = n_batches*batch_size*seq_length,
    # we won't have enough data for the last target!
    # We want only full batches. So, if
    # len(int_text) = n_batches*batch_size*seq_length,
    # DECREMENT n_batches by 1.
    if len(int_text) == n_batches*batch_size*seq_length:
        n_batches -= 1   

    # Loop over BATCHES.
    # Submission #2: Cast to int rather than the default float,
    # since word tokens/indices are integers.
    Batches = np.ndarray(shape=(n_batches,2,batch_size,seq_length), dtype=int) # Initialize output numpy ndarray.
    for batch_num in range(n_batches):

        # Loop over SEQUENCES within batch.
        for seq_num in range(batch_size):
            # Grab the next input and target sequences.
            idx1 = (batch_num*seq_length) + (seq_num*seq_length)
            idx2 = idx1 + seq_length
            idy1 = idx1 + 1
            idy2 = idy1 + seq_length
            Batches[batch_num,0,seq_num,] = int_text[idx1:idx2]
            Batches[batch_num,1,seq_num,] = int_text[idy1:idy2]
    return Batches

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)


# ## Neural Network Training
# ### Hyperparameters
# Tune the following parameters:
# 
# - Set `num_epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `seq_length` to the length of sequence.
# - Set `learning_rate` to the learning rate.
# - Set `show_every_n_batches` to the number of batches the neural network should print progress.

# In[18]:

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 256
# RNN Size (number of units in the LSTM cell, not number of layers)
rnn_size = 512
# Sequence Length
seq_length = 64
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 10

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'


# ### Build the Graph
# Build the graph using the neural network you implemented.

# In[19]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)


# ## Train
# Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.

# In[20]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# ## Save Parameters
# Save `seq_length` and `save_dir` for generating a new TV script.

# In[21]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))


# # Checkpoint

# In[22]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# ## Implement Generate Functions
# ### Get Tensors
# Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
# - "input:0"
# - "initial_state:0"
# - "final_state:0"
# - "probs:0"
# 
# Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 

# In[23]:

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # Get tensors from loaded_graph using the function get_tensor_by_name().
    # Submission #2: use 'inputs_data' rather than 'input' for the
    # variable name. I tried to also change the TensorFlow label to
    # 'inputs_data', but the unit test was specifically
    # looking for the name 'input:0'. I hacked the unit test to
    # allow the change in the TensorFlow label, but this resulted
    # in obtaining training errors of NaN, and I could not figure
    # out why!
    inputs_data   = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state   = loaded_graph.get_tensor_by_name('final_state:0')
    probs         = loaded_graph.get_tensor_by_name('probs:0')
    return (inputs_data, initial_state, final_state, probs)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)


# ### Choose Word
# Implement the `pick_word()` function to select the next word using `probabilities`.

# In[24]:

def pick_word(probabilities, int_to_vocab, top_n=5):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # Implement the pick_word() function to select the next word using probabilities.
    # Adapting Mat's strategy to "reduce noise and make things a little less random"
    # when sampling characters in the Anna KaRNNa notebook -- sample only from the
    # top N words.
    
    # Set all probabilities below the top 'N' to 0.
    probabilities[np.argsort(probabilities)[:-top_n]] = 0
    
    # Re-normalize probabilities so they sum to 1.
    probabilities = probabilities / np.sum(probabilities)
    
    # Sample from the top 'N' words. Add 1 when indexing into int_to_vocab, 
    # since indexing in that dictionary is 1-offset. (int_to_vocab is
    # 1-offset because we used Counter to create it.)
    vocab_size   = len(int_to_vocab)
    idx          = np.random.choice(vocab_size, 1, p=probabilities)[0]
    selectedWord = int_to_vocab[idx+1] # add 1
    return selectedWord


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)


# ## Generate TV Script
# This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.

# In[25]:

gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)


# # The TV Script is Nonsensical
# It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# # Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
