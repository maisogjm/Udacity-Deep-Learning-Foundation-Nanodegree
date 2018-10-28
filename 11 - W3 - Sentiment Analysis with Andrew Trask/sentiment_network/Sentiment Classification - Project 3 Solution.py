
# coding: utf-8

# # Sentiment Classification & How To "Frame Problems" for a Neural Network
# 
# by Andrew Trask
# 
# - **Twitter**: @iamtrask
# - **Blog**: http://iamtrask.github.io

# ### What You Should Already Know
# 
# - neural networks, forward and back-propagation
# - stochastic gradient descent
# - mean squared error
# - and train/test splits
# 
# ### Where to Get Help if You Need it
# - Re-watch previous Udacity Lectures
# - Leverage the recommended Course Reading Material - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) (40% Off: **traskud17**)
# - Shoot me a tweet @iamtrask
# 
# 
# ### Tutorial Outline:
# 
# - Intro: The Importance of "Framing a Problem"
# 
# 
# - Curate a Dataset
# - Developing a "Predictive Theory"
# - **PROJECT 1**: Quick Theory Validation
# 
# 
# - Transforming Text to Numbers
# - **PROJECT 2**: Creating the Input/Output Data
# 
# 
# - Putting it all together in a Neural Network
# - **PROJECT 3**: Building our Neural Network
# 
# 
# - Understanding Neural Noise
# - **PROJECT 4**: Making Learning Faster by Reducing Noise
# 
# 
# - Analyzing Inefficiencies in our Network
# - **PROJECT 5**: Making our Network Train and Run Faster
# 
# 
# - Further Noise Reduction
# - **PROJECT 6**: Reducing Noise by Strategically Reducing the Vocabulary
# 
# 
# - Analysis: What's going on in the weights?

# # Lesson: Curate a Dataset

# In[1]:

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()


# In[2]:

len(reviews)


# In[5]:

reviews[0]


# In[6]:

labels[0]


# # Lesson: Develop a Predictive Theory

# In[7]:

print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)


# # Project 1: Quick Theory Validation

# In[9]:

from collections import Counter
import numpy as np


# In[10]:

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()


# In[11]:

for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1


# In[12]:

positive_counts.most_common()


# In[20]:

pos_neg_ratios = Counter()

for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

for word,ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))


# In[21]:

# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()


# In[22]:

# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]


# # Transforming Text into Numbers

# In[26]:

from IPython.display import Image

review = "This was a horrible, terrible movie."

Image(filename='sentiment_network.png')


# In[27]:

review = "The movie was excellent"

Image(filename='sentiment_network_pos.png')


# # Project 2: Creating the Input/Output Data

# In[74]:

vocab = set(total_counts.keys())
vocab_size = len(vocab)
print(vocab_size)


# In[75]:

list(vocab)


# In[46]:

import numpy as np

layer_0 = np.zeros((1,vocab_size))
layer_0


# In[47]:

from IPython.display import Image
Image(filename='sentiment_network.png')


# In[48]:

word2index = {}

for i,word in enumerate(vocab):
    word2index[word] = i
word2index


# In[49]:

def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])


# In[33]:

layer_0


# In[51]:

def get_target_for_label(label):
    if(label == 'POSITIVE'):
        return 1
    else:
        return 0


# In[54]:

labels[0]


# In[52]:

get_target_for_label(labels[0])


# In[55]:

labels[1]


# In[53]:

get_target_for_label(labels[1])


# # Project 3: Building a Neural Network

# - Start with your neural network from the last chapter
# - 3 layer neural network
# - no non-linearity in hidden layer
# - use our functions to create the training data
# - create a "pre_process_data" function to create vocabulary for our training data generating functions
# - modify "train" to train over the entire corpus

# ### Where to Get Help if You Need it
# - Re-watch previous week's Udacity Lectures
# - Chapters 3-5 - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) - (40% Off: **traskud17**)

# In[86]:

import time
import sys
import numpy as np

# Let's tweak our network from before to model these phenomena
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
       
        # set our random number generator 
        np.random.seed(1)
    
        self.pre_process_data(reviews, labels)
        
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
        
        
    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        self.label_vocab = list(label_vocab)
        
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        self.learning_rate = learning_rate
        
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] += 1
                
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, training_reviews, training_labels):
        
        assert(len(training_reviews) == len(training_labels))
        
        correct_so_far = 0
        
        start = time.time()
        
        for i in range(len(training_reviews)):
            
            review = training_reviews[i]
            label = training_labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            self.update_input_layer(review)

            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # TODO: Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # TODO: Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            if(np.abs(layer_2_error) < 0.5):
                correct_so_far += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        
        correct = 0
        
        start = time.time()
        
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            reviews_per_second = i / float(time.time() - start)
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                             + "% #Correct:" + str(correct) + " #Tested:" + str(i+1) + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if(layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
        


# In[87]:

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)


# In[61]:

# evaluate our model before training (just to show how horrible it is)
mlp.test(reviews[-1000:],labels[-1000:])


# In[62]:

# train the network
mlp.train(reviews[:-1000],labels[:-1000])


# In[63]:

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)


# In[64]:

# train the network
mlp.train(reviews[:-1000],labels[:-1000])


# In[65]:

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)


# In[66]:

# train the network
mlp.train(reviews[:-1000],labels[:-1000])

