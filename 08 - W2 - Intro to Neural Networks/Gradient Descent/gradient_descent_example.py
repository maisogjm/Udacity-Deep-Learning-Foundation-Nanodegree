# Defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([0.1, 0.3])
y = 0.2
weights = np.array([-0.8, 0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# The neural network output
nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])
# or nn_output = sigmoid(np.dot(x, w))

# output error
error = y - nn_output

# error gradient
error_grad = error * sigmoid_prime(np.dot(x,w))

# Gradient descent step
del_w = [ learnrate * error_grad * x[0],
          learnrate * error_grad * x[1]]
# or del_w = learnrate * error_grad * x