import numpy as np
import math

class Mlp:
    def __init__(self, input_nodes = 2, hidden_nodes = 2, output_nodes = 1, learning_rate = .2):
        # Number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize random weights for each layer
        self.weights_hidden = np.random.uniform(-1,1, size=(hidden_nodes, input_nodes))
        self.weights_output = np.random.uniform(-1,1, size=(output_nodes, hidden_nodes))

        # Initialize random bias for each layer
        self.bias_hidden = np.random.uniform(-1,1, size=(hidden_nodes,1))
        self.bias_output = np.random.uniform(-1,1, size=(output_nodes,1))

        # How much every train iteration contributes to changing weights
        self.learning_rate = learning_rate

    def sigmoid_f(self, x):
        # Standard sigmoid function
        return 1 / (1 + math.exp(-x))
    
    def sigmoid(self, x):
        sigmoider = lambda x: self.sigmoid_f(x)
        sig = np.vectorize(sigmoider)
        return sig(x)
    
    def feed_forward(self, inp):
        inp = np.matrix(inp).transpose()
        # Calculate hidden layer linear outputs
        hidden_outputs = np.dot(self.weights_hidden, inp)
        # Add hidden layer bias
        hidden_outputs = hidden_outputs + self.bias_hidden
        # Apply squashing function
        hidden_outputs = self.sigmoid(hidden_outputs)

        # Calculate output layer linear outputs
        output = np.dot(self.weights_output, hidden_outputs)
        # Add output layer bias
        output = output + self.bias_output
        # Apply squashing function
        output = self.sigmoid(output)

        # Return the layers' outputs, the hidden one is needed during backpropagation
        return (hidden_outputs, output)

    # This trai function uses stochastic gradient descent instead of batch
    def train(self, inp, targets):
        # Calculate output with given input
        hidden_outputs, output = self.feed_forward(inp)
        inp = np.matrix(inp).transpose()
        # Calculate errors between predicted and real values
        output_errors = targets - output

        # Calculate gradient, first sigmoid derivative
        gradient = np.multiply( output, (1-output) )
        gradient = np.multiply(output_errors, gradient)
        # Multiply by the learing rate
        gradient = np.multiply(gradient, self.learning_rate)
        # Calculate hidden to output weights changes 
        delta_weights_output = np.dot(gradient, hidden_outputs.transpose())
        # The new weights are just the sum between the old ones and the deltas
        self.weights_output = self.weights_output + delta_weights_output
        # Changes in bias are just the gradient
        self.bias_output = self.bias_output + gradient
        
        # Calculate the errors in input to hidden layer
        weights_output_t = self.weights_output.transpose()
        hidden_errors = np.dot(weights_output_t, output_errors)

        # Same calculus as before, first the gradient
        hidden_gradient = np.multiply( hidden_outputs, (1-hidden_outputs) )
        hidden_gradient = np.multiply(hidden_errors, hidden_gradient)
        hidden_gradient = np.multiply(hidden_gradient, self.learning_rate)
        # Weights changes in input to hidden
        delta_weights_hidden = np.dot(hidden_gradient, inp.transpose())
        # Sum the error
        self.weights_hidden = self.weights_hidden + delta_weights_hidden
        # The bias change according to the gradient
        self.bias_hidden = self.bias_hidden + hidden_gradient
        

    def predict(self, inp):
        hidden_outputs, output = self.feed_forward(inp)
        return output