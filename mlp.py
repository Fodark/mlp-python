import numpy as np
import math


class Mlp:
    def __init__(self, input_nodes: int = 2, hidden_nodes: int = 2, output_nodes: int = 1,
                 learning_rate: float = .2) -> object:
        # Number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.number_of_nodes = [input_nodes, hidden_nodes, output_nodes]

        self.weights = []
        self.weights.append(np.random.randn(hidden_nodes, input_nodes) * np.sqrt(2 / (hidden_nodes + input_nodes)))
        self.weights.append(np.random.randn(output_nodes, hidden_nodes) * np.sqrt(2 / (output_nodes + hidden_nodes)))

        self.biases = []
        self.biases.append(np.random.uniform(0, 0, size=(hidden_nodes, 1)))
        self.biases.append(np.random.uniform(0, 0, size=(output_nodes, 1)))

        self.output = []
        self.errors = []

        # How much every train iteration contributes to changing weights
        self.learning_rate = learning_rate

    def add_layer(self, number_of_nodes):
        self.number_of_nodes.insert(-1, number_of_nodes)
        self.weights.append([])
        for i in range(2):
            self.weights[-1-i] = np.random.randn(self.number_of_nodes[-1-i], self.number_of_nodes[-2-i]) * np.sqrt(2 / (self.number_of_nodes[-1-i] + self.number_of_nodes[-2-i]))
        self.biases.insert(-1, np.random.uniform(0, 0, size=(number_of_nodes, 1)))

    @staticmethod
    def sigmoid_f(x):
        # Standard sigmoid function
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))

    def sigmoid(self, x):
        sig = np.vectorize(lambda y: self.sigmoid_f(y))
        return sig(x)

    def feed_forward(self, inp):
        outputs = [np.matrix(inp).transpose()]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(self.sigmoid((np.dot(self.weights[i], outputs[-1]) + self.biases[i])))

        return outputs

    # This training function uses stochastic gradient descent instead of batch
    def train(self, inp, targets):
        targets = np.matrix(targets).transpose()
        # Calculate output with given input
        outputs = self.feed_forward(inp)

        errors = [np.subtract(targets, outputs[-1])]
        for i in range(len(self.weights) - 1):
            errors.insert(0, np.dot(self.weights[-1-i].transpose(), errors[0]))

        for i in range(len(self.weights)):
            tmp = np.multiply(outputs[-1-i], (1 - outputs[-1-i]))
            gradient = np.multiply(errors[-1-i], tmp)
            gradient *= self.learning_rate
            self.biases[-1-i] += gradient
            delta_w  = np.dot(gradient, outputs[-2-i].transpose())
            self.weights[-1-i] += delta_w

    def predict(self, inp):
        outputs = self.feed_forward(inp)
        return outputs[-1]
