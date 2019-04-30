import numpy as np
import math
import operator

class Mlp:
    def __init__(self, init_nodes: int = 0, learning_rate: float = .2) -> None:
        self.number_of_nodes = []
        if init_nodes > 0:
            self.number_of_nodes.append(init_nodes)
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate

    def add_layer(self, number_of_nodes: int, weights = None, bias = None):
        self.number_of_nodes.append(number_of_nodes)
        if not weights is None:
            self.weights.append(weights)
        elif len(self.number_of_nodes) > 1:
            self.weights.append(np.random.randn(self.number_of_nodes[-1], self.number_of_nodes[-2]) * np.sqrt(2 / (self.number_of_nodes[-1] + self.number_of_nodes[-2])))
        
        if not bias is None:
            self.biases.append(bias)
        elif len(self.number_of_nodes) > 1:
            self.biases.append(np.random.uniform(0, 0, size=(number_of_nodes, 1)))
    
    def save(self, location):
        f = open(location, "w+")
        for i in self.number_of_nodes:
            f.write(str(i) + " ")
        f.write("\n")
        for i in self.weights:
            for j in i:
                for k in j:
                    f.write(str(k) + " ")
                f.write("\t")
            f.write("\n")
        for b in self.biases:
            for i in b:
                for k in i:
                    f.write(str(k) + " ")
            f.write("\n")
        f.close()

    @staticmethod
    def load(location):
        f = open(location, "r")
        lines = f.readlines()
    
        number_of_nodes = np.vectorize(lambda x: int(x))( lines[0].strip().split() )
        weigths = []
        for i in range(1, len(number_of_nodes)):
            m = lines[i].strip().split("\t")
            for j in range(len(m)):
                m[j] = m[j].split()
            m = np.vectorize(lambda x: float(x))(np.matrix(m))
            weigths.append(m)
        biases = []
        for i in range(len(number_of_nodes), len(lines)):
            b = lines[i].strip().split("\t")
            for j in range(len(b)):
                b[j] = b[j].split()
            b = np.vectorize(lambda x: float(x))(np.matrix(b))
            biases.append(b)
        nn = Mlp()
        for i in range(len(number_of_nodes)):
            if i > 0:
                nn.add_layer(number_of_nodes[i], weigths[i-1], biases[i-1])
            else:
                nn.add_layer(number_of_nodes[i])
        return nn

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
        outputs = [np.matrix(inp).T]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(self.sigmoid((np.dot(self.weights[i], outputs[-1]) + self.biases[i])))

        return outputs

    # This training function uses stochastic gradient descent instead of batch
    def train(self, inp, targets):
        targets = np.matrix(targets).T
        # Calculate output with given input
        outputs = self.feed_forward(inp)

        # Calculate each layer error
        errors = [np.subtract(targets, outputs[-1])]
        for i in range(len(self.weights) - 1):
            errors.insert(0, np.dot(self.weights[-1-i].T, errors[0]))

        for i in range(len(self.weights)):
            # Calculate gradient and weight correction
            gradient = np.multiply(errors[-1-i], np.multiply(outputs[-1-i], (1 - outputs[-1-i])))
            gradient *= self.learning_rate
            self.biases[-1-i] += gradient
            delta_w  = np.dot(gradient, outputs[-2-i].T)
            self.weights[-1-i] += delta_w

    def predict(self, inp):
        output = self.feed_forward(inp)[-1]
        output = dict(enumerate(output.A1))
        out_class = max(output.items(), key=operator.itemgetter(1))[0]
        out_prob = output[out_class]
        
        return out_class, out_prob
