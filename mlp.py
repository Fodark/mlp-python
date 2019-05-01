import numpy as np
import math
import operator
import re

class Mlp:
    def __init__(self, init_nodes: int = 0, learning_rate: float = .2) -> None:
        self.number_of_nodes = []
        if init_nodes > 0:
            self.number_of_nodes.append(init_nodes)
        self.weights = []
        self.biases = []
        self.functions = []
        self.learning_rate = learning_rate

    def add_layer(self, number_of_nodes: int, weights = None, bias = None, function="sigmoid"):
        self.number_of_nodes.append(number_of_nodes)
        if not weights is None:
            self.weights.append(weights)
            self.functions.append(function)
        elif len(self.number_of_nodes) > 1:
            self.weights.append(np.random.randn(self.number_of_nodes[-1], self.number_of_nodes[-2]) * np.sqrt(2 / (self.number_of_nodes[-1] + self.number_of_nodes[-2])))
            self.functions.append(function)

        if not bias is None:
            self.biases.append(bias)
        elif len(self.number_of_nodes) > 1:
            self.biases.append(np.random.uniform(0, 0, size=(number_of_nodes, 1)))
    
    def save(self, location):
        f = open(location, "w+")
        for i in self.number_of_nodes:
            f.write(str(i) + " ")
        f.write("\t")
        for i in self.functions:
            f.write(i + " ")
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
        f_l = lines[0].strip()
        f_l = re.split(r'\t+', f_l)
        number_of_nodes = np.vectorize(lambda x: int(x))( f_l[0].split() )
        functions = f_l[1].split()
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
            b = np.vectorize(lambda x: float(x))(np.matrix(b).T)
            biases.append(b)
        nn = Mlp()
        for i in range(len(number_of_nodes)):
            if i > 0:
                nn.add_layer(number_of_nodes=number_of_nodes[i], weights=weigths[i-1], bias=biases[i-1], function=functions[i-1])
            else:
                nn.add_layer(number_of_nodes[i])
        return nn

    @staticmethod
    def soft_plus(x):
        sp = np.vectorize(lambda y: math.log(1 + math.exp(y)))
        return sp(x)

    @staticmethod
    def relu(x):
        re = np.vectorize(lambda y: max(0, y))
        return re(x)

    @staticmethod
    def sigmoid(x):
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)
    
    @staticmethod
    def squash(x, function):
        if function == "sigmoid":
            return Mlp.sigmoid(x)
        elif function == "soft_plus":
            return Mlp.soft_plus(x)
        elif function == "relu":
            return Mlp.relu(x)

    @staticmethod
    def derivative(x, function):
        if function == "sigmoid":
            return np.multiply(x, (1-x))
        elif function == "soft_plus":
            return Mlp.sigmoid(x)
        elif function == "relu":
            d_relu = np.vectorize(lambda y: 1 if y > 0 else 0)
            return d_relu(x)

    def feed_forward(self, inp):
        outputs = [np.matrix(inp).T]

        for i in range(len(self.number_of_nodes) - 1):
            outputs.append(Mlp.squash((np.dot(self.weights[i], outputs[-1]) + self.biases[i]), self.functions[i]))

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
            gradient = np.multiply(errors[-1-i], Mlp.derivative(outputs[-1-i], self.functions[-1-i]))
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
