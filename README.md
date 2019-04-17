# Simple MLP
MLP (Multi-Layer Perceptron) is an ANN (Artificial Neural Network) which has its fundamentals in human brain, where each neuron (here perceptron or node) fires an output depending on the input and its internal weights, and then squashing it through a function which contrains the output range.

Organizing these neurons in layers permits to determine non-linear relationships between inputs.

## Structure of the MLP
Only one hidden layer, with weights initialized with random values.

The number of nodes of input, hidden layer and output layer are configurable when instantiating the network.

The train functions performs a stochastic gradient descent but I'm planning to implement also batch gradient descent.

The squashing function used is a sigmoid.

## Dependencies
- Python3
- numpy, for matrix computation
- math, for exponential function

## Demo
The demo file contains an example of training the network to perform XOR operations, which is not a linear-separable problem and thus needs a MLP.

The networks performs poorly sometimes because of random starting weights and biases.

Run the demo by typing in your console:

```python demo.py```