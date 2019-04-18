# Simple MLP
MLP (Multi-Layer Perceptron) is an ANN (Artificial Neural Network) which has its fundamentals in human brain, where each neuron (here perceptron or node) fires an output depending on the input and its internal weights, and then squashing it through a function which contrains the output range.

Organizing these neurons in layers permits to determine non-linear relationships between inputs.

## Structure of the MLP
Only one hidden layer, with weights initialized with normal distribution values.

The number of nodes of input, hidden layer and output layer are configurable when instantiating the network.

The train functions performs a stochastic gradient descent but I'm planning to implement also batch gradient descent.

The squashing function used is a sigmoid.

## Dependencies
- Python3
- numpy, for matrix computation
- math, for exponential function

## Demo
The demo file contains an example of training the network to perform XOR operations, which is not a linear-separable problem and thus needs a MLP and a second example for classifying hand-written numeric characters provided as 28x28 pixel matrix (786 inputs).

## Performance
For the OCR part the results are:

- Training set: **5000** samples - Testing set: **1000** samples
    - Recall: 86% 
    - Precision: 87%
    - **Accuracy:** 87%

- Training set: **10000** samples - Testing set: **2000** samples
    - Recall: 89%
    - Precision: 89%
    - **Accuracy:** 89%
    
- Training set: **20000** samples - Testing set: **4000** samples
    - Recall: 91%
    - Precision: 91%
    - **Accuracy:** 91%

The dataset used is available [here](https://www.kaggle.com/bagusn1367/mnist-data/version/1#train.csv)

Run the demo by typing in your console:

```python demo.py```
