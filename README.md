# Simple MLP
MLP (Multi-Layer Perceptron) is an ANN (Artificial Neural Network) which has its fundamentals in human brain, where each neuron (here perceptron or node) fires an output depending on the input and its internal weights, and then squashing it through a function which contrains the output range.

Organizing these neurons in layers permits to determine non-linear relationships between inputs.

## Structure of the MLP
The network starts empty or only input layer if specified.

Calling `add_layer(number_of_nodes)` allows to add a new layer as the last, optional weigths and bias for this layer can be specified.

Calling `save(file_name)` allows to save a file containing the network structure which can be loaded with the static method `Mlp.load(file_name)`

The train functions performs a stochastic gradient descent but I'm planning to implement also batch gradient descent.

The squashing function used is a sigmoid.

## Dependencies
- Python3
- numpy, for matrix computation
- pandas, for loading external data
- math, for exponential function

## Demo
The demo file contains an example of training the network to perform XOR operations, which is not a linear-separable problem and thus needs a MLP and a second example for classifying hand-written numeric characters provided as 28x28 pixel matrix (786 inputs).

## Performance
For the OCR part the results are:

**One hidden layer with 50 nodes**

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
    
**Two hidden layer, first 150 nodes, second 50 nodes**

- Training set: **50000** samples - Testing set: **5000** samples
    - Recall: 95% 
    - Precision: 95%
    - **Accuracy:** 95%

The dataset used is available [here](https://www.kaggle.com/bagusn1367/mnist-data/version/1#train.csv)

Run the demo by typing in your console:

```python demo.py```
